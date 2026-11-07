#!/usr/bin/env python3
"""Exact TT application of G^{-1} using the element structure of the DPG Gram matrix, at a cost independent of the
number of x and t cells.

G is block diagonal over the 4D elements, and on uniform x and t meshes the element block G_K depends on K only through
its velocity cell v (gram_block_sharing.py).  Order the TT modes cells first, then local basis:
    (c_x, c_t, c_ux, c_uz, b_x, b_t, b_ux, b_uz).
A TT vector z with rank r_cb at the cells|basis cut restricts to element K as z_K = sum_a c_a(K) Z_a (Z_a: r_cb fixed
local vectors), so y = G^{-1} z restricts to y_K = sum_a c_a(K) G_v^{-1} Z_a: n_v * r_cb local solves, and y is the TT
    sum_v [cell cores with c_ux, c_uz restricted to v] (x) [G_v^{-1} Z],
built exactly (rank n_v * r_cb at the cut) and then rounded.  Nothing is evaluated per element.

Usage: tt_ginv_structured.py <prefix> [--tau 1e-6]   (element-block dump; prints errors vs the element-wise G^{-1} z)
"""
import argparse, os, sys, time
import numpy as np, scipy.linalg as sla, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schur_krylov_ranks import load_sparse


def tt_svd(t, tol):
    """TT-SVD of a full tensor (first index fastest); relative Frobenius tolerance; returns cores (r_{k-1}, n_k, r_k)."""
    sizes = t.shape; d = len(sizes); norm = np.linalg.norm(t)
    delta = tol * norm / np.sqrt(max(d - 1, 1)); cores = []; r = 1; C = t.reshape(sizes[0], -1, order='F')
    for k in range(d - 1):
        C = C.reshape(r * sizes[k], -1, order='F'); U, s, Vt = np.linalg.svd(C, full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]; rk = max(1, int(np.sum(tail > delta)))
        cores.append(U[:, :rk].reshape(r, sizes[k], rk, order='F')); C = s[:rk, None] * Vt[:rk]; r = rk
    cores.append(C.reshape(r, sizes[-1], 1, order='F')); return cores


def tt_full(cores):
    """Contract TT cores; a left rank r0 > 1 becomes a leading index (shape [r0] + sizes), first index fastest."""
    r0 = cores[0].shape[0]
    out = cores[0].reshape(r0 * cores[0].shape[1], -1, order='F')
    for G in cores[1:]:
        out = (out @ G.reshape(G.shape[0], -1, order='F')).reshape(-1, G.shape[2], order='F')
    sizes = [G.shape[1] for G in cores]
    return out.reshape(([r0] if r0 > 1 else []) + sizes, order='F')


def tt_round(cores, tol):
    """Standard TT rounding: right-to-left orthogonalization, then left-to-right truncated SVDs."""
    cores = [c.copy() for c in cores]; d = len(cores)
    for k in range(d - 1, 0, -1):
        r0, n, r1 = cores[k].shape
        Q, R = np.linalg.qr(cores[k].reshape(r0, n * r1, order='F').T)
        cores[k] = Q.T.reshape(-1, n, r1, order='F')
        cores[k - 1] = np.einsum('anb,cb->anc', cores[k - 1], R)
    norm = np.linalg.norm(cores[0]); delta = tol * norm / np.sqrt(max(d - 1, 1))
    for k in range(d - 1):
        r0, n, r1 = cores[k].shape
        U, s, Vt = np.linalg.svd(cores[k].reshape(r0 * n, r1, order='F'), full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]; rk = max(1, int(np.sum(tail > delta)))
        cores[k] = U[:, :rk].reshape(r0, n, rk, order='F')
        cores[k + 1] = np.einsum('ab,bnc->anc', s[:rk, None] * Vt[:rk], cores[k + 1])
    return cores


class StructuredGinv:
    """G^-1 applied to TT vectors (cells-first mode order) using one Cholesky factor per velocity cell."""
    def __init__(self, prefix, blk, modes):
        info = [int(x) for x in open(prefix + "_Gelem.txt").read().split()]
        self.cells = cells = info[0:4]; ne = int(np.prod(cells)); tsz = modes[0]
        self.nbt = nbt = [tsz[d] // cells[d] for d in range(4)]; self.shape8 = cells + nbt
        grid = np.indices(self.shape8).reshape(8, -1, order='F')
        self.gidx = np.zeros(grid.shape[1], dtype=np.int64); stride = 1
        for d in range(4):
            self.gidx += (grid[d] * nbt[d] + grid[4 + d]) * stride; stride *= tsz[d]
        self.Gv_f = {}
        for k, (_, Gk) in enumerate(blk[0, 0]):
            c = np.unravel_index(k, cells, order='F')
            if (c[2], c[3]) not in self.Gv_f: self.Gv_f[(c[2], c[3])] = sla.cho_factor(0.5 * (Gk + Gk.T))
    def to_tt(self, z, tau):
        return tt_svd(z[self.gidx].reshape(self.shape8, order='F'), tau)
    def from_tt(self, cores):
        y = np.empty(len(self.gidx)); y[self.gidx] = tt_full(cores).reshape(-1, order='F'); return y
    def apply(self, cores, tau):
        cells, nbt = self.cells, self.nbt; rcb = cores[3].shape[2]
        Z = tt_full([cores[4]] + cores[5:]).reshape(rcb, -1, order='F')
        C3, C4 = cores[2], cores[3]; nux, nuz = cells[2], cells[3]; r2, r3 = C3.shape[0], C3.shape[2]
        K3 = np.zeros((r2, nux, nux * r3))
        for i in range(nux): K3[:, i, i * r3:(i + 1) * r3] = C3[:, i, :]
        K4 = np.zeros((nux * r3, nuz, nux * nuz * rcb)); W = np.zeros((nux * nuz * rcb, Z.shape[1]))
        for i in range(nux):
            for j in range(nuz):
                col = (j * nux + i) * rcb
                K4[i * r3:(i + 1) * r3, j, col:col + rcb] = C4[:, j, :]
                W[col:col + rcb] = sla.cho_solve(self.Gv_f[(i, j)], Z.T).T
        Wc = tt_svd(W.reshape([W.shape[0]] + nbt, order='F'), 1e-15)
        return tt_round([cores[0], cores[1], K3, np.einsum('abc,cd->abd', K4, Wc[0][0])] + Wc[1:], tau)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix"); ap.add_argument("--tau", type=float, default=1e-6)
    a = ap.parse_args()
    blk, rhs, modes, nb = load_sparse(a.prefix)
    info = [int(x) for x in open(a.prefix + "_Gelem.txt").read().split()]
    cells = info[0:4]; ne = int(np.prod(cells)); tsz = modes[0]; nbt = [tsz[d] // cells[d] for d in range(4)]
    idx = np.fromfile(a.prefix + "_Gelem_idx.bin", dtype=np.int32).reshape(ne, -1)
    Gblocks = [Gk for _, Gk in blk[0, 0]]
    B = sp.hstack([blk[0, j] for j in range(1, nb)]).tocsr()

    # global test index (4-mode, mode d index = cell_d * nbt_d + b_d, first mode fastest) <-> 8-mode (cells, basis)
    shape8 = cells + nbt
    grid = np.indices(shape8).reshape(8, -1, order='F')
    gidx = np.zeros(grid.shape[1], dtype=np.int64); stride = 1
    for d in range(4):
        gidx += (grid[d] * nbt[d] + grid[4 + d]) * stride; stride *= tsz[d]
    # element k's local ordering: check it is Fortran order of (b_x, b_t, b_ux, b_uz) within the cell
    k = ne // 2; ck = np.unravel_index(k, cells, order='F')
    loc = np.indices(nbt).reshape(4, -1, order='F'); exp = np.zeros(loc.shape[1], dtype=np.int64); stride = 1
    for d in range(4):
        exp += (ck[d] * nbt[d] + loc[d]) * stride; stride *= tsz[d]
    assert np.array_equal(idx[k], exp), "element-local ordering is not Fortran order of the local basis indices"

    # element classes: G_K must depend only on the velocity cell (uniform x, t); check and collect G_v
    nv = cells[2] * cells[3]; Gv = {}; worst = 0.0
    for k in range(ne):
        c = np.unravel_index(k, cells, order='F'); v = (c[2], c[3])
        if v not in Gv: Gv[v] = Gblocks[k]
        else: worst = max(worst, np.linalg.norm(Gblocks[k] - Gv[v]) / np.linalg.norm(Gv[v]))
    print(f"cells {cells}, {ne} elements, {nv} velocity-cell Gram blocks; max deviation within a velocity cell {worst:.1e}")
    Gv_f = {v: sla.cho_factor(0.5 * (G + G.T)) for v, G in Gv.items()}

    def exact_Ginv(z):
        y = np.empty_like(z)
        for k in range(ne):
            c = np.unravel_index(k, cells, order='F'); y[idx[k]] = sla.cho_solve(Gv_f[(c[2], c[3])], z[idx[k]])
        return y

    # test vectors: z = l (the Riesz RHS) and z = B x for a smooth trial vector (x = c = B^T G^-1 l)
    l = rhs[0]; xs = B.T @ exact_Ginv(l)
    for name, z in (("l", l), ("B c", B @ xs)):
        t0 = time.time()
        z8 = z[gidx].reshape(shape8, order='F')
        cores = tt_svd(z8, a.tau)                                  # setup only: in a TT code z is already a TT
        ranks = [c.shape[2] for c in cores[:-1]]; rcb = ranks[3]
        t1 = time.time()
        # structured G^{-1}: basis part Z (r_cb x nloc), one solve block per velocity cell
        Z = tt_full([cores[4]] + cores[5:]).reshape(rcb, -1, order='F')        # r_cb x 625 (b_x fastest)
        new = [cores[0], cores[1]]
        C3, C4 = cores[2], cores[3]; nux, nuz = cells[2], cells[3]; r2, r3 = C3.shape[0], C3.shape[2]
        # c_ux core: carry the u_x cell index in the rank: (r2, nux, nux*r3)
        K3 = np.zeros((r2, nux, nux * r3))
        for i in range(nux): K3[:, i, i * r3:(i + 1) * r3] = C3[:, i, :]
        # c_uz core: carry (i, j) in the rank: (nux*r3, nuz, nux*nuz*rcb)
        K4 = np.zeros((nux * r3, nuz, nux * nuz * rcb))
        for i in range(nux):
            for j in range(nuz):
                col = (j * nux + i) * rcb
                K4[i * r3:(i + 1) * r3, j, col:col + rcb] = C4[:, j, :]
        # basis part: rows (i, j, a) -> G_v^{-1} Z_a
        W = np.zeros((nux * nuz * rcb, Z.shape[1]))
        for i in range(nux):
            for j in range(nuz):
                col = (j * nux + i) * rcb
                W[col:col + rcb] = sla.cho_solve(Gv_f[(i, j)], Z.T).T
        Wc = tt_svd(W.reshape([W.shape[0]] + nbt, order='F'), 1e-15)           # exact split of the basis part
        first = np.einsum('abc,cd->abd', K4, Wc[0][0])          # contract the stacked (i, j, a) index into the c_uz core
        ycores = tt_round(new + [K3, first] + Wc[1:], a.tau)
        t2 = time.time()
        y_tt = tt_full(ycores).reshape(-1, order='F'); y = np.empty_like(z); y[gidx] = y_tt
        y_ex = exact_Ginv(z)
        err = np.linalg.norm(y - y_ex) / np.linalg.norm(y_ex)
        print(f"z = {name}: TT ranks (cells|basis order) {ranks}; structured G^-1: {nv * rcb} local solve columns, "
              f"y ranks {[c.shape[2] for c in ycores[:-1]]}, rel. error vs element-wise {err:.1e}, "
              f"time {t2 - t1:.2f} s (TT-SVD of z {t1 - t0:.2f} s, setup only)", flush=True)


if __name__ == "__main__":
    main()
