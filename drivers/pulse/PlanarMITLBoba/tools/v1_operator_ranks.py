#!/usr/bin/env python3
"""V1 (BoBa status review): TT ranks of the DPG operators versus refinement, for the L^2, N5 (direction-separable) and
graph test norms, with constant or (x, t)-varying prescribed fields.

Ultraweak DPG on a tensor mesh of boxes K (directions x, t, u_x, u_z).  Transport beta = (v_x, 1, a_ux, a_uz),
a_ux = (q/m)(E_x - v_z B_y), a_uz = (q/m) v_x B_y, with E_x(x, t), B_y(x, t).  Per element, with physical scaling:
    graph: G_K = int (beta.grad w)(beta.grad v) + alpha_K w v
    N5:    G_K = alpha_K int w v + sum_d s_d(K)^2 int d_d w d_d v,  s = (v_x,max(K), 1, a_bounds)
    L2:    G_K = alpha_K int w v
alpha_K the transport-scale weight (alpha0 + C sum_d (s_d / h_d)^2).  D_K = int (beta.grad w) phi_f (test x field, the
volume part of the ultraweak form), and the element-local Schur complement S_K = B_K^T G_K^{-1} B_K, B_K = [-D_K, face
terms of the traces t_x, t_t, t_ux, t_uz (degree p)]; S = sum_K S_K.  We measure the TT-matrix ranks of each of the 15
block pairs of blockdiag(S_K) (field-field, and the maximum over all pairs) and, with --ginv, of blockdiag(G_K^{-1}), in the interleaved
layout (c_x, i_x, j_x) (c_t, i_t, j_t) (c_ux, i_ux, j_ux) (c_uz, i_uz, j_uz): the ranks a TT operator would carry.
(Field basis degree p - 1, test degree p + dp, Lagrange at GLL points; ranks do not depend on the per-direction basis.)

Fields ("--fields"): const: E_x = E0, B_y = B0; vary: E_x = E0 (1 + 0.5 cos(pi x)) (1 + 0.5 t / T), B_y = B0 (0.5 + x)
(1 - 0.3 t / T), on x in [0, 1], t in [0, T].  Acceleration bounds for N5 and alpha_K: (|q/m|(max|E| + max|B|),
|q/m| max|B|).

Usage: v1_operator_ranks.py [--p 2] [--dp 1] [--meshes 8,2,8,8;16,2,8,8;...] [--fields const,vary] [--norms L2,N5,graph]
"""
import argparse, itertools, os, sys, time
import numpy as np, scipy.linalg as sla
import numpy.polynomial.legendre as L
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frozen_gram_study as F


def lagrange_at(k, x):
    """Lagrange basis of degree k at GLL nodes, values at x."""
    nodes = F.gll(k)
    V = np.ones((len(x), k + 1))
    for j in range(k + 1):
        for m in range(k + 1):
            if m != j: V[:, j] *= (x - nodes[m]) / (nodes[j] - nodes[m])
    return V


class Spaces:
    def __init__(self, p, dp):
        self.ref = F.Reference(p + dp)                                  # test space and quadrature
        r = self.ref; nq = r.nq
        phi_f = lagrange_at(p - 1, r.q) if p > 1 else np.ones((nq, 1))
        self.nf1 = phi_f.shape[1]
        kron4 = lambda m: np.kron(m[3], np.kron(m[2], np.kron(m[1], m[0])))
        self.PhiF = kron4([phi_f] * 4)                                  # (quadrature points) x (field dofs)
        # traces (degree p, GLL Lagrange) on the faces normal to direction d: B_tr,d = prod_{e != d} (h_e/2) kron4(m),
        # m_d = [-w(-1), +w(+1)] (test values at the two faces; GLL Lagrange: unit vectors), m_e = 1D test-trace mass
        phi_tr = lagrange_at(p, r.q); self.ntr1 = p + 1
        Mtr = r.phi.T @ (r.w[:, None] * phi_tr)
        ends = np.zeros((r.n1, 2)); ends[0, 0] = -1.0; ends[-1, 1] = 1.0
        self.Btr = [kron4([ends if e == d else Mtr for e in range(4)]) for d in range(4)]
        self.kron4 = kron4


def fields(kind, E0, B0, T):
    if kind == "const": return (lambda x, t: E0 + 0 * x), (lambda x, t: B0 + 0 * x)
    return (lambda x, t: E0 * (1 + 0.5 * np.cos(np.pi * x)) * (1 + 0.5 * t / T)), (lambda x, t: B0 * (0.5 + x) * (1 - 0.3 * t / T))


def element_blocks(sp, box, Ef, Bf, qm, norm, alpha0, C, a_bounds, want_ginv):
    r = sp.ref; h = np.array([b[1] - b[0] for b in box]); mid = [(b[0] + b[1]) / 2 for b in box]
    P = [mid[d] + h[d] / 2 * r.Q[d] for d in range(4)]                   # physical quadrature points
    g = np.sqrt(1 + P[2] ** 2 + P[3] ** 2); vx, vz = P[2] / g, P[3] / g
    E, B = Ef(P[0], P[1]), Bf(P[0], P[1])
    beta = [vx, np.ones_like(vx), qm * (E - vz * B), qm * vx * B]
    aK = F.alpha_transport(alpha0, C, h[0], h[1], h[2], h[3], box[2], box[3], a_bounds)
    A = sum((2 * beta[d] / h[d])[:, None] * r.dPhi[d] for d in range(4))
    if norm == "graph": G = A.T @ (r.W[:, None] * A) + aK * r.M
    elif norm == "N5":
        umax = max(abs(box[2][0]), abs(box[2][1])); uzmin = 0.0 if box[3][0] <= 0 <= box[3][1] else min(abs(box[3][0]), abs(box[3][1]))
        s = [umax / np.sqrt(1 + umax ** 2 + uzmin ** 2), 1.0, a_bounds[0], a_bounds[1]]
        G = aK * r.M + sum((2 * s[d] / h[d]) ** 2 * r.K[d][d] for d in range(4))
    else: G = aK * r.M
    D = A.T @ (r.W[:, None] * sp.PhiF)
    vol = np.prod(h) / 16
    # ultraweak: b(f, w) = -(f, beta.grad w) + <t_hat, w>; physical B_K = vol * [-D, (2/h_d) Btr_d]
    Ball = np.hstack([-D] + [(2 / h[d]) * sp.Btr[d] for d in range(4)])
    cf = sla.cho_factor(G)
    S = vol * (Ball.T @ sla.cho_solve(cf, Ball))
    Gi = sla.cho_solve(cf, np.eye(G.shape[0])) / vol if want_ginv else None
    return S, Gi


def tt_ranks(T, tols):
    out = {}
    for tol in tols:
        sizes = T.shape; d = len(sizes); delta = tol * np.linalg.norm(T) / np.sqrt(d - 1)
        Cm = T.reshape(sizes[0], -1); r = 1; ranks = []
        for k in range(d - 1):
            Cm = Cm.reshape(r * sizes[k], -1)
            if Cm.shape[0] > Cm.shape[1]: U, sv, Vt = np.linalg.svd(Cm, full_matrices=False)
            else:                                                       # wide: eig of the small Gram
                w, U = np.linalg.eigh(Cm @ Cm.T); w = np.clip(w[::-1], 0, None); U = U[:, ::-1]; sv = np.sqrt(w); Vt = None
            tail = np.sqrt(np.cumsum(sv[::-1] ** 2))[::-1]; rk = max(1, int(np.sum(tail > delta)))
            ranks.append(rk); Cm = (U[:, :rk].T @ Cm) if Vt is None else sv[:rk, None] * Vt[:rk]; r = rk
        out[tol] = ranks
    return out


def blockdiag_tensor(blocks, cells, na, nb=None):
    """blocks[cx, ct, cux, cuz] (rows: local sizes na[d], columns nb[d], first axis fastest) -> interleaved TT-operator
    tensor with modes (c_d, i_d, j_d), d = x, t, u_x, u_z.  (Assembly of traces from (cell, face) to mesh vertices is a
    mode-wise linear map and does not change these ranks.)"""
    nc = list(cells); na = [na] * 4 if np.isscalar(na) else list(na); nb = na if nb is None else ([nb] * 4 if np.isscalar(nb) else list(nb))
    Tt = blocks.reshape(nc + na[::-1] + nb[::-1])  # [c.., i3,i2,i1,i0, j3,j2,j1,j0] (C order local index)
    perm = []
    for d in range(4): perm += [d, 4 + (3 - d), 8 + (3 - d)]
    Tt = np.transpose(Tt, perm)
    return Tt.reshape([nc[d] * na[d] * nb[d] for d in range(4)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=2); ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--meshes", default="4,2,8,8;8,2,8,8;16,2,8,8;8,2,16,16")
    ap.add_argument("--fields", default="const,vary"); ap.add_argument("--norms", default="L2,N5,graph")
    ap.add_argument("--E0", type=float, default=-1.0); ap.add_argument("--B0", type=float, default=0.5)
    ap.add_argument("--uMax", type=float, default=2.0); ap.add_argument("--T", type=float, default=0.1)
    ap.add_argument("--alpha0", type=float, default=1e2); ap.add_argument("--C", type=float, default=1.6)
    ap.add_argument("--ginv", action="store_true", help="also the ranks of blockdiag(G^{-1})")
    ap.add_argument("--tols", default="1e-2,1e-4,1e-6")
    a = ap.parse_args()
    sp = Spaces(a.p, a.dp); n1t = sp.ref.n1; n1f = sp.nf1; qm = -1.0
    tols = [float(t) for t in a.tols.split(",")]
    print(f"p = {a.p}, dp = {a.dp}: test {n1t}^4, field {n1f}^4; E0 = {a.E0}, B0 = {a.B0}, u in [-{a.uMax}, {a.uMax}]^2, "
          f"T = {a.T}", flush=True)
    for kind in a.fields.split(","):
        Ef, Bf = fields(kind, a.E0, a.B0, a.T)
        xs, ts = np.linspace(0, 1, 101), np.linspace(0, a.T, 11); X, Tm = np.meshgrid(xs, ts)
        Em, Bm = np.abs(Ef(X, Tm)).max(), np.abs(Bf(X, Tm)).max()
        a_bounds = (abs(qm) * (Em + Bm), abs(qm) * Bm)
        for mesh in a.meshes.split(";"):
            cells = [int(c) for c in mesh.split(",")]
            verts = [np.linspace(0, 1, cells[0] + 1), np.linspace(0, a.T, cells[1] + 1),
                     np.linspace(-a.uMax, a.uMax, cells[2] + 1), np.linspace(-a.uMax, a.uMax, cells[3] + 1)]
            for norm in a.norms.split(","):
                t0 = time.time()
                sizes = [[n1f] * 4] + [[2 if e == d else sp.ntr1 for e in range(4)] for d in range(4)]
                offs = np.cumsum([0] + [int(np.prod(z)) for z in sizes]); nS = offs[-1]
                Sb = np.empty(cells + [nS, nS]); Gb = np.empty(cells + [n1t ** 4, n1t ** 4]) if a.ginv else None
                for idx in itertools.product(*[range(c) for c in cells]):
                    box = [(verts[d][idx[d]], verts[d][idx[d] + 1]) for d in range(4)]
                    S, Gi = element_blocks(sp, box, Ef, Bf, qm, norm, a.alpha0, a.C, a_bounds, a.ginv)
                    Sb[idx] = S
                    if a.ginv: Gb[idx] = Gi
                names = ["f", "tx", "tt", "tux", "tuz"]; worst = {t: [0, 0, 0] for t in tols}; ff = None
                for ia, ib in itertools.combinations_with_replacement(range(5), 2):
                    blk = Sb[..., offs[ia]:offs[ia + 1], offs[ib]:offs[ib + 1]]
                    if np.linalg.norm(blk) == 0: continue
                    R = tt_ranks(blockdiag_tensor(np.ascontiguousarray(blk), cells, sizes[ia], sizes[ib]), tols)
                    if ia == ib == 0: ff = R
                    for t in tols: worst[t] = [max(u, v) for u, v in zip(worst[t], R[t])]
                line = (f"  {kind:5s} mesh {mesh:12s} {norm:5s}: S_ff " + "; ".join(f"{t:g}: {ff[t]}" for t in tols)
                        + " | max over the 15 S blocks " + "; ".join(f"{t:g}: {worst[t]}" for t in tols))
                if a.ginv:
                    RG = tt_ranks(blockdiag_tensor(Gb, cells, n1t), tols)
                    line += "  | G^-1 ranks " + "; ".join(f"{t:g}: {RG[t]}" for t in tols)
                print(line + f"  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
