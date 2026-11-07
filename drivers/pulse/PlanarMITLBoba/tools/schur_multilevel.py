#!/usr/bin/env python3
"""Two-level (p-coarsened) preconditioning of CG on the DPG Schur complement S = B^T G^{-1} B, and the TT ranks of the
vectors S must be applied to.

Question: a TT Krylov method must form y = G^{-1} B v for every v it applies S to.  For raw CG directions y has rank
200-370 (schur_krylov_ranks.py).  Does a two-level preconditioner make the search directions smooth enough that y stays
low rank, as it is for the iterate (G^{-1} B u has rank ~30-50)?

Input: an element-block dump (PLANAR_MITL_DUMP_SPARSE) made with HIERARCHICAL field and trace bases, so that within each
cell the 1D trial basis functions are ordered by degree and a p-coarse space is a column selection: keep local degree
<= --kmax in every basis direction (all face indices of a flux block's normal direction).  The coarse operator is the
Galerkin S_c = P^T S P = (B P)^T G^{-1} (B P), assembled element by element and factored directly.

Preconditioners (--precs): jacobi; add = omega D^{-1} + P S_c^{-1} P^T; vcycle = symmetric multiplicative two-level
cycle (nu Jacobi sweeps, coarse correction, nu Jacobi sweeps; S applied inside).  For each, PCG to --rtol (exact
residual), reporting iterations and, every --every iterations, the TT ranks (tau) of the iterate, residual, direction, and
of G^{-1} B v for the vectors v that S was applied to in that iteration.

Usage: schur_multilevel.py <prefix> [--kmax 1] [--precs jacobi,add,vcycle] [--nu 1] [--omega 0.6] [--rtol 1e-6]
"""
import argparse, os, sys, time
import numpy as np, scipy.linalg as sla, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schur_krylov_ranks import load_sparse, tt_round, Trial


def coarse_columns(T, cells, kmax):
    """Indices (into the stacked trial vector) of the p-coarse dofs: local basis index <= kmax on every basis axis."""
    keep = []
    for b, sz in enumerate(T.sizes):
        masks = []
        for d in range(4):
            n = sz[d]
            if cells[d] > 1 and n == cells[d] - 1:       # face axis (normal direction of a flux block): keep all
                masks.append(np.ones(n, bool))
            else:
                nb = n // cells[d]
                masks.append((np.arange(n) % nb) <= kmax)  # index = cell * nb + local, local ordered by degree
        m = np.einsum('i,j,k,l->ijkl', *masks).reshape(-1, order='F')
        keep.append(T.offsets[b] + np.flatnonzero(m))
    return np.concatenate(keep)


def legendre_prolongation(ncells, nb):
    """1D prolongation for Legendre P_0..P_{nb-1} per cell (Camellia's hierarchical L2 basis is exactly these):
    coarse cell k = fine cells 2k, 2k+1; coefficients by exact Legendre projection."""
    from numpy.polynomial import legendre as L
    xg, wg = L.leggauss(nb + 2)
    P = np.zeros((ncells * nb, (ncells // 2) * nb))
    for k in range(ncells // 2):
        for child, shift in ((0, -1.0), (1, 1.0)):
            xc = (xg + shift) / 2                               # fine reference point -> coarse reference point
            for i in range(nb):
                ci = L.legval(xc, np.eye(nb)[i])
                for j in range(nb):
                    P[(2 * k + child) * nb + j, k * nb + i] = (2 * j + 1) / 2 * np.sum(wg * ci * L.legval(xg, np.eye(nb)[j]))
    return P


def h_prolongation(T, cells, axes):
    """Block-diagonal (over trial blocks) Kronecker prolongation coarsening the given axes by cell pairs."""
    blocks = []
    for b, sz in enumerate(T.sizes):
        mats = []
        for d in range(4):
            n = sz[d]
            if cells[d] > 1 and n == cells[d] - 1:                       # face axis
                if d in axes:
                    Pd = np.zeros((n, cells[d] // 2 - 1))
                    for k in range(cells[d] // 2 - 1): Pd[2 * k + 1, k] = 1.0   # coarse face k = fine face 2k+1
                else: Pd = np.eye(n)
            else:
                nbd = n // cells[d]
                Pd = legendre_prolongation(cells[d], nbd) if (d in axes and cells[d] > 1) else np.eye(n)
            mats.append(sp.csr_matrix(Pd))
        blocks.append(sp.kron(mats[3], sp.kron(mats[2], sp.kron(mats[1], mats[0]))))   # first index fastest
    return sp.block_diag(blocks).tocsr()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix"); ap.add_argument("--kmax", type=int, default=1)
    ap.add_argument("--precs", default="jacobi,add,vcycle"); ap.add_argument("--nu", type=int, default=1)
    ap.add_argument("--omega", type=float, default=0.0, help="Jacobi damping (<= 0: 1/lambda_max(D^-1 S))"); ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--iters", type=int, default=400); ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--coarsen", default="p", help="p (degree <= kmax) or h:<axes>, e.g. h:0,2 coarsens x and u_x by cell pairs")
    a = ap.parse_args()

    blk, rhs, modes, nb = load_sparse(a.prefix)
    info = [int(x) for x in open(a.prefix + "_Gelem.txt").read().split()]
    cells = info[0:4]
    B = sp.hstack([blk[0, j] for j in range(1, nb)]).tocsr()
    elems = [(idx, sla.cho_factor(0.5 * (Gk + Gk.T))) for idx, Gk in blk[0, 0]]
    T = Trial(modes, nb); test_sizes = modes[0]; n = B.shape[1]
    Bk = [(idx, B[idx]) for idx, _ in elems]

    def Ginv(v):
        out = np.empty_like(v)
        for idx, f in elems: out[idx] = sla.cho_solve(f, v[idx])
        return out
    applied = []                                   # vectors S was applied to in the current iteration
    def S(v, record=True):
        if record: applied.append(v)
        out = np.zeros(n)
        for (idx, f), (_, Bi) in zip(elems, Bk): out += Bi.T @ sla.cho_solve(f, Bi @ v)
        return out
    dS = np.zeros(n)
    for (idx, f), (_, Bi) in zip(elems, Bk):
        Bd = Bi.toarray(); dS += np.einsum('ij,ij->j', Bd, sla.cho_solve(f, Bd))
    c = B.T @ Ginv(rhs[0])

    # Galerkin coarse operator on the p-coarse columns, element by element
    t0 = time.time()
    if a.coarsen == 'p':
        cc = coarse_columns(T, cells, a.kmax)
        P = sp.csr_matrix((np.ones(len(cc)), (cc, np.arange(len(cc)))), shape=(n, len(cc)))
        label = f"p-coarse, degree <= {a.kmax}"
    else:
        axes = [int(x) for x in a.coarsen.split(':')[1].split(',')]
        P = h_prolongation(T, cells, axes); label = f"h-coarse by cell pairs along axes {axes}"
    nc = P.shape[1]; BP = (B @ P).tocsr()
    rows, cols, vals = [], [], []
    for idx, f in elems:
        Bc = BP[idx].tocsc(); nzc = np.flatnonzero(np.diff(Bc.indptr))
        if len(nzc) == 0: continue
        Bd = Bc[:, nzc].toarray(); blockS = Bd.T @ sla.cho_solve(f, Bd)
        rr, ccol = np.meshgrid(nzc, nzc, indexing='ij'); rows.append(rr.ravel()); cols.append(ccol.ravel()); vals.append(blockS.ravel())
    Sc = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nc, nc))
    lu = spla.splu(Sc)
    print(f"trial dofs {n}, coarse ({label}) {nc} ({nc/n:.1%}); coarse operator nnz {Sc.nnz}, "
          f"assembled+factored in {time.time()-t0:.1f} s", flush=True)
    print(f"Jacobi diagonal TT ranks (tau {a.tau:g}): {T.ranks(dS, a.tau)}", flush=True)
    # largest eigenvalue of D^-1 S (power iteration on the symmetric D^-1/2 S D^-1/2) -> Jacobi damping
    v = np.random.default_rng(0).standard_normal(n)
    for _ in range(30):
        w = S(v / np.sqrt(dS), record=False) / np.sqrt(dS); lam = np.linalg.norm(w) / np.linalg.norm(v); v = w
    if a.omega <= 0: a.omega = 1.0 / lam
    print(f"lambda_max(D^-1 S) ~ {lam:.3f}; Jacobi omega = {a.omega:.3f} (convergent smoother needs < {2/lam:.3f})", flush=True)

    def coarse(r):
        return P @ lu.solve(P.T @ r)
    def jacobi(r): return r / dS
    def add(r): return a.omega * r / dS + coarse(r)
    def vcycle(r):
        x = np.zeros(n)
        for _ in range(a.nu): x = x + a.omega * (r - (S(x) if x.any() else 0)) / dS
        x = x + coarse(r - S(x))
        for _ in range(a.nu): x = x + a.omega * (r - S(x)) / dS
        return x
    precs = {'jacobi': jacobi, 'add': add, 'vcycle': vcycle}

    def yrank(v):
        tt_round(Ginv(B @ v), test_sizes, a.tau); return max(tt_round.last_ranks)
    nc_ = np.linalg.norm(c)
    for name in a.precs.split(','):
        M = precs[name]
        print(f"--- {name}" + (f" (omega {a.omega}, nu {a.nu})" if name != 'jacobi' else ""), flush=True)
        x = np.zeros(n); r = c.copy(); applied.clear(); z = M(r); zapps = list(applied); p = z.copy(); rz = r @ z
        napply = len(applied)
        for k in range(1, a.iters + 1):
            applied.clear()
            Sp = S(p); al = rz / (p @ Sp); x += al * p; r -= al * Sp
            res = np.linalg.norm(r) / nc_
            show = (k % a.every == 0) or k == 1 or res < a.rtol
            if show:
                inner = [yrank(v) for v in zapps] if zapps else []
                print(f"  it {k:4d}  |r|/|c| {res:.2e}  rank(x) {T.ranks(x, a.tau)}  rank(r) {T.ranks(r, a.tau)}"
                      f"  rank(p) {T.ranks(p, a.tau)}  rank(G^-1 B p) {yrank(p)}"
                      + (f"  rank(G^-1 B v) inside M: {inner}" if inner else "")
                      + f"  rank(G^-1 B x) {yrank(x)}", flush=True)
            if res < a.rtol: break
            applied.clear(); z = M(r); zapps = list(applied); napply += 1 + len(zapps)
            rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
        true = np.linalg.norm(c - S(x, record=False)) / nc_
        print(f"  {name}: {k} iterations, true |r|/|c| {true:.2e}, {napply} applications of S", flush=True)


if __name__ == "__main__":
    main()
