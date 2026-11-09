#!/usr/bin/env python3
"""Can the structured G^{-1} avoid storing dense element factors?  Iteration counts of preconditioned (block) CG for the
local solves G_v X = Z^T that the TT Schur-Chebyshev solver performs every iteration, on the real right-hand sides.

Runs the structured Chebyshev iteration (dense Cholesky for the actual solves) and, every --every iterations, for
--sample velocity cells, re-solves the local systems by PCG to --rtol (max relative residual over the columns):
  cold: X0 = 0;   warm: X0 = X_prev Q, Q = argmin |Z_prev^T Q - Z^T|  (previous local solutions re-expressed in the new
  local vectors, as a TT code could do).
Preconditioners: jacobi; kron1 = nearest Kronecker product G_v ~ A_3 (x) A_2 (x) A_1 (x) A_0 (higher-order power method on
the rearranged block), applied by 1D solves.  Also reports the Kronecker rank R of G_v over the velocity cells (rank of
the stacked blocks), i.e. how many local operators span all blocks.

Usage: local_solve_study.py <prefix> [--iters 60] [--every 10] [--sample 6] [--tau 1e-6] [--rtol 1e-8]
"""
import argparse, os, sys
import numpy as np, scipy.linalg as sla, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schur_krylov_ranks import load_sparse
from tt_ginv_structured import StructuredGinv, tt_full, tt_round


def nearest_kronecker(G, nb):
    """G (n x n, local index first mode fastest) ~ kron(A_3, A_2, A_1, A_0) by the higher-order power method."""
    d = len(nb)
    T = G.reshape(nb + nb, order='F')                         # (i0..i3, j0..j3)
    T = np.transpose(T, [k for m in range(d) for k in (m, d + m)]).reshape([n * n for n in nb], order='F')
    A = [np.eye(n).reshape(-1, order='F') for n in nb]
    for _ in range(30):
        for m in range(d):
            v = T                                                 # contract all modes but m (highest first)
            for q in sorted([q for q in range(d) if q != m], reverse=True):
                v = np.tensordot(v, A[q], axes=([q], [0]))
            A[m] = v / np.linalg.norm(v)
    scale = T
    for q in reversed(range(d)): scale = np.tensordot(scale, A[q], axes=([q], [0]))
    mats = [a.reshape(n, n, order='F') for a, n in zip(A, nb)]
    mats = [0.5 * (M + M.T) for M in mats]
    mats = [M if np.trace(M) > 0 else -M for M in mats]
    mats[0] = mats[0] * abs(float(scale))
    return mats


def kron_solve(mats, R):
    """Apply (A_3 (x) A_2 (x) A_1 (x) A_0)^{-1} to the columns of R (local index first mode fastest)."""
    nb = [M.shape[0] for M in mats]; m = R.shape[1]
    X = R.reshape(nb + [m], order='F')
    for q, M in enumerate(mats):
        X = np.moveaxis(np.tensordot(np.linalg.inv(M), X, axes=([1], [q])), 0, q)
    return X.reshape(-1, m, order='F')


def pcg(G, B, X0, prec, rtol, maxit=2000):
    X = X0.copy(); R = B - G @ X; nb = np.linalg.norm(B, axis=0) + 1e-300
    Z = prec(R); P = Z.copy(); rz = np.sum(R * Z, axis=0)
    for k in range(maxit + 1):
        if np.max(np.linalg.norm(R, axis=0) / nb) < rtol: return k
        Q = G @ P; al = rz / np.sum(P * Q, axis=0)
        X += P * al; R -= Q * al; Z = prec(R); rzn = np.sum(R * Z, axis=0)
        P = Z + P * (rzn / rz); rz = rzn
    return maxit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix"); ap.add_argument("--iters", type=int, default=60); ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--sample", type=int, default=6); ap.add_argument("--tau", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-8)
    a = ap.parse_args()
    blk, rhs, modes, nb = load_sparse(a.prefix)
    B = sp.hstack([blk[0, j] for j in range(1, nb)]).tocsr(); n = B.shape[1]
    elems = [(idx, sla.cho_factor(0.5 * (G + G.T))) for idx, G in blk[0, 0]]
    def Ginv_full(v):
        o = np.empty_like(v)
        for idx, f in elems: o[idx] = sla.cho_solve(f, v[idx])
        return o
    def S(v): return B.T @ Ginv_full(B @ v)
    dS = np.zeros(n)
    for idx, f in elems:
        Bd = B[idx].toarray(); dS += np.einsum('ij,ij->j', Bd, sla.cho_solve(f, Bd))
    sg = StructuredGinv(a.prefix, blk, modes)
    nbt = sg.nbt; nv = len(sg.Gv_f)
    Gv = {}
    for k, (_, Gk) in enumerate(blk[0, 0]):
        c = np.unravel_index(k, sg.cells, order='F')
        Gv.setdefault((c[2], c[3]), 0.5 * (Gk + Gk.T))
    keys = sorted(Gv)
    # Kronecker rank of the family {G_v}
    Fam = np.array([Gv[k].reshape(-1) for k in keys])
    sv = np.linalg.svd(Fam, compute_uv=False)
    print(f"{nv} velocity-cell blocks of size {Gv[keys[0]].shape[0]}; rank of the family {{G_v}} at 1e-4/1e-8/1e-12: "
          + "/".join(str(int(np.sum(sv > t * sv[0]))) for t in (1e-4, 1e-8, 1e-12)), flush=True)
    kron = {k: nearest_kronecker(Gv[k], nbt) for k in keys}
    for k in keys[:2]:
        P = kron[k][3]
        for M in reversed(kron[k][:3]): P = np.kron(P, M)
        ev = np.linalg.eigvals(np.linalg.solve(P, Gv[k])).real
        print(f"  cell {k}: cond G_v {np.linalg.cond(Gv[k]):.1e}; kron1-preconditioned spectrum [{ev.min():.2e}, {ev.max():.2e}]", flush=True)
    sample = [keys[i] for i in np.linspace(0, nv - 1, a.sample).astype(int)]
    precs = {'jacobi': lambda k: (lambda R, d=np.diag(Gv[k]): R / d[:, None]),
             'kron1': lambda k: (lambda R, m=kron[k]: kron_solve(m, R))}
    # Chebyshev on S with Jacobi, structured G^-1 (dense factors), recording local systems
    c = B.T @ Ginv_full(rhs[0]); nc = np.linalg.norm(c)
    v = np.random.default_rng(0).standard_normal(n)
    for _ in range(30): w = S(v / np.sqrt(dS)) / np.sqrt(dS); lmax = np.linalg.norm(w) / np.linalg.norm(v); v = w
    lmax *= 1.05; lmin = 0.03 * lmax / 10        # rough; only the right-hand sides matter here
    x = np.zeros(n); r = c.copy(); th, de = (lmax + lmin) / 2, (lmax - lmin) / 2; sig = th / de; rho = 1 / sig; dvec = (r / dS) / th
    prev = {}
    print(f"--- local PCG iterations to {a.rtol:g} (max over columns), sampled velocity cells {sample}", flush=True)
    for k in range(1, a.iters + 1):
        x = x + dvec
        zc = sg.to_tt(B @ x, a.tau)
        rcb = zc[3].shape[2]
        Z = tt_full([zc[4]] + zc[5:]).reshape(rcb, -1, order='F')
        if k % a.every == 0 or k == 1:
            out = []
            for key in sample:
                Rhs = Z.T; G = Gv[key]; Xex = sla.cho_solve(sg.Gv_f[key], Rhs)
                line = []
                for pname, pf in precs.items():
                    cold = pcg(G, Rhs, np.zeros_like(Rhs), pf(key), a.rtol)
                    if key in prev:
                        Zp, Xp = prev[key]; Q = np.linalg.lstsq(Zp.T, Rhs, rcond=None)[0]
                        warm = pcg(G, Rhs, Xp @ Q, pf(key), a.rtol)
                    else: warm = None
                    line.append(f"{pname} {cold}/{warm}")
                out.append(f"{key}: " + ", ".join(line))
                prev[key] = (Z, Xex)
            print(f"  it {k:3d} r_cb {rcb}: " + " | ".join(out), flush=True)
        else:
            for key in sample: prev[key] = (Z, sla.cho_solve(sg.Gv_f[key], Z.T))
        y = sg.from_tt(sg.apply(zc, a.tau)); r = c - B.T @ y
        rn = 1 / (2 * sig - rho); dvec = rn * rho * dvec + (2 * rn / de) * (r / dS); rho = rn
    print("(cold/warm iteration counts; warm = previous iteration's local solutions re-expressed in the new local vectors)")


if __name__ == "__main__":
    main()
