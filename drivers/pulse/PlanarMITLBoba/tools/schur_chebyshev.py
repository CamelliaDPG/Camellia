#!/usr/bin/env python3
"""Chebyshev semi-iteration on the DPG Schur complement S = B^T G^{-1} B that applies S only to ITERATES.

Why: a TT code must form y = G^{-1} B v for every v it applies S to.  For CG search directions y is nearly full rank
(G^{-1} B p: 285-400 of 400 on (16,1,8,4)), but for an iterate y = G^{-1} B x is as compressible as G^{-1} l
(rank ~36 at 1e-4, ~85 at 1e-6).  Chebyshev iteration needs no inner products with S p:
    r_k = c - T_ty( S x_k )          (S applied to the rounded iterate; G^{-1} B x_k rounded relative to its own norm)
    x_{k+1} = T_tx( x_k + alpha_k D^{-1} r_k + beta_k (x_k - x_{k-1}) )
with the standard three-term Chebyshev coefficients for the spectrum [lmin, lmax] of D^{-1} S.  lmax: power iteration;
lmin: Ritz values from a short Lanczos (Jacobi-PCG) run.  Errors are reported with the exact S.

Usage: schur_chebyshev.py <prefix> [--tx 1e-5] [--ty 1e-5] [--iters 600] [--lanczos 40] [--lmin-safety 0.8]
"""
import argparse, os, sys
import numpy as np, scipy.linalg as sla, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schur_krylov_ranks import load_sparse, tt_round, Trial


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix"); ap.add_argument("--tx", type=float, default=1e-5, help="iterate rounding (0: none)")
    ap.add_argument("--ty", type=float, default=1e-5, help="rounding of G^-1 B x, relative to its norm (0: none)")
    ap.add_argument("--iters", type=int, default=600); ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--lanczos", type=int, default=40); ap.add_argument("--lmin-safety", type=float, default=0.8)
    ap.add_argument("--rtol", type=float, default=1e-6)
    a = ap.parse_args()

    blk, rhs, modes, nb = load_sparse(a.prefix)
    B = sp.hstack([blk[0, j] for j in range(1, nb)]).tocsr()
    elems = [(idx, sla.cho_factor(0.5 * (G + G.T))) for idx, G in blk[0, 0]]; Bk = [B[idx] for idx, _ in elems]
    n = B.shape[1]; T = Trial(modes, nb); ts = modes[0]
    def Ginv(v):
        o = np.empty_like(v)
        for idx, f in elems: o[idx] = sla.cho_solve(f, v[idx])
        return o
    def S(v):
        o = np.zeros(n)
        for (idx, f), Bi in zip(elems, Bk): o += Bi.T @ sla.cho_solve(f, Bi @ v)
        return o
    yr = []
    def S_tt(x):                               # S x as a TT code would form it: y = G^-1 B x rounded, then B^T y
        y = Ginv(B @ x)
        if a.ty > 0:
            y, _ = tt_round(y, ts, a.ty); yr.append(max(tt_round.last_ranks))
        return B.T @ y
    dS = np.zeros(n)
    for (idx, f), Bi in zip(elems, Bk):
        Bd = Bi.toarray(); dS += np.einsum('ij,ij->j', Bd, sla.cho_solve(f, Bd))
    c = B.T @ Ginv(rhs[0]); nc = np.linalg.norm(c)

    # spectrum bounds of D^-1 S: lmax by power iteration, lmin from Lanczos (Jacobi-PCG coefficients)
    v = np.random.default_rng(0).standard_normal(n)
    for _ in range(30):
        w = S(v / np.sqrt(dS)) / np.sqrt(dS); lmax = np.linalg.norm(w) / np.linalg.norm(v); v = w
    lmax *= 1.05
    x = np.zeros(n); r = c.copy(); z = r / dS; p = z.copy(); rz = r @ z; al_prev = None; be_prev = 0; d, e = [], []
    for k in range(a.lanczos):
        Sp = S(p); al = rz / (p @ Sp); x += al * p; r -= al * Sp; z = r / dS; rzn = r @ z; be = rzn / rz
        d.append(1 / al + (be_prev / al_prev if al_prev else 0)); e.append(np.sqrt(be) / al)
        p = z + be * p; rz = rzn; al_prev = al; be_prev = be
    ritz = np.linalg.eigvalsh(np.diag(d) + np.diag(e[:-1], 1) + np.diag(e[:-1], -1))
    lmin = ritz[0] * a.lmin_safety
    print(f"D^-1 S spectrum: Ritz [{ritz[0]:.3e}, {ritz[-1]:.3f}] from {a.lanczos} Lanczos steps; power lmax {lmax/1.05:.3f}; "
          f"using [{lmin:.3e}, {lmax:.3f}], kappa {lmax/lmin:.0f}", flush=True)

    # Chebyshev semi-iteration (Saad, Alg. 12.1 form), S applied only to iterates
    theta, delta = (lmax + lmin) / 2, (lmax - lmin) / 2; sigma = theta / delta
    x = np.zeros(n); r = c.copy(); rho = 1 / sigma; dvec = (r / dS) / theta
    print(f"--- Chebyshev, tx {a.tx:g}, ty {a.ty:g}", flush=True)
    for k in range(1, a.iters + 1):
        x = x + dvec
        if a.tx > 0: x, _ = T.round(x, a.tx)
        r = c - S_tt(x)
        rho_new = 1 / (2 * sigma - rho)
        dvec = rho_new * rho * dvec + (2 * rho_new / delta) * (r / dS)
        rho = rho_new
        if k % a.every == 0 or k == 1:
            true = np.linalg.norm(c - S(x)) / nc
            print(f"  it {k:4d}  true |r|/|c| {true:.2e}  rank(x) {T.ranks(x, 1e-4)}"
                  + (f"  rank(G^-1 B x) {yr[-1]} (max so far {max(yr)})" if yr else ""), flush=True)
            if true < a.rtol: break


if __name__ == "__main__":
    main()
