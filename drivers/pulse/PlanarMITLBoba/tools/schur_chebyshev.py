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
from schur_multilevel import h_prolongation
from tt_ginv_structured import StructuredGinv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix"); ap.add_argument("--tx", type=float, default=1e-5, help="iterate rounding (0: none)")
    ap.add_argument("--ty", type=float, default=1e-5, help="rounding of G^-1 B x, relative to its norm (0: none)")
    ap.add_argument("--iters", type=int, default=600); ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--lanczos", type=int, default=40); ap.add_argument("--lmin-safety", type=float, default=0.8)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--coarsen", default="", help="h:<axes> coarse space (cell pairs) for --prec add and/or --x0 coarse")
    ap.add_argument("--prec", default="jacobi", help="jacobi, or add = omega D^-1 + P S_c^-1 P^T (never applies S)")
    ap.add_argument("--continuation", default="", help="comma-separated tolerances, e.g. 1e-4,1e-5,1e-6: converge at "
                    "each (tx = ty = tol) until |r|/|c| < 2 tol or --stage-iters, restarting the recurrence from the iterate")
    ap.add_argument("--stage-iters", type=int, default=200)
    ap.add_argument("--ginv", default="exact", help="exact (element-wise; y rounded in 4-mode TT) or structured (z = B x "
                    "rounded in cells-first TT, structured TT G^-1, y rounded there)")
    ap.add_argument("--write-kron", default="", help="with --prec kron1: write the per-mode diagonal factors here")
    ap.add_argument("--x0", default="zero", help="zero, or coarse = P S_c^-1 P^T c (nested iteration warm start)")
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
    sg = StructuredGinv(a.prefix, blk, modes) if a.ginv == 'structured' else None
    zr, tg = [], []
    def S_tt(x):                               # S x as a TT code would form it: y = G^-1 B x rounded, then B^T y
        if sg is not None:
            import time as _t
            zc = sg.to_tt(B @ x, a.ty if a.ty > 0 else 1e-14); t0 = _t.time()
            yc = sg.apply(zc, a.ty if a.ty > 0 else 1e-14); tg.append(_t.time() - t0)
            zr.append(max(c.shape[2] for c in zc[:-1])); yr.append(max(c.shape[2] for c in yc[:-1]))
            return B.T @ sg.from_tt(yc)
        y = Ginv(B @ x)
        if a.ty > 0:
            y, rk = tt_round(y, ts, a.ty); yr.append(rk)
        return B.T @ y
    dS = np.zeros(n)
    for (idx, f), Bi in zip(elems, Bk):
        Bd = Bi.toarray(); dS += np.einsum('ij,ij->j', Bd, sla.cho_solve(f, Bd))
    c = B.T @ Ginv(rhs[0]); nc = np.linalg.norm(c)

    if a.coarsen:
        import scipy.sparse.linalg as spla
        axes = [int(t) for t in a.coarsen.split(':')[1].split(',')]
        cells = [int(t) for t in open(a.prefix + "_Gelem.txt").read().split()][0:4]
        P = h_prolongation(T, cells, axes); BP = (B @ P).tocsr(); rows, cols, vals = [], [], []
        for idx, f in elems:
            Bc = BP[idx].tocsc(); nzc = np.flatnonzero(np.diff(Bc.indptr))
            if len(nzc) == 0: continue
            Bd = Bc[:, nzc].toarray(); rr, cc_ = np.meshgrid(nzc, nzc, indexing='ij')
            rows.append(rr.ravel()); cols.append(cc_.ravel()); vals.append((Bd.T @ sla.cho_solve(f, Bd)).ravel())
        lu = spla.splu(sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(P.shape[1],) * 2))
        coarse = lambda r: P @ lu.solve(P.T @ r)
        print(f"coarse space h{axes}: {P.shape[1]} dofs ({P.shape[1]/n:.1%})", flush=True)
    # preconditioner M^-1 (symmetric, never applies S)
    lam_d = None
    if a.prec == 'add':
        v = np.random.default_rng(1).standard_normal(n)
        for _ in range(30):
            w = S(v / np.sqrt(dS)) / np.sqrt(dS); lam_d = np.linalg.norm(w) / np.linalg.norm(v); v = w
        Minv = lambda r: r / (lam_d * dS) + coarse(r)
    elif a.prec == 'kron1':
        # rank-one Kronecker (per trial block) approximation of the Jacobi diagonal, as a TT code would store it:
        # one diagonal factor per mode, fitted to log(diag S) in least squares (sums of per-mode log factors)
        dK = np.empty(n); kfactors = []
        for b, sz in enumerate(T.sizes):
            seg = slice(T.offsets[b], T.offsets[b + 1]); L = np.log(dS[seg]).reshape(sz, order='F'); mean = L.mean()
            fs = [L.mean(axis=tuple(e for e in range(4) if e != d)) - mean for d in range(4)]
            fit = mean + sum(np.expand_dims(fs[d], tuple(e for e in range(4) if e != d)) for d in range(4))
            dK[seg] = np.exp(fit).reshape(-1, order='F'); kfactors.append((mean, fs))
        rel = np.abs(dK / dS - 1)
        print(f"rank-one Kronecker Jacobi: |d_K/d - 1| median {np.median(rel):.2f}, max {rel.max():.2f}", flush=True)
        if a.write_kron:
            with open(a.write_kron, 'w') as f:
                for b, (mean, fs) in enumerate(kfactors):
                    for d in range(4):    # mode d factor of block b, as exp(mean/4 + f_d)
                        f.write(f"{b} {d} {len(fs[d])} " + " ".join(f"{np.exp(mean/4 + v):.17g}" for v in fs[d]) + "\n")
            print(f"wrote {a.write_kron}", flush=True)
        Minv = lambda r: r / dK
    elif a.prec == 'kron8':
        # rank-one Kronecker Jacobi in the cells-first 8-mode layout (c_x, c_t, c_ux, c_uz, b_x, b_t, b_ux, b_uz):
        # log diag S fitted by main effects (least squares on the full grid); flux normal axes: c = faces, b = 1
        cells = [int(t) for t in open(a.prefix + "_Gelem.txt").read().split()][0:4]
        dK = np.empty(n); lines = []
        for b, sz in enumerate(T.sizes):
            seg = slice(T.offsets[b], T.offsets[b + 1]); shp = []
            for d in range(4):
                face = cells[d] > 1 and sz[d] == cells[d] - 1
                nbd, ncd = (1, sz[d]) if face else (sz[d] // cells[d], cells[d]); shp += [nbd, ncd]
            L = np.log(dS[seg]).reshape(shp, order='F'); mean = L.mean(); fit = np.full(L.shape, mean)
            eff = {}
            for ax in range(8):
                e = L.mean(axis=tuple(q for q in range(8) if q != ax)) - mean; eff[ax] = e
                fit = fit + np.expand_dims(e, tuple(q for q in range(8) if q != ax))
            dK[seg] = np.exp(fit).reshape(-1, order='F')
            order = [1, 3, 5, 7, 0, 2, 4, 6]          # cells-first mode m <- tensor axis order[m]
            for m, ax in enumerate(order):
                lines.append(f"{b} {m} {len(eff[ax])} " + " ".join(f"{np.exp(mean / 8 + v):.17g}" for v in eff[ax]))
        rel = np.abs(dK / dS - 1)
        print(f"rank-one 8-mode Kronecker Jacobi: |d_K/d - 1| median {np.median(rel):.2f}, max {rel.max():.2f}", flush=True)
        if a.write_kron:
            open(a.write_kron, 'w').write("\n".join(lines) + "\n"); print(f"wrote {a.write_kron}", flush=True)
        Minv = lambda r: r / dK
    else:
        Minv = lambda r: r / dS

    # spectrum bounds of D^-1 S: lmax by power iteration, lmin from Lanczos (Jacobi-PCG coefficients)
    x = np.zeros(n); r = c.copy(); z = Minv(r); p = z.copy(); rz = r @ z; al_prev = None; be_prev = 0; d, e = [], []
    for k in range(a.lanczos):
        Sp = S(p); al = rz / (p @ Sp); x += al * p; r -= al * Sp; z = Minv(r); rzn = r @ z; be = rzn / rz
        d.append(1 / al + (be_prev / al_prev if al_prev else 0)); e.append(np.sqrt(be) / al)
        p = z + be * p; rz = rzn; al_prev = al; be_prev = be
    ritz = np.linalg.eigvalsh(np.diag(d) + np.diag(e[:-1], 1) + np.diag(e[:-1], -1))
    lmin = ritz[0] * a.lmin_safety; lmax = ritz[-1] * 1.05
    print(f"M^-1 S spectrum ({a.prec}): Ritz [{ritz[0]:.3e}, {ritz[-1]:.3f}] from {a.lanczos} Lanczos steps; "
          f"using [{lmin:.3e}, {lmax:.3f}], kappa {lmax/lmin:.0f}", flush=True)

    # Chebyshev semi-iteration (Saad, Alg. 12.1 form), S applied only to iterates
    theta, delta = (lmax + lmin) / 2, (lmax - lmin) / 2; sigma = theta / delta
    if a.x0 == 'coarse':
        x = coarse(c)
        if a.tx > 0: x, _ = T.round(x, a.tx)
        r = c - S_tt(x)
        print(f"warm start x0 = P S_c^-1 P^T c: rank {T.ranks(x, 1e-4)}, |r0|/|c| {np.linalg.norm(c - S(x))/nc:.2e}"
              + (f", rank(G^-1 B x0) {yr[-1]}" if yr else ""), flush=True)
    else:
        x = np.zeros(n); r = c.copy()
    stages = [float(t) for t in a.continuation.split(',')] if a.continuation else [None]
    total = 0
    for tol in stages:
        if tol is not None:
            a.tx = a.ty = tol; r = c - S_tt(x); yr.clear() if total == 0 else None
        stage_peak0 = len(yr)
        rho = 1 / sigma; dvec = Minv(r) / theta
        print(f"--- Chebyshev ({a.prec}, x0 {a.x0}), tx {a.tx:g}, ty {a.ty:g}", flush=True)
        iters = a.stage_iters if tol is not None else a.iters
        for k in range(1, iters + 1):
            total += 1
            x = x + dvec
            if a.tx > 0: x, _ = T.round(x, a.tx)
            r = c - S_tt(x)
            rho_new = 1 / (2 * sigma - rho)
            dvec = rho_new * rho * dvec + (2 * rho_new / delta) * Minv(r)
            rho = rho_new
            if k % a.every == 0 or k == 1:
                true = np.linalg.norm(c - S(x)) / nc
                print(f"  it {k:4d} (total {total:4d})  true |r|/|c| {true:.2e}  rank(x) {T.ranks(x, 1e-4)}"
                      + (f"  rank(G^-1 B x) {yr[-1]} (stage max {max(yr[stage_peak0:])})" if yr else "")
                      + (f"  [cells-first: max rank B x {zr[-1]}, structured G^-1 {tg[-1]:.2f} s]" if zr else ""), flush=True)
                if true < (2 * tol if tol is not None else a.rtol): break


if __name__ == "__main__":
    main()
