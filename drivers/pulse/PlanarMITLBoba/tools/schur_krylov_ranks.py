#!/usr/bin/env python3
"""Rank growth and rank control in Krylov iterations on the DPG Schur complement S = B^T G^{-1} B.

Input: a global block dump from PlanarMITLBobaDriver (PLANAR_MITL_DUMP_GLOBAL=<prefix>, which also writes the RHS blocks
and per-block TT mode sizes).  Block 0 is the test (Riesz) space; blocks 1.. are the trial blocks (f, fluxes).  The
arithmetic is done on full vectors (exact S via a Cholesky factor of G), but every trial vector is measured -- and, for the
truncated methods, rounded -- in TT format block by block, so the ranks are what a TT implementation would see.

Methods (all on S u = c, c = B^T G^{-1} l, which is SPD):
  cg       exact CG; reports the TT ranks the iterate, residual, and direction would need (no truncation).
  tcg      CG with x, r, p rounded to tolerance tau after every update (truncated CG).
  tproj    truncated subspace projection (Sands et al.): new direction = rounded preconditioned residual; the step is the
           S-Galerkin solution in the span of the last m rounded directions (Gram system), x rounded, r recomputed.
  pstep    "perturbed step" (Saibal's idea): exact CG direction d and step beta; the new iterate is the best of a few
           rank-reduced candidates T_tau(x + s*beta*d), s in {0.5, 0.75, 1, 1.25}, tau in {tau, tau/10}, chosen by the
           Jacobi-weighted residual norm; the residual is recomputed and the direction rounded.
Preconditioner: Jacobi (diagonal of S) unless --noprec.

  pstepr   pstep with the residual updated recursively and rounded (true residual every 10 its), as a TT code must.
Usage: schur_krylov_ranks.py <prefix> [--tau 1e-4] [--iters 150] [--methods cg,tcg,tproj,pstep,pstepr] [--noprec]
"""
import argparse, numpy as np, scipy.linalg as sla

def load(prefix):
    meta = [l.split() for l in open(prefix + "_blocks.txt")]
    nb = max(int(m[0]) for m in meta) + 1
    blk = {}
    for i, j, r, c, *_ in meta:
        if int(i) == 0 and int(j) == 0:      # memory-map the (large) Gram block; it is only sliced, by element blocks
            blk[0, 0] = np.memmap(f"{prefix}_0_0.bin", dtype=np.float64, mode='r', shape=(int(r), int(c)), order='F')
        elif int(i) == 0 or int(j) == 0:      # keep only the test x trial blocks (row 0); B^T is their transpose
            if int(i) == 0:
                blk[0, int(j)] = np.fromfile(f"{prefix}_{i}_{j}.bin").reshape(int(r), int(c), order='F')
    rhs = [np.fromfile(f"{prefix}_rhs_{b}.bin") for b in range(nb)]
    modes = {int(l.split()[0]): [int(x) for x in l.split()[2:6]] for l in open(prefix + "_modes.txt")}
    return blk, rhs, modes, nb

def tt_round(v, sizes, tau):
    """TT-SVD of a full vector (first index fastest) at relative tolerance tau; returns (rounded full vector, max rank)."""
    d = len(sizes); t = v.reshape(sizes, order='F'); norm = np.linalg.norm(v)
    if norm == 0: return v.copy(), 0
    delta = tau * norm / np.sqrt(max(d - 1, 1))
    cores, r_prev, C = [], 1, t.reshape(sizes[0], -1, order='F')
    ranks = []
    for k in range(d - 1):
        C = C.reshape(r_prev * sizes[k], -1, order='F')
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]          # tail[i] = norm of s[i:]
        r = max(1, int(np.sum(tail > delta)))
        cores.append(U[:, :r].reshape(r_prev, sizes[k], r, order='F'))
        C = s[:r, None] * Vt[:r]; r_prev = r; ranks.append(r)
    cores.append(C.reshape(r_prev, sizes[-1], 1, order='F'))
    out = cores[0].reshape(sizes[0], -1, order='F')
    for k in range(1, d):
        out = (out @ cores[k].reshape(cores[k].shape[0], -1, order='F')).reshape(-1, cores[k].shape[2], order='F')
    return out.reshape(-1, order='F'), max(ranks) if ranks else 1

class Trial:
    """Stacked trial vector made of TT-shaped blocks."""
    def __init__(self, modes, nb):
        self.sizes = [modes[b] for b in range(1, nb)]
        self.offsets = np.cumsum([0] + [int(np.prod(s)) for s in self.sizes])
    def round(self, v, tau):
        out = np.empty_like(v); ranks = []
        for b, sz in enumerate(self.sizes):
            seg = slice(self.offsets[b], self.offsets[b + 1])
            out[seg], r = tt_round(v[seg], sz, tau); ranks.append(r)
        return out, ranks
    def ranks(self, v, tau):
        return self.round(v, tau)[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix"); ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=150); ap.add_argument("--methods", default="cg,tcg,tproj,pstep")
    ap.add_argument("--noprec", action="store_true"); ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--m", type=int, default=8, help="subspace size for tproj")
    a = ap.parse_args()
    blk, rhs, modes, nb = load(a.prefix)
    G = blk[0, 0]; B = np.hstack([blk[0, j] for j in range(1, nb)])          # test x trial
    l = rhs[0]
    # G is block diagonal over elements in a suitable ordering; factor it sparsely (it is far too large to factor
    # densely at 40k).  Connected components of its sparsity graph are the element blocks.
    import scipy.sparse as sp, scipy.sparse.csgraph as csg
    n_test = G.shape[0]; rows, cols = [], []
    for j0 in range(0, n_test, 2000):                     # scan column chunks of the memory-mapped G for nonzeros
        chunk = np.asarray(G[:, j0:j0 + 2000])
        ii, jj = np.nonzero(np.abs(chunk) > 0)
        rows.append(ii); cols.append(jj + j0)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    Gs = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=G.shape)
    ncomp, labels = csg.connected_components(Gs, directed=False)
    factors = []
    for k in range(ncomp):
        idx = np.flatnonzero(labels == k)
        Gk = np.asarray(G[np.ix_(idx, idx)])
        factors.append((idx, sla.cho_factor(0.5 * (Gk + Gk.T))))
    del Gs, rows, cols
    def Ginv(v):
        out = np.empty_like(v)
        for idx, f in factors: out[idx] = sla.cho_solve(f, v[idx])
        return out
    print(f"G: {G.shape[0]} dofs in {ncomp} diagonal blocks (sizes {min(len(i) for i,_ in factors)}-{max(len(i) for i,_ in factors)})", flush=True)
    S = B.T @ Ginv(B); c = B.T @ Ginv(l)
    u_star = np.linalg.solve(S, c)
    T = Trial(modes, nb)
    fseg = slice(T.offsets[0], T.offsets[1])
    ferr = lambda u: np.linalg.norm(u[fseg] - u_star[fseg]) / np.linalg.norm(u_star[fseg])
    eS = lambda u: np.sqrt((u - u_star) @ S @ (u - u_star) / (u_star @ S @ u_star))
    dS = np.diag(S); M = (lambda r: r) if a.noprec else (lambda r: r / dS)
    ev = np.linalg.eigvalsh(S); evp = np.linalg.eigvalsh(S / np.sqrt(np.outer(dS, dS)))
    print(f"S: n={S.shape[0]}, cond {ev.max()/ev.min():.2e}, Jacobi-scaled cond {evp.max()/evp.min():.2e}; "
          f"exact solution ranks (tau={a.tau:g}): {T.ranks(u_star, a.tau)}", flush=True)
    for method in a.methods.split(','):
        u = np.zeros_like(c); r = c.copy(); z = M(r); p = z.copy(); rz = r @ z
        P, SP = [], []
        print(f"--- {method} (tau {a.tau:g}{', no preconditioner' if a.noprec else ', Jacobi'})", flush=True)
        for k in range(1, a.iters + 1):
            if method in ("cg", "tcg"):
                Sp = S @ p; beta = rz / (p @ Sp)
                u = u + beta * p; r = r - beta * Sp
                if method == "tcg":
                    u, _ = T.round(u, a.tau); r = c - S @ u                     # rounded iterate, true residual
                z = M(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
                if method == "tcg": p, _ = T.round(p, a.tau)
            elif method == "tproj":
                d, _ = T.round(M(r), a.tau)
                P.append(d); SP.append(S @ d); P, SP = P[-a.m:], SP[-a.m:]
                Pm = np.column_stack(P); SPm = np.column_stack(SP)
                y = np.linalg.lstsq(Pm.T @ SPm, Pm.T @ r, rcond=None)[0]        # S-Galerkin on span(P)
                u, _ = T.round(u + Pm @ y, a.tau); r = c - S @ u
            elif method == "pstepr":
                # perturbed step, with the residual updated recursively and rounded (as a TT code would have to),
                # plus a true-residual replacement every 10 iterations.
                Sp = S @ p; beta = rz / (p @ Sp)
                best = None
                for s_ in (0.5, 0.75, 1.0, 1.25):
                    for t in (a.tau, a.tau / 10):
                        cand, _ = T.round(u + s_ * beta * p, t)
                        res = c - S @ cand; score = res @ (res / dS)
                        if best is None or score < best[0]: best = (score, cand, s_)
                u_new = best[1]
                r = r - S @ (u_new - u); u = u_new
                r, _ = T.round(r, a.tau)
                if k % 10 == 0: r = c - S @ u
                z = M(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
                p, _ = T.round(p, a.tau)
            elif method == "pstep":
                Sp = S @ p; beta = rz / (p @ Sp)
                best = None
                for s in (0.5, 0.75, 1.0, 1.25):
                    for t in (a.tau, a.tau / 10):
                        cand, rk = T.round(u + s * beta * p, t)
                        res = c - S @ cand; score = res @ (res / dS)            # Jacobi-weighted residual norm
                        if best is None or score < best[0]: best = (score, cand, s, t)
                u = best[1]; r = c - S @ u
                z = M(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
                p, _ = T.round(p, a.tau)
            if k % a.every == 0 or k == 1:
                extra = f"  rank(r) {T.ranks(r, a.tau)} rank(p) {T.ranks(p, a.tau)}" if method in ("cg", "tcg", "pstep", "pstepr") else ""
                print(f"  it {k:4d}  f err {ferr(u):.2e}  S-err {eS(u):.2e}  |r|/|c| {np.linalg.norm(c - S @ u)/np.linalg.norm(c):.2e}"
                      f"  rank(x) {T.ranks(u, a.tau)}{extra}", flush=True)

if __name__ == "__main__":
    main()
