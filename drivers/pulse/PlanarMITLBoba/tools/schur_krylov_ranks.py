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
--tau-y applies S inexactly in every method (the test-space intermediate G^-1 B p rounded in TT); the printed residual
and errors always use the exact S.
Usage: schur_krylov_ranks.py <prefix> [--tau 1e-4] [--iters 150] [--methods cg,tcg,tproj,pstep,pstepr] [--noprec]
"""
import argparse, numpy as np, scipy.linalg as sla

def load_sparse(prefix):
    """Sparse row-0 dump (PLANAR_MITL_DUMP_SPARSE): G and test x trial blocks as COO, RHS blocks, TT modes."""
    import scipy.sparse as sp
    def coo(name):
        with open(name, 'rb') as f:
            r, c, n = np.fromfile(f, dtype=np.int64, count=3)
            ri = np.fromfile(f, dtype=np.int32, count=n); ci = np.fromfile(f, dtype=np.int32, count=n)
            v = np.fromfile(f, dtype=np.float64, count=n)
        return sp.csr_matrix((v, (ri, ci)), shape=(r, c))
    meta = [l.split() for l in open(prefix + "_blocks.txt")]
    blk = {(0, int(m[1])): coo(f"{prefix}_0_{m[1]}.coo") for m in meta}
    nb = 1 + len([m for m in meta if int(m[1]) > 0])
    import os
    if os.path.exists(prefix + "_Gelem.txt"):
        # element blocks of G: list of (global indices, dense block); G itself is never assembled
        info = [int(x) for x in open(prefix + "_Gelem.txt").read().split()]
        bs = info[8]; ne = info[0] * info[1] * info[2] * info[3]
        data = np.fromfile(prefix + "_Gelem.bin").reshape(ne, bs, bs)          # each block column-major
        idx = np.fromfile(prefix + "_Gelem_idx.bin", dtype=np.int32).reshape(ne, bs)
        blk[0, 0] = [(idx[k], data[k].T) for k in range(ne)]
    rhs = [np.fromfile(f"{prefix}_rhs_{b}.bin") for b in range(nb)]
    modes = {int(l.split()[0]): [int(x) for x in l.split()[2:6]] for l in open(prefix + "_modes.txt")}
    return blk, rhs, modes, nb

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
    tt_round.last_ranks = ranks
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
    def cut_ranks(self, v, tau):
        """Per block: list of (rank, max possible rank) at each of the three cuts."""
        out = []
        for b, sz in enumerate(self.sizes):
            seg = slice(self.offsets[b], self.offsets[b + 1])
            tt_round(v[seg], sz, tau)
            mx = [min(int(np.prod(sz[:k + 1])), int(np.prod(sz[k + 1:]))) for k in range(len(sz) - 1)]
            out.append(list(zip(tt_round.last_ranks, mx)))
        return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix"); ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=150); ap.add_argument("--methods", default="cg,tcg,tproj,pstep")
    ap.add_argument("--noprec", action="store_true"); ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--m", type=int, default=8, help="subspace size for tproj")
    ap.add_argument("--percut", action="store_true", help="print per-cut ranks with the maximum possible rank")
    ap.add_argument("--matfree", action="store_true", help="never form S: apply sum_K B_K^T G_K^{-1} B_K (large cases)")
    ap.add_argument("--tau-y", type=float, default=0.0,
                    help="inexact S, as a TT code would apply it: round y = G^-1 B p in the test-space TT modes at this "
                         "relative tolerance before B^T y (0: exact).  Convergence is still reported with the exact S.")
    ap.add_argument("--relax", action="store_true",
                    help="with --tau-y: relaxed inexact Krylov (Simoncini-Szyld), tau_y,k = min(tau_y |c| / |r_k|, 0.5)")
    ap.add_argument("--diag-ranks", action="store_true", help="report TT ranks of the Jacobi diagonal of S per block")
    ap.add_argument("--noref", action="store_true", help="with --matfree: skip the reference solve (errors reported as nan)")
    a = ap.parse_args()
    import os, scipy.sparse as sp, scipy.sparse.csgraph as csg
    sparse = os.path.exists(a.prefix + "_0_1.coo")
    blk, rhs, modes, nb = (load_sparse if sparse else load)(a.prefix)
    G = blk[0, 0]; l = rhs[0]
    B = sp.hstack([blk[0, j] for j in range(1, nb)]).tocsr() if sparse else np.hstack([blk[0, j] for j in range(1, nb)])
    # G is block diagonal over elements in a suitable ordering; factor it sparsely (it is far too large to factor
    # densely at 40k).  Connected components of its sparsity graph are the element blocks.
    n_test = len(l)
    if isinstance(G, list):
        factors = [(idx, sla.cho_factor(0.5 * (Gk + Gk.T))) for idx, Gk in G]
        ncomp = len(factors)
    elif sparse:
        Gs = G
    else:
        rows, cols = [], []
        for j0 in range(0, n_test, 2000):                 # scan column chunks of the memory-mapped G for nonzeros
            chunk = np.asarray(G[:, j0:j0 + 2000])
            ii, jj = np.nonzero(np.abs(chunk) > 0)
            rows.append(ii); cols.append(jj + j0)
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        Gs = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=G.shape)
    if not isinstance(G, list):
        ncomp, labels = csg.connected_components(Gs, directed=False)
        factors = []
        for k in range(ncomp):
            idx = np.flatnonzero(labels == k)
            Gk = (G[idx][:, idx].toarray() if sparse else np.asarray(G[np.ix_(idx, idx)]))
            factors.append((idx, sla.cho_factor(0.5 * (Gk + Gk.T))))
    def Ginv(v):
        out = np.empty_like(v)
        for idx, f in factors: out[idx] = sla.cho_solve(f, v[idx])
        return out
    print(f"G: {n_test} dofs in {ncomp} diagonal blocks (sizes {min(len(i) for i,_ in factors)}-{max(len(i) for i,_ in factors)})", flush=True)
    if a.matfree:
        # Element-local, matrix-free S p = sum_K B_K^T G_K^{-1} B_K p (B_K = rows of B on element K, sparse).
        Bks = [(idx, B[idx], f) for idx, f in factors]
        class MatFreeS:
            shape = (B.shape[1], B.shape[1])
            def __matmul__(self, v):
                out = np.zeros(B.shape[1])
                for idx, Bk, f in Bks: out += Bk.T @ sla.cho_solve(f, Bk @ v)
                return out
            def diagonal(self):
                d = np.zeros(B.shape[1])
                for idx, Bk, f in Bks:
                    Bd = Bk.toarray(); d += np.einsum('ij,ij->j', Bd, sla.cho_solve(f, Bd))
                return d
        S = MatFreeS()
    elif sparse:
        # S = B^T G^{-1} B, built by element blocks (G^{-1} block diagonal): sum_k B_k^T G_k^{-1} B_k
        S = np.zeros((B.shape[1], B.shape[1]))
        for idx, f in factors:
            Bk = B[idx].toarray(); S += Bk.T @ sla.cho_solve(f, Bk)
    else:
        S = B.T @ Ginv(B)
    c = B.T @ Ginv(l)
    test_sizes = modes[0]
    if a.matfree and a.noref:
        u_star = None
    elif a.matfree:
        # reference solution: Jacobi-PCG on the matrix-free S to a tight tolerance
        dS0 = S.diagonal(); u_star = np.zeros_like(c); r0 = c.copy(); z0 = r0 / dS0; p0 = z0.copy(); rz0 = r0 @ z0
        for it in range(5000):
            Sp0 = S @ p0; al = rz0 / (p0 @ Sp0); u_star += al * p0; r0 -= al * Sp0
            if np.linalg.norm(r0) <= 1e-13 * np.linalg.norm(c): break
            z0 = r0 / dS0; rzn = r0 @ z0; p0 = z0 + (rzn / rz0) * p0; rz0 = rzn
        print(f"reference solution: {it+1} Jacobi-PCG iterations to 1e-13", flush=True)
    else:
        u_star = np.linalg.solve(S, c)
    T = Trial(modes, nb)
    fseg = slice(T.offsets[0], T.offsets[1])
    ferr = lambda u: np.nan if u_star is None else np.linalg.norm(u[fseg] - u_star[fseg]) / np.linalg.norm(u_star[fseg])
    eS = lambda u: np.nan if u_star is None else np.sqrt((u - u_star) @ (S @ (u - u_star)) / (u_star @ (S @ u_star)))
    dS = S.diagonal() if a.matfree else np.diag(S); M = (lambda r: r) if a.noprec else (lambda r: r / dS)
    class InexactS:
        """S p = B^T T_tau_y(G^-1 B p): the test-space intermediate rounded in TT, as a TT implementation must form it."""
        shape = (B.shape[1], B.shape[1]); ranks = []; tol = a.tau_y
        def __matmul__(self, v):
            y, r = tt_round(Ginv(B @ v), test_sizes, self.tol); self.ranks.append(r)
            return B.T @ y
    Sit = InexactS() if a.tau_y > 0 else S
    if a.diag_ranks:
        for t in (1e-2, 1e-3, 1e-4):
            print(f"Jacobi diagonal TT ranks (tau {t:g}): {T.ranks(dS, t)}", flush=True)
    if a.matfree:
        print(f"S: n={S.shape[0]} (matrix-free); exact solution ranks (tau={a.tau:g}): {T.ranks(u_star, a.tau) if u_star is not None else '-'}", flush=True)
    else:
        ev = np.linalg.eigvalsh(S); evp = np.linalg.eigvalsh(S / np.sqrt(np.outer(dS, dS)))
        print(f"S: n={S.shape[0]}, cond {ev.max()/ev.min():.2e}, Jacobi-scaled cond {evp.max()/evp.min():.2e}; "
              f"exact solution ranks (tau={a.tau:g}): {T.ranks(u_star, a.tau)}", flush=True)
    for method in a.methods.split(','):
        u = np.zeros_like(c); r = c.copy(); z = M(r); p = z.copy(); rz = r @ z
        P, SP = [], []
        print(f"--- {method} (tau {a.tau:g}{', no preconditioner' if a.noprec else ', Jacobi'}"
              f"{f', inexact S: tau_y {a.tau_y:g}' if a.tau_y > 0 else ''})", flush=True)
        if a.tau_y > 0: Sit.ranks.clear()
        for k in range(1, a.iters + 1):
            if a.tau_y > 0 and a.relax:
                Sit.tol = min(a.tau_y * np.linalg.norm(c) / max(np.linalg.norm(r), 1e-300), 0.5)
            if method in ("cg", "tcg"):
                Sp = Sit @ p; beta = rz / (p @ Sp)
                u = u + beta * p; r = r - beta * Sp
                if method == "tcg":
                    u, _ = T.round(u, a.tau); r = c - Sit @ u                     # rounded iterate, true residual
                z = M(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
                if method == "tcg": p, _ = T.round(p, a.tau)
            elif method == "tproj":
                d, _ = T.round(M(r), a.tau)
                P.append(d); SP.append(Sit @ d); P, SP = P[-a.m:], SP[-a.m:]
                Pm = np.column_stack(P); SPm = np.column_stack(SP)
                y = np.linalg.lstsq(Pm.T @ SPm, Pm.T @ r, rcond=None)[0]        # S-Galerkin on span(P)
                u, _ = T.round(u + Pm @ y, a.tau); r = c - Sit @ u
            elif method == "pstepr":
                # perturbed step, with the residual updated recursively and rounded (as a TT code would have to),
                # plus a true-residual replacement every 10 iterations.
                Sp = Sit @ p; beta = rz / (p @ Sp)
                best = None
                for s_ in (0.5, 0.75, 1.0, 1.25):
                    for t in (a.tau, a.tau / 10):
                        cand, _ = T.round(u + s_ * beta * p, t)
                        res = c - Sit @ cand; score = res @ (res / dS)
                        if best is None or score < best[0]: best = (score, cand, s_)
                u_new = best[1]
                r = r - Sit @ (u_new - u); u = u_new
                r, _ = T.round(r, a.tau)
                if k % 10 == 0: r = c - Sit @ u
                z = M(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
                p, _ = T.round(p, a.tau)
            elif method == "pstep":
                Sp = Sit @ p; beta = rz / (p @ Sp)
                best = None
                for s in (0.5, 0.75, 1.0, 1.25):
                    for t in (a.tau, a.tau / 10):
                        cand, rk = T.round(u + s * beta * p, t)
                        res = c - Sit @ cand; score = res @ (res / dS)            # Jacobi-weighted residual norm
                        if best is None or score < best[0]: best = (score, cand, s, t)
                u = best[1]; r = c - Sit @ u
                z = M(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
                p, _ = T.round(p, a.tau)
            if a.percut and (k % a.every == 0):
                for name, vec in (("x", u), ("r", r)):
                    cr = T.cut_ranks(vec, a.tau)
                    print(f"    per-cut (rank/max) {name}: " + "  ".join(
                        "[" + ",".join(f"{rk}/{mx}" for rk, mx in blk) + "]" for blk in cr), flush=True)
                tst = tt_round(Ginv(B @ p), test_sizes, a.tau)
                mxt = [min(int(np.prod(test_sizes[:k + 1])), int(np.prod(test_sizes[k + 1:]))) for k in range(3)]
                print("    per-cut (rank/max) G^-1 B p: [" + ",".join(f"{rk}/{mx}" for rk, mx in zip(tt_round.last_ranks, mxt)) + "]", flush=True)
            if k % a.every == 0 or k == 1:
                # rank of the intermediate G^{-1} B p (the vector a TT code must form to apply S), test-space modes
                if method in ("cg", "tcg", "pstep", "pstepr"):
                    _, rGBp = tt_round(Ginv(B @ p), test_sizes, a.tau)
                else:
                    rGBp = '-'
                extra0 = f"  rank(G^-1 B p) {rGBp}  rank(S p) {T.ranks(S @ p, a.tau)}" if rGBp != '-' else ""
                extra = f"  rank(r) {T.ranks(r, a.tau)} rank(p) {T.ranks(p, a.tau)}" if method in ("cg", "tcg", "pstep", "pstepr") else ""
                if a.tau_y > 0 and Sit.ranks:
                    print(f"    inexact S: rank of rounded G^-1 B p: max {max(Sit.ranks)}, last {Sit.ranks[-1]} (tol {Sit.tol:.1e}); "
                          f"{len(Sit.ranks)} applications", flush=True)
                print(f"  it {k:4d}  f err {ferr(u):.2e}  S-err {eS(u):.2e}  |r|/|c| {np.linalg.norm(c - S @ u)/np.linalg.norm(c):.2e}"
                      f"  rank(x) {T.ranks(u, a.tau)}{extra}{extra0}", flush=True)

if __name__ == "__main__":
    main()
