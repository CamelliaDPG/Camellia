#!/usr/bin/env python3
"""Strategy N5 checks on production meshes.

(A1) The direction-separable (N5) element Gram, alpha_K M + sum_d s_d(K)^2 int d_d w d_d v with s = (v_x,max(K), 1, a_bounds),
     as a preconditioner for the true graph-norm element Gram: spectrum of G_N5^{-1} G_graph and local PCG counts.
(D)  The fast-diagonalization diagonal of the N5 inverse, 1 / (|K| alpha_K/16 * (1 + sum_d xi_d(K)^2 mu^(d)_{i_d})), xi_d =
     2 s_d / (h_d sqrt(alpha_K)), mu^(d) the generalized eigenvalues of (K_1, M_1) in direction d: its TT ranks over
     (cells x eigen-indices) on the production mesh (8 bisection-graded x cells refined once, 2 t cells, the 22 x 20
     graded velocity mesh), cells-first and interleaved mode orders.

Usage: n5_checks.py [--p 3] [--dp 1] [--stride 1]
"""
import argparse, itertools, os, sys
import numpy as np, scipy.linalg as sla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frozen_gram_study as F


def n5_gram(ref, box, aK, a_bounds, c=1.0):
    h = [b[1] - b[0] for b in box]
    umax = max(abs(box[2][0]), abs(box[2][1])); uzmin = 0.0 if box[3][0] <= 0 <= box[3][1] else min(abs(box[3][0]), abs(box[3][1]))
    s = [umax / np.sqrt(1 + (umax ** 2 + uzmin ** 2) / c ** 2), 1.0, a_bounds[0], a_bounds[1]]
    xi = [2 * s[d] / (h[d] * np.sqrt(aK)) for d in range(4)]
    return ref.M + sum(xi[d] ** 2 * ref.K[d][d] for d in range(4)), xi


def tt_ranks(T, tols):
    """TT-SVD ranks at each cut for each relative tolerance."""
    out = {}
    for tol in tols:
        sizes = T.shape; d = len(sizes); delta = tol * np.linalg.norm(T) / np.sqrt(d - 1)
        C = T.reshape(sizes[0], -1, order='F'); r = 1; ranks = []
        for k in range(d - 1):
            C = C.reshape(r * sizes[k], -1, order='F'); U, sv, Vt = np.linalg.svd(C, full_matrices=False)
            tail = np.sqrt(np.cumsum(sv[::-1] ** 2))[::-1]; rk = max(1, int(np.sum(tail > delta)))
            ranks.append(rk); C = sv[:rk, None] * Vt[:rk]; r = rk
        out[tol] = ranks
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=3); ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--C", type=float, default=1.6)
    ap.add_argument("--part", default="A,D")
    ap.add_argument("--vrefine", type=int, default=0, help="D: extra uniform refinements of the velocity mesh")
    ap.add_argument("--xrefine", type=int, default=0, help="D: extra uniform refinements of the x mesh")
    a = ap.parse_args()
    ref = F.Reference(a.p + a.dp); k1 = ref.n1
    print(f"p = {a.p}, dp = {a.dp}: local test dimension {ref.n}", flush=True)
    umax = 0.995 / np.sqrt(1 - 0.995 ** 2); sg, u0 = 0.0497003, 0.198801
    uxv = F.refine(F.graded(-umax, umax, u0, 2 * sg, 2 * sg, 3.0)); uzv = F.refine(F.graded(-umax, umax, 0.0, 2 * sg, 2 * sg, 3.0))
    xb = sorted({0.0, 1.0} | {0.5 ** i for i in range(1, 8)}); xv = F.refine(np.array(xb))     # production x mesh
    tv = np.array([0.0, 0.05, 0.1]); a_bounds = (1.8, 0.65); qm = -1.0
    for _ in range(a.vrefine): uxv, uzv = F.refine(uxv), F.refine(uzv)
    for _ in range(a.xrefine): xv = F.refine(xv)
    if "A" in a.part:
        print("--- A1: N5 Gram as a preconditioner for the graph-norm element Gram", flush=True)
        for fname, (Ex, By) in {"comparison (Ex=-1, By=0.5)": (-1.0, 0.5), "production-scale (Ex=2.4, By=4.2)": (2.4, 4.2)}.items():
            for hx in (1 / 256, 1 / 32, 1 / 4):
                its, lo, hi = [], [], []
                for cnt, (i, j) in enumerate(itertools.product(range(len(uxv) - 1), range(len(uzv) - 1))):
                    if cnt % a.stride: continue
                    box = [(0, hx), (0, 0.05), (uxv[i], uxv[i + 1]), (uzv[j], uzv[j + 1])]
                    Gt, _, _ = F.element_grams(ref, box, qm, Ex, By, 1e2, a.C, a_bounds)
                    h = [b[1] - b[0] for b in box]
                    aK = F.alpha_transport(1e2, a.C, h[0], h[1], h[2], h[3], box[2], box[3], a_bounds)
                    G5, _ = n5_gram(ref, box, aK, a_bounds)
                    ev = sla.eigh(Gt, G5, eigvals_only=True); lo.append(ev[0]); hi.append(ev[-1])
                    f = sla.cho_factor(G5); its.append(F.pcg_iters(Gt, lambda R: sla.cho_solve(f, R), m=5))
                its = np.array(its)
                print(f"  {fname}, h_x = {hx:.4g}: G_N5^-1 G_graph spectrum in [{min(lo):.3f}, {max(hi):.3f}]; "
                      f"PCG its to 1e-8: median {int(np.median(its))}, max {its.max()} over {len(its)} elements", flush=True)
    if "D" in a.part:
        print("--- D: TT ranks of the N5 fast-diagonalization diagonal on the production mesh", flush=True)
        mu = sla.eigh(_k1(ref), _m1(ref), eigvals_only=True)          # 1D generalized eigenvalues of (K_1, M_1)
        cells = [len(xv) - 1, len(tv) - 1, len(uxv) - 1, len(uzv) - 1]
        D = np.empty(cells + [k1] * 4)
        for cx, ct, i, j in itertools.product(*[range(n) for n in cells]):
            box = [(xv[cx], xv[cx + 1]), (tv[ct], tv[ct + 1]), (uxv[i], uxv[i + 1]), (uzv[j], uzv[j + 1])]
            h = [b[1] - b[0] for b in box]
            aK = F.alpha_transport(1e2, a.C, h[0], h[1], h[2], h[3], box[2], box[3], a_bounds)
            _, xi = n5_gram(ref, box, aK, a_bounds)
            lam = 1 + sum(xi[d] ** 2 * mu.reshape([-1 if e == d else 1 for e in range(4)]) for d in range(4))
            D[cx, ct, i, j] = 1.0 / (np.prod(h) / 16 * aK * lam)
        print(f"  tensor {D.shape} ({D.size:.2e} entries); dynamic range of the diagonal {D.max() / D.min():.2e}", flush=True)
        tols = (1e-2, 1e-4, 1e-6)
        for name, perm in (("interleaved (cx,bx,ct,bt,cux,bux,cuz,buz)", [0, 4, 1, 5, 2, 6, 3, 7]),):
            R = tt_ranks(np.transpose(D, perm), tols)
            print(f"  {name}: " + "; ".join(f"{t:g}: {R[t]}" for t in tols), flush=True)



def _m1(ref):
    w = ref.w; return ref.phi.T @ (w[:, None] * ref.phi)


def _k1(ref):
    w = ref.w; return ref.dphi.T @ (w[:, None] * ref.dphi)


if __name__ == "__main__":
    main()
