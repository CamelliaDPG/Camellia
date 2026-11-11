#!/usr/bin/env python3
"""TT-operator ranks of the DPG Schur complement S = sum_K B_K^T G_K^{-1} B_K on the PRODUCTION mesh: 16 graded x cells
(8 bisection-graded, refined once), 2 t cells (h_t = 0.05), 22 x 20 graded velocity cells with returning electrons
(u in [-9.95, 9.95]^2), (x, t)-varying fields; N5 and graph test norms.  Question: is forming S explicitly in TT worth it
(rank of S vs the ranks of the test-space intermediates B x, G^{-1} B x, which reach 700 at tau = 1e-6)?

Element-local S_K over field + the four trace families (as v1_operator_ranks.py); the 15 block pairs are processed in
passes (all of S at once would be ~40 GB at p = 3), recomputing the element blocks in each pass.  Ranks in the
interleaved (c_d, i_d, j_d) layout (assembly of the traces is mode-wise and does not raise them).

Usage: v1_production_S_ranks.py [--p 3] [--dp 1] [--norms N5,graph] [--variation 0.5] [--tols 1e-2,1e-4,1e-6]
"""
import argparse, itertools, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frozen_gram_study as F
import v1_operator_ranks as V


def production_vertices():
    umax = 0.995 / np.sqrt(1 - 0.995 ** 2); sg, u0 = 0.0497003, 0.198801
    uxv = F.refine(F.graded(-umax, umax, u0, 2 * sg, 2 * sg, 3.0)); uzv = F.refine(F.graded(-umax, umax, 0.0, 2 * sg, 2 * sg, 3.0))
    xv = F.refine(np.array(sorted({0.0, 1.0} | {0.5 ** i for i in range(1, 8)})))
    tv = np.array([0.0, 0.05, 0.1])
    return [xv, tv, uxv, uzv]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=3); ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--norms", default="N5,graph"); ap.add_argument("--variation", type=float, default=0.5)
    ap.add_argument("--E0", type=float, default=-1.0); ap.add_argument("--B0", type=float, default=0.5)
    ap.add_argument("--alpha0", type=float, default=1e2); ap.add_argument("--C", type=float, default=1.6)
    ap.add_argument("--bounds", default="3.6,0.98", help="acceleration bounds (u_x, u_z), as in the TT runs")
    ap.add_argument("--tols", default="1e-2,1e-4,1e-6")
    ap.add_argument("--passGB", type=float, default=6.0, help="memory budget per pass for the stored block pairs")
    a = ap.parse_args()
    sp = V.Spaces(a.p, a.dp); n1f = sp.nf1; qm = -1.0
    tols = [float(t) for t in a.tols.split(",")]
    verts = production_vertices(); cells = [len(v) - 1 for v in verts]
    T = verts[1][-1]
    # field profile as PlanarMITLBoba::FieldProfile (x in [0, 1], t in [0, T])
    v = a.variation; E0, B0 = a.E0, a.B0
    Ef = lambda x, t: E0 * (1 + v * np.cos(np.pi * x)) * (1 + v * t / T)
    Bf = lambda x, t: B0 * (1 + v * (2 * x - 1)) * (1 - 0.6 * v * t / T)
    a_bounds = tuple(float(b) for b in a.bounds.split(","))
    sizes = [[n1f] * 4] + [[2 if e == d else sp.ntr1 for e in range(4)] for d in range(4)]
    offs = np.cumsum([0] + [int(np.prod(z)) for z in sizes])
    names = ["f", "tx", "tt", "tux", "tuz"]
    pairs = list(itertools.combinations_with_replacement(range(5), 2))
    nel = int(np.prod(cells))
    print(f"production mesh cells {cells} ({nel} elements), p = {a.p}, dp = {a.dp}, local S size {offs[-1]}, variation {v}, "
          f"bounds {a_bounds}", flush=True)
    # passes: greedy by memory
    passes, cur, curGB = [], [], 0.0
    for (ia, ib) in pairs:
        gb = nel * (offs[ia + 1] - offs[ia]) * (offs[ib + 1] - offs[ib]) * 8 / 2 ** 30
        if cur and curGB + gb > a.passGB: passes.append(cur); cur, curGB = [], 0.0
        cur.append((ia, ib)); curGB += gb
    passes.append(cur)
    for norm in a.norms.split(","):
        worst = {t: [0, 0, 0] for t in tols}; t0 = time.time()
        for ps in passes:
            store = {pr: np.empty(cells + [offs[pr[0] + 1] - offs[pr[0]], offs[pr[1] + 1] - offs[pr[1]]]) for pr in ps}
            for idx in itertools.product(*[range(c) for c in cells]):
                box = [(verts[d][idx[d]], verts[d][idx[d] + 1]) for d in range(4)]
                S, _ = V.element_blocks(sp, box, Ef, Bf, qm, norm, a.alpha0, a.C, a_bounds, False)
                for (ia, ib) in ps: store[(ia, ib)][idx] = S[offs[ia]:offs[ia + 1], offs[ib]:offs[ib + 1]]
            for (ia, ib) in ps:
                blk = store.pop((ia, ib))
                if np.linalg.norm(blk) == 0: continue
                R = V.tt_ranks(V.blockdiag_tensor(np.ascontiguousarray(blk), cells, sizes[ia], sizes[ib]), tols)
                del blk
                for t in tols: worst[t] = [max(u, w) for u, w in zip(worst[t], R[t])]
                print(f"  {norm:5s} S[{names[ia]},{names[ib]}]: " + "; ".join(f"{t:g}: {R[t]}" for t in tols)
                      + f"  ({time.time() - t0:.0f} s)", flush=True)
        print(f"{norm}: max over the 15 blocks " + "; ".join(f"{t:g}: {worst[t]}" for t in tols)
              + f"  (cuts (x)|(t), (x,t)|(u_x), (x,t,u_x)|(u_z); {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
