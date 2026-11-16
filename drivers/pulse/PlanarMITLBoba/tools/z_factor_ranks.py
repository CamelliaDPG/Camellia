#!/usr/bin/env python3
"""TT-operator ranks of the separable-graph-norm DPG operator factors on the production-shape mesh: B (test x trial),
the fast-diagonalization diagonal D^{-1/2}, and the half factor

    Z = D^{-1/2} (V^{(x)4})^T B,   S = Z^T Z = B^T G^{-1} B,

(V: the shared 1D generalized eigenbasis, V^T M V = I, V^T K V = Lambda), against the ranks of S itself.  Question: can
the explicit-S solver store Z (or B and D) instead of S -- a least-squares formulation with local QR solves -- at
comparable TT ranks?

Mesh: level 0 of the production shape by default (8 graded x, 1 t, 11 x 10 graded velocity cells; --refine 1 for level
1); prescribed constant fields (default: the first substep's vacuum fields).  Element blocks are processed one trial
block at a time (Z_K is test x trial, 2401 x 1256 at p = 4, dp = 2).  Ranks in the interleaved (c_d, i_d, j_d) layout,
cuts (x)|(t), (x,t)|(u_x), (x,t,u_x)|(u_z); relative Frobenius TT-SVD tolerances, as v1_production_S_ranks.py.

Usage: z_factor_ranks.py [--p 4] [--dp 2] [--refine 0] [--tols 1e-2,1e-4,1e-6]
"""
import argparse, itertools, os, sys, time
import numpy as np
import scipy.linalg as sla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frozen_gram_study as F
import v1_operator_ranks as V


def mesh(refine):
    umax = 0.995 / np.sqrt(1 - 0.995 ** 2); sg, u0 = 0.0497003, 0.198801
    def ens0(v):                                    # PlanarMITL::ensureVertexAt(v, 0)
        v = list(v)
        for i in range(len(v) - 1):
            a, b = v[i], v[i + 1]
            if 0 <= a or 0 >= b: continue
            w = b - a
            if -a < 0.25 * w: v[i] = 0.0
            elif b < 0.25 * w: v[i + 1] = 0.0
            else: v.insert(i + 1, 0.0)
            break
        return np.array(v)
    ux = ens0(F.graded(-umax, umax, u0, 2 * sg, 2 * sg, 3.0)); uz = ens0(F.graded(-umax, umax, 0.0, 2 * sg, 2 * sg, 3.0))
    x = np.array(sorted({0.0, 1.0} | {0.5 ** i for i in range(1, 8)})); t = np.array([0.0, 0.1])
    v = [x, t, ux, uz]
    for _ in range(refine): v = [F.refine(a) for a in v]
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=4); ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--E0", type=float, default=-1.95695); ap.add_argument("--B0", type=float, default=-4.17409)
    ap.add_argument("--alpha0", type=float, default=1e2); ap.add_argument("--C", type=float, default=1.6)
    ap.add_argument("--bounds", default="1.8,0.65"); ap.add_argument("--tols", default="1e-2,1e-4,1e-6")
    ap.add_argument("--blocks", default="f,tx,tt,tux,tuz", help="trial blocks to measure B and Z for")
    a = ap.parse_args()
    tols = [float(t) for t in a.tols.split(",")]; a_bounds = tuple(float(b) for b in a.bounds.split(","))
    sp = V.Spaces(a.p, a.dp); r = sp.ref; n1 = r.n1; qm = -1.0
    verts = mesh(a.refine); cells = [len(v) - 1 for v in verts]; nel = int(np.prod(cells))
    Ef = lambda x, t: a.E0 + 0 * x; Bf = lambda x, t: a.B0 + 0 * x
    # 1D generalized eigenproblem of the reference test space (nodal GLL Lagrange, degree p + dp)
    M1 = r.phi.T @ (r.w[:, None] * r.phi); K1 = r.dphi.T @ (r.w[:, None] * r.dphi)
    lam, V1 = sla.eigh(K1, M1)
    V4 = sp.kron4([V1] * 4)
    lam4 = [sp.kron4([(lam if d == e else np.ones(n1))[:, None] for e in range(4)]).ravel() for d in range(4)]
    sizes = [[sp.nf1] * 4] + [[2 if e == d else sp.ntr1 for e in range(4)] for d in range(4)]
    offs = np.cumsum([0] + [int(np.prod(z)) for z in sizes]); names = ["f", "tx", "tt", "tux", "tuz"]
    print(f"cells {cells} ({nel} elements), p = {a.p}, dp = {a.dp}: test {r.n}, trial {offs[-1]} per element; "
          f"E0 {a.E0}, B0 {a.B0}, bounds {a_bounds}", flush=True)

    def element(idx):
        box = [(verts[d][idx[d]], verts[d][idx[d] + 1]) for d in range(4)]
        h = np.array([b[1] - b[0] for b in box]); mid = [(b[0] + b[1]) / 2 for b in box]
        P = [mid[d] + h[d] / 2 * r.Q[d] for d in range(4)]
        g = np.sqrt(1 + P[2] ** 2 + P[3] ** 2); vx, vz = P[2] / g, P[3] / g
        E, B = Ef(P[0], P[1]), Bf(P[0], P[1])
        beta = [vx, np.ones_like(vx), qm * (E - vz * B), qm * vx * B]
        aK = F.alpha_transport(a.alpha0, a.C, h[0], h[1], h[2], h[3], box[2], box[3], a_bounds)
        umax = max(abs(box[2][0]), abs(box[2][1])); uzmin = 0.0 if box[3][0] <= 0 <= box[3][1] else min(abs(box[3][0]), abs(box[3][1]))
        s = [umax / np.sqrt(1 + umax ** 2 + uzmin ** 2), 1.0, a_bounds[0], a_bounds[1]]
        Dvec = aK + sum((2 * s[d] / h[d]) ** 2 * lam4[d] for d in range(4))        # G = V^{-T} diag(D) V^{-1}
        A = sum((2 * beta[d] / h[d])[:, None] * r.dPhi[d] for d in range(4))
        Dm = A.T @ (r.W[:, None] * sp.PhiF)
        Ball = np.hstack([-Dm] + [(2 / h[d]) * sp.Btr[d] for d in range(4)])
        vol = np.prod(h) / 16
        Z = np.sqrt(vol) * (V4.T @ Ball) / np.sqrt(Dvec)[:, None]
        return Ball, Dvec, Z, vol, aK, s, h

    # sanity: Z^T Z = vol B^T G^{-1} B on one element
    Ball, Dvec, Z, vol, aK, s, h = element((0, 0, 0, 0))
    G = aK * r.M + sum((2 * s[d] / h[d]) ** 2 * r.K[d][d] for d in range(4))
    S_ref = vol * Ball.T @ sla.cho_solve(sla.cho_factor(G), Ball)
    print(f"check ||Z^T Z - S|| / ||S|| = {np.linalg.norm(Z.T @ Z - S_ref) / np.linalg.norm(S_ref):.2e}", flush=True)

    t0 = time.time()
    # D^{-1/2}: a function of (cell, eigen-index) in each direction
    Dt = np.empty(cells + [n1] * 4)
    for idx in itertools.product(*[range(c) for c in cells]):
        Dt[idx] = (1 / np.sqrt(element(idx)[1])).reshape([n1] * 4, order="F")
    perm = [];  [perm.extend([d, 4 + d]) for d in range(4)]
    R = V.tt_ranks(np.transpose(Dt, perm).reshape([cells[d] * n1 for d in range(4)]), tols)
    print("D^{-1/2}: " + "; ".join(f"{t:g}: {R[t]}" for t in tols) + f"  ({time.time() - t0:.0f} s)", flush=True)
    del Dt

    for name in a.blocks.split(","):
        ib = names.index(name); nb = offs[ib + 1] - offs[ib]
        Sst = np.empty(cells + [nb, nb]) if ib == 0 else None
        for label in ("Z", "B"):                     # one stored operator at a time (each is nel x 2401 x nb doubles)
            st = np.empty(cells + [r.n, nb])
            for idx in itertools.product(*[range(c) for c in cells]):
                Ball, Dvec, Z, *_ = element(idx)
                st[idx] = (Z if label == "Z" else Ball)[:, offs[ib]:offs[ib + 1]]
                if label == "Z" and Sst is not None: Sst[idx] = st[idx].T @ st[idx]
            R = V.tt_ranks(V.blockdiag_tensor(st, cells, [n1] * 4, sizes[ib]), tols)
            print(f"{label}[test,{name}]: " + "; ".join(f"{t:g}: {R[t]}" for t in tols) + f"  ({time.time() - t0:.0f} s)", flush=True)
            del st
        if Sst is not None:
            R = V.tt_ranks(V.blockdiag_tensor(np.ascontiguousarray(Sst), cells, sizes[0], sizes[0]), tols)
            print(f"S[f,f] (element-local, same tool): " + "; ".join(f"{t:g}: {R[t]}" for t in tols) + f"  ({time.time() - t0:.0f} s)", flush=True)
            del Sst


if __name__ == "__main__":
    main()
