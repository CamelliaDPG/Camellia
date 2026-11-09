#!/usr/bin/env python3
"""Strategy N1: can element Gram inverses be represented on a fixed, mesh-independent parameter set?

Graph-norm element Gram on K = box with widths h (directions x, t, u_x, u_z), test basis Lagrange at GLL points of degree
k = p + dp, quadrature Gauss:
    G_K = int_K (beta.grad w)(beta.grad v) + alpha_K w v,   beta = (v_x(u), 1, a_ux, a_uz),
    v = u / gamma, a = (q/m) (E + v x B):  a_ux = (q/m)(E_x - v_z B_y),  a_uz = (q/m) v_x B_y.
With beta frozen at the element center, G_K = (|K|/16) alpha_K H(xi), H(xi) = M + sum_ij xi_i xi_j D_i^T M D_j (reference
matrices), xi_i = 2 beta_i / (h_i sqrt(alpha_K)); with the transport-scale weight |xi|^2 <= 4/C.

Part 1: frozen vs true G_K over production-like elements: spectrum of G_frozen^{-1} G_true, PCG iterations (1e-8).
Part 2: the xi range that occurs, and (a) PCG iterations for H(xi) with the factor of H at the nearest node of a grid
of spacing d in each xi component, (b) Chebyshev interpolation of H(xi)^{-1} along rays, error vs number of nodes.

Usage: frozen_gram_study.py [--p 3] [--dp 1] [--part 1,2]
"""
import argparse, itertools
import numpy as np, numpy.polynomial.legendre as L, scipy.linalg as sla


def gll(k):
    """GLL nodes of degree k on [-1, 1]."""
    if k == 1: return np.array([-1.0, 1.0])
    inner = np.real(L.legroots(L.legder(np.eye(k + 1)[k])))
    return np.concatenate([[-1.0], np.sort(inner), [1.0]])


def lagrange_1d(k, q):
    """Values and derivatives (reference) of the degree-k GLL Lagrange basis at points q: (nq x k+1) each."""
    x = gll(k); V = L.legvander(x, k); C = np.linalg.inv(V)                 # basis_j = sum_m C[m, j] P_m
    Pq = L.legvander(q, k); dPq = np.stack([L.legval(q, L.legder(np.eye(k + 1)[m])) for m in range(k + 1)], axis=1)
    return Pq @ C, dPq @ C


class Reference:
    def __init__(self, k, nq_extra=4):
        self.k = k; self.n1 = k + 1; self.nq = k + nq_extra
        self.q, self.w = L.leggauss(self.nq)
        self.phi, self.dphi = lagrange_1d(k, self.q)
        self.n = self.n1 ** 4
        # 4D values and reference derivatives at quadrature points (point index and local index: first axis fastest)
        def kron4(m): return np.kron(m[3], np.kron(m[2], np.kron(m[1], m[0])))
        self.Phi = kron4([self.phi] * 4)
        self.dPhi = [kron4([self.dphi if d == e else self.phi for e in range(4)]) for d in range(4)]
        self.W = kron4([self.w[:, None]] * 4).ravel()
        self.Q = [kron4([(self.q if d == e else np.ones(self.nq))[:, None] for e in range(4)]).ravel() for d in range(4)]
        self.M = self.Phi.T @ (self.W[:, None] * self.Phi)
        self.K = [[self.dPhi[i].T @ (self.W[:, None] * self.dPhi[j]) for j in range(4)] for i in range(4)]

    def H(self, xi):
        return self.M + sum(xi[i] * xi[j] * self.K[i][j] for i in range(4) for j in range(4))


def beta_at(u_x, u_z, qm, Ex, By, c=1.0):
    g = np.sqrt(1 + (u_x ** 2 + u_z ** 2) / c ** 2); vx, vz = u_x / g, u_z / g
    return [vx, np.ones_like(vx), qm * (Ex - vz * By), qm * vx * By]


def alpha_transport(alpha, C, hx, ht, hux, huz, ux_box, uz_box, a_bounds, c=1.0):
    umax = max(abs(ux_box[0]), abs(ux_box[1])); uzmin = 0.0 if uz_box[0] <= 0 <= uz_box[1] else min(abs(uz_box[0]), abs(uz_box[1]))
    vmax = umax / np.sqrt(1 + (umax ** 2 + uzmin ** 2) / c ** 2)
    return alpha + C * ((vmax / hx) ** 2 + 1 / ht ** 2 + (a_bounds[0] / hux) ** 2 + (a_bounds[1] / huz) ** 2)


def element_grams(ref, box, qm, Ex, By, alpha, C, a_bounds):
    """True and frozen (center) graph-norm Grams on box = [(lo, hi)] * 4 (reference-scaled, without |K|/16)."""
    h = [b[1] - b[0] for b in box]; mid = [(b[0] + b[1]) / 2 for b in box]
    aK = alpha_transport(alpha, C, h[0], h[1], h[2], h[3], box[2], box[3], a_bounds)
    ux = mid[2] + h[2] / 2 * ref.Q[2]; uz = mid[3] + h[3] / 2 * ref.Q[3]
    beta = beta_at(ux, uz, qm, Ex, By)
    A = sum((2 * beta[d] / h[d])[:, None] * ref.dPhi[d] for d in range(4))
    Gtrue = A.T @ (ref.W[:, None] * A) + aK * ref.M
    bc = beta_at(np.array([mid[2]]), np.array([mid[3]]), qm, Ex, By)
    xi = np.array([2 * bc[d][0] / (h[d] * np.sqrt(aK)) for d in range(4)])
    return Gtrue / aK, ref.H(xi), xi


def pcg_iters(G, P, rtol=1e-8, m=10, maxit=2000, seed=0):
    R0 = np.random.default_rng(seed).standard_normal((G.shape[0], m)); X = np.zeros_like(R0); R = R0.copy()
    Z = P(R); D = Z.copy(); rz = np.sum(R * Z, 0); n0 = np.linalg.norm(R0, axis=0)
    for k in range(maxit):
        if np.max(np.linalg.norm(R, axis=0) / n0) < rtol: return k
        Q = G @ D; al = rz / np.sum(D * Q, 0); X += D * al; R -= Q * al; Z = P(R); rzn = np.sum(R * Z, 0); D = Z + D * (rzn / rz); rz = rzn
    return maxit


def graded(left, right, center, core_w, core_hw, ratio):
    """PlanarMITL::gradedVertices (production velocity grading)."""
    cl, cr = max(left, center - core_hw), min(right, center + core_hw)
    def side(start, end):
        out, length, d, pos, w = [], abs(end - start), 1.0 if end > start else -1.0, 0.0, core_w * ratio
        while length - pos > 0:
            if length - pos <= w * 1.5: out.append(end); break
            pos += w; out.append(start + d * pos); w *= ratio
        return out
    v = list(reversed(side(cl, left)))
    nc = max(1, int(np.ceil((cr - cl) / core_w - 1e-10)))
    v += [cl + (cr - cl) * i / nc for i in range(nc + 1)] + side(cr, right)
    if not any(abs(x) < 1e-14 for x in v):
        i = next(j for j in range(len(v) - 1) if v[j] < 0 < v[j + 1]); w = v[i + 1] - v[i]
        if abs(v[i]) < w / 4: v[i] = 0.0
        elif abs(v[i + 1]) < w / 4: v[i + 1] = 0.0
        else: v.insert(i + 1, 0.0)
    return np.array(sorted(v))


def refine(v): return np.sort(np.concatenate([v, 0.5 * (v[1:] + v[:-1])]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=3); ap.add_argument("--dp", type=int, default=1); ap.add_argument("--part", default="1,2")
    ap.add_argument("--stride", type=int, default=1, help="Part 1: use every stride-th velocity cell")
    ap.add_argument("--C", type=float, default=1.6); ap.add_argument("--alpha", type=float, default=1e2)
    a = ap.parse_args()
    ref = Reference(a.p + a.dp); print(f"p = {a.p}, dp = {a.dp}: local test dimension {ref.n}", flush=True)
    umax = 0.995 / np.sqrt(1 - 0.995 ** 2); sigma, u0 = 0.0497003, 0.198801
    uxv = refine(graded(-umax, umax, u0, 2 * sigma, 2 * sigma, 3.0)); uzv = refine(graded(-umax, umax, 0.0, 2 * sigma, 2 * sigma, 3.0))
    print(f"production velocity mesh (refined): {len(uxv) - 1} x {len(uzv) - 1} cells", flush=True)
    a_bounds = (1.8, 0.65); qm = -1.0
    fields = {"comparison (Ex=-1, By=0.5)": (-1.0, 0.5), "production-scale (Ex=2.4, By=4.2)": (2.4, 4.2)}
    hxs = [1 / 256, 1 / 32, 1 / 4]; ht = 0.05
    all_xi = []
    if "1" in a.part:
        print("--- Part 1: frozen-center Gram as a preconditioner for the true element Gram", flush=True)
        for fname, (Ex, By) in fields.items():
            for hx in hxs:
                its, lo, hi = [], [], []
                for cnt, (i, j) in enumerate(itertools.product(range(len(uxv) - 1), range(len(uzv) - 1))):
                    if cnt % a.stride: continue
                    box = [(0, hx), (0, ht), (uxv[i], uxv[i + 1]), (uzv[j], uzv[j + 1])]
                    Gt, Gf, xi = element_grams(ref, box, qm, Ex, By, a.alpha, a.C, a_bounds); all_xi.append(xi)
                    f = sla.cho_factor(Gf)
                    if True:
                        ev = sla.eigh(Gt, Gf, eigvals_only=True); lo.append(ev[0]); hi.append(ev[-1])
                        its.append(pcg_iters(Gt, lambda R: sla.cho_solve(f, R)))
                its = np.array(its)
                print(f"  {fname}, h_x = {hx:.4g}: G_f^-1 G_t spectrum in [{min(lo):.3f}, {max(hi):.3f}]; PCG its to 1e-8: "
                      f"median {int(np.median(its))}, max {its.max()} over {len(its)} elements", flush=True)
    if "2" in a.part:
        xi = np.array(all_xi) if all_xi else None
        if xi is not None:
            print(f"--- Part 2: xi range over those elements: |xi| <= {np.linalg.norm(xi, axis=1).max():.3f} (bound 2/sqrt(C) = {2/np.sqrt(a.C):.3f}); "
                  f"per component max |xi_i| = {np.abs(xi).max(axis=0).round(3)}", flush=True)
        rng = np.random.default_rng(1); R = 2 / np.sqrt(a.C)
        # (a) nearest-node preconditioner on a uniform grid of spacing d in each component
        samples = []
        while len(samples) < 12:
            z = rng.uniform(-R, R, 4)
            if np.linalg.norm(z) <= R: samples.append(z)
        for d in (0.4, 0.2, 0.1, 0.05):
            its = []
            for z in samples:
                node = np.round(z / d) * d
                f = sla.cho_factor(ref.H(node)); its.append(pcg_iters(ref.H(z), lambda Rm: sla.cho_solve(f, Rm), m=5))
            nodes = int(np.prod([2 * np.floor(R / d) + 1] * 4) * np.pi ** 2 / 32)   # nodes inside the ball ~ ball/box volume
            print(f"  (a) grid spacing {d}: PCG its with the nearest node's factor: median {int(np.median(its))}, max {max(its)}"
                  f"  (~{nodes} nodes in the |xi| <= {R:.2f} ball)", flush=True)
        # (b) Chebyshev interpolation of H^-1 along rays, in s^2 (H depends on xi_i xi_j)
        dirs = [np.eye(4)[0], np.eye(4)[2], np.array([1, 1, 1, 1.0]) / 2, rng.standard_normal(4)]
        for e in dirs:
            e = e / np.linalg.norm(e); test_s = np.linspace(0, R, 23)[1:]
            exact = [np.linalg.inv(ref.H(s * e)) for s in test_s]
            line = []
            for nn in (4, 8, 16, 32):
                t = (np.cos(np.pi * (np.arange(nn) + 0.5) / nn) + 1) / 2 * R ** 2             # Chebyshev nodes in s^2
                Hinv = [np.linalg.inv(ref.H(np.sqrt(tt) * e)) for tt in t]
                err = 0.0
                for s, X in zip(test_s, exact):
                    wts = np.array([np.prod([(s ** 2 - t[m]) / (t[j] - t[m]) for m in range(nn) if m != j]) for j in range(nn)])
                    approx = sum(w * Hj for w, Hj in zip(wts, Hinv))
                    err = max(err, np.linalg.norm(approx - X, 2) / np.linalg.norm(X, 2))
                line.append(f"{nn}: {err:.1e}")
            print(f"  (b) ray {np.round(e, 2)}: Chebyshev interpolation of H^-1 in |xi|^2, rel. error vs nodes  " + "  ".join(line), flush=True)


if __name__ == "__main__":
    main()
