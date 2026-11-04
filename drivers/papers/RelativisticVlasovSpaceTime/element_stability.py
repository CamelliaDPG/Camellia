#!/usr/bin/env python3
"""Element stability constants for the ultraweak DPG transport form on one hypercube element, with frozen (constant)
coefficients: the d-dimensional analogue of the alpha_h and beta_h computations in Demkowicz, Roberts & Munoz-Matute,
"The DPG method for the convection-reaction problem, revisited" (Secs. 6-7).

Transport  beta . grad f = 0  on K = prod_d [0, h_d], beta constant (for 1D1T2V Vlasov, beta = (1, v_x, a_x, a_y) in
(t, x, u_x, u_y)).  Ultraweak form, with broken test function w:

    b((f, t_hat), w) = -(f, beta . grad w)_K + <t_hat, w>_{dK},        t_hat = (beta . n) f on dK.

Discretization as in Camellia for H1 order k (--p): f in Q_{k-1}, t_hat in Q_k on each face, w in Q_{k+dp} (H1 within
K).  Test inner product

    (w, z)_V = (beta . grad w, beta . grad z)_K + alpha (w, z)_K          (graph; alpha -> infinity is the L^2 norm,
                                                                           up to the scale alpha).

Reported, as in the paper, from the generalized eigenproblem  B^T G^{-1} B y = lambda^2 M y:

  alpha_h  (field):  inf over f in Q_{k-1} of  sup_w b(f, w) / (||w||_V ||f||_K),  with t_hat = 0 on the inflow and
                     w restricted to vanish on the outflow faces (the paper's construction), so it measures L^2
                     control of f;
  beta_h   (trace):  inf over outflow traces of sup_w <t_hat, w>_{dK+} / (||w||_V ||t_hat||_{w b_n}),  w unrestricted;
                     beta_h near 1 is the paper's condition for global discrete stability (product over layers).
  gamma_h  (full):   the discrete inf-sup constant of the element form in the field and outflow traces together
                     (inflow traces are data), in the norm ||f||^2 + sum_{outflow faces} ||t_hat||^2 / (beta.n).

Each constant is also reported relative to its value with a much larger test degree (--dpref), a proxy for the
ideal method with the same test norm: a ratio near 1 means the enriched test space realizes the ideal method.  With
the L^2 norm there is no ideal trace constant (a trace functional is unbounded on L^2(K)); beta_h then grows like
(k + dp)/sqrt(h): the trace penalty of a least-squares method, set by dp.

The graph-norm scale: alpha has units of 1/time^2 here; the natural comparison is alpha h_t^2 (h_t the element extent
in time, beta_t = 1), and the transport term ||beta.grad w||^2 ~ ((k+dp)^2 |beta_d| / h_d)^2 ||w||^2 for polynomial w.

Usage: element_stability.py [--k 3] [--dp 1 2 3 4] [--alpha 1 1e2 1e4 1e6 inf] [--case ...]
"""
import argparse
import itertools
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh, cho_factor, cho_solve

# --------------------------------------------------------------------------------------------------------------------
# 1D building blocks on [0, h]: Legendre bases (orthogonal, well conditioned), Gauss quadrature.
# --------------------------------------------------------------------------------------------------------------------

def legendre_values(n, x, h):
    """values and derivatives of Legendre P_0..P_n mapped to [0, h], at points x in [0, h]: (n+1, len(x)) arrays"""
    s = 2.0 * x / h - 1.0
    P = np.zeros((n + 1, len(x)))
    dP = np.zeros((n + 1, len(x)))
    P[0] = 1.0
    if n >= 1:
        P[1] = s
        dP[1] = 1.0
    for j in range(1, n):
        P[j + 1] = ((2 * j + 1) * s * P[j] - j * P[j - 1]) / (j + 1)
        dP[j + 1] = dP[j - 1] + (2 * j + 1) * P[j]
    return P, dP * (2.0 / h)


class Line:
    """1D matrices on [0, h] for test degree r and trial degrees used for field and trace."""
    def __init__(self, h, r, kf):
        self.h = h
        q, wq = leggauss(r + kf + 4)
        x = 0.5 * h * (q + 1.0)
        wq = 0.5 * h * wq
        V, dV = legendre_values(r, x, h)          # test basis
        F, _ = legendre_values(kf, x, h)          # field basis
        self.M = (V * wq) @ V.T                   # test mass
        self.D = (dV * wq) @ V.T                  # (dV_i, V_j): D[i, j] = int dV_i V_j
        self.K = (dV * wq) @ dV.T                 # (dV_i, dV_j)
        self.FV = (F * wq) @ V.T                  # (F_a, V_j)
        self.FdV = (F * wq) @ dV.T                # (F_a, dV_j)
        self.Mf = (F * wq) @ F.T                  # field mass
        e0, _ = legendre_values(r, np.array([0.0]), h)
        e1, _ = legendre_values(r, np.array([h]), h)
        self.end = [e0[:, 0], e1[:, 0]]           # test basis values at x = 0, x = h


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def element_matrices(h, beta, k, dp, alpha):
    """G (test Gram), Bf (field block, test x field), and per-face (Bt, Mt, normal flux) on the hypercube."""
    d = len(h)
    r = k + dp
    kf = k - 1
    lines = [Line(h[i], r, max(kf, k)) for i in range(d)]
    # field and trace bases: restrict the generic Line (built with degree max(kf, k)) to the needed degrees
    nf = kf + 1
    nt = k + 1
    # Gram: (beta.grad w, beta.grad z) + alpha (w, z)
    G = np.zeros(((r + 1) ** d,) * 2)
    for i in range(d):
        for j in range(d):
            if beta[i] == 0 or beta[j] == 0:
                continue
            mats = []
            for m in range(d):
                L = lines[m]
                if i == j == m:
                    mats.append(L.K)
                elif m == i and m != j:
                    mats.append(L.D)                    # dW_i in direction i (row), plain in direction j
                elif m == j and m != i:
                    mats.append(L.D.T)
                else:
                    mats.append(L.M)
            G += beta[i] * beta[j] * kron_all(mats)
    if np.isfinite(alpha):
        G += alpha * kron_all([L.M for L in lines])
    else:
        G = kron_all([L.M for L in lines])         # pure L^2 (scale irrelevant for the constants)
    # field block: b(f, w) = -(f, beta . grad w)  ->  Bf[w, f]
    Bf = np.zeros(((r + 1) ** d, nf ** d))
    for i in range(d):
        if beta[i] == 0:
            continue
        mats = [(lines[m].FdV[:nf].T if m == i else lines[m].FV[:nf].T) for m in range(d)]
        Bf += -beta[i] * kron_all(mats)
    # faces: <t_hat, w>_face, t_hat in Q_k on the face; flux sign b_n = beta . n
    faces = []
    for i in range(d):
        for side in (0, 1):
            bn = beta[i] * (1.0 if side == 1 else -1.0)
            mats, mass = [], []
            for m in range(d):
                L = lines[m]
                if m == i:
                    mats.append(L.end[side][:, None])     # w evaluated on the face
                else:
                    mats.append(L.FV[:nt].T)               # (t basis, w basis) along the face
                    mass.append(L.Mf[:nt, :nt])
            Bt = kron_all(mats)                            # (test, trace)
            Mt = kron_all(mass) if mass else np.ones((1, 1))
            faces.append(dict(dir=i, side=side, bn=bn, B=Bt, M=Mt))
    Mf = kron_all([L.Mf[:nf, :nf] for L in lines])
    return G, Bf, faces, Mf


def element_matrices_varying(h, origin, beta_fn, k, dp, alpha, nq_extra=3):
    """As element_matrices(), for beta varying within the element: beta_fn(points) -> (npts, d) array, points in
    physical coordinates (npts, d).  Assembled by full tensor-product Gauss quadrature, so affordable only for small
    test spaces (d = 4: test degree <= 5)."""
    d = len(h)
    r = k + dp
    kf = k - 1
    nf, nt = kf + 1, k + 1
    nq = r + kf + nq_extra
    q, wq = leggauss(nq)
    xs, ws, V, dV, F, Ft = [], [], [], [], [], []
    for m in range(d):
        x = origin[m] + 0.5 * h[m] * (q + 1.0)
        xs.append(x)
        ws.append(0.5 * h[m] * wq)
        Vm, dVm = legendre_values(r, x - origin[m], h[m])
        Fm, _ = legendre_values(max(kf, k), x - origin[m], h[m])
        V.append(Vm); dV.append(dVm); F.append(Fm[:nf]); Ft.append(Fm[:nt])
    grids = np.meshgrid(*xs, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1)
    W = ws[0]
    for m in range(1, d):
        W = np.multiply.outer(W, ws[m])
    W = W.ravel()
    b = beta_fn(pts)                                         # (npts, d)
    # test values and transport derivative at points: (ntest, npts)
    Vall = V[0]
    for m in range(1, d):
        Vall = np.einsum("ip,jq->ijpq", Vall.reshape(-1, Vall.shape[-1]) if Vall.ndim == 2 else Vall, V[m]) \
            if False else None
    # build by Kronecker-structured einsum, direction by direction
    def tensor_eval(factors):
        out = factors[0]
        for f in factors[1:]:
            out = np.einsum("ap,bq->abpq", out, f).reshape(out.shape[0] * f.shape[0], out.shape[1] * f.shape[1])
        return out
    Vt = tensor_eval(V)                                      # (ntest, npts)
    LV = np.zeros_like(Vt)
    for m in range(d):
        Dm = tensor_eval([dV[j] if j == m else V[j] for j in range(d)])
        LV += Dm * b[:, m][None, :]
    G = (LV * W) @ LV.T
    Mtest = (Vt * W) @ Vt.T
    G = Mtest.copy() if not np.isfinite(alpha) else G + alpha * Mtest
    Fv = tensor_eval(F)                                      # (nfield, npts)
    Bf = -(LV * W) @ Fv.T                                    # (test, field)
    Mf = (Fv * W) @ Fv.T
    faces = []
    for i in range(d):
        for side in (0, 1):
            xf = origin[i] + (h[i] if side == 1 else 0.0)
            Ve, _ = legendre_values(r, np.array([xf - origin[i]]), h[i])
            tang = [m for m in range(d) if m != i]
            fac_test = [Ve if m == i else V[m] for m in range(d)]
            Vface = tensor_eval(fac_test)                    # (ntest, nq^(d-1))
            Tface = tensor_eval([Ft[m] for m in tang])       # (ntrace, nq^(d-1))
            Wf = ws[tang[0]]
            for m in tang[1:]:
                Wf = np.multiply.outer(Wf, ws[m])
            Wf = Wf.ravel()
            fgrids = np.meshgrid(*[np.array([xf]) if m == i else xs[m] for m in range(d)], indexing="ij")
            fpts = np.stack([g.ravel() for g in fgrids], axis=1)
            bn = beta_fn(fpts)[:, i] * (1.0 if side == 1 else -1.0)
            Bt = (Vface * Wf) @ Tface.T
            Mt = (Tface * Wf) @ Tface.T
            faces.append(dict(dir=i, side=side, bn=float(np.mean(bn)), bn_min=float(bn.min()), bn_max=float(bn.max()),
                              B=Bt, M=Mt, Mw=(Tface * (Wf / np.abs(bn))) @ Tface.T))
    return G, Bf, faces, Mf


def constants_varying(h, origin, beta_fn, k, dp, alpha):
    """constants() for varying beta: outflow faces are those with b_n > 0 throughout (faces where b_n changes sign
    are reported and skipped); trace norm weight 1/b_n pointwise"""
    G, Bf, faces, Mf = element_matrices_varying(h, origin, beta_fn, k, dp, alpha)
    mixed = [f for f in faces if f["bn_min"] < 0 < f["bn_max"]]
    outflow = [f for f in faces if f["bn_min"] > 0]
    factor = cho_factor(G)
    Bo = np.hstack([f["B"] for f in outflow])
    Mo = block_mass([f["Mw"] for f in outflow])
    GiBo = cho_solve(factor, Bo)
    beta_h = smallest_sqrt_eig(Bo.T @ GiBo, Mo)
    Ball = np.hstack([Bf, Bo])
    gamma_h = smallest_sqrt_eig(Ball.T @ cho_solve(factor, Ball), block_mass([Mf, Mo]))
    return np.array([beta_h, gamma_h]), len(outflow), len(mixed)


def smallest_sqrt_eig(S, M):
    S = 0.5 * (S + S.T)
    lam = eigh(S, 0.5 * (M + M.T), eigvals_only=True, subset_by_index=[0, 0])
    return np.sqrt(max(lam[0], 0.0))


def block_mass(masses):
    n = sum(m.shape[0] for m in masses)
    out = np.zeros((n, n))
    o = 0
    for m in masses:
        out[o:o + m.shape[0], o:o + m.shape[0]] = m
        o += m.shape[0]
    return out


def constants(h, beta, k, dp, alpha):
    """(alpha_h, beta_h, gamma_h) for one element; inflow traces are data (zero), as in the layer-by-layer argument"""
    G, Bf, faces, Mf = element_matrices(h, beta, k, dp, alpha)
    outflow = [f for f in faces if f["bn"] > 0]
    factor = cho_factor(G)
    Bo = np.hstack([f["B"] for f in outflow])
    Mo = block_mass([f["M"] / f["bn"] for f in outflow])     # t_hat = b_n u:  ||u||^2_{b_n} = ||t_hat||^2 / b_n
    GiBo = cho_solve(factor, Bo)
    GiBf = cho_solve(factor, Bf)
    # beta_h: outflow traces against the full enriched test space
    beta_h = smallest_sqrt_eig(Bo.T @ GiBo, Mo)
    # alpha_h: field, against test functions with <t_hat, w>_{dK+} = 0 for all discrete outflow traces (the discrete
    # analogue of w = 0 on dK+): restrict G^{-1} to that subspace by the Schur-complement formula
    So = Bo.T @ GiBo
    Sff = Bf.T @ GiBf - (Bf.T @ GiBo) @ np.linalg.solve(So, GiBo.T @ Bf)
    alpha_h = smallest_sqrt_eig(Sff, Mf)
    # gamma_h: field and outflow traces together
    Ball = np.hstack([Bf, Bo])
    gamma_h = smallest_sqrt_eig(Ball.T @ cho_solve(factor, Ball), block_mass([Mf, Mo]))
    return np.array([alpha_h, beta_h, gamma_h])


CASES = {
    # name: (h, beta) in (t, x, u_x, u_y); beta_t = 1 always
    "1-outflow (t only)":       ((0.1, 0.25, 0.1, 0.1), (1.0, 0.0, 0.0, 0.0)),
    "2-outflow (t, x)":         ((0.1, 0.25, 0.1, 0.1), (1.0, 0.5, 0.0, 0.0)),
    "3-outflow (t, x, ux)":     ((0.1, 0.25, 0.1, 0.1), (1.0, 0.5, 1.0, 0.0)),
    "4-outflow (t, x, ux, uy)": ((0.1, 0.25, 0.1, 0.1), (1.0, 0.5, 1.0, 0.5)),
    "4-outflow, fine core cell": ((0.1, 0.25, 0.0994, 0.0994), (1.0, 0.1, 2.0, 2.0)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=3, help="H1 order (Camellia --p); field degree k-1, trace degree k")
    ap.add_argument("--dp", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--dpref", type=int, default=8, help="test enrichment for the reference (near-ideal) constants")
    ap.add_argument("--alpha", type=float, nargs="+", default=[1.0, 1e2, 1e4, 1e6, np.inf])
    ap.add_argument("--case", nargs="*", default=list(CASES))
    ap.add_argument("--dim", type=int, default=4, help="use the first DIM directions of each case (2-4)")
    args = ap.parse_args()
    for name in args.case:
        h, beta = CASES[name]
        h, beta = h[:args.dim], beta[:args.dim]
        print(f"== {name}: h = {h}, beta = {beta}, k = {args.k}, reference dp = {args.dpref}")
        print(f"   {'alpha':>8s} {'dp':>3s} {'alpha_h':>9s} {'beta_h':>9s} {'gamma_h':>9s}   {'ratios to reference':>27s}")
        for alpha in args.alpha:
            ref = constants(h, beta, args.k, args.dpref, alpha)
            for dp in args.dp:
                c = constants(h, beta, args.k, dp, alpha)
                r = c / ref
                print(f"   {alpha:8.0e} {dp:3d} {c[0]:9.4f} {c[1]:9.4f} {c[2]:9.4f}   {r[0]:8.3f} {r[1]:8.3f} {r[2]:8.3f}",
                      flush=True)
            print(f"   {alpha:8.0e} ref {ref[0]:9.4f} {ref[1]:9.4f} {ref[2]:9.4f}", flush=True)


if __name__ == "__main__":
    main()
