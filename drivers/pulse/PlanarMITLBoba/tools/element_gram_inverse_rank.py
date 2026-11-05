"""Supporting numbers for the G^{-1} ~ M^{-1}/alpha + X analysis, one 4D element (element_stability.py matrices)."""
import sys, numpy as np
import os
# element_stability.py lives in drivers/papers/RelativisticVlasovSpaceTime (branch 316, commit 43e0600d); override with
# ELEMENT_STABILITY_DIR if it is elsewhere.
sys.path.insert(0, os.environ.get("ELEMENT_STABILITY_DIR",
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../papers/RelativisticVlasovSpaceTime")))
import element_stability as es
from scipy.linalg import eigh
def kron_sv(A, n1):
    R = A.reshape(n1, n1, n1, n1).transpose(0, 2, 1, 3).reshape(n1*n1, n1*n1)
    s = np.linalg.svd(R, compute_uv=False); return s / s[0]
k = 3
for name, (h, beta) in es.CASES.items():
    for dp in (1, 2):
        K = es.element_matrices(h, beta, k, dp, 0.0)[0]
        M = es.element_matrices(h, beta, k, dp, np.inf)[0]
        lam = eigh(K, M, eigvals_only=True)
        N = len(lam); r = k + dp + 1
        tol = 1e-10 * lam.max()
        nz = lam[lam > tol]
        print(f"== {name} dp={dp}: N={N} rank(K)={len(nz)} null={N-len(nz)} "
              f"min nonzero lam={nz.min():.3e} (x h_t^2 = {nz.min()*h[0]**2:.2f}) max lam={lam.max():.3e}")
        Minv = np.linalg.inv(M)
        for a in (1.0, 1e2, 1e4, 1e6):
            # eigenvalues of alpha*G^{-1} in M-orthonormal coordinates are 1/(1+lam/a); X's are -lam/(a+lam) (times 1/a)
            mu = lam / (a + lam)
            mu_sorted = np.sort(mu)[::-1]
            fro = np.sqrt(np.cumsum(mu_sorted**2) / np.sum(mu_sorted**2)) if mu_sorted.sum() > 0 else np.ones(N)
            def rank_for(eps):  # smallest rank capturing X to relative Frobenius error eps (M-orthonormal coords)
                tail = np.sqrt(np.maximum(0, 1 - fro**2)); return int(np.argmax(tail <= eps)) + 1 if mu_sorted.sum() > 0 else 0
            Gi = np.linalg.inv(K + a * M)
            s = kron_sv(Gi, r * r)
            print(f"   a={a:.0e}: #lam>a={int(np.sum(lam>a)):4d} #lam>a/10={int(np.sum(lam>a/10)):4d} "
                  f"rank(X) for rel err 1e-1/1e-2/1e-4: {rank_for(1e-1)}/{rank_for(1e-2)}/{rank_for(1e-4)} | "
                  f"Kron rank (tx|uu) of G^-1 at 1e-2/1e-4/1e-6: {np.sum(s>1e-2)}/{np.sum(s>1e-4)}/{np.sum(s>1e-6)}", flush=True)
