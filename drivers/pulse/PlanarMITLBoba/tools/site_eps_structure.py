"""Is the site's Riesz solution eps well captured by A00^{-1} (projected G^{-1}) applied to the trial image A01 u?
At a dumped site: solve the scaled site system exactly; eps = x[:n0], u = x[n0:].  Then (exactly, from the first block
row) A00 eps = b0 - A01 u.  Compare the singular values of eps, of the trial contribution A00^{-1} A01 u, and of the
RHS part A00^{-1} b0, each reshaped (rx*n) x rxp1 (the cut AMEn splits at, sweeping left to right)."""
import sys, numpy as np, scipy.linalg as sla
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from site_preconditioners import load
A, b, off, rf, phiL, Ac, phiR = load(sys.argv[1])
n0 = off[1]; rx = phiL.shape[0]; nr = Ac.shape[1]; rxp1 = phiR.shape[0]
x = sla.lu_solve(sla.lu_factor(A), b); eps, u = x[:n0], x[n0:]
c = sla.cho_factor(0.5*(A[:n0,:n0]+A[:n0,:n0].T))
trial_part = -sla.cho_solve(c, A[:n0, n0:] @ u); rhs_part = sla.cho_solve(c, b[:n0])
print(f"site {sys.argv[1].split('/')[-1]}: eps block {rx}x{nr}x{rxp1}; |eps - (rhs+trial)|/|eps| = "
      f"{np.linalg.norm(eps-(rhs_part+trial_part))/np.linalg.norm(eps):.1e}")
print(f"norms: eps {np.linalg.norm(eps):.3e}, rhs part {np.linalg.norm(rhs_part):.3e}, trial part {np.linalg.norm(trial_part):.3e}")
for name, v in [("eps", eps), ("A00^-1 b0", rhs_part), ("-A00^-1 A01 u", trial_part)]:
    for cut, shape in [("left|right (rx*n | rxp1)", (rx*nr, rxp1)), ("left|right (rx | n*rxp1)", (rx, nr*rxp1))]:
        sv = np.linalg.svd(v.reshape(shape, order='F'), compute_uv=False); sv = sv/sv[0]
        ranks = [int(np.sum(sv > t)) for t in (1e-2, 1e-3, 1e-5)]
        print(f"  {name:15s} {cut:26s} ranks at 1e-2/1e-3/1e-5: {ranks} of {min(shape)}")
