"""Cost of direct strategies on a big dumped site: dense LU of the whole site vs Cholesky of the eps block + Schur complement."""
import sys, time, numpy as np, scipy.linalg as sla
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from site_preconditioners import load
A, b, off, rf, *_ = load(sys.argv[1])
n0 = off[1]; n = A.shape[0]
print(f"n={n}, eps={n0}, trial={n-n0}", flush=True)
t = time.time(); x_lu = sla.lu_solve(sla.lu_factor(A), b); t_lu = time.time() - t
print(f"dense LU of whole site: {t_lu:.1f} s, res {np.linalg.norm(A@x_lu-b)/np.linalg.norm(b):.1e}", flush=True)
t = time.time()
A00 = A[:n0,:n0]; A01 = A[:n0,n0:]; A10 = A[n0:,:n0]; A11 = A[n0:,n0:]
c = sla.cho_factor(0.5*(A00+A00.T)); Y = sla.cho_solve(c, A01); S = A11 - A10 @ Y
y0 = sla.cho_solve(c, b[:n0]); x1 = sla.lu_solve(sla.lu_factor(S), b[n0:] - A10 @ y0); x0 = y0 - Y @ x1
t_ch = time.time() - t; x = np.concatenate([x0, x1])
print(f"Cholesky(eps) + Schur: {t_ch:.1f} s, res {np.linalg.norm(A@x-b)/np.linalg.norm(b):.1e}, |x-x_lu|/|x_lu| {np.linalg.norm(x-x_lu)/np.linalg.norm(x_lu):.1e}", flush=True)
