"""Inner preconditioned CG on the eps block (Kronecker-term preconditioner), and the outer block-triangular GMRES
with an inexact inner solve.  Also checks the (0,0) Kronecker reconstruction ordering."""
import sys, glob, itertools, numpy as np, scipy.sparse.linalg as spla, scipy.linalg as sla
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from site_preconditioners import load, gmres_count

base = sys.argv[1]
A, b, off, rf, phiL, Ac, phiR = load(base)
n0 = off[1]; n = A.shape[0]; A00 = A[:n0, :n0]
rx, _, ra = phiL.shape; _, nr, nc, rap1 = Ac.shape; rxp1 = phiR.shape[0]
# Try the index orderings of the reconstruction and keep the one matching A00 best.
best = None
for perm in itertools.permutations('ikm'):
    rows = ''.join(perm); cols = rows.replace('i', 'j').replace('k', 'l').replace('m', 'n')
    G = np.einsum('ija,aklb,nbm->' + rows + cols, phiL, Ac, phiR)
    G = G.reshape(n0, n0)  # C order: first letter slowest
    e = np.linalg.norm(A00 - rf[0] * G) / np.linalg.norm(A00)
    if best is None or e < best[0]: best = (e, rows)
print(f"best ordering {best[1]} (C order, first slowest): mismatch {best[0]:.2e}", flush=True)
# Interface and mode sizes: is rx the left rank and rxp1 the right?  dims:
print(f"sizes rx={rx} n={nr} rxp1={rxp1}; n0={n0} vs rx*n*rxp1={rx*nr*rxp1}", flush=True)

# Inner solves on A00 (SPD): CG with (a) no preconditioner, (b) Jacobi, (c) dominant Kronecker term P (exact inverse of P
# via factor LUs), (d) sum of the top-k Kronecker terms is not cheaply invertible, so skip.
ev = np.linalg.eigvalsh(A00); print(f"A00 eigs: min {ev.min():.3e} max {ev.max():.3e} cond {ev.max()/ev.min():.2e}", flush=True)
terms = [(phiL[:, :, a], Ac[a, :, :, c], phiR[:, c, :].T) for a in range(ra) for c in range(rap1)]
norms = [np.linalg.norm(L)*np.linalg.norm(K)*np.linalg.norm(R) for L, K, R in terms]
L, K, R = terms[int(np.argmax(norms))]
# P = rf0 * kron(R, kron(K, L)); inverse via factor inverses (symmetrize each factor)
iL, iK, iR = [np.linalg.inv(0.5*(X+X.T)) for X in (L, K, R)]
def Pinv(v):
    V = v.reshape(rx, nr, rxp1, order='F')
    V = np.einsum('ij,jkl->ikl', iL, V); V = np.einsum('ij,kjl->kil', iK, V); V = np.einsum('ij,klj->kli', iR, V)
    return V.reshape(-1, order='F') / rf[0]
def cg_count(Mfun, tol=1e-6):
    rhs = np.random.default_rng(0).standard_normal(n0); it = [0]
    M = None if Mfun is None else spla.LinearOperator((n0, n0), matvec=Mfun)
    x, info = spla.cg(A00, rhs, M=M, rtol=tol, maxiter=2000, callback=lambda _: it.__setitem__(0, it[0]+1))
    return it[0], info
d = np.diag(A00)
print("inner CG (rtol 1e-6) none / Jacobi / Kronecker-term:", cg_count(None), cg_count(lambda v: v/d), cg_count(Pinv), flush=True)

# Outer: block-triangular preconditioner with inexact inner eps solves (PCG with the Kronecker term), Schur complement
# formed from the exact A00^{-1} (to isolate the effect of the inner solve) and from the Kronecker P^{-1}.
A01 = A[:n0, n0:]; A10 = A[n0:, :n0]; A11 = A[n0:, n0:]
PcM = spla.LinearOperator((n0, n0), matvec=Pinv)
def inner(v, tol):
    x, info = spla.cg(A00, v, M=PcM, rtol=tol, maxiter=500)
    return x
lu00 = sla.lu_factor(A00)
S_exact = A11 - A10 @ sla.lu_solve(lu00, A01)
S_kron = A11 - A10 @ np.column_stack([Pinv(A01[:, j]) for j in range(A01.shape[1])])
for sname, S in [("exact S", S_exact), ("Kron-P S", S_kron)]:
    luS = sla.lu_factor(S)
    for tol in (1e-2, 1e-3, 1e-6):
        def tri(v, tol=tol, luS=luS):
            v0, v1 = v[:n0], v[n0:]
            y1 = sla.lu_solve(luS, v1 - A10 @ inner(v0, tol)); y0 = inner(v0 - A01 @ y1, tol)
            return np.concatenate([y0, y1])
        M = spla.LinearOperator((n, n), matvec=tri)
        print(f"outer FGMRES-like (GMRES) {sname}, inner PCG rtol {tol:.0e}: its/info/res =", gmres_count(A, b, M, tol=1e-6, maxiter=200), flush=True)
