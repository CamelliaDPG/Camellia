"""Offline study of preconditioned GMRES on dumped block-AMEn site systems (BOBA_AMEN_DUMP_DIR)."""
import sys, glob, numpy as np, scipy.sparse.linalg as spla, scipy.linalg as sla

def load(base):
    meta = {}
    for line in open(base + "_meta.txt"):
        k, *v = line.split(); meta[k] = [float(x) for x in v]
    n = int(meta['rows'][0]); off = [int(x) for x in meta['offsets']]
    A = np.fromfile(base + "_A.bin").reshape(n, n, order='F')
    b = np.fromfile(base + "_b.bin")
    sh = lambda k: tuple(int(x) for x in meta[k])
    phiL = np.fromfile(base + "_phiL.bin").reshape(sh('phiL'), order='F')   # (rx, rx_, ra)
    Ac = np.fromfile(base + "_Acore.bin").reshape(sh('Acore'), order='F')   # (ra, row, col, rap1)
    phiR = np.fromfile(base + "_phiR.bin").reshape(sh('phiR'), order='F')   # (rxp1, rap1, rxp1_)
    return A, b, off, meta['row_factors'], phiL, Ac, phiR

def gmres_count(A, b, M=None, tol=1e-6, maxiter=400):
    it = [0]
    def cb(_): it[0] += 1
    x, info = spla.gmres(A, b, M=M, rtol=tol, restart=maxiter, maxiter=maxiter, callback=cb, callback_type='pr_norm')
    return it[0], info, np.linalg.norm(A @ x - b) / np.linalg.norm(b)

bases = sorted(set(f[:-6] for f in glob.glob(sys.argv[1] + "/*_A.bin")), key=lambda b: int(b.split('_n')[-1]))
for base in bases:
    A, b, off, rf, phiL, Ac, phiR = load(base)
    n0 = off[1]; n = A.shape[0]
    rx, _, ra = phiL.shape; _, nr, nc, rap1 = Ac.shape; rxp1 = phiR.shape[0]
    # local unknown ordering (rx, n, rxp1), first index fastest; matrix rows (rx,n,rxp1), cols (rx_,n_,rxp1_)
    G = np.einsum('ija,aklb,nbm->ikmjln', phiL, Ac, phiR).reshape(rx*nr*rxp1, rx*nc*rxp1, order='F')
    print(f"== {base.split('/')[-1]}: n={n} blocks={np.diff(off).tolist()} eps-block {rx}x{nr}x{rxp1}, op ranks {ra},{rap1}")
    print(f"   (0,0) block check: |A00 - rf0*G|/|A00| = {np.linalg.norm(A[:n0,:n0]-rf[0]*G)/np.linalg.norm(A[:n0,:n0]):.2e}")
    A00 = A[:n0, :n0]
    print(f"   A00 symmetry |A00-A00^T|/|A00| = {np.linalg.norm(A00-A00.T)/np.linalg.norm(A00):.1e}", flush=True)
    # nearest single Kronecker term of G as (rx x rx) (x) (n x n) (x) (rxp1 x rxp1): sum over a,b of phiL[:,:,a] (x) Ac[a,:,:,b] (x) phiR[:,b,:]
    # Use the dominant (a,b) term structure via SVD of the operator-rank sums: approximate each factor by its term-weighted sum.
    terms = [(phiL[:, :, a], Ac[a, :, :, c], phiR[:, c, :].T) for a in range(ra) for c in range(rap1)]
    norms = [np.linalg.norm(L)*np.linalg.norm(Ak)*np.linalg.norm(R) for L, Ak, R in terms]
    L, Ak, R = terms[int(np.argmax(norms))]
    print(f"   dominant Kronecker term carries {max(norms)/sum(norms):.2f} of the summed norms ({len(terms)} terms)")
    # exact eps-block inverse (dense, small enough here) and Kronecker approx inverse
    lu00 = sla.lu_factor(A00)
    P00 = rf[0] * np.kron(np.kron(R, Ak), L)  # ordering: (rx fastest) -> kron(R, kron(Ak, L))
    luP = sla.lu_factor(P00)
    print(f"   |A00 - P00|/|A00| = {np.linalg.norm(A00-P00)/np.linalg.norm(A00):.2f}")
    A01 = A[:n0, n0:]; A10 = A[n0:, :n0]; A11 = A[n0:, n0:]
    for label, inv00 in [("exact A00", lambda v: sla.lu_solve(lu00, v)), ("Kron P00", lambda v: sla.lu_solve(luP, v))]:
        S = A11 - A10 @ inv00(A01)
        luS = sla.lu_factor(S)
        def tri(v, inv00=inv00, luS=luS):
            v0, v1 = v[:n0], v[n0:]
            y1 = sla.lu_solve(luS, v1 - A10 @ inv00(v0)); y0 = inv00(v0 - A01 @ y1)
            return np.concatenate([y0, y1])
        def diag(v, inv00=inv00, luS=luS):
            return np.concatenate([inv00(v[:n0]), sla.lu_solve(luS, v[n0:])])
        for pname, f in [("block-LU (exact if exact A00)", tri), ("block-diagonal", diag)]:
            M = spla.LinearOperator((n, n), matvec=f)
            print(f"   {label:9s} {pname:30s}: GMRES its/info/res = {gmres_count(A, b, M)}", flush=True)
    print(f"   unpreconditioned: GMRES its/info/res = {gmres_count(A, b)}")
