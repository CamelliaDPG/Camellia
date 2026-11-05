"""Mixed-system conditioning after symmetric diagonal (Jacobi) scaling and after block-diagonal scaling with the exact
G block and the exact DPG Schur complement on the trial block, for every basis combination."""
import sys, numpy as np, scipy.linalg as sla
d = sys.argv[1]
def blocks(tag):
    meta = [l.split() for l in open(f"{d}/{tag}_blocks.txt")]
    B = {}
    for i, j, r, c, *_ in meta:
        B[int(i), int(j)] = np.fromfile(f"{d}/{tag}_{i}_{j}.bin").reshape(int(r), int(c), order='F')
    return B, max(int(m[0]) for m in meta) + 1
print(f"{'test/field/trace':20s} {'mixed, diag-scaled':>19s} {'mixed, |diag| of G and of K':>28s}")
for tag in sys.argv[2:]:
    B, nb = blocks(tag)
    G = B[0, 0]; Bt = np.hstack([B[0, j] for j in range(1, nb)]); m = Bt.shape[1]
    mixed = np.block([[G, Bt], [Bt.T, np.zeros((m, m))]])
    K = Bt.T @ sla.cho_solve(sla.cho_factor(G), Bt)
    # scaling 1: G's diagonal for the test rows, K's diagonal for the trial rows (zero diagonal in mixed)
    s = np.concatenate([1/np.sqrt(np.diag(G)), 1/np.sqrt(np.diag(K))])
    M1 = s[:, None]*mixed*s[None, :]
    sv = np.linalg.svd(M1, compute_uv=False)
    sv0 = np.linalg.svd(mixed, compute_uv=False)
    name = '/'.join({'N': 'nodal', 'H': 'hier'}[x] for x in tag)
    print(f"{name:20s} {sv0[0]/sv0[-1]:19.2e} {sv[0]/sv[-1]:28.2e}", flush=True)
