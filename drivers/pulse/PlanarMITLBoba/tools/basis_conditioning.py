"""Conditioning of the TT-assembled DPG blocks for each 1D basis-family combination (test, field, trace)."""
import sys, numpy as np, scipy.linalg as sla
d = sys.argv[1]
def blocks(tag):
    meta = [l.split() for l in open(f"{d}/{tag}_blocks.txt")]
    B = {}
    for i, j, r, c, *_ in meta:
        B[int(i), int(j)] = np.fromfile(f"{d}/{tag}_{i}_{j}.bin").reshape(int(r), int(c), order='F')
    return B, max(int(m[0]) for m in meta) + 1
print(f"{'test/field/trace':18s} {'cond G':>9s} {'cond mixed':>11s} {'cond DPG K':>11s} {'cond K (Jacobi)':>15s}")
for t in 'NH':
    for f in 'NH':
        for r in 'NH':
            tag = t + f + r
            B, nb = blocks(tag)
            G = B[0, 0]
            Bt = np.hstack([B[0, j] for j in range(1, nb)])          # test x trial
            mixed = np.block([[G, Bt], [Bt.T, np.zeros((Bt.shape[1], Bt.shape[1]))]])
            K = Bt.T @ sla.cho_solve(sla.cho_factor(G), Bt)          # DPG stiffness
            dK = 1/np.sqrt(np.diag(K)); KJ = dK[:, None]*K*dK[None, :]
            ev = np.linalg.eigvalsh(G); eK = np.linalg.eigvalsh(K); eKJ = np.linalg.eigvalsh(KJ)
            sm = np.linalg.svd(mixed, compute_uv=False)
            name = '/'.join({'N': 'nodal', 'H': 'hier'}[x] for x in tag)
            print(f"{name:18s} {ev.max()/ev.min():9.2e} {sm[0]/sm[-1]:11.2e} {eK.max()/eK.min():11.2e} {eKJ.max()/eKJ.min():15.2e}", flush=True)
