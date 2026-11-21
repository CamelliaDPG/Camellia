#!/usr/bin/env python3
"""Low-rank tensor fits of the exact DPG diagonal (from --matrixFreeDiagonalFit's dump).

Each free dof gets a place in a tensor-product layout, per direction k (x, t, u_x, u_y):
  - a field dof, or a flux dof in a tangential direction: index e_k * L_k + l_k, e_k the cell's element index in that
    direction and l_k the label of its one-dimensional component (L_k labels);
  - a flux dof in its normal direction: the face's node index.
The dofs are split into blocks: fields, and per normal direction the fluxes on the lower boundary, the interior
faces, and the upper boundary, each further split by the sign of the cell's velocity box where the boundary block is
only partly free (inflow faces are Dirichlet).  Each block is a full grid (checked).  The fits are of g = d^(-1/2), as on
the TT side, and the fitted diagonal is 1 / g_fit^2.

usage: fit_diagonal.py dump outdir
writes outdir/<variant>.txt ("lid value") and outdir/fits.ready-list ("variant file" lines), and prints fit statistics.
"""
import sys, os
import numpy as np

dump, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
rows = np.loadtxt(dump, comments='#')
lid = rows[:, 0].astype(int)
dirichlet = rows[:, 1].astype(int)
diag = rows[:, 2]
normal = rows[:, 3].astype(int)
endpoint = rows[:, 4].astype(int)
dim = (rows.shape[1] - 5) // 3
lower = rows[:, 5:5 + dim]
width = rows[:, 5 + dim:5 + 2 * dim]
label = rows[:, 5 + 2 * dim:5 + 3 * dim].astype(int)
free = (dirichlet == 0) & (label[:, 0] != -2)
print(f"{len(lid)} dofs, {free.sum()} free with a layout, dim {dim}")

def key(v):
    return np.round(v * 1e9).astype(np.int64)

# element index per direction, from the cells' lower corners; node index from face coordinates
elementIndex = np.zeros((len(lid), dim), dtype=int)
nodeIndex = np.full(len(lid), -1)
for k in range(dim):
    lo = key(lower[:, k])
    uniq = np.unique(lo)
    elementIndex[:, k] = np.searchsorted(uniq, lo)
# nodes per normal direction
faceCoord = np.zeros(len(lid))
for k in range(dim):
    sel = normal == k
    faceCoord[sel] = lower[sel, k] + endpoint[sel] * width[sel, k]
nodeCount = {}
for k in range(dim):
    sel = normal == k
    if not sel.any():
        continue
    allNodes = np.unique(np.concatenate([key(lower[:, k]), key(lower[:, k] + width[:, k])]))
    nodeIndex[sel] = np.searchsorted(allNodes, key(faceCoord[sel]))
    nodeCount[k] = len(allNodes)

# local labels per (block kind, direction) -> 0..L-1
def localLabels(sel, k):
    uniq = np.unique(label[sel, k])
    return uniq, np.searchsorted(uniq, label[sel, k])

blocks = []  # (name, indices into rows, mode index arrays)
for n in [-1] + list(range(dim)):
    selN = free & (normal == n)
    if not selN.any():
        continue
    if n == -1:
        parts = [("field", selN)]
    else:
        parts = []
        last = nodeCount[n] - 1
        for where, cond in (("lower", nodeIndex == 0), ("interior", (nodeIndex > 0) & (nodeIndex < last)), ("upper", nodeIndex == last)):
            sel = selN & cond
            if sel.any():
                parts.append((f"normal{n}-{where}", sel))
    for name, sel in parts:
        modes = []
        for k in range(dim):
            if k == n:
                modes.append(nodeIndex[sel])
            else:
                uniqLabels, l = localLabels(sel, k)
                modes.append(elementIndex[sel, k] * len(uniqLabels) + l)
        # compress each mode to the indices present (a boundary block covers a sub-box of the cells)
        compressed = []
        for m in modes:
            uniq = np.unique(m)
            compressed.append(np.searchsorted(uniq, m))
        shape = tuple(int(c.max()) + 1 for c in compressed)
        size = int(np.prod(shape))
        count = int(sel.sum())
        flat = np.ravel_multi_index(tuple(compressed), shape)
        if len(np.unique(flat)) != count:
            print(f"  block {name}: repeated tensor indices ({count} dofs, {len(np.unique(flat))} distinct): layout error")
            sys.exit(1)
        blocks.append((name, np.nonzero(sel)[0], shape, flat, count / size))
        print(f"  block {name}: {count} dofs, shape {shape}, fill {count / size:.4f}")

def ttsvd(T, eps=None, rmax=None):
    """TT-SVD of T with relative Frobenius tolerance eps (split evenly over the d-1 truncations) and/or rank cap."""
    shape = T.shape
    d = len(shape)
    delta = (eps / np.sqrt(d - 1)) * np.linalg.norm(T) if eps is not None else 0.0
    cores = []
    C = T.copy()
    r = 1
    ranks = []
    for k in range(d - 1):
        C = C.reshape(r * shape[k], -1)
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        # smallest rank with tail <= delta
        tail = np.sqrt(np.cumsum((s ** 2)[::-1]))[::-1]  # tail[j] = norm of s[j:]
        rk = len(s)
        for j in range(1, len(s) + 1):
            if j == len(s) or tail[j] <= delta:
                rk = j
                break
        if rmax is not None:
            rk = min(rk, rmax)
        cores.append(U[:, :rk].reshape(r, shape[k], rk))
        C = s[:rk, None] * Vt[:rk, :]
        r = rk
        ranks.append(rk)
    cores.append(C.reshape(r, shape[-1], 1))
    return cores, ranks

def ttfull(cores):
    T = cores[0]
    for G in cores[1:]:
        T = np.tensordot(T, G, axes=([-1], [0]))
    return T.reshape(T.shape[1:-1])

def rank1als(T, sweeps=50):
    """best CP rank-one approximation (alternating least squares from the TT-SVD start)."""
    cores, _ = ttsvd(T, rmax=1)
    vs = [c.reshape(-1) for c in cores]
    d = T.ndim
    for _ in range(sweeps):
        for k in range(d):
            M = np.moveaxis(T, k, 0).reshape(T.shape[k], -1)
            other = np.ones(1)
            for j in range(d):
                if j != k:
                    other = np.kron(other, vs[j])
            denom = other @ other
            vs[k] = M @ other / denom
    R = vs[0]
    for v in vs[1:]:
        R = np.multiply.outer(R, v)
    return R

def additiveLog(T):
    """least squares fit of log T by a sum of one-dimensional terms (a rank-one product in T): ANOVA main effects."""
    L = np.log(T)
    F = np.full(T.shape, L.mean())
    for k in range(T.ndim):
        axes = tuple(j for j in range(T.ndim) if j != k)
        effect = L.mean(axis=axes) - L.mean()
        shape = [1] * T.ndim
        shape[k] = -1
        F = F + effect.reshape(shape)
    return np.exp(F)

variants = {
    "exact": None,
    "rank1-als": ("rank1",),
    "rank1-logadditive": ("logadd",),
    "tt-rank2": ("rmax", 2),
    "tt-rank4": ("rmax", 4),
    "tt-tol0.1": ("eps", 0.1),
    "tt-tol0.03": ("eps", 0.03),
    "tt-tol0.01": ("eps", 0.01),
}
readyLines = []
for vname, spec in variants.items():
    fitted = diag.copy()
    stats = []
    for name, idx, shape, flat, fill in blocks:
        g = np.full(int(np.prod(shape)), np.nan)
        g[flat] = diag[idx] ** -0.5
        if fill < 1:
            g[np.isnan(g)] = np.nanmean(g)  # not expected (checked above)
        T = g.reshape(shape)
        if spec is None:
            F = T
            info = ""
        elif spec[0] == "rank1":
            F = rank1als(T)
            info = "rank 1"
        elif spec[0] == "logadd":
            F = additiveLog(T)
            info = "rank 1 (log-additive)"
        elif spec[0] == "rmax":
            cores, ranks = ttsvd(T, rmax=spec[1])
            F = ttfull(cores)
            info = f"ranks {ranks}"
        else:
            cores, ranks = ttsvd(T, eps=spec[1])
            F = ttfull(cores)
            info = f"ranks {ranks}"
        err = np.linalg.norm(F - T) / np.linalg.norm(T)
        gf = F.reshape(-1)[flat]
        ratio = (gf / g[flat]) ** -2  # fitted diagonal / exact diagonal
        nonpositive = int((np.abs(gf) < 1e-300).sum())
        fitted[idx] = 1.0 / np.maximum(gf ** 2, 1e-300)
        stats.append(f"{name}: {info} err {err:.3g}, d_fit/d in [{ratio.min():.3g}, {ratio.max():.3g}]" + (f", {nonpositive} zero" if nonpositive else ""))
    path = os.path.join(outdir, vname + ".txt")
    with open(path, "w") as f:
        for i in np.nonzero(free)[0]:
            f.write(f"{lid[i]} {fitted[i]:.17g}\n")
    readyLines.append(f"{vname} {path}")
    print(f"variant {vname}:")
    for s in stats:
        print("   ", s)
with open(os.path.join(outdir, "fits.ready-list"), "w") as f:
    f.write("\n".join(readyLines) + "\n")
