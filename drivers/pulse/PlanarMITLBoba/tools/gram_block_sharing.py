#!/usr/bin/env python3
"""How many distinct element Gram blocks G_K are there, up to scaling, and up to a diagonal (per-axis scaling)
congruence?  Input: an element-block dump (<prefix>_Gelem.{txt,bin}, from PLANAR_MITL_DUMP_SPARSE).

For the Schur-complement idea of sharing element solves: if G_K depends on the element only through a few parameters,
G_K^{-1} can be shared (or cheaply transformed) across elements instead of factored per element.

Usage: gram_block_sharing.py <prefix> [--tol 1e-10]
"""
import argparse, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--tol", type=float, default=1e-10)
a = ap.parse_args()
info = [int(x) for x in open(a.prefix + "_Gelem.txt").read().split()]
cells = info[0:4]; bs = info[8]; ne = int(np.prod(cells))
data = np.fromfile(a.prefix + "_Gelem.bin").reshape(ne, bs, bs)

def cell_of(k):
    c = []
    for n in cells: c.append(k % n); k //= n
    return tuple(c)

def classes(normalize):
    reps, labels = [], []
    for k in range(ne):
        Gk = normalize(data[k])
        for ci, R in enumerate(reps):
            if np.linalg.norm(Gk - R) <= a.tol * np.linalg.norm(R): labels.append(ci); break
        else:
            reps.append(Gk); labels.append(len(reps) - 1)
    return reps, labels

# (1) identical up to a scalar
reps, labels = classes(lambda G: G / np.linalg.norm(G))
print(f"{ne} elements (cells {cells}); distinct blocks up to a scalar: {len(reps)}")
# (2) up to symmetric diagonal scaling D G D (absorbs per-axis Jacobian scalings of the basis): normalize by the diagonal
reps2, labels2 = classes(lambda G: G / np.sqrt(np.outer(np.diag(G), np.diag(G))))
print(f"distinct blocks up to diagonal congruence D G D: {len(reps2)}")
# which cell indices do the classes depend on?
for name, lab in [("scalar", labels), ("diagonal", labels2)]:
    dep = []
    for d in range(4):
        # does the class change when only cell index d changes?
        changes = False
        for k in range(ne):
            c = list(cell_of(k))
            if c[d] + 1 < cells[d]:
                c2 = c.copy(); c2[d] += 1
                k2 = sum(c2[i] * int(np.prod(cells[:i])) for i in range(4))
                if lab[k] != lab[k2]: changes = True; break
        dep.append(changes)
    print(f"  classes ({name}) depend on cell index in (x, t, ux, uz): {dep}")
