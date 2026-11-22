#!/usr/bin/env python3
"""Per-slab summary of a coupled run, for the cathode/anode negative-density mode tests.

usage: mode_summary.py run_dir [run_dir ...]

For each slab: the local side's measure (at the slab's final time, the integral over x of n and of its negative part,
over the gap, the cathode cells x < 0.0234 and the anode side x >= 0.5, and min n there); then over the whole slab: min n_e
and the x where it occurs (from density_slab<N>.csv), the integral of the negative part of n_e
over the slab's (x, t) (cell means times the cell area), f min (console), int f^- over phase space at the slab end
(phase_reductions.csv), the emitted density and E(0) at the slab end, and the relative charge-balance error.
"""
import sys, os, re, glob
import numpy as np

def console(run):
    text = open(os.path.join(run, "console.log")).read()
    fmin = {int(s): float(v) for s, v in re.findall(r"Time slab (\d+), f min: (\S+)", text)}
    emitted = [float(v) for v in re.findall(r"emitted density (\S+) n0", text)]
    e0 = [float(v) for v in re.findall(r"E\(0\) at final time is \S+ \((\S+) E0\)", text)]
    charge = {int(s): float(v) for s, v in re.findall(r"CHARGE_BALANCE slab=(\d+) .*? relative=(\S+)", text)}
    return fmin, emitted, e0, charge

def reductions(run):
    path = os.path.join(run, "phase_reductions.csv")
    if not os.path.exists(path): return {}
    rows = np.atleast_2d(np.genfromtxt(path, delimiter=",", names=True))
    return {int(r["slab"]): float(r["int_f_negative_part"]) for r in rows.ravel()}

def density(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1, names=True)
    n = d["n_e"]
    i = int(np.argmin(n))
    neg = 0.0
    for c in np.unique(d["cell"]):
        sel = d["cell"] == c
        hx = np.ptp(d["x"][sel]); ht = np.ptp(d["t"][sel])
        neg += np.minimum(n[sel], 0).mean() * hx * ht
    # the local side's measure: at the slab's final time, the integral over x of n and of its negative part, over the gap,
    # the cathode cells (x < 0.0234) and the anode side (x >= 0.5); trapezoidal rule on each cell's points
    tEnd = d["t"].max()
    end = np.abs(d["t"] - tEnd) < 1e-12
    regions = {"gap": (0.0, 2.0), "cathode": (0.0, 0.0234375), "anode": (0.5, 2.0)}
    endInt = {k: [0.0, 0.0] for k in regions}
    nEndMin = np.inf
    for c in np.unique(d["cell"][end]):
        sel = end & (d["cell"] == c)
        order = np.argsort(d["x"][sel])
        x = d["x"][sel][order]; nn = n[sel][order]
        nEndMin = min(nEndMin, nn.min())
        xc = 0.5 * (x[0] + x[-1])
        for k, (lo, hi) in regions.items():
            if lo <= xc < hi:
                endInt[k][0] += np.trapezoid(nn, x)
                endInt[k][1] += np.trapezoid(np.minimum(nn, 0), x)
    return n[i], d["x"][i], d["t"][i], neg, endInt, nEndMin

for run in sys.argv[1:]:
    fmin, emitted, e0, charge = console(run)
    red = reductions(run)
    print(f"== {os.path.basename(os.path.normpath(run))}")
    print(" slab  | at slab end: int n   int n^- gap  cathode     anode    min n |  slab: min n_e  at x  (t)   f min     int f^-"
          "   emitted/n0  E(0)/E0  charge rel")
    files = sorted(glob.glob(os.path.join(run, "density_slab*.csv")), key=lambda p: int(re.search(r"slab(\d+)", p).group(1)))
    for p in files:
        s = int(re.search(r"slab(\d+)", p).group(1))
        nmin, x, t, neg, endInt, nEndMin = density(p)
        em = emitted[s] if s < len(emitted) else float("nan")
        e = e0[s] if s < len(e0) else float("nan")
        print(f" {s:4d}  | {endInt['gap'][0]:13.4g} {endInt['gap'][1]:11.3g} {endInt['cathode'][1]:9.3g} {endInt['anode'][1]:9.3g}"
              f" {nEndMin:8.3g} | {nmin:9.4g} {x:7.4f} ({t:5.2f}) {fmin.get(s, float('nan')):8.4g} {red.get(s, float('nan')):9.4g}"
              f" {em:10.4g} {e:10.3g} {charge.get(s, float('nan')):10.3g}")
