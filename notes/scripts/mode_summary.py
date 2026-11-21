#!/usr/bin/env python3
"""Per-slab summary of a coupled run, for the cathode/anode negative-density mode tests.

usage: mode_summary.py run_dir [run_dir ...]

For each slab: min n_e and the x where it occurs (from density_slab<N>.csv), the integral of the negative part of n_e
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
    return n[i], d["x"][i], d["t"][i], neg

for run in sys.argv[1:]:
    fmin, emitted, e0, charge = console(run)
    red = reductions(run)
    print(f"== {os.path.basename(os.path.normpath(run))}")
    print(" slab   min n_e  at x      (t)     int n_e^-     f min     int f^-   emitted/n0  E(0)/E0   charge rel")
    files = sorted(glob.glob(os.path.join(run, "density_slab*.csv")), key=lambda p: int(re.search(r"slab(\d+)", p).group(1)))
    for p in files:
        s = int(re.search(r"slab(\d+)", p).group(1))
        nmin, x, t, neg = density(p)
        em = emitted[s] if s < len(emitted) else float("nan")
        e = e0[s] if s < len(e0) else float("nan")
        print(f" {s:4d} {nmin:9.4g} {x:8.4f} ({t:6.3f}) {neg:11.4g} {fmin.get(s, float('nan')):10.4g} {red.get(s, float('nan')):10.4g}"
              f" {em:10.4g} {e:10.3g} {charge.get(s, float('nan')):10.3g}")
