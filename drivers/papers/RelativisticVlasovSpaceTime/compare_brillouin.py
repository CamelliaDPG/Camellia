#!/usr/bin/env python3
"""Compare a computed planar MITL electron-density profile with the Lau et al. (2021) planar Brillouin flow.

Usage:
  compare_brillouin.py <outputDir or density_slab*.csv> [--V 1e6] [--ratio 1.5] [--slabs 5] [--out brillouin.png]

The Brillouin flow is evaluated from Lau et al., IEEE TPS 2021, Eqs. (2.12)-(2.30) at normalized gap voltage
V/(mc^2/e) and flux ratio A_a/A_a^min; lengths are in units of the gap, density in units of n_s = eps0 m c^2/(e^2 D^2)
(the driver's density scaling).  The computed profile is the time average of n_e(x) over the last few slabs (default 5),
evaluated exactly from each cell's polynomial (see plot_density.py).
"""

import argparse
import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_density as pd  # noqa: E402

MC2_OVER_E = 510998.95  # V


def brillouin(V, ratio):
    """Lau et al. (2021) planar Brillouin flow; returns (kappa, chi_b, x_b, dict of scalars)."""
    gamma_a = V + 1.0                                 # (2.27)
    A_min = math.sqrt(gamma_a * gamma_a - 1.0)        # (2.28), = sqrt(V^2 + 2V)
    A = ratio * A_min
    chi_b = math.atanh((gamma_a * A - math.sqrt(A * A - A_min * A_min)) / (A * A + 1.0))  # (2.26)
    kappa = (V - (math.cosh(chi_b) - 1.0) + chi_b * math.sinh(chi_b)) / math.sinh(chi_b)  # (2.29)
    x_b = chi_b / kappa                               # (2.30)
    return kappa, chi_b, x_b, {
        "A_min": A_min, "A_a": A, "beta_b": math.tanh(chi_b), "n_cathode": kappa ** 2,
        "n_hub_top": kappa ** 2 * math.cosh(chi_b), "sheath_charge": kappa * math.sinh(chi_b),
        "I_e_over_I_a": (math.cosh(chi_b) - 1.0) / math.cosh(chi_b),
    }


def brillouin_density(x, kappa, x_b):
    """n(x) = d/dx [chi' sinh(chi)] with chi = kappa x (2.18-2.19): kappa^2 cosh(kappa x) inside the hub, 0 outside."""
    x = np.asarray(x)
    return np.where(x <= x_b, kappa ** 2 * np.cosh(kappa * x), 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--V", type=float, default=1e6, help="gap voltage [V] (default 1e6)")
    parser.add_argument("--ratio", type=float, default=1.5, help="A_a / A_a^min (default 1.5)")
    parser.add_argument("--slabs", type=int, default=5, help="average over this many final slabs (default 5)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    paths = []
    for entry in args.inputs:
        if os.path.isdir(entry):
            found = glob.glob(os.path.join(entry, "density_slab*.csv"))
            paths += sorted(found, key=lambda p: int(os.path.basename(p)[len("density_slab"):-len(".csv")]))
        else:
            paths.append(entry)
    paths = paths[-args.slabs:]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(paths[-1])), "brillouin-comparison.png")

    V = args.V / MC2_OVER_E
    kappa, chi_b, x_b, scalars = brillouin(V, args.ratio)
    print("Lau Brillouin flow at V/(mc^2/e) = %.4f, A_a/A_a^min = %.3f:" % (V, args.ratio))
    print("  kappa = %.4f, chi_b = %.4f, hub height x_b = %.4f D" % (kappa, chi_b, x_b))
    for key, value in scalars.items():
        print("  %s = %.4f" % (key, value))

    # time-averaged computed profile: evaluate each cell polynomial at its own rows over the slab, then average
    xGrid = np.linspace(0.0, 1.0, 801)
    profiles = []
    for path in paths:
        pointsPerDim, degree, cells = pd.read_csvs([path])
        if degree is not None and pointsPerDim > degree:
            pd.fit_cell_polynomials(cells, pointsPerDim, degree)
            tValues = np.linspace(min(c["tRange"][0] for c in cells), max(c["tRange"][1] for c in cells), 9)[1:]
            for t in tValues:
                segments, _ = pd.profile_at_time(cells, t, pointsPerCell=200)
                xs = np.concatenate([s[0] for s in segments]); ns = np.concatenate([s[1] for s in segments])
                order = np.argsort(xs)
                profiles.append(np.interp(xGrid, xs[order], ns[order]))
        else:
            tFinal = max(c["t"].max() for c in cells)
            segments, _ = pd.profile_at_time(cells, tFinal)
            xs = np.concatenate([s[0] for s in segments]); ns = np.concatenate([s[1] for s in segments])
            order = np.argsort(xs)
            profiles.append(np.interp(xGrid, xs[order], ns[order]))
    computed = np.mean(profiles, axis=0)
    print("computed (mean over %d slabs): n(0) = %.3f, integral of n over the gap = %.4f"
          % (len(paths), computed[0], np.trapezoid(computed, xGrid) if hasattr(np, "trapezoid") else np.trapz(computed, xGrid)))
    halfMaxIndex = np.argmax(computed < 0.5 * computed.max())
    print("computed sheath half-density point: x = %.3f D" % xGrid[halfMaxIndex])

    plt.rcParams.update({"font.size": 10, "figure.facecolor": pd.SURFACE})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pd.style_axes(ax)
    ax.plot(xGrid, computed, color=pd.BLUE_RAMP[-2], linewidth=2, label="computed (mean of last %d slabs)" % len(paths))
    ax.plot(xGrid, brillouin_density(xGrid, kappa, x_b), color=pd.TEXT_SECONDARY, linewidth=1.5, linestyle="--",
            label="Brillouin flow (Lau et al. 2021)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("x / D")
    ax.set_ylabel("$n_e / n_s$")
    ax.grid(True, color=pd.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(pd.TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
