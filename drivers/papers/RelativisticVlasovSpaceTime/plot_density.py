#!/usr/bin/env python3
"""Plot the configuration-space electron density written by PlanarMITLRelativisticSpaceTime --densityDiagnostic.

Each density_slab<N>.csv holds, for every configuration (x,t) cell, a uniform pointsPerDim x pointsPerDim grid of points
(including the cell boundaries) at which n_e is the exact velocity moment of the polynomial f.  Plotting is per cell,
so discontinuities across cell boundaries (n_e comes from an L^2 field) are shown rather than smoothed over.

Usage:
  plot_density.py <outputDir or CSV files...> [--times t1 t2 ...] [--log] [--out density.png] [--no-cells]

Produces <out> with two panels: an x-t map of n_e (cell boundaries overlaid), and n_e(x) at several times (by
default, the end of each time slab).  A profile time snaps to the nearest sample row, so request times on the sample
grid (or raise --densityDiagnosticPointsPerDim) for exact times; the legend shows the time actually plotted.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import numpy as np

# single-hue sequential ramp (light -> dark) for magnitude
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d4"


def read_csvs(paths):
    """Returns (pointsPerDim, list of per-cell dicts with x, t, n_e as (pointsPerDim, pointsPerDim) arrays [t, x])."""
    cells = []
    pointsPerDim = None
    for path in paths:
        with open(path) as f:
            header = f.readline()
            if not header.startswith("# pointsPerDim="):
                sys.exit(f"{path}: not a density diagnostic file")
            n = int(header.split("=")[1])
            if pointsPerDim is not None and n != pointsPerDim:
                sys.exit(f"{path}: pointsPerDim {n} differs from {pointsPerDim}")
            pointsPerDim = n
        data = np.genfromtxt(path, delimiter=",", skip_header=2)
        data = data.reshape(-1, n * n, data.shape[-1])
        for cell in data:
            shape = (n, n)  # rows are t-major, x-minor
            cells.append({"x": cell[:, 1].reshape(shape), "t": cell[:, 2].reshape(shape),
                          "n_e": cell[:, 3].reshape(shape), "Jx": cell[:, 4].reshape(shape),
                          "E_x": cell[:, 5].reshape(shape)})
    return pointsPerDim, cells


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(TEXT_SECONDARY)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)
    ax.title.set_color(TEXT_PRIMARY)


def profile_at_time(cells, t):
    """n_e(x) on the cells containing time t, using each cell's sample row nearest t; returns (segments, actual t)."""
    # Each cell covers (t0, t1]; at a cell (or slab) boundary this takes the earlier cell's final state, so a profile at
    # the end of a slab is that slab's solution, not the next slab's initial one.  The first cells also cover t0.
    tFirst = min(cell["t"].min() for cell in cells)
    segments = []
    actualTimes = []
    for cell in cells:
        tRows = cell["t"][:, 0]
        t0, t1 = tRows.min(), tRows.max()
        tolerance = 1e-12 * max(1.0, abs(t1))
        inCell = (t0 + tolerance < t <= t1 + tolerance) or (abs(t - tFirst) <= tolerance and abs(t0 - tFirst) <= tolerance)
        if not inCell:
            continue
        row = int(np.argmin(np.abs(tRows - t)))
        segments.append((cell["x"][row], cell["n_e"][row]))
        actualTimes.append(tRows[row])
    if not segments:
        return [], None
    segments.sort(key=lambda seg: seg[0][0])
    return segments, actualTimes[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="output directory (uses density_slab*.csv) or CSV files")
    parser.add_argument("--times", type=float, nargs="*", help="times for n_e(x) profiles (default: end of each slab)")
    parser.add_argument("--log", action="store_true", help="logarithmic color scale for the x-t map")
    parser.add_argument("--floor", type=float, default=None, help="lower limit of the log color scale (default: max/1e4)")
    parser.add_argument("--no-cells", action="store_true", help="do not overlay configuration cell boundaries")
    parser.add_argument("--out", default=None, help="output image (default: <first input dir>/density.png)")
    args = parser.parse_args()

    paths = []
    for entry in args.inputs:
        if os.path.isdir(entry):
            found = glob.glob(os.path.join(entry, "density_slab*.csv"))
            paths += sorted(found, key=lambda p: int(os.path.basename(p)[len("density_slab"):-len(".csv")]))
        else:
            paths.append(entry)
    if not paths:
        sys.exit("no density_slab*.csv files found")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(paths[0])), "density.png")

    pointsPerDim, cells = read_csvs(paths)
    nMax = max(cell["n_e"].max() for cell in cells)
    nMin = min(cell["n_e"].min() for cell in cells)
    tMin = min(cell["t"].min() for cell in cells)
    tMax = max(cell["t"].max() for cell in cells)
    xMin = min(cell["x"].min() for cell in cells)
    xMax = max(cell["x"].max() for cell in cells)

    if args.times:
        times = args.times
    else:
        # ends of the time slabs: the top edge of the cells in each file
        times = sorted({round(float(cell["t"].max()), 12) for cell in cells})
        if len(times) > 6:
            times = [times[i] for i in np.linspace(0, len(times) - 1, 6).round().astype(int)]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": SURFACE})
    fig, (axMap, axProfile) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [3, 2]})
    for ax in (axMap, axProfile):
        style_axes(ax)

    cmap = LinearSegmentedColormap.from_list("density", BLUE_RAMP)
    if args.log:
        floor = args.floor if args.floor else nMax * 1e-4
        norm = LogNorm(vmin=floor, vmax=nMax)
        cmap.set_under(SURFACE)
    else:
        norm = Normalize(vmin=min(0.0, nMin), vmax=nMax)
    for cell in cells:
        values = np.maximum(cell["n_e"], norm.vmin * 0.5) if args.log else cell["n_e"]
        mesh = axMap.pcolormesh(cell["x"], cell["t"], values, shading="gouraud", cmap=cmap, norm=norm, rasterized=True)
        if not args.no_cells:
            x0, x1 = cell["x"][0, 0], cell["x"][0, -1]
            t0, t1 = cell["t"][0, 0], cell["t"][-1, 0]
            axMap.plot([x0, x1, x1, x0, x0], [t0, t0, t1, t1, t0], color="#ffffff", linewidth=0.4, alpha=0.6)
    colorbar = fig.colorbar(mesh, ax=axMap, pad=0.02)
    colorbar.set_label("electron density $n_e$", color=TEXT_PRIMARY)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
    colorbar.outline.set_edgecolor(TEXT_SECONDARY)
    axMap.set_xlim(xMin, xMax)
    axMap.set_ylim(tMin, tMax)
    axMap.set_xlabel("x")
    axMap.set_ylabel("t")
    axMap.set_title(f"Electron density (exact moment of f, {pointsPerDim}x{pointsPerDim} samples per cell)", loc="left")

    # profiles: ordered times use an ordinal ramp of the same hue, light (early) -> dark (late)
    ordinal = BLUE_RAMP[2:] if len(times) <= 5 else BLUE_RAMP[1:]
    colors = [ordinal[i] for i in np.linspace(0, len(ordinal) - 1, len(times)).round().astype(int)] if len(times) > 1 \
        else [BLUE_RAMP[-2]]
    for t, color in zip(times, colors):
        segments, actual = profile_at_time(cells, t)
        if not segments:
            print(f"warning: no cells contain t = {t}", file=sys.stderr)
            continue
        for i, (x, n) in enumerate(segments):
            axProfile.plot(x, n, color=color, linewidth=2, solid_capstyle="round",
                           label=(f"t = {actual:.4g}" if i == 0 else None))
    axProfile.set_xlim(xMin, xMax)
    axProfile.grid(True, color=GRID, linewidth=0.6)
    axProfile.set_axisbelow(True)
    axProfile.set_xlabel("x")
    axProfile.set_ylabel("$n_e$")
    axProfile.set_title("Density profiles", loc="left")
    if axProfile.get_legend_handles_labels()[0]:
        legend = axProfile.legend(frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(TEXT_PRIMARY)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
