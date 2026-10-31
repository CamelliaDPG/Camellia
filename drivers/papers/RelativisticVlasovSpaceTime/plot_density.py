#!/usr/bin/env python3
"""Plot the configuration-space electron density written by PlanarMITLRelativisticSpaceTime --densityDiagnostic.

Each density_slab<N>.csv holds, for every configuration (x,t) cell, a uniform pointsPerDim x pointsPerDim grid of points
(including the cell boundaries) at which n_e is the exact velocity moment of the polynomial f.  Configuration cells are
everywhere at least as fine as the phase-space cells, so on each cell n_e is a single polynomial of degree at most p (the
degree of f, recorded in the file header) in each of x and t.  When pointsPerDim > p, that polynomial is recovered
exactly from the samples (a least-squares fit in a tensor Legendre basis, which reproduces it to rounding) and plotted
at whatever resolution is wanted, so what is shown is the discrete solution itself.  A fit residual above rounding level
(which would mean the polynomial assumption failed) is reported.  With --sampled, or for files without a degree, the
samples are plotted directly, with linear interpolation between them.

Plotting is per cell, so discontinuities across cell boundaries (n_e comes from an L^2 field) are shown rather than
smoothed over.

Usage:
  plot_density.py <outputDir or CSV files...> [--times t1 t2 ...] [--log] [--out density.png] [--no-cells] [--sampled]
                  [--movie density.mp4 [--frames N] [--fps F]]

Produces <out> with two panels: an x-t map of n_e (cell boundaries overlaid), and n_e(x) at several times (by
default, the end of each time slab).  In polynomial mode, profiles are evaluated at exactly the requested times; in
sampled mode, a time snaps to the nearest sample row (the legend shows the time actually plotted).

With --movie, also writes an animation: the x-t map with a line at the current time, above n_e(x) at that time, on a
fixed n_e scale.  Frames are uniformly spaced in t over the whole run.  The format follows the extension: .mp4 needs
ffmpeg on the PATH; .gif uses Pillow (no other dependencies).
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import numpy as np
from numpy.polynomial import legendre

# single-hue sequential ramp (light -> dark) for magnitude
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d4"


def read_csvs(paths):
    """Returns (pointsPerDim, degree or None, list of per-cell dicts with x, t, n_e as (pointsPerDim, pointsPerDim) arrays
    [t, x])."""
    cells = []
    pointsPerDim = None
    degree = None
    for path in paths:
        with open(path) as f:
            header = f.readline()
            if not header.startswith("# pointsPerDim="):
                sys.exit(f"{path}: not a density diagnostic file")
            fields = dict(entry.split("=") for entry in header[1:].split())
            n = int(fields["pointsPerDim"])
            if pointsPerDim is not None and n != pointsPerDim:
                sys.exit(f"{path}: pointsPerDim {n} differs from {pointsPerDim}")
            pointsPerDim = n
            if "degree" in fields:
                degree = max(degree or 0, int(fields["degree"]))
        data = np.genfromtxt(path, delimiter=",", skip_header=2)
        data = data.reshape(-1, n * n, data.shape[-1])
        for cell in data:
            shape = (n, n)  # rows are t-major, x-minor
            cells.append({"x": cell[:, 1].reshape(shape), "t": cell[:, 2].reshape(shape),
                          "n_e": cell[:, 3].reshape(shape), "Jx": cell[:, 4].reshape(shape),
                          "E_x": cell[:, 5].reshape(shape)})
    return pointsPerDim, degree, cells


def fit_cell_polynomials(cells, pointsPerDim, degree):
    """Adds to each cell the tensor-Legendre coefficients (degree+1, degree+1) [t, x] of n_e on the reference cell, and its
    x and t extents.  Returns the maximum fit residual over all cells."""
    reference = np.linspace(-1.0, 1.0, pointsPerDim)
    vandermonde = legendre.legvander(reference, degree)  # (pointsPerDim, degree+1)
    pseudoinverse = np.linalg.pinv(vandermonde)
    maxResidual = 0.0
    for cell in cells:
        cell["xRange"] = (cell["x"][0, 0], cell["x"][0, -1])
        cell["tRange"] = (cell["t"][0, 0], cell["t"][-1, 0])
        # samples are on a tensor grid, so the 2D least-squares fit factors into 1D fits in each direction
        coefficients = pseudoinverse @ cell["n_e"] @ pseudoinverse.T
        cell["coefficients"] = coefficients
        residual = vandermonde @ coefficients @ vandermonde.T - cell["n_e"]
        maxResidual = max(maxResidual, float(np.abs(residual).max()))
    return maxResidual


def to_reference(value, interval):
    return 2.0 * (np.asarray(value) - interval[0]) / (interval[1] - interval[0]) - 1.0


def evaluate_cell(cell, x, t):
    """n_e of the cell's polynomial on the tensor grid t (rows) x x (columns), both physical."""
    return legendre.legval2d(*np.meshgrid(to_reference(t, cell["tRange"]), to_reference(x, cell["xRange"]), indexing="ij"),
                             cell["coefficients"])


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(TEXT_SECONDARY)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)
    ax.title.set_color(TEXT_PRIMARY)


def profile_at_time(cells, t, pointsPerCell=None):
    """n_e(x) on the cells containing time t; returns (segments, actual t).  With pointsPerCell, evaluates each cell's
    polynomial exactly at t; otherwise uses each cell's sample row nearest t."""
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
        if pointsPerCell:
            x = np.linspace(*cell["xRange"], pointsPerCell)
            segments.append((x, evaluate_cell(cell, x, [t])[0]))
            actualTimes.append(t)
        else:
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
    parser.add_argument("--vmax", type=float, default=None,
                        help="upper limit of the color scale (default: max of n_e; values above it saturate)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="lower limit of the linear color scale (default: min(0, min of n_e); values below it saturate)")
    parser.add_argument("--no-cells", action="store_true", help="do not overlay configuration cell boundaries")
    parser.add_argument("--out", default=None, help="output image (default: <first input dir>/density.png)")
    parser.add_argument("--sampled", action="store_true",
                        help="plot the samples with linear interpolation, rather than the recovered polynomials")
    parser.add_argument("--movie", default=None, help="also write an animation of n_e(x) over time (.mp4 or .gif)")
    parser.add_argument("--frames", type=int, default=200, help="number of movie frames (default 200)")
    parser.add_argument("--fps", type=int, default=20, help="movie frames per second (default 20)")
    parser.add_argument("--resolution", type=int, default=1200,
                        help="approximate evaluation points across the x and t extents for the polynomial map (default 1200)")
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

    pointsPerDim, degree, cells = read_csvs(paths)
    polynomial = not args.sampled
    if polynomial and degree is None:
        print("warning: no polynomial degree in the file header (older driver); plotting samples", file=sys.stderr)
        polynomial = False
    if polynomial and pointsPerDim <= degree:
        print(f"warning: pointsPerDim = {pointsPerDim} cannot determine a degree-{degree} polynomial; plotting samples",
              file=sys.stderr)
        polynomial = False
    if polynomial:
        maxResidual = fit_cell_polynomials(cells, pointsPerDim, degree)
        scale = max(float(np.abs(cell["n_e"]).max()) for cell in cells)
        if maxResidual > 1e-8 * scale:
            print(f"warning: polynomial fit residual {maxResidual:.3g} (relative {maxResidual / scale:.3g}); n_e is not a "
                  f"degree-{degree} polynomial on every configuration cell, so the polynomial plot is not exact",
                  file=sys.stderr)
    nMax = max(cell["n_e"].max() for cell in cells)
    nMin = min(cell["n_e"].min() for cell in cells)
    tMin = min(cell["t"].min() for cell in cells)
    tMax = max(cell["t"].max() for cell in cells)
    xMin = min(cell["x"].min() for cell in cells)
    xMax = max(cell["x"].max() for cell in cells)

    if polynomial:
        # extrema of the polynomials can lie between samples; take them from a finer evaluation
        fine = [evaluate_cell(cell, np.linspace(*cell["xRange"], 4 * pointsPerDim), np.linspace(*cell["tRange"], 4 * pointsPerDim))
                for cell in cells]
        nMax = max(nMax, max(float(values.max()) for values in fine))
        nMin = min(nMin, min(float(values.min()) for values in fine))

    if args.times:
        times = args.times
    else:
        # ends of the time slabs: the top edge of the cells in each file
        times = sorted({round(float(cell["t"].max()), 12) for cell in cells})
        if len(times) > 6:
            times = [times[i] for i in np.linspace(0, len(times) - 1, 6).round().astype(int)]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": SURFACE})
    cmap = LinearSegmentedColormap.from_list("density", BLUE_RAMP)
    if args.vmax is not None:
        nMax = args.vmax
    if args.log:
        floor = args.floor if args.floor else nMax * 1e-4
        norm = LogNorm(vmin=floor, vmax=nMax)
        cmap.set_under(SURFACE)
    else:
        norm = Normalize(vmin=(args.vmin if args.vmin is not None else min(0.0, nMin)), vmax=nMax)

    def new_figure():
        fig, (axMap, axProfile) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [3, 2]})
        for ax in (axMap, axProfile):
            style_axes(ax)
        draw_map(fig, axMap)
        axProfile.set_xlim(xMin, xMax)
        axProfile.grid(True, color=GRID, linewidth=0.6)
        axProfile.set_axisbelow(True)
        axProfile.set_xlabel("x")
        axProfile.set_ylabel("$n_e$")
        return fig, axMap, axProfile

    def draw_map(fig, axMap):
        for cell in cells:
            if polynomial:
                # enough points per cell that the interpolation between them is below the image resolution
                nx = max(pointsPerDim, int(np.ceil(args.resolution * (cell["xRange"][1] - cell["xRange"][0]) / (xMax - xMin))) + 1)
                nt = max(pointsPerDim, int(np.ceil(args.resolution * (cell["tRange"][1] - cell["tRange"][0]) / (tMax - tMin))) + 1)
                x, t = np.linspace(*cell["xRange"], nx), np.linspace(*cell["tRange"], nt)
                X, T = np.meshgrid(x, t)
                values = evaluate_cell(cell, x, t)
            else:
                X, T, values = cell["x"], cell["t"], cell["n_e"]
            if args.log:
                values = np.maximum(values, norm.vmin * 0.5)
            mesh = axMap.pcolormesh(X, T, values, shading="gouraud", cmap=cmap, norm=norm, rasterized=True)
            if not args.no_cells:
                x0, x1 = cell["x"][0, 0], cell["x"][0, -1]
                t0, t1 = cell["t"][0, 0], cell["t"][-1, 0]
                axMap.plot([x0, x1, x1, x0, x0], [t0, t0, t1, t1, t0], color="#ffffff", linewidth=0.4, alpha=0.6)
        colorbar = fig.colorbar(mesh, ax=axMap, pad=0.02, extend=("max" if args.vmax is not None else "neither"))
        colorbar.set_label("electron density $n_e$", color=TEXT_PRIMARY)
        colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
        colorbar.outline.set_edgecolor(TEXT_SECONDARY)
        axMap.set_xlim(xMin, xMax)
        axMap.set_ylim(tMin, tMax)
        axMap.set_xlabel("x")
        axMap.set_ylabel("t")
        if polynomial:
            axMap.set_title(f"Electron density (exact moment of f: degree-{degree} polynomial per cell)", loc="left")
        else:
            axMap.set_title(f"Electron density (exact moment of f, {pointsPerDim}x{pointsPerDim} samples per cell, "
                            "linearly interpolated)", loc="left")

    fig, axMap, axProfile = new_figure()

    # profiles: ordered times use an ordinal ramp of the same hue, light (early) -> dark (late)
    ordinal = BLUE_RAMP[2:] if len(times) <= 5 else BLUE_RAMP[1:]
    colors = [ordinal[i] for i in np.linspace(0, len(ordinal) - 1, len(times)).round().astype(int)] if len(times) > 1 \
        else [BLUE_RAMP[-2]]
    for t, color in zip(times, colors):
        segments, actual = profile_at_time(cells, t, pointsPerCell=(200 if polynomial else None))
        if not segments:
            print(f"warning: no cells contain t = {t}", file=sys.stderr)
            continue
        for i, (x, n) in enumerate(segments):
            axProfile.plot(x, n, color=color, linewidth=2, solid_capstyle="round",
                           label=(f"t = {actual:.4g}" if i == 0 else None))
    axProfile.set_title("Density profiles", loc="left")
    if axProfile.get_legend_handles_labels()[0]:
        legend = axProfile.legend(frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(TEXT_PRIMARY)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")

    if args.movie:
        write_movie(args, new_figure, cells, polynomial, tMin, tMax, nMin, nMax)


def write_movie(args, new_figure, cells, polynomial, tMin, tMax, nMin, nMax):
    if args.movie.endswith(".gif"):
        writer = animation.PillowWriter(fps=args.fps)
    elif animation.FFMpegWriter.isAvailable():
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    else:
        sys.exit("ffmpeg not found: install it, or write a .gif movie instead")

    fig, axMap, axProfile = new_figure()
    padding = 0.05 * (nMax - min(0.0, nMin))
    axProfile.set_ylim(min(0.0, nMin) - padding, nMax + padding)
    timeLine = axMap.axhline(tMin, color=TEXT_PRIMARY, linewidth=1.2)
    profileColor = BLUE_RAMP[-2]
    lines = [axProfile.plot([], [], color=profileColor, linewidth=2, solid_capstyle="round")[0] for _ in cells]
    title = axProfile.set_title("", loc="left")
    fig.tight_layout()

    times = np.linspace(tMin, tMax, args.frames)
    pointsPerCell = 200 if polynomial else None

    def update(t):
        segments, actual = profile_at_time(cells, t, pointsPerCell=pointsPerCell)
        for line, segment in zip(lines, segments + [(np.array([]), np.array([]))] * (len(lines) - len(segments))):
            line.set_data(*segment)
        timeLine.set_ydata([actual, actual])
        title.set_text(f"Density profile, t = {actual:.4g}")
        return lines + [timeLine, title]

    with writer.saving(fig, args.movie, dpi=120):
        for frame, t in enumerate(times):
            update(t)
            writer.grab_frame()
            if (frame + 1) % 50 == 0:
                print(f"  frame {frame + 1} of {len(times)}", file=sys.stderr)
    plt.close(fig)
    print(f"wrote {args.movie}")


if __name__ == "__main__":
    main()
