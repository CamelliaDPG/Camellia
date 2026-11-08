#!/usr/bin/env python3
"""Velocity-space figures for the planar MITL runs, from PlanarMITLRelativisticSpaceTime's --phaseSpaceDiagnostic and
--densityDiagnostic output.

Usage:
  plot_phase_space.py <outputDir> [--slabs 0 1 2 5 10 19] [--x 0 0.05 0.15 0.3] [--uxlim -1.1 1.5] [--uzlim 1.3]
                      [--average 10] [--out <dir>]

Inputs in <outputDir>:
  phase_slab<N>_part<r>.npy, phase_slab<N>_meta.txt  samples of f at the end of slab N on every phase-space cell, on a
                                                       uniform (n x n x n) grid in (x, u_x, u_z) (n = degree + 1 by default)
  phase_reductions.csv                                 integrals of f at every slab end
  density_slab<N>.csv                                  n_e, Jx, E_x, Jz (Jz since 2026-11-07)

Each cell's samples determine its polynomial f exactly, so everything below is evaluated from the polynomial (Lagrange
interpolation on the uniform points), not from a coarse interpolant.  The driver's second velocity coordinate (stored
as u_y) is the physical u_z; it is labeled u_z here.  Units: proper velocity u = gamma v in units of c; x in units of
the gap D; f in the driver's density scaling.

Figures written to <out> (default <outputDir>/phase-space):
  portrait_slab<N>.png      the (x, u_x) phase portrait, int f du_z; negative values in red (diverging scale)
  portraits.png             the same for all requested slabs, formation sequence (shared scale)
  velocity_slab<N>.png      f(u_x, u_z) at the requested x positions
  negative_slab<N>.png      where f < 0: int of the negative part of f over u_z, on (x, u_x)
  moments_slab<N>.png       n, mean drift <v_z>, and velocity spreads sigma_ux, sigma_uz vs x, from the samples
  drift.png                 mean drift <v_z> = Jz / (q n_e) vs x (mean of the last --average slabs, from density CSVs),
                            with the Brillouin flow's tanh(kappa x) (Lau et al. 2021)
  returning.png             fraction of the electron charge moving back toward the cathode (u_x < 0), overall and near
                            the cathode, and the negative part of f, vs t (from phase_reductions.csv)
"""
import argparse
import glob
import os
import sys

import numpy as np
from numpy.polynomial import legendre
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_density as pd  # noqa: E402  (shared style and density-file reader)

# Diverging map for signed f: red (negative) -> neutral gray -> the blue sequential ramp of plot_density (positive).
NEUTRAL = "#f0efec"
RED_ARM = ["#b02c2c", "#e34948", "#f0a3a2"]
DIVERGING = LinearSegmentedColormap.from_list(
    "f_signed", list(zip([0.0, 0.25, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         RED_ARM + [NEUTRAL, pd.BLUE_RAMP[1], pd.BLUE_RAMP[3], pd.BLUE_RAMP[4], pd.BLUE_RAMP[5],
                                    pd.BLUE_RAMP[6]])))
SEQUENTIAL = LinearSegmentedColormap.from_list("f_positive", [pd.SURFACE] + pd.BLUE_RAMP)
REDS = LinearSegmentedColormap.from_list("f_negative", [pd.SURFACE] + RED_ARM[::-1])
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order (blue, orange, aqua, yellow)


# ---------------------------------------------------------------------------------------------------------------------
# reading and per-cell polynomial machinery

def lagrange_matrix(n, s):
    """(len(s), n) matrix evaluating the degree-(n-1) interpolant through n equally spaced points on [-1,1] at s."""
    nodes = np.linspace(-1.0, 1.0, n)
    s = np.atleast_1d(s)
    L = np.ones((len(s), n))
    for j in range(n):
        for m in range(n):
            if m != j:
                L[:, j] *= (s - nodes[m]) / (nodes[j] - nodes[m])
    return L


def newton_cotes(n):
    """Weights on [-1,1] for n equally spaced points (exact for degree n-1), from Legendre moments."""
    nodes = np.linspace(-1.0, 1.0, n)
    V = np.vander(nodes, n, increasing=True).T
    moments = np.array([2.0 / (k + 1) if k % 2 == 0 else 0.0 for k in range(n)])
    return np.linalg.solve(V, moments)


class Slab:
    """f at the end of one slab: cells with their (x, u_x, u_z) extents and samples f[k, j, i] (u_z, u_x, x)."""

    def __init__(self, directory, slab):
        meta = dict(line.split(None, 1) for line in open(os.path.join(directory, f"phase_slab{slab}_meta.txt")))
        self.n = int(meta["pointsPerDim"])
        self.t = float(meta["t"])
        self.slab = slab
        parts = sorted(glob.glob(os.path.join(directory, f"phase_slab{slab}_part*.npy")))
        if not parts:
            sys.exit(f"no phase_slab{slab}_part*.npy in {directory}")
        data = np.concatenate([np.load(p) for p in parts if os.path.getsize(p) > 128])
        n = self.n
        blocks = data.reshape(-1, n * n * n, 5)
        self.f = blocks[:, :, 4].reshape(-1, n, n, n)          # [cell, k (u_z), j (u_x), i (x)]
        self.x0, self.x1 = blocks[:, 0, 1], blocks[:, -1, 1]
        self.ux0, self.ux1 = blocks[:, 0, 2], blocks[:, -1, 2]
        self.uz0, self.uz1 = blocks[:, 0, 3], blocks[:, -1, 3]
        self.w = newton_cotes(n)

    def xcells(self):
        return np.unique(np.round(np.stack([self.x0, self.x1], 1), 12), axis=0)

    def portrait(self, m):
        """int f du_z on each (x-cell, u_x-cell), sampled on an (m x m) uniform grid per cell.  Returns list of
        (xgrid, uxgrid, values[j, i]); negative part separately (sampled on the same fine grid, so approximate)."""
        n = self.n
        Lm = lagrange_matrix(n, np.linspace(-1, 1, m))
        Lz = lagrange_matrix(n, np.linspace(-1, 1, m))   # u_z fine grid for the negative part
        wz_fine = newton_cotes(m)
        key = {}
        for c in range(len(self.f)):
            k = (round(self.x0[c], 12), round(self.ux0[c], 12))
            hz = self.uz1[c] - self.uz0[c]
            integral = np.tensordot(self.w * 0.5 * hz, self.f[c], axes=(0, 0))          # [j, i]
            fine = np.einsum("ak,bj,ci,kji->abc", Lz, Lm, Lm, self.f[c])              # [uz, ux, x] on fine grid
            negative = np.tensordot(wz_fine * 0.5 * hz, np.minimum(fine, 0.0), axes=(0, 0))
            if k not in key:
                key[k] = [self.x0[c], self.x1[c], self.ux0[c], self.ux1[c], np.zeros((n, n)), np.zeros((m, m))]
            key[k][4] += integral
            key[k][5] += negative
        out = []
        for x0, x1, u0, u1, values, negative in key.values():
            out.append((np.linspace(x0, x1, m), np.linspace(u0, u1, m), Lm @ values @ Lm.T, negative))
        return out

    def at_x(self, x, m):
        """f(u_x, u_z) at position x: list of (uxgrid, uzgrid, values[uz, ux]) per velocity cell."""
        n = self.n
        Lm = lagrange_matrix(n, np.linspace(-1, 1, m))
        out = []
        for c in np.nonzero((self.x0 <= x + 1e-12) & (x - 1e-12 <= self.x1))[0]:
            s = 2.0 * (x - self.x0[c]) / (self.x1[c] - self.x0[c]) - 1.0
            Lx = lagrange_matrix(n, [s])[0]
            g = np.tensordot(self.f[c], Lx, axes=(2, 0))                               # [k, j]
            out.append((np.linspace(self.ux0[c], self.ux1[c], m), np.linspace(self.uz0[c], self.uz1[c], m), Lm @ g @ Lm.T))
        # a point on a cell boundary belongs to two x-cells: keep one set of velocity cells
        seen, unique = set(), []
        for ux, uz, v in out:
            k = (round(ux[0], 12), round(uz[0], 12))
            if k not in seen:
                seen.add(k); unique.append((ux, uz, v))
        return unique

    def moments(self, pointsPerXCell=12, gauss=8, xGauss=False):
        """n, <v_z>, sigma_ux, sigma_uz on a fine x grid, by Gauss quadrature in velocity of the exact polynomial.  With
        xGauss, the x points are Gauss points and the x quadrature weights are returned as a sixth value (exact
        integrals over x of polynomial moments, for consistency checks)."""
        n = self.n
        gs, gw = legendre.leggauss(gauss)
        Lg = lagrange_matrix(n, gs)
        if xGauss:
            sx, sxw = legendre.leggauss(pointsPerXCell)
        else:
            sx = np.linspace(-1, 1, pointsPerXCell)
        Lx = lagrange_matrix(n, sx)
        acc = {}
        for c in range(len(self.f)):
            hx, hu, hz = self.x1[c] - self.x0[c], self.ux1[c] - self.ux0[c], self.uz1[c] - self.uz0[c]
            ux = self.ux0[c] + 0.5 * (gs + 1) * hu
            uz = self.uz0[c] + 0.5 * (gs + 1) * hz
            UX, UZ = np.meshgrid(ux, uz, indexing="xy")                                # [uz, ux]
            gamma = np.sqrt(1.0 + UX ** 2 + UZ ** 2)
            W = np.outer(gw, gw) * 0.25 * hu * hz                                     # [uz, ux]
            fg = np.einsum("ak,bj,xi,kji->xab", Lg, Lg, Lx, self.f[c])                # [x, uz, ux]
            k = round(self.x0[c], 12)
            if k not in acc:
                acc[k] = [self.x0[c] + 0.5 * (sx + 1) * hx, np.zeros((6, pointsPerXCell)),
                          (0.5 * hx * sxw) if xGauss else None]
            a = acc[k][1]
            a[0] += np.einsum("xab,ab->x", fg, W)
            a[1] += np.einsum("xab,ab->x", fg, W * UZ / gamma)
            a[2] += np.einsum("xab,ab->x", fg, W * UX)
            a[3] += np.einsum("xab,ab->x", fg, W * UX ** 2)
            a[4] += np.einsum("xab,ab->x", fg, W * UZ ** 2)
            a[5] += np.einsum("xab,ab->x", fg, W * UZ)
        xs, cols, weights = [], [], []
        for k in sorted(acc):
            xs.append(acc[k][0]); cols.append(acc[k][1]); weights.append(acc[k][2])
        x = np.concatenate(xs); a = np.concatenate(cols, axis=1)
        dens = a[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            vz = a[1] / dens
            mux = a[2] / dens
            # where negative f makes a variance negative, the spread is undefined: NaN (a gap in the plots), not 0
            varx = a[3] / dens - mux ** 2
            sux = np.where(varx >= 0, np.sqrt(np.abs(varx)), np.nan)
            muz = a[5] / dens
            varz = a[4] / dens - muz ** 2
            suz = np.where(varz >= 0, np.sqrt(np.abs(varz)), np.nan)
        if xGauss:
            return x, dens, vz, sux, suz, np.concatenate(weights)
        return x, dens, vz, sux, suz


# ---------------------------------------------------------------------------------------------------------------------
# figures

def styled(figsize):
    plt.rcParams.update({"font.size": 10, "figure.facecolor": pd.SURFACE})
    fig, ax = plt.subplots(figsize=figsize)
    pd.style_axes(ax)
    return fig, ax


def draw_portrait(ax, blocks, norm, uxlim):
    mesh = None
    for xg, ug, values, _ in blocks:
        if ug[-1] < uxlim[0] or ug[0] > uxlim[1]:
            continue
        mesh = ax.pcolormesh(xg, ug, values.T, cmap=DIVERGING, norm=norm, shading="gouraud", rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(*uxlim)
    ax.axhline(0, color=pd.TEXT_SECONDARY, lw=0.6, ls=":")
    return mesh


def signed_norm(lo, hi):
    lo = min(lo, -1e-12 * max(abs(hi), 1.0)); hi = max(hi, 1e-12)
    return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)


def portrait_figures(slabs, out, uxlim, m=9):
    data = {s.slab: s.portrait(m) for s in slabs}
    for s in slabs:
        blocks = [b for b in data[s.slab] if b[1][-1] >= uxlim[0] and b[1][0] <= uxlim[1]]
        lo = min(b[2].min() for b in blocks); hi = max(b[2].max() for b in blocks)
        fig, ax = styled((7.5, 4.2))
        mesh = draw_portrait(ax, data[s.slab], signed_norm(lo, hi), uxlim)
        cb = fig.colorbar(mesh, ax=ax); cb.set_label(r"$\int f\, du_z$")
        ax.set_xlabel("x / D"); ax.set_ylabel(r"$u_x / c$")
        ax.set_title(f"Phase portrait at t = {s.t:.2f} (end of slab {s.slab}); red: negative", loc="left", fontsize=10,
                     color=pd.TEXT_PRIMARY)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"portrait_slab{s.slab}.png"), dpi=200); plt.close(fig)
    # formation sequence, shared scale (clip the colour range to the later slabs' maximum so the sheath is visible)
    k = len(slabs); cols = min(3, k); rows = int(np.ceil(k / cols))
    plt.rcParams.update({"font.size": 9, "figure.facecolor": pd.SURFACE})
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols + 1.0, 2.9 * rows), squeeze=False, sharex=True, sharey=True,
                             layout="constrained")
    his = [max(b[2].max() for b in data[s.slab] if b[1][-1] >= uxlim[0] and b[1][0] <= uxlim[1]) for s in slabs]
    los = [min(b[2].min() for b in data[s.slab] if b[1][-1] >= uxlim[0] and b[1][0] <= uxlim[1]) for s in slabs]
    norm = signed_norm(min(los), float(np.median(his)))
    for idx, s in enumerate(slabs):
        ax = axes[idx // cols][idx % cols]; pd.style_axes(ax)
        mesh = draw_portrait(ax, data[s.slab], norm, uxlim)
        ax.set_title(f"t = {s.t:.1f}", loc="left", fontsize=9, color=pd.TEXT_PRIMARY)
    for idx in range(k, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    for ax in axes[-1]: ax.set_xlabel("x / D")
    for row in axes: row[0].set_ylabel(r"$u_x / c$")
    fig.colorbar(mesh, ax=axes, shrink=0.9, aspect=30, extend="max", label=r"$\int f\, du_z$ (red: negative)")
    fig.savefig(os.path.join(out, "portraits.png"), dpi=200); plt.close(fig)


def negative_figures(slabs, out, uxlim, m=9):
    for s in slabs:
        blocks = s.portrait(m)
        lo = min(b[3].min() for b in blocks)
        fig, ax = styled((7.5, 4.2))
        mesh = None
        for xg, ug, _, negative in blocks:
            if ug[-1] < uxlim[0] or ug[0] > uxlim[1]:
                continue
            mesh = ax.pcolormesh(xg, ug, -negative.T, cmap=REDS, vmin=0, vmax=max(-lo, 1e-12), shading="gouraud",
                                 rasterized=True)
        ax.set_xlim(0, 1); ax.set_ylim(*uxlim); ax.axhline(0, color=pd.TEXT_SECONDARY, lw=0.6, ls=":")
        cb = fig.colorbar(mesh, ax=ax); cb.set_label(r"$-\int \min(f, 0)\, du_z$")
        ax.set_xlabel("x / D"); ax.set_ylabel(r"$u_x / c$")
        ax.set_title(f"Where f < 0, t = {s.t:.2f}", loc="left", fontsize=10, color=pd.TEXT_PRIMARY)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"negative_slab{s.slab}.png"), dpi=200); plt.close(fig)


def velocity_figures(slabs, out, xs, uxlim, uzlim, m=9):
    for s in slabs:
        plt.rcParams.update({"font.size": 9, "figure.facecolor": pd.SURFACE})
        fig, axes = plt.subplots(1, len(xs), figsize=(3.2 * len(xs), 3.2), squeeze=False, sharey=True)
        for a, x in zip(axes[0], xs):
            pd.style_axes(a)
            blocks = [b for b in s.at_x(x, m) if b[0][-1] >= uxlim[0] and b[0][0] <= uxlim[1]
                      and b[1][-1] >= -uzlim and b[1][0] <= uzlim]
            lo = min(b[2].min() for b in blocks); hi = max(b[2].max() for b in blocks)
            norm = signed_norm(lo, hi)
            mesh = None
            for ux, uz, v in blocks:
                mesh = a.pcolormesh(ux, uz, v, cmap=DIVERGING, norm=norm, shading="gouraud", rasterized=True)
            a.set_xlim(*uxlim); a.set_ylim(-uzlim, uzlim); a.set_aspect("equal")
            a.axhline(0, color=pd.TEXT_SECONDARY, lw=0.5, ls=":"); a.axvline(0, color=pd.TEXT_SECONDARY, lw=0.5, ls=":")
            a.set_title(f"x = {x:g}", loc="left", fontsize=9, color=pd.TEXT_PRIMARY)
            a.set_xlabel(r"$u_x / c$")
            fig.colorbar(mesh, ax=a, shrink=0.75)
        axes[0][0].set_ylabel(r"$u_z / c$")
        fig.suptitle(f"f(u_x, u_z) at t = {s.t:.2f} (end of slab {s.slab}); red: negative", x=0.01, ha="left",
                     fontsize=10, color=pd.TEXT_PRIMARY)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"velocity_slab{s.slab}.png"), dpi=200); plt.close(fig)


def moment_figures(slabs, out, kappa, x_b):
    for s in slabs:
        x, dens, vz, sux, suz = s.moments()
        mask = dens > 0.03 * np.nanmax(dens)
        plt.rcParams.update({"font.size": 9, "figure.facecolor": pd.SURFACE})
        fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        for a in axes: pd.style_axes(a); a.grid(True, color=pd.GRID, lw=0.6); a.set_axisbelow(True)
        axes[0].plot(x, dens, color=CATEGORICAL[0], lw=2); axes[0].set_ylabel(r"$n_e$")
        axes[0].set_title(f"Velocity moments at t = {s.t:.2f} (end of slab {s.slab})", loc="left", fontsize=10,
                          color=pd.TEXT_PRIMARY)
        axes[1].plot(x[mask], np.abs(vz[mask]), color=CATEGORICAL[0], lw=2, label=r"$|\langle v_z \rangle|$")
        xb = np.linspace(0, x_b, 100)
        axes[1].plot(xb, np.tanh(kappa * xb), color=pd.TEXT_SECONDARY, lw=1.5, ls="--", label="Brillouin flow")
        axes[1].set_ylabel(r"drift / c"); axes[1].legend(frameon=False, fontsize=8)
        axes[2].plot(x[mask], sux[mask], color=CATEGORICAL[0], lw=2, label=r"$\sigma_{u_x}$")
        axes[2].plot(x[mask], suz[mask], color=CATEGORICAL[1], lw=2, label=r"$\sigma_{u_z}$")
        axes[2].set_ylabel("spread / c"); axes[2].set_xlabel("x / D"); axes[2].legend(frameon=False, fontsize=8)
        axes[2].set_xlim(0, 1)
        fig.text(0.01, 0.005, "drift and spreads shown where n_e > 3% of its maximum; gaps: variance < 0 (negative f)",
                 fontsize=7, color=pd.TEXT_SECONDARY)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"moments_slab{s.slab}.png"), dpi=200); plt.close(fig)


def drift_figure(directory, out, average, q, kappa, x_b):
    paths = sorted(glob.glob(os.path.join(directory, "density_slab*.csv")), key=lambda p: int(p.rsplit("slab", 1)[1][:-4]))
    if not paths:
        return None
    xg = np.linspace(0, 1, 801); profiles = []
    for path in paths[-average:]:
        ppd, deg, cells = pd.read_csvs([path])
        if "Jz" not in cells[0]:
            print(f"{path}: no Jz column (run predates it); skipping drift.png"); return None
        tEnd = max(c["t"].max() for c in cells)
        xs, n, jz = [], [], []
        for c in cells:
            row = np.argmin(np.abs(c["t"][:, 0] - tEnd))
            if abs(c["t"][row, 0] - tEnd) > 1e-9:
                continue
            xs.append(c["x"][row]); n.append(c["n_e"][row]); jz.append(c["Jz"][row])
        xs, n, jz = map(np.concatenate, (xs, n, jz)); o = np.argsort(xs)
        profiles.append((np.interp(xg, xs[o], n[o]), np.interp(xg, xs[o], jz[o])))
    n = np.mean([p[0] for p in profiles], axis=0); jz = np.mean([p[1] for p in profiles], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vz = jz / (q * n)
    mask = n > 0.03 * n.max()
    fig, ax = styled((7, 4.2))
    ax.plot(xg[mask], np.abs(vz[mask]), color=CATEGORICAL[0], lw=2,
            label=r"computed $|\langle v_z \rangle| = |J_z / (q n_e)|$" + f" (last {len(profiles)} slabs)")
    xb = np.linspace(0, x_b, 100)
    ax.plot(xb, np.tanh(kappa * xb), color=pd.TEXT_SECONDARY, lw=1.5, ls="--", label=r"Brillouin flow $\tanh(\kappa x)$")
    ax.axvline(x_b, color=pd.GRID, lw=1.0)
    ax.set_xlim(0, 1); ax.set_xlabel("x / D"); ax.set_ylabel(r"mean drift $/ c$")
    ax.grid(True, color=pd.GRID, lw=0.6); ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts(): t.set_color(pd.TEXT_PRIMARY)
    fig.text(0.01, 0.005, "shown where n_e > 3% of its maximum; vertical line: Brillouin hub height", fontsize=7,
             color=pd.TEXT_SECONDARY)
    fig.tight_layout(); fig.savefig(os.path.join(out, "drift.png"), dpi=200); plt.close(fig)
    return xg, n, vz


def returning_figure(directory, out):
    path = os.path.join(directory, "phase_reductions.csv")
    if not os.path.exists(path):
        return None
    r = np.genfromtxt(path, delimiter=",", names=True)
    r = np.atleast_1d(r)
    plt.rcParams.update({"font.size": 9, "figure.facecolor": pd.SURFACE})
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
    for a in axes: pd.style_axes(a); a.grid(True, color=pd.GRID, lw=0.6); a.set_axisbelow(True)
    with np.errstate(divide="ignore", invalid="ignore"):
        axes[0].plot(r["t"], r["int_f_ux_neg"] / r["int_f"], color=CATEGORICAL[0], lw=2, marker="o", ms=3,
                     label="whole gap")
        axes[0].plot(r["t"], r["int_f_near_ux_neg"] / r["int_f_near"], color=CATEGORICAL[1], lw=2, marker="o", ms=3,
                     label=f"x < {r['x_near'][0]:g}")
    axes[0].set_ylabel(r"fraction with $u_x < 0$"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Electrons moving back toward the cathode", loc="left", fontsize=10, color=pd.TEXT_PRIMARY)
    axes[1].plot(r["t"], -r["int_f_negative_part"], color=CATEGORICAL[0], lw=2, marker="o", ms=3)
    axes[1].set_ylabel(r"$-\int \min(f,0)$"); axes[1].set_xlabel("t")
    axes[1].set_title("Negative part of f (sampled)", loc="left", fontsize=10, color=pd.TEXT_PRIMARY)
    fig.tight_layout(); fig.savefig(os.path.join(out, "returning.png"), dpi=200); plt.close(fig)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--slabs", type=int, nargs="*", default=None, help="slabs to plot (default: all sampled)")
    parser.add_argument("--x", type=float, nargs="*", default=[0.0, 0.05, 0.15, 0.3], help="x positions for f(u_x, u_z)")
    parser.add_argument("--uxlim", type=float, nargs=2, default=[-1.1, 1.5])
    parser.add_argument("--uzlim", type=float, default=1.3)
    parser.add_argument("--average", type=int, default=10, help="slabs averaged for drift.png")
    parser.add_argument("--q", type=float, default=-1.0, help="species charge in the driver's scaling (electrons: -1)")
    parser.add_argument("--V", type=float, default=1e6); parser.add_argument("--ratio", type=float, default=1.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(args.directory, "phase-space")
    os.makedirs(out, exist_ok=True)

    import compare_brillouin as cb
    kappa, chi_b, x_b, _ = cb.brillouin(args.V / cb.MC2_OVER_E, args.ratio)

    sampled = sorted(int(os.path.basename(p)[len("phase_slab"):].split("_")[0])
                     for p in glob.glob(os.path.join(args.directory, "phase_slab*_meta.txt")))
    slabNumbers = [s for s in (args.slabs if args.slabs is not None else sampled) if s in sampled]
    slabs = [Slab(args.directory, s) for s in slabNumbers]
    print("sampled slabs:", sampled, "plotting:", slabNumbers)

    r = returning_figure(args.directory, out)
    d = drift_figure(args.directory, out, args.average, args.q, kappa, x_b)
    if slabs:
        portrait_figures(slabs, out, args.uxlim)
        negative_figures(slabs, out, args.uxlim)
        velocity_figures(slabs, out, args.x, args.uxlim, args.uzlim)
        moment_figures(slabs, out, kappa, x_b)
        # consistency checks: the samples' integrals against the reductions, and the sampled drift against Jz/(q n)
        for s in slabs:
            x, dens, vz, sux, suz, wx = s.moments(pointsPerXCell=s.n, xGauss=True)
            total = float(np.dot(dens, wx))
            line = f"slab {s.slab}: int n dx from samples {total:.6f}"
            if r is not None:
                row = r[r["slab"] == s.slab]
                if len(row):
                    line += f", phase_reductions int_f {float(row['int_f'][0]):.6f}"
            print(line)
    print("wrote figures to", out)


if __name__ == "__main__":
    main()
