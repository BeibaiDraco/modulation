#!/usr/bin/env python3
"""
Reproduce paper panels B, C, D from saved CSV/JSON data.

This script is intentionally standalone: it depends only on standard scientific
Python libraries (numpy, matplotlib) and the standard library (json, csv, argparse, pathlib).

It expects that you have already run the experiments to produce the CSV/JSON data:
- TRIAD outputs folder (e.g., outputs/<tag>/):
    - panel_b_points.csv
        columns: state, pc1, pc2, pc3, shape, color
        where state ∈ {"color", "shape"} refers to the attention state the grid was computed under
    - panel_b_axes.json
        keys: target_pc, color_axis_color_pc, color_axis_shape_pc,
              shape_axis_shape_pc, shape_axis_color_pc
    - panel_c_data.csv
        columns: neuron, selectivity, g_color, g_shape, ratio, log2_ratio

- TRIAD-SWEEP outputs folder (e.g., outputs/<tag>_triad_sweep/):
    - panel_d_data.csv
        columns:
          range, impr_color_deg, impr_shape_deg,
          impr_color_shuf_mean_or_single, impr_shape_shuf_mean_or_single,
          impr_color_shuf_sem, impr_shape_shuf_sem

What each panel shows (undirectional axes: angles are in [0, 90] degrees):

Panel B (3D embedding, single axes):
    • Two clouds of points in the unmodulated PCA basis:
        - color-attend grid (marker 'o')
        - shape-attend grid (marker '^')
      Each point is colored bivariately:
        - hue ~ color stimulus (0→1 wraps hue)
        - lightness ~ shape stimulus (darker→lighter as shape increases)
    • Lines from the origin:
        - Target axis (crimson)
        - Color axis under color gains (darkorange, solid)
        - Color axis under shape gains (darkorange, dashed)
        - Shape axis under shape gains (seagreen,  solid)
        - Shape axis under color gains (seagreen,  dashed)

Panel C (gain comparison):
    • x-axis: selectivity = (color - shape) for each neuron
    • y-axis: log2(g_color / g_shape) by default
    • Scatter colored by selectivity (bwr cmap), dashed y=0 line

Panel D (range sweep: improvement-to-target metric):
    For each constraint range, define improvement for each axis as:
        Δ_color = angle(color axis under shape gains, target) − angle(color axis under color gains, target)
        Δ_shape = angle(shape axis under color gains, target) − angle(shape axis under shape gains, target)
      Positive values mean the axis is better aligned to target under its "own" attention.
    • Solid lines: Δ_color and Δ_shape vs range
    • Dashed lines: shuffled mean (if present) with shaded 95% CI = 1.96 * SEM

Usage:
    python reproduce_panels_from_csv.py \
        --triad-dir outputs/paper_main \
        --sweep-dir outputs/paper_main_triad_sweep \
        --out-dir outputs/paper_main/repro \
        --zlim -1 1 --elev 25 --azim 135 --dpi 300 --transparent

Notes:
    • This script DOES NOT recompute any model outputs; it only re-renders figures from saved CSV/JSON.
    • Angles are treated as axes (orientation-invariant). The saved numbers already follow this convention.
"""

import argparse
import csv
import json
import math
import colorsys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------- helpers ----------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_csv(path: Path) -> List[List[str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # handle potential newlines in last column
            rows.append(line.rstrip("\n").split(","))
    return rows

def _parse_header_and_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    rows = _read_csv(path)
    if not rows:
        return [], []
    return rows[0], rows[1:]

def _to_float_or_none(s: str) -> Optional[float]:
    s = s.strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def line3d(ax, vec3: np.ndarray, two_sided: bool = False, linewidth: float = 2.0, color=None) -> None:
    """Draw a simple 3D line from origin along 'vec3' (PC coordinates)."""
    v = np.asarray(vec3).ravel()
    if v.size != 3:
        v = v[:3]
    p0 = -v if two_sided else np.zeros(3)
    p1 = v
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], lw=linewidth, color=color)

def bivariate_colors(shape_list, color_list):
    """
    Map (shape,color) in [0,1]x[0,1] to RGB via HLS:
      - Hue encodes 'color' (wraps around)
      - Lightness encodes 'shape' (darker->lighter as shape increases)
      - Saturation fixed at ~0.9
    """
    shape = np.asarray(shape_list, dtype=float)
    color = np.asarray(color_list, dtype=float)
    L = np.clip(0.35 + 0.45 * shape, 0.0, 1.0)
    S = 0.90 * np.ones_like(L)
    H = np.mod(color, 1.0)
    rgb = [colorsys.hls_to_rgb(h, l, s) for h, l, s in zip(H, L, S)]
    return np.asarray(rgb)

def draw_bivariate_legend(fig, *, loc=(0.72, 0.68, 0.24, 0.24)):
    """Inset legend: shape↑ (lightness), color→ (hue)."""
    N = 64
    xs = np.linspace(0,1,N)  # color
    ys = np.linspace(0,1,N)  # shape
    grid = np.zeros((N,N,3))
    for i, y in enumerate(ys):
        row = bivariate_colors(np.full(N, y), xs)
        grid[N-1-i, :, :] = row  # shape increases upward
    a = fig.add_axes(loc)
    a.imshow(grid, origin="lower", extent=(0,1,0,1))
    a.set_xticks([0,1]); a.set_yticks([0,1])
    a.set_xticklabels(["0","1"], fontsize=8)
    a.set_yticklabels(["0","1"], fontsize=8)
    a.set_xlabel("color", fontsize=9)
    a.set_ylabel("shape", fontsize=9)
    a.tick_params(length=2, pad=2)
    for s in a.spines.values():
        s.set_linewidth(0.8)


# ---------------------- panel renderers ----------------------

def render_panel_b(triad_dir: Path, outdir: Path, zlim: Optional[Tuple[float,float]]=None,
                   elev: Optional[float]=None, azim: Optional[float]=None, dpi: int=300, transparent: bool=True):
    """
    Reproduce Panel B from:
      triad_dir / panel_b_points.csv
      triad_dir / panel_b_axes.json
    """
    pts_csv = triad_dir / "panel_b_points.csv"
    axes_json = triad_dir / "panel_b_axes.json"
    if not pts_csv.exists() or not axes_json.exists():
        raise FileNotFoundError(f"Missing panel_b data: {pts_csv} and/or {axes_json}")

    header, rows = _parse_header_and_rows(pts_csv)
    # expected header: state,pc1,pc2,pc3,shape,color
    col_idx = {name: i for i, name in enumerate(header)}
    Zc, Zs, shape_list, color_list = [], [], [], []
    for r in rows:
        state = r[col_idx["state"]]
        pc1, pc2, pc3 = map(float, (r[col_idx["pc1"]], r[col_idx["pc2"]], r[col_idx["pc3"]]))
        shp = float(r[col_idx["shape"]]); col = float(r[col_idx["color"]])
        if state == "color":
            Zc.append((pc1, pc2, pc3))
        elif state == "shape":
            Zs.append((pc1, pc2, pc3))
        shape_list.append(shp); color_list.append(col)
    Zc = np.array(Zc, dtype=float); Zs = np.array(Zs, dtype=float)
    shape_arr = np.array(shape_list[:len(Zc)], dtype=float)  # assuming same length/order for two clouds
    color_arr = np.array(color_list[:len(Zc)], dtype=float)
    bicolors = bivariate_colors(shape_arr, color_arr)

    with open(axes_json, "r", encoding="utf-8") as f:
        axes = json.load(f)
    t_pc   = np.array(axes["target_pc"], dtype=float)
    col_c  = np.array(axes["color_axis_color_pc"], dtype=float)
    col_s  = np.array(axes["color_axis_shape_pc"], dtype=float)
    shp_s  = np.array(axes["shape_axis_shape_pc"], dtype=float)
    shp_c  = np.array(axes["shape_axis_color_pc"], dtype=float)

    _ensure_dir(outdir)
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Zc[:,0], Zc[:,1], Zc[:,2], s=22, marker='o', c=bicolors, edgecolors='none', alpha=0.95, label="color-attend")
    ax.scatter(Zs[:,0], Zs[:,1], Zs[:,2], s=22, marker='^', c=bicolors, edgecolors='none', alpha=0.95, label="shape-attend")

    line3d(ax, t_pc, two_sided=False, linewidth=2.5, color='crimson')
    ax.plot([0, col_c[0]], [0, col_c[1]], [0, col_c[2]], lw=2.2, color='darkorange', label="color axis (color)")
    ax.plot([0, col_s[0]], [0, col_s[1]], [0, col_s[2]], lw=2.2, color='darkorange', ls='--', label="color axis (shape)")
    ax.plot([0, shp_s[0]], [0, shp_s[1]], [0, shp_s[2]], lw=2.2, color='seagreen',  label="shape axis (shape)")
    ax.plot([0, shp_c[0]], [0, shp_c[1]], [0, shp_c[2]], lw=2.2, color='seagreen',  ls='--', label="shape axis (color)")

    if zlim is not None:
        ax.set_zlim(zlim)
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

    if elev is not None or azim is not None:
        ax.view_init(elev=elev if elev is not None else ax.elev,
                     azim=azim if azim is not None else ax.azim)

    legend_elems = [
        Line2D([0],[0], color='crimson',    lw=2.5, label='Target'),
        Line2D([0],[0], color='darkorange', lw=2.2, label='Color axis (color)'),
        Line2D([0],[0], color='darkorange', lw=2.2, ls='--', label='Color axis (shape)'),
        Line2D([0],[0], color='seagreen',   lw=2.2, label='Shape axis (shape)'),
        Line2D([0],[0], color='seagreen',   lw=2.2, ls='--', label='Shape axis (color)'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', frameon=True, fontsize=9)

    draw_bivariate_legend(fig)

    plt.tight_layout()
    for ext in ("png","pdf","svg"):
        plt.savefig(outdir / f"panel_b_repro.{ext}", dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def render_panel_c(triad_dir: Path, outdir: Path, mode: str="ratio", logratio: bool=True,
                   dpi: int=300, transparent: bool=True):
    """
    Reproduce Panel C from triad_dir / panel_c_data.csv

    By default plots y = log2(g_color/g_shape) vs selectivity = color - shape.
    """
    csv_path = triad_dir / "panel_c_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing panel_c data: {csv_path}")

    header, rows = _parse_header_and_rows(csv_path)
    idx = {name: i for i, name in enumerate(header)}
    sel = []; y = []
    for r in rows:
        s = float(r[idx["selectivity"]])
        if mode == "ratio":
            if logratio and "log2_ratio" in idx:
                val = float(r[idx["log2_ratio"]])
            else:
                # fall back to raw ratio if needed
                val = float(r[idx["ratio"]])
        else:
            # not standard for the paper, but supported
            gc = float(r[idx["g_color"]]); gs = float(r[idx["g_shape"]])
            val = gc - gs
        sel.append(s); y.append(val)
    sel = np.asarray(sel); y = np.asarray(y)

    fig = plt.figure(figsize=(5.6, 4.6))
    plt.axhline(0.0, color='k', lw=1.0, ls='--', alpha=0.6)
    sc = plt.scatter(sel, y, c=sel, cmap='bwr', alpha=0.8, edgecolors='none', s=22)
    cb = plt.colorbar(sc); cb.set_label('Color - Shape selectivity')
    plt.xlabel('Selectivity (color - shape)')
    plt.ylabel('log2 gain ratio (color / shape)' if mode=="ratio" and logratio else ('gain ratio (color / shape)' if mode=="ratio" else 'gain difference (color - shape)'))
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    _ensure_dir(outdir)
    for ext in ("png","pdf","svg"):
        plt.savefig(outdir / f"panel_c_repro.{ext}", dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def render_panel_d(sweep_dir: Path, outdir: Path, dpi: int=300, transparent: bool=True):
    """
    Reproduce Panel D from sweep_dir / panel_d_data.csv

    Plots improvement-to-target curves:
        Δ_color = angle(color axis under shape gains, target) − angle(color axis under color gains, target)
        Δ_shape = angle(shape axis under color gains, target) − angle(shape axis under shape gains, target)
    If shuffled mean/SEM present, overlays dashed mean and 95% CI (1.96 * SEM).
    """
    csv_path = sweep_dir / "panel_d_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing panel_d data: {csv_path}")

    header, rows = _parse_header_and_rows(csv_path)
    idx = {name: i for i, name in enumerate(header)}

    xs = []; impr_c = []; impr_s = []
    impr_c_mean = []; impr_s_mean = []
    impr_c_sem  = []; impr_s_sem  = []
    for r in rows:
        xs.append(float(r[idx["range"]]))
        impr_c.append(float(r[idx["impr_color_deg"]]))
        impr_s.append(float(r[idx["impr_shape_deg"]]))
        impr_c_mean.append(_to_float_or_none(r[idx["impr_color_shuf_mean_or_single"]]))
        impr_s_mean.append(_to_float_or_none(r[idx["impr_shape_shuf_mean_or_single"]]))
        impr_c_sem.append(_to_float_or_none(r[idx["impr_color_shuf_sem"]]))
        impr_s_sem.append(_to_float_or_none(r[idx["impr_shape_shuf_sem"]]))

    xs = np.asarray(xs)
    impr_c = np.asarray(impr_c)
    impr_s = np.asarray(impr_s)
    cmean = np.array([np.nan if v is None else v for v in impr_c_mean], dtype=float)
    smean = np.array([np.nan if v is None else v for v in impr_s_mean], dtype=float)
    csem  = np.array([np.nan if v is None else v for v in impr_c_sem], dtype=float)
    ssem  = np.array([np.nan if v is None else v for v in impr_s_sem], dtype=float)

    fig = plt.figure(figsize=(6.2, 4.8))
    plt.plot(xs, impr_c, marker="o", lw=2.0, label="Color-axis: (shape) - (color)")
    plt.plot(xs, impr_s, marker="s", lw=2.0, label="Shape-axis: (color) - (shape)")

    has_mean = np.isfinite(cmean).any() or np.isfinite(smean).any()
    if has_mean:
        if np.isfinite(cmean).all():
            plt.plot(xs, cmean, marker="^", lw=1.6, ls="--", label="Color-axis (shuf mean)")
            if np.isfinite(csem).all():
                plt.fill_between(xs, cmean - 1.96*csem, cmean + 1.96*csem, alpha=0.15, linewidth=0)
        if np.isfinite(smean).all():
            plt.plot(xs, smean, marker="v", lw=1.6, ls="--", label="Shape-axis (shuf mean)")
            if np.isfinite(ssem).all():
                plt.fill_between(xs, smean - 1.96*ssem, smean + 1.96*ssem, alpha=0.15, linewidth=0)

    plt.axhline(0.0, color='k', lw=1.0, ls='--', alpha=0.5)
    plt.xlabel("Constraint range")
    plt.ylabel("Δ angle-to-target (deg)")
    plt.title("Δ to-target under own vs other attention")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    _ensure_dir(outdir)
    for ext in ("png","pdf","svg"):
        plt.savefig(outdir / f"panel_d_repro.{ext}", dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser(description="Reproduce paper panels B/C/D from saved CSV/JSON data.")
    ap.add_argument("--triad-dir", required=True, type=Path,
                    help="Directory containing panel_b_points.csv, panel_b_axes.json, panel_c_data.csv")
    ap.add_argument("--sweep-dir", required=True, type=Path,
                    help="Directory containing panel_d_data.csv (e.g., outputs/<tag>_triad_sweep)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Directory to write output images (panel_b_repro.*, panel_c_repro.*, panel_d_repro.*)")
    ap.add_argument("--zlim", nargs=2, type=float, metavar=("ZMIN","ZMAX"),
                    help="Optional z-limits for 3D plot (panel B), e.g., --zlim -1 1")
    ap.add_argument("--elev", type=float, help="Elevation angle for 3D plot (panel B)")
    ap.add_argument("--azim", type=float, help="Azimuth angle for 3D plot (panel B)")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    ap.add_argument("--transparent", action="store_true", help="Save with transparent background")
    args = ap.parse_args()

    # Panel B
    render_panel_b(args.triad_dir, args.out_dir,
                   zlim=tuple(args.zlim) if args.zlim else None,
                   elev=args.elev, azim=args.azim,
                   dpi=args.dpi, transparent=args.transparent)

    # Panel C
    render_panel_c(args.triad_dir, args.out_dir,
                   mode="ratio", logratio=True,
                   dpi=args.dpi, transparent=args.transparent)

    # Panel D
    render_panel_d(args.sweep_dir, args.out_dir,
                   dpi=args.dpi, transparent=args.transparent)

if __name__ == "__main__":
    main()
