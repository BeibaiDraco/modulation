from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.typing import NDArray
import colorsys, json

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def line3d(ax, vec_pc: NDArray[np.float64], two_sided: bool = False,
           linewidth: float = 2.0, color=None) -> None:
    """Draw a simple 3D line along a vector in PC space (your original style)."""
    v = np.asarray(vec_pc).ravel()
    if v.size != 3:
        v = v[:3]
    p0 = -v if two_sided else np.zeros(3)
    p1 = v
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            lw=linewidth, color=color)

def plot_original_two_panel(
    Z_unmod: NDArray[np.float64],
    Z_mod: NDArray[np.float64],
    pca_components: NDArray[np.float64],     # 3 x N
    target_vec_neuron: NDArray[np.float64],  # N
    d_unmod_neuron: NDArray[np.float64],     # N
    d_mod_neuron: NDArray[np.float64],       # N
    shape_vals: Sequence[float],
    color_vals: Sequence[float],
    outdir: Path,
    tag: str,
    *,
    zlim: Optional[tuple[float, float]] = None,  # e.g., (-1, 1)
    elev: Optional[float] = None,                # starting elevation (deg)
    azim: Optional[float] = None,                # starting azimuth (deg)
    show: bool = False,                          # open interactive window
) -> None:
    """Two-panel 3D plot with colorbars and target/axis lines."""
    _ensure_dir(outdir)

    comps = pca_components[:3, :]                # 3 x N
    target_pc_coords = comps @ target_vec_neuron # (3,)
    axis_unmod_pc    = comps @ d_unmod_neuron    # (3,)
    axis_mod_pc      = comps @ d_mod_neuron      # (3,)

    mins = np.minimum(Z_unmod.min(axis=0), Z_mod.min(axis=0))
    maxs = np.maximum(Z_unmod.max(axis=0), Z_mod.max(axis=0))

    # For each shape, iterate color (your original ordering)
    color_list = np.array([c for _s in shape_vals for c in color_vals])

    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    sc1 = ax1.scatter(Z_unmod[:, 0], Z_unmod[:, 1], Z_unmod[:, 2],
                      c=color_list, cmap='viridis', s=20)
    sc2 = ax2.scatter(Z_mod[:, 0], Z_mod[:, 1], Z_mod[:, 2],
                      c=color_list, cmap='viridis', s=20)

    # Lines: target + axes
    line3d(ax1, target_pc_coords, two_sided=False, linewidth=2, color='crimson')
    line3d(ax1, axis_unmod_pc,   two_sided=False, linewidth=2, color='black')
    line3d(ax2, target_pc_coords, two_sided=False, linewidth=2, color='crimson')
    line3d(ax2, axis_mod_pc,      two_sided=False, linewidth=2, color='darkorange')

    for ax in (ax1, ax2):
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])
        if zlim is not None:
            ax.set_zlim(zlim)  # <- fixed Z range if provided
        else:
            ax.set_zlim([mins[2], maxs[2]])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_box_aspect((1, 1, 1))
        if (elev is not None) or (azim is not None):
            ax.view_init(elev=elev if elev is not None else ax.elev,
                         azim=azim if azim is not None else ax.azim)

    ax1.set_title('Unmodulated Responses')
    ax2.set_title('Modulated Responses')

    legend_elements = [
        Line2D([0], [0], color='crimson',    lw=2, label='Target axis'),
        Line2D([0], [0], color='black',      lw=2, label='Color axis (pre)'),
        Line2D([0], [0], color='darkorange', lw=2, label='Color axis (post)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.7); cb1.set_label('Color value')
    cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.7); cb2.set_label('Color value')

    plt.tight_layout()
    png = outdir / f"{tag}_embedding_twopanel.png"
    pdf = outdir / f"{tag}_embedding_twopanel.pdf"
    plt.savefig(png, dpi=200); plt.savefig(pdf)

    if show:
        plt.show()
    plt.close(fig)

def plot_embedding_3d(Z0: NDArray[np.float64], Z1: NDArray[np.float64],
                      pca_components, target_vec_neuron: NDArray[np.float64],
                      d_unmod_neuron: NDArray[np.float64], d_opt_neuron: NDArray[np.float64],
                      outdir: Path, tag: str) -> None:
    """Single-axes plot (non-interactive save)."""
    _ensure_dir(outdir)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Z0[:,0], Z0[:,1], Z0[:,2], alpha=0.5, s=20, label="pre (g=1)")
    ax.scatter(Z1[:,0], Z1[:,1], Z1[:,2], alpha=0.5, s=20, label="post (g=opt)", marker="^")

    comps = pca_components[:3, :]
    def to_pc_space(vN): return (comps @ vN)

    t_pc = to_pc_space(target_vec_neuron)
    u_pc = to_pc_space(d_unmod_neuron)
    v_pc = to_pc_space(d_opt_neuron)

    def norm3(x):
        n = np.linalg.norm(x)+1e-12
        return x / n

    t_pc = norm3(t_pc); u_pc = norm3(u_pc); v_pc = norm3(v_pc)

    ori = np.zeros(3)
    ax.plot([ori[0], 1.2*t_pc[0]], [ori[1], 1.2*t_pc[1]], [ori[2], 1.2*t_pc[2]], lw=3.0, label="target")
    ax.plot([ori[0], 1.2*u_pc[0]], [ori[1], 1.2*u_pc[1]], [ori[2], 1.2*u_pc[2]], lw=2.0, label="axis (pre)")
    ax.plot([ori[0], 1.2*v_pc[0]], [ori[1], 1.2*v_pc[1]], [ori[2], 1.2*v_pc[2]], lw=2.0, ls="--", label="axis (post)")

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"Embedding & alignment: {tag}")
    ax.legend(loc="best")
    plt.tight_layout()

    png = outdir / f"{tag}_embedding3d.png"
    pdf = outdir / f"{tag}_embedding3d.pdf"
    plt.savefig(png, dpi=200); plt.savefig(pdf)
    plt.close(fig)

def plot_range_vs_degree(rows: Iterable[dict], outdir: Path, tag: str,
                         ylabel: str = "Δ angle (deg)") -> None:
    _ensure_dir(outdir)
    xs = [r["range"] for r in rows]
    ys = [r["delta_opt_vs_unmod_deg"] for r in rows]
    fig = plt.figure(figsize=(6,4.5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Constraint range")
    plt.ylabel(ylabel)
    plt.title(f"Range vs degree change — {tag}")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    png = outdir / f"{tag}_range_vs_degree.png"
    pdf = outdir / f"{tag}_range_vs_degree.pdf"
    plt.savefig(png, dpi=200); plt.savefig(pdf)
    plt.close(fig)

def plot_gains_vs_selectivity(S: NDArray[np.float64], g_opt: NDArray[np.float64],
                              outdir: Path, tag: str) -> None:
    """'Gains vs selectivity bias' scatter (bwr colormap)."""
    _ensure_dir(outdir)
    color_diff = S[:, 1] - S[:, 0]
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(color_diff, g_opt, c=color_diff, cmap='bwr', alpha=0.7, edgecolors='none')
    plt.xlabel('Color - Shape selectivity')
    plt.ylabel('Optimized gain $g_i$')
    plt.title('Gains vs. selectivity bias')
    cb = plt.colorbar(); cb.set_label('Color - Shape')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    png = outdir / f"{tag}_gains_vs_selectivity.png"
    pdf = outdir / f"{tag}_gains_vs_selectivity.pdf"
    plt.savefig(png, dpi=200); plt.savefig(pdf)
    plt.close(fig)


def plot_triad_three_panel(
    Z_unmod, Z_color, Z_shape,
    pca_components, target_vec_neuron,
    d_color_unmod, d_color_opt, d_shape_unmod, d_shape_opt,
    shape_vals, color_vals,
    outdir: Path, tag: str,
    *, zlim=None, elev=None, azim=None, show=False
):
    from matplotlib.lines import Line2D
    _ensure_dir(outdir)
    comps = pca_components[:3, :]
    t_pc      = comps @ target_vec_neuron
    col_pre   = comps @ d_color_unmod
    col_post  = comps @ d_color_opt
    shp_pre   = comps @ d_shape_unmod
    shp_post  = comps @ d_shape_opt

    color_list = np.array([c for _s in shape_vals for c in color_vals])

    fig = plt.figure(figsize=(19, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    sc1 = ax1.scatter(Z_unmod[:,0], Z_unmod[:,1], Z_unmod[:,2], c=color_list, cmap='viridis', s=20)
    sc2 = ax2.scatter(Z_color[:,0],  Z_color[:,1],  Z_color[:,2],  c=color_list, cmap='viridis', s=20)
    sc3 = ax3.scatter(Z_shape[:,0],  Z_shape[:,1],  Z_shape[:,2],  c=color_list, cmap='viridis', s=20)

    # lines
    line3d(ax1, t_pc,    color='crimson');    line3d(ax1, col_pre, color='black');     line3d(ax1, shp_pre, color='steelblue')
    line3d(ax2, t_pc,    color='crimson');    line3d(ax2, col_post, color='darkorange')
    line3d(ax3, t_pc,    color='crimson');    line3d(ax3, shp_post, color='seagreen')

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
        if zlim is not None: ax.set_zlim(zlim)
        ax.set_box_aspect((1,1,1))
        if (elev is not None) or (azim is not None):
            ax.view_init(elev=elev if elev is not None else ax.elev,
                         azim=azim if azim is not None else ax.azim)

    ax1.set_title('Unmodulated (pre)')
    ax2.set_title('Color-aligned (post)')
    ax3.set_title('Shape-aligned (post)')

    legend_elems = [
        Line2D([0],[0], color='crimson',    lw=2, label='Target axis'),
        Line2D([0],[0], color='black',      lw=2, label='Color axis (pre)'),
        Line2D([0],[0], color='steelblue',  lw=2, label='Shape axis (pre)'),
        Line2D([0],[0], color='darkorange', lw=2, label='Color axis (post)'),
        Line2D([0],[0], color='seagreen',   lw=2, label='Shape axis (post)'),
    ]
    ax3.legend(handles=legend_elems, loc='upper right')

    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.7); cb1.set_label('Color value')
    cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.7); cb2.set_label('Color value')
    cb3 = fig.colorbar(sc3, ax=ax3, shrink=0.7); cb3.set_label('Color value')

    plt.tight_layout()
    out_png = outdir / f"{tag}_triad_threepanel.png"
    out_pdf = outdir / f"{tag}_triad_threepanel.pdf"
    plt.savefig(out_png, dpi=200); plt.savefig(out_pdf)
    if show: plt.show()
    plt.close(fig)

def plot_gains_vs_selectivity_pair(
    S: NDArray[np.float64], g_color: NDArray[np.float64], g_shape: NDArray[np.float64],
    outdir: Path, tag: str, *,
    mode: str = "ratio", logratio: bool = False, show: bool = False
):
    _ensure_dir(outdir)
    color_diff = S[:,1] - S[:,0]
    eps = 1e-12

    if mode == "ratio":
        y = (g_color + eps) / (g_shape + eps)
        label = "Gain ratio (color / shape)"
        if logratio:
            # only safe if all positive; otherwise you’ll see negative/invalid logs
            y = np.log2(y)
            label = "log2 gain ratio (color / shape)"
    elif mode == "diff":
        y = g_color - g_shape
        label = "Gain difference (color - shape)"
    else:
        raise ValueError("mode must be 'ratio' or 'diff'")

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(color_diff, y, c=color_diff, cmap='bwr', alpha=0.7, edgecolors='none')
    plt.xlabel('Color - Shape selectivity')
    plt.ylabel(label)
    plt.title('Gains comparison vs. selectivity')
    cb = plt.colorbar(); cb.set_label('Color - Shape')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    png = outdir / f"{tag}_gains_{mode}_color_over_shape.png"
    pdf = outdir / f"{tag}_gains_{mode}_color_over_shape.pdf"
    plt.savefig(png, dpi=200); plt.savefig(pdf)
    if show: plt.show()
    plt.close(fig)


def plot_triad_cross_sweep(rows, outdir: Path, tag: str, ylabel: str = "Cross-attend angle (deg)") -> None:
    _ensure_dir(outdir)
    xs = [r["range"] for r in rows]
    ys_color = [r["color_cross_attend_deg"] for r in rows]
    ys_shape = [r["shape_cross_attend_deg"] for r in rows]

    fig = plt.figure(figsize=(6.5, 4.5))
    plt.plot(xs, ys_color, marker="o", label="Color axis (color vs shape gains)")
    plt.plot(xs, ys_shape, marker="s", label="Shape axis (color vs shape gains)")
    plt.xlabel("Constraint range")
    plt.ylabel(ylabel)
    plt.title(f"Cross-attend angles vs range — {tag}")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    png = outdir / f"{tag}_triad_cross_sweep.png"
    pdf = outdir / f"{tag}_triad_cross_sweep.pdf"
    plt.savefig(png, dpi=200); plt.savefig(pdf)
    plt.close(fig)

def plot_triad_cross_sweep(rows, outdir: Path, tag: str, ylabel: str = "Cross-attend angle (deg)") -> None:
    _ensure_dir(outdir)
    xs = [r["range"] for r in rows]
    ys_color = [r["color_cross_attend_deg"] for r in rows]
    ys_shape = [r["shape_cross_attend_deg"] for r in rows]
    has_shuf = "color_cross_attend_deg_shuf" in rows[0]

    fig = plt.figure(figsize=(7.0, 4.8))
    plt.plot(xs, ys_color, marker="o", label="Color cross-attend")
    plt.plot(xs, ys_shape, marker="s", label="Shape cross-attend")

    if has_shuf:
        ys_color_shuf = [r["color_cross_attend_deg_shuf"] for r in rows]
        ys_shape_shuf = [r["shape_cross_attend_deg_shuf"] for r in rows]
        plt.plot(xs, ys_color_shuf, marker="^", linestyle="--", label="Color cross-attend (shuffled)")
        plt.plot(xs, ys_shape_shuf, marker="v", linestyle="--", label="Shape cross-attend (shuffled)")

    plt.xlabel("Constraint range")
    plt.ylabel(ylabel)
    plt.title(f"Cross-attend angles vs range — {tag}")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    png = outdir / f"{tag}_triad_cross_sweep.png"
    pdf = outdir / f"{tag}_triad_cross_sweep.pdf"
    plt.savefig(png, dpi=200); plt.savefig(pdf)
    plt.close(fig)




def _save_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def _save_json(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def bivariate_colors(shape_list, color_list):
    shape = np.asarray(shape_list, dtype=float)
    color = np.asarray(color_list, dtype=float)
    L = np.clip(0.35 + 0.45 * shape, 0.0, 1.0)   # lightness encodes shape
    S = 0.90 * np.ones_like(L)                   # fixed saturation
    H = np.mod(color, 1.0)                       # hue encodes color
    rgb = [colorsys.hls_to_rgb(h, l, s) for h, l, s in zip(H, L, S)]
    return np.asarray(rgb)

def draw_bivariate_legend(fig, ax, *, loc=(0.72, 0.68, 0.24, 0.24)):
    N = 64
    xs = np.linspace(0, 1, N)   # color
    ys = np.linspace(0, 1, N)   # shape
    grid = np.zeros((N, N, 3))
    for i, y in enumerate(ys):
        row = bivariate_colors(np.full(N, y), xs)
        grid[N - 1 - i, :, :] = row  # shape increases upward
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

def _panel_b_save_data(
    outdir: Path,
    Z_color: np.ndarray,
    Z_shape: np.ndarray,
    shape_list: np.ndarray,
    color_list: np.ndarray,
    target_pc: np.ndarray,
    col_c_pc: np.ndarray,
    col_s_pc: np.ndarray,
    shp_c_pc: np.ndarray,
    shp_s_pc: np.ndarray,
) -> None:
    rows = []
    N = Z_color.shape[0]
    for i in range(N):
        rows.append((
            "color",
            Z_color[i, 0], Z_color[i, 1], Z_color[i, 2],
            float(shape_list[i]), float(color_list[i])
        ))
    for i in range(N):
        rows.append((
            "shape",
            Z_shape[i, 0], Z_shape[i, 1], Z_shape[i, 2],
            float(shape_list[i]), float(color_list[i])
        ))
    _save_csv(
        outdir / "panel_b_points.csv",
        ["state", "pc1", "pc2", "pc3", "shape", "color"],
        rows,
    )
    _save_json(outdir / "panel_b_axes.json", {
        "target_pc": target_pc.tolist(),
        "color_axis_color_pc": col_c_pc.tolist(),
        "color_axis_shape_pc": col_s_pc.tolist(),
        "shape_axis_shape_pc": shp_s_pc.tolist(),
        "shape_axis_color_pc": shp_c_pc.tolist(),
    })


def _draw_offset_line(ax, v: np.ndarray, Z_all: np.ndarray,
                      scale: float = 1.8, offset_frac: float = 0.12,
                      color: str = 'crimson', lw: float = 2.8) -> None:
    v = np.asarray(v).ravel()
    if np.linalg.norm(v) < 1e-12:
        return
    v = v / (np.linalg.norm(v) + 1e-12)
    mins = Z_all.min(axis=0)
    maxs = Z_all.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    start = np.array([0.0, mins[1] + offset_frac * span[1], 0.0], dtype=float)
    length = scale * np.max(span) * 0.25
    p0 = start
    p1 = start + length * v
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], lw=lw, color=color)


def plot_panel_b_color_state(
    Z_color: np.ndarray,
    Z_shape: np.ndarray,
    pca_components,
    target_vec_neuron: np.ndarray,
    color_axis_color: np.ndarray,
    color_axis_shape: np.ndarray,
    shape_axis_color: np.ndarray,
    shape_axis_shape: np.ndarray,
    shape_vals: Sequence[float],
    color_vals: Sequence[float],
    outdir: Path,
    tag: str,
    *,
    zlim=None,
    elev=None,
    azim=None,
    show: bool = False,
    save_data: bool = False,
) -> None:
    _ensure_dir(outdir)
    comps = pca_components[:3, :]
    t_pc = comps @ target_vec_neuron
    col_c = comps @ color_axis_color
    col_s = comps @ color_axis_shape
    shp_c = comps @ shape_axis_color
    shp_s = comps @ shape_axis_shape

    shape_list = np.array([s for s in shape_vals for _c in color_vals], dtype=float)
    color_list = np.array([c for _s in shape_vals for c in color_vals], dtype=float)
    bicolors = bivariate_colors(shape_list, color_list)

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        Z_color[:, 0], Z_color[:, 1], Z_color[:, 2],
        s=28, marker='o', c=bicolors, edgecolors='none', alpha=0.95,
        label="color-attend"
    )

    Z_all = np.vstack([Z_color, Z_shape])
    _draw_offset_line(ax, t_pc, Z_all, color='crimson', lw=2.8)

    ax.plot([0, col_c[0]], [0, col_c[1]], [0, col_c[2]],
            lw=2.2, color='darkorange', label="Color axis (color)")
    ax.plot([0, shp_c[0]], [0, shp_c[1]], [0, shp_c[2]],
            lw=2.2, color='seagreen', label="Shape axis (color)")

    if zlim is not None:
        ax.set_zlim(zlim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    elev_used = 30 if elev is None else elev
    azim_used = 166 if azim is None else azim
    ax.view_init(elev=elev_used, azim=azim_used)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='crimson', lw=2.8, label='Target (offset)'),
        Line2D([0], [0], color='darkorange', lw=2.2, label='Color axis (color)'),
        Line2D([0], [0], color='seagreen', lw=2.2, label='Shape axis (color)'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', frameon=True, fontsize=9)
    draw_bivariate_legend(fig, ax)
    plt.title(f"Panel B — Color attend ({tag})")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_b_color.{ext}", dpi=300,
                    bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close(fig)

    if save_data:
        _panel_b_save_data(
            outdir,
            Z_color, Z_shape,
            shape_list, color_list,
            t_pc, col_c, col_s, shp_c, shp_s
        )


def plot_panel_b_shape_state(
    Z_color: np.ndarray,
    Z_shape: np.ndarray,
    pca_components,
    target_vec_neuron: np.ndarray,
    color_axis_color: np.ndarray,
    color_axis_shape: np.ndarray,
    shape_axis_color: np.ndarray,
    shape_axis_shape: np.ndarray,
    shape_vals: Sequence[float],
    color_vals: Sequence[float],
    outdir: Path,
    tag: str,
    *,
    zlim=None,
    elev=None,
    azim=None,
    show: bool = False,
) -> None:
    _ensure_dir(outdir)
    comps = pca_components[:3, :]
    t_pc = comps @ target_vec_neuron
    col_c = comps @ color_axis_color
    col_s = comps @ color_axis_shape
    shp_c = comps @ shape_axis_color
    shp_s = comps @ shape_axis_shape

    shape_list = np.array([s for s in shape_vals for _c in color_vals], dtype=float)
    color_list = np.array([c for _s in shape_vals for c in color_vals], dtype=float)
    bicolors = bivariate_colors(shape_list, color_list)

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        Z_shape[:, 0], Z_shape[:, 1], Z_shape[:, 2],
        s=28, marker='^', c=bicolors, edgecolors='none', alpha=0.95,
        label="shape-attend"
    )

    Z_all = np.vstack([Z_color, Z_shape])
    _draw_offset_line(ax, t_pc, Z_all, color='crimson', lw=2.8)

    ax.plot([0, col_s[0]], [0, col_s[1]], [0, col_s[2]],
            lw=2.2, color='darkorange', label="Color axis (shape)")
    ax.plot([0, shp_s[0]], [0, shp_s[1]], [0, shp_s[2]],
            lw=2.2, color='seagreen', label="Shape axis (shape)")

    if zlim is not None:
        ax.set_zlim(zlim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    elev_used = 30 if elev is None else elev
    azim_used = 166 if azim is None else azim
    ax.view_init(elev=elev_used, azim=azim_used)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='crimson', lw=2.8, label='Target (offset)'),
        Line2D([0], [0], color='darkorange', lw=2.2, label='Color axis (shape)'),
        Line2D([0], [0], color='seagreen', lw=2.2, label='Shape axis (shape)'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', frameon=True, fontsize=9)
    draw_bivariate_legend(fig, ax)
    plt.title(f"Panel B — Shape attend ({tag})")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_b_shape.{ext}", dpi=300,
                    bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close(fig)


def plot_panel_b(
    Z_color, Z_shape,
    pca_components,
    target_vec_neuron,
    color_axis_color, color_axis_shape,
    shape_axis_color, shape_axis_shape,
    shape_vals, color_vals,
    outdir: Path,
    *, zlim=None, elev=None, azim=None, show=False, tag: str = "experiment"
):
    """
    Backwards-compatible entry point: render both color- and shape-state views.
    """
    plot_panel_b_color_state(
        Z_color, Z_shape, pca_components, target_vec_neuron,
        color_axis_color, color_axis_shape,
        shape_axis_color, shape_axis_shape,
        shape_vals, color_vals, outdir, tag,
        zlim=zlim, elev=elev, azim=azim, show=show, save_data=True
    )
    plot_panel_b_shape_state(
        Z_color, Z_shape, pca_components, target_vec_neuron,
        color_axis_color, color_axis_shape,
        shape_axis_color, shape_axis_shape,
        shape_vals, color_vals, outdir, tag,
        zlim=zlim, elev=elev, azim=azim, show=show
    )

def plot_panel_c_activity(net: Any, cfg: Any,
                          g_color: NDArray[np.float64], g_shape: NDArray[np.float64],
                          outdir: Path, tag: Optional[str] = None,
                          logratio: bool = False, show: bool = False) -> None:
    """
    Panel C (activity-derived, linear):
      x = selectivity (shape - color)
      y = |Δr_color(g_color)| / |Δr_shape(g_shape)|
    """
    _ensure_dir(outdir)

    shape_const = cfg.objective.shape_for_color_line
    color_const = cfg.objective.color_for_shape_line

    # color modulation under color-attend
    r_c0 = net.response(shape_const, 0.0, g=g_color)
    r_c1 = net.response(shape_const, 1.0, g=g_color)
    d_color_col = r_c1 - r_c0

    # shape modulation under shape-attend
    r_s0 = net.response(0.0, color_const, g=g_shape)
    r_s1 = net.response(1.0, color_const, g=g_shape)
    d_shape_shp = r_s1 - r_s0

    eps = 1e-12
    ratio = (np.abs(d_color_col) + eps) / (np.abs(d_shape_shp) + eps)
    sel_sc = net.S[:, 0] - net.S[:, 1]

    fig = plt.figure(figsize=(5.8, 4.8))
    plt.axhline(1.0, color='k', lw=1.0, ls='--', alpha=0.6)
    sc = plt.scatter(sel_sc, ratio, c=sel_sc, cmap='bwr', alpha=0.85, edgecolors='none', s=24)
    cb = plt.colorbar(sc); cb.set_label('Selectivity (shape - color)')
    plt.xlabel('Selectivity (shape - color)')
    plt.ylabel('Gain ratio (color / shape) from activity')
    if tag:
        plt.title(f"Activity-derived gain ratio — {tag}")
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_c.{ext}", dpi=300, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close(fig)

    rows = []
    for i in range(len(sel_sc)):
        rows.append((
            i,
            float(sel_sc[i]),
            float(d_color_col[i]),
            float(d_shape_shp[i]),
            float(ratio[i]),
        ))
    _save_csv(
        outdir / "panel_c_activity_data.csv",
        ["neuron",
         "selectivity_shape_minus_color",
         "delta_r_color_under_color",
         "delta_r_shape_under_shape",
         "ratio_color_over_shape_activity"],
        rows,
    )


def plot_panel_c_gopt(S: NDArray[np.float64],
                      g_color: NDArray[np.float64], g_shape: NDArray[np.float64],
                      outdir: Path, tag: Optional[str] = None,
                      logratio: bool = False, show: bool = False) -> None:
    """
    Panel C (optimized-g, linear):
      x = selectivity (shape - color)
      y = g_color / g_shape
    """
    _ensure_dir(outdir)
    eps = 1e-12
    ratio = (g_color + eps) / (g_shape + eps)
    sel_sc = S[:, 0] - S[:, 1]

    fig = plt.figure(figsize=(5.8, 4.8))
    plt.axhline(1.0, color='k', lw=1.0, ls='--', alpha=0.6)
    sc = plt.scatter(sel_sc, ratio, c=sel_sc, cmap='bwr', alpha=0.85, edgecolors='none', s=24)
    cb = plt.colorbar(sc); cb.set_label('Selectivity (shape - color)')
    plt.xlabel('Selectivity (shape - color)')
    plt.ylabel('Gain ratio (color / shape) from optimized g')
    if tag:
        plt.title(f"Optimized gains — {tag}")
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_c_gopt.{ext}", dpi=300, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close(fig)

    rows = []
    for i in range(len(sel_sc)):
        rows.append((
            i,
            float(sel_sc[i]),
            float(g_color[i]),
            float(g_shape[i]),
            float(ratio[i]),
        ))
    _save_csv(
        outdir / "panel_c_gopt_data.csv",
        ["neuron",
         "selectivity_shape_minus_color",
         "g_color",
         "g_shape",
         "ratio_color_over_shape_gopt"],
        rows,
    )


def plot_panel_c(S: NDArray[np.float64], g_color: NDArray[np.float64], g_shape: NDArray[np.float64],
                 outdir: Path, *, mode: str = "ratio", logratio: bool = False,
                 show: bool = False, tag: str = "experiment"):
    """
    Backwards-compatible wrapper that now emits the optimized-g view.
    """
    plot_panel_c_gopt(S, g_color, g_shape, outdir, tag=tag,
                      logratio=False, show=show)

def plot_panel_d_full(rows: list[dict], outdir: Path, tag: str,
                      ylabel: str = "Δ angle-to-target (deg)") -> None:
    _ensure_dir(outdir)
    xs = [r["range"] for r in rows]
    yc = [r["impr_color_deg"] for r in rows]  # color-axis improvement
    ys = [r["impr_shape_deg"] for r in rows]  # shape-axis improvement

    fig = plt.figure(figsize=(6.2, 4.8))
    plt.plot(xs, yc, marker="o", lw=2.0, label="Color-axis: (shape) - (color)")
    plt.plot(xs, ys, marker="s", lw=2.0, label="Shape-axis: (color) - (shape)")

    has_mean = ("impr_color_deg_shuf_mean" in rows[0])
    if has_mean:
        yc_m  = np.array([r["impr_color_deg_shuf_mean"] for r in rows])
        ys_m  = np.array([r["impr_shape_deg_shuf_mean"] for r in rows])
        yc_se = np.array([r.get("impr_color_deg_shuf_sem", 0.0) for r in rows])
        ys_se = np.array([r.get("impr_shape_deg_shuf_sem", 0.0) for r in rows])
        plt.plot(xs, yc_m, marker="^", lw=1.6, ls="--", label="Color-axis (shuf mean)")
        plt.plot(xs, ys_m, marker="v", lw=1.6, ls="--", label="Shape-axis (shuf mean)")
        plt.fill_between(xs, yc_m - 1.96*yc_se, yc_m + 1.96*yc_se, alpha=0.15, linewidth=0)
        plt.fill_between(xs, ys_m - 1.96*ys_se, ys_m + 1.96*ys_se, alpha=0.15, linewidth=0)

    plt.axhline(0.0, color='k', lw=1.0, ls='--', alpha=0.5)
    plt.xlabel("Constraint range")
    plt.ylabel(ylabel)
    plt.title(f"Δ to-target under own vs other attention — {tag}")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    for ext in ("png","pdf","svg"):
        plt.savefig(outdir / f"panel_d_full.{ext}", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # Save data
    out_rows = []
    for r in rows:
        out_rows.append((
            r["range"],
            r["impr_color_deg"], r["impr_shape_deg"],
            r.get("impr_color_deg_shuf_mean", r.get("impr_color_deg_shuf","")),
            r.get("impr_shape_deg_shuf_mean", r.get("impr_shape_deg_shuf","")),
            r.get("impr_color_deg_shuf_sem", ""),
            r.get("impr_shape_deg_shuf_sem", "")
        ))
    _save_csv(outdir / "panel_d_data.csv",
              ["range","impr_color_deg","impr_shape_deg",
               "impr_color_shuf_mean_or_single","impr_shape_shuf_mean_or_single",
               "impr_color_shuf_sem","impr_shape_shuf_sem"],
              out_rows)


def plot_panel_d_shape_only(rows: list[dict], outdir: Path, tag: str,
                            ylabel: str = "Δ angle-to-target (deg)") -> None:
    """
    COLOR-ONLY panel D: keeps color-axis improvement + shuffled statistics.
    """
    _ensure_dir(outdir)
    import numpy as np
    xs = [r["range"] for r in rows]
    yc = [r["impr_color_deg"] for r in rows]

    fig = plt.figure(figsize=(6.2, 4.8))
    plt.plot(xs, yc, marker="o", lw=2.0, label="Color-axis: (shape) - (color)")

    has_mean = ("impr_color_deg_shuf_mean" in rows[0])
    if has_mean:
        yc_m = np.array([r["impr_color_deg_shuf_mean"] for r in rows])
        yc_se = np.array([r.get("impr_color_deg_shuf_sem", 0.0) for r in rows])
        plt.plot(xs, yc_m, marker="^", lw=1.6, ls="--", label="Color-axis (shuf mean)")
        if np.isfinite(yc_se).all():
            plt.fill_between(xs, yc_m - 1.96 * yc_se, yc_m + 1.96 * yc_se, alpha=0.15, linewidth=0)

    plt.axhline(0.0, color='k', lw=1.0, ls='--', alpha=0.5)
    plt.xlabel("Constraint range")
    plt.ylabel(ylabel)
    plt.title(f"Δ to-target (color-axis) — {tag}")
    plt.ylim(0.0, 50.0)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    for ext in ("png","pdf","svg"):
        plt.savefig(outdir / f"panel_d.{ext}", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
