from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.typing import NDArray

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
) -> None:
    """Your two-panel 3D plot with colorbars and axis/target lines."""
    _ensure_dir(outdir)

    comps = pca_components[:3, :]                # 3 x N
    target_pc_coords = comps @ target_vec_neuron # (3,)
    axis_unmod_pc    = comps @ d_unmod_neuron    # (3,)
    axis_mod_pc      = comps @ d_mod_neuron      # (3,)

    mins = np.minimum(Z_unmod.min(axis=0), Z_mod.min(axis=0))
    maxs = np.maximum(Z_unmod.max(axis=0), Z_mod.max(axis=0))

    # Match your ordering: for each shape, iterate color
    color_list = np.array([c for _s in shape_vals for c in color_vals])

    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    sc1 = ax1.scatter(Z_unmod[:, 0], Z_unmod[:, 1], Z_unmod[:, 2],
                      c=color_list, cmap='viridis', s=20)
    sc2 = ax2.scatter(Z_mod[:, 0], Z_mod[:, 1], Z_mod[:, 2],
                      c=color_list, cmap='viridis', s=20)

    # Lines: target + axes (colors match your snippet)
    line3d(ax1, target_pc_coords, two_sided=False, linewidth=2, color='crimson')
    line3d(ax1, axis_unmod_pc,   two_sided=False, linewidth=2, color='black')
    line3d(ax2, target_pc_coords, two_sided=False, linewidth=2, color='crimson')
    line3d(ax2, axis_mod_pc,      two_sided=False, linewidth=2, color='darkorange')

    for ax in (ax1, ax2):
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])
        ax.set_zlim([mins[2], maxs[2]])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_box_aspect((1, 1, 1))

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
    plt.close(fig)

def plot_embedding_3d(Z0: NDArray[np.float64], Z1: NDArray[np.float64],
                      pca_components, target_vec_neuron: NDArray[np.float64],
                      d_unmod_neuron: NDArray[np.float64], d_opt_neuron: NDArray[np.float64],
                      outdir: Path, tag: str) -> None:
    """Keeps the combined single-axes plot (if you still want it)."""
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
    """Your 'gains vs selectivity bias' scatter (bwr colormap)."""
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
