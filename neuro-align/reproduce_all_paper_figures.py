#!/usr/bin/env python3
"""
Script to reproduce all paper figures from saved CSV data.

This is a standalone script that reads data from the figs_data/ directory
and generates all publication-ready figures in the figs_paper/ directory.

Required directory structure:
    figs_data/
        panel_b_points.csv
        panel_b_axes.json
        panel_c_gopt_data.csv
        panel_d_data.csv
    reproduce_all_paper_figures.py (this script)
    
Output:
    figs_paper/ - All generated figures in PNG, PDF, and SVG formats
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import json
import csv

# ========== Configuration ==========
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Output directory for figures
OUTPUT_DIR = SCRIPT_DIR / "figs_paper"

# Input data directory (standalone data files)
DATA_DIR = SCRIPT_DIR / "figs_data"

# ========== Utility Functions ==========
def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def bivariate_colors(shape_list, color_list):
    """Generate bivariate colors for shape/color encoding."""
    shape = np.asarray(shape_list, dtype=float)
    color = np.asarray(color_list, dtype=float)
    L = np.clip(0.35 + 0.45 * shape, 0.0, 1.0)   # lightness encodes shape
    S = 0.90 * np.ones_like(L)                   # fixed saturation
    H = 0.67 - 0.5 * color                       # Viridis-like: blue → green → yellow
    rgb = [colorsys.hls_to_rgb(h, l, s) for h, l, s in zip(H, L, S)]
    return np.asarray(rgb)

def set_3d_view_angle(ax, elev=None, azim=None):
    """Set 3D view angle (default: elev=8, azim=160)."""
    elev_used = 8 if elev is None else elev
    azim_used = 160 if azim is None else azim
    ax.view_init(elev=elev_used, azim=azim_used)
    return elev_used, azim_used

def draw_offset_line(ax, v: np.ndarray, Z_all: np.ndarray,
                     scale: float = 1.8, offset_frac: float = 0.12,
                     color: str = 'crimson', lw: float = 2.8) -> None:
    """Draw an offset line for the target axis."""
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

# ========== Panel B Functions ==========
def create_panel_b_colormap(outdir: Path) -> None:
    """Create a separate colormap plot for Panel B."""
    ensure_dir(outdir)
    N = 128
    xs = np.linspace(0, 1, N)   # color
    ys = np.linspace(0, 1, N)   # shape
    grid = np.zeros((N, N, 3))
    for i, y in enumerate(ys):
        row = bivariate_colors(np.full(N, y), xs)
        grid[N - 1 - i, :, :] = row  # shape increases upward
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(grid, origin="lower", extent=(0, 1, 0, 1))
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=16, fontweight='bold')
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=16, fontweight='bold')
    ax.set_xlabel("Color", fontsize=18, fontweight='bold')
    ax.set_ylabel("Shape", fontsize=18, fontweight='bold')
    ax.tick_params(length=4, pad=4)
    for s in ax.spines.values():
        s.set_linewidth(2.0)
    
    plt.title("Bivariate Color Legend", fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_b_colormap.{ext}", dpi=300,
                    bbox_inches="tight", transparent=True)
    plt.close(fig)

def create_panel_b_legend(outdir: Path, state: str) -> None:
    """Create a separate legend plot for Panel B."""
    ensure_dir(outdir)
    from matplotlib.lines import Line2D
    
    if state == "color":
        legend_elems = [
            Line2D([0], [0], color='crimson', lw=4.0, label='Target (offset)'),
            Line2D([0], [0], color='darkorange', lw=4.0, label='Color axis (color)'),
            Line2D([0], [0], color='seagreen', lw=4.0, label='Shape axis (color)'),
        ]
    else:  # shape
        legend_elems = [
            Line2D([0], [0], color='crimson', lw=4.0, label='Target (offset)'),
            Line2D([0], [0], color='darkorange', lw=4.0, label='Color axis (shape)'),
            Line2D([0], [0], color='seagreen', lw=4.0, label='Shape axis (shape)'),
        ]
    
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')  # Hide axes
    
    legend = ax.legend(handles=legend_elems, loc='center', frameon=True, 
                      fontsize=16, framealpha=1.0, edgecolor='none')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0)
    
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_b_{state}_legend.{ext}", dpi=300,
                    bbox_inches="tight", transparent=True)
    plt.close(fig)

def plot_panel_b_color_from_csv(csv_path: Path, axes_data: dict, outdir: Path,
                                 tag: str = "paper_b", elev=None, azim=None) -> None:
    """Recreate Panel B color state from CSV data."""
    ensure_dir(outdir)
    
    # Read CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Extract color-state and shape-state data
    color_data = [r for r in rows if r['state'] == 'color']
    shape_data = [r for r in rows if r['state'] == 'shape']
    
    Z_color = np.array([[float(r['pc1']), float(r['pc2']), float(r['pc3'])] for r in color_data])
    shape_list = np.array([float(r['shape']) for r in color_data])
    color_list = np.array([float(r['color']) for r in color_data])
    
    # Get axes from JSON
    t_pc = np.array(axes_data['target_pc'])
    col_c = np.array(axes_data['color_axis_color_pc'])
    shp_c = np.array(axes_data['shape_axis_color_pc'])
    
    # Generate bivariate colors
    bicolors = bivariate_colors(shape_list, color_list)
    
    # Create figure
    fig = plt.figure(figsize=(8.0, 7.0), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    
    # Set white background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(True, color='lightgray', alpha=0.3)
    
    # Scatter plot
    ax.scatter(Z_color[:, 0], Z_color[:, 1], Z_color[:, 2],
               s=80, marker='o', c=bicolors, edgecolors='none', alpha=0.95,
               label="color-attend")
    
    # Draw axes
    Z_shape = np.array([[float(r['pc1']), float(r['pc2']), float(r['pc3'])] for r in shape_data])
    Z_all = np.vstack([Z_color, Z_shape])
    draw_offset_line(ax, t_pc, Z_all, color='crimson', lw=3.5)
    
    ax.plot([0, col_c[0]], [0, col_c[1]], [0, col_c[2]],
            lw=3.0, color='darkorange', label="Color axis (color)")
    ax.plot([0, shp_c[0]], [0, shp_c[1]], [0, shp_c[2]],
            lw=3.0, color='seagreen', label="Shape axis (color)")
    
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('PC1', fontsize=20, fontweight='bold', labelpad=20)
    ax.set_ylabel('PC2', fontsize=20, fontweight='bold', labelpad=20)
    ax.set_zlabel('PC3', fontsize=20, fontweight='bold', labelpad=20)
    
    ax.tick_params(axis='x', which='major', labelsize=16, pad=10)
    ax.tick_params(axis='y', which='major', labelsize=16, pad=10)
    ax.tick_params(axis='z', which='major', labelsize=16, pad=10)
    
    set_3d_view_angle(ax, elev=elev, azim=azim)
    
    plt.title(f"Panel B — Color attend ({tag})", fontsize=22, fontweight='bold', pad=25)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_b_color.{ext}", dpi=300,
                    bbox_inches="tight", transparent=True, facecolor='white', pad_inches=0.3)
    plt.close(fig)

def plot_panel_b_shape_from_csv(csv_path: Path, axes_data: dict, outdir: Path,
                                 tag: str = "paper_b", elev=None, azim=None) -> None:
    """Recreate Panel B shape state from CSV data."""
    ensure_dir(outdir)
    
    # Read CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Extract shape-state and color-state data
    color_data = [r for r in rows if r['state'] == 'color']
    shape_data = [r for r in rows if r['state'] == 'shape']
    
    Z_shape = np.array([[float(r['pc1']), float(r['pc2']), float(r['pc3'])] for r in shape_data])
    shape_list = np.array([float(r['shape']) for r in shape_data])
    color_list = np.array([float(r['color']) for r in shape_data])
    
    # Get axes from JSON
    t_pc = np.array(axes_data['target_pc'])
    col_s = np.array(axes_data['color_axis_shape_pc'])
    shp_s = np.array(axes_data['shape_axis_shape_pc'])
    
    # Generate bivariate colors
    bicolors = bivariate_colors(shape_list, color_list)
    
    # Create figure
    fig = plt.figure(figsize=(8.0, 7.0), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    
    # Set white background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(True, color='lightgray', alpha=0.3)
    
    # Scatter plot
    ax.scatter(Z_shape[:, 0], Z_shape[:, 1], Z_shape[:, 2],
               s=80, marker='^', c=bicolors, edgecolors='none', alpha=0.95,
               label="shape-attend")
    
    # Draw axes
    Z_color = np.array([[float(r['pc1']), float(r['pc2']), float(r['pc3'])] for r in color_data])
    Z_all = np.vstack([Z_color, Z_shape])
    draw_offset_line(ax, t_pc, Z_all, color='crimson', lw=3.5)
    
    ax.plot([0, col_s[0]], [0, col_s[1]], [0, col_s[2]],
            lw=3.0, color='darkorange', label="Color axis (shape)")
    ax.plot([0, shp_s[0]], [0, shp_s[1]], [0, shp_s[2]],
            lw=3.0, color='seagreen', label="Shape axis (shape)")
    
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('PC1', fontsize=20, fontweight='bold', labelpad=20)
    ax.set_ylabel('PC2', fontsize=20, fontweight='bold', labelpad=20)
    ax.set_zlabel('PC3', fontsize=20, fontweight='bold', labelpad=20)
    
    ax.tick_params(axis='x', which='major', labelsize=16, pad=10)
    ax.tick_params(axis='y', which='major', labelsize=16, pad=10)
    ax.tick_params(axis='z', which='major', labelsize=16, pad=10)
    
    set_3d_view_angle(ax, elev=elev, azim=azim)
    
    plt.title(f"Panel B — Shape attend ({tag})", fontsize=22, fontweight='bold', pad=25)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_b_shape.{ext}", dpi=300,
                    bbox_inches="tight", transparent=True, facecolor='white', pad_inches=0.3)
    plt.close(fig)

# ========== Panel C Functions ==========
def plot_panel_c_gopt_from_csv(csv_path: Path, outdir: Path, tag: str = "paper_c") -> None:
    """Recreate Panel C from CSV data.
    Note: CSV column is named 'selectivity_shape_minus_color' but actual calculation 
    in plotting.py is S[:, 1] - S[:, 0] which is color - shape (opposite of name).
    """
    ensure_dir(outdir)
    
    # Read CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Note: The column name says "shape_minus_color" but it's actually color - shape
    sel_sc = np.array([float(r['selectivity_shape_minus_color']) for r in rows])
    ratio = np.array([float(r['ratio_color_over_shape_gopt_adjusted']) for r in rows])
    
    # Create square figure with white background
    fig = plt.figure(figsize=(6.0, 6.0), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Add horizontal reference line at y=0
    plt.axhline(0.0, color='k', lw=2.0, ls='--', alpha=0.7)
    
    # Create scatter plot with grey dots
    plt.scatter(sel_sc, ratio, color='grey', alpha=0.7, edgecolors='none', s=60)
    
    # Set X-axis range from -1.0 to 1.0
    plt.xlim(-1.0, 1.0)
    plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Set labels
    plt.xlabel('Feature selectivity index', fontsize=24, fontweight='bold')
    plt.ylabel('Gain modulation', fontsize=24, fontweight='bold')
    
    # Customize tick labels
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Add grid
    plt.grid(True, linestyle=':', alpha=0.5, color='lightgray')
    
    # Adjust layout
    plt.subplots_adjust(left=0.22, right=0.95, top=0.95, bottom=0.22)
    
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_c_gopt.{ext}", dpi=300, bbox_inches="tight", 
                    transparent=True, facecolor='white', pad_inches=0.2)
    plt.close(fig)

# ========== Panel D Functions ==========
def plot_panel_d_from_csv(csv_path: Path, outdir: Path, tag: str = "paper_panel_d",
                          ylabel: str = "Δ angle-to-target (deg)") -> None:
    """Recreate Panel D from CSV data (color-axis only view)."""
    ensure_dir(outdir)
    
    # Read CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    xs = np.array([float(r['range']) for r in rows])
    yc = np.array([float(r['impr_color_deg']) for r in rows])
    x_plot = np.sqrt(xs + 1.0) - 1.0
    
    fig = plt.figure(figsize=(6.5, 5.5), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")
    
    plt.plot(x_plot, yc, marker="o", lw=3.0, color="darkorange",
             markersize=8, label="Color axis change")
    
    # Check if shuffle data is available
    if 'impr_color_shuf_mean_or_single' in rows[0]:
        try:
            yc_m = np.array([float(r['impr_color_shuf_mean_or_single']) for r in rows])
            if 'impr_color_shuf_sem' in rows[0]:
                yc_se = np.array([float(r['impr_color_shuf_sem']) for r in rows])
                plt.plot(x_plot, yc_m, marker="^", lw=2.5, ls="--", color="grey",
                         markersize=6, label="Mean of shuffle")
                if np.isfinite(yc_se).all():
                    plt.fill_between(x_plot, yc_m - 1.96 * yc_se, yc_m + 1.96 * yc_se,
                                    alpha=0.15, color="grey", linewidth=0)
        except (ValueError, KeyError):
            pass  # Skip if shuffle data is not properly formatted
    
    plt.axhline(0.0, color="k", lw=2.0, ls="--", alpha=0.7)
    plt.xlabel("Constraint range", fontsize=20, fontweight="bold")
    plt.ylabel(ylabel, fontsize=20, fontweight="bold")
    plt.ylim(-2.0, 65.0)
    plt.grid(True, linestyle=":", alpha=0.5, color="lightgray")
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.legend(fontsize=16, frameon=True, facecolor="white", edgecolor="none")
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    for ext in ("png", "pdf", "svg"):
        plt.savefig(outdir / f"panel_d.{ext}", dpi=300, bbox_inches="tight",
                    transparent=True, facecolor="white", pad_inches=0.2)
    plt.close(fig)

# ========== Main Execution ==========
def main():
    """Main function to recreate all figures."""
    print("=" * 70)
    print("Reproducing all paper figures from CSV data")
    print("=" * 70)
    
    ensure_dir(OUTPUT_DIR)
    
    # ===== Panel B =====
    print("\n[1/3] Processing Panel B...")
    panel_b_csv = DATA_DIR / "panel_b_points.csv"
    panel_b_axes = DATA_DIR / "panel_b_axes.json"
    
    if panel_b_csv.exists() and panel_b_axes.exists():
        print(f"  Reading: {panel_b_csv}")
        print(f"  Reading: {panel_b_axes}")
        with open(panel_b_axes, 'r') as f:
            axes_data = json.load(f)
        
        print("  Creating Panel B color state...")
        plot_panel_b_color_from_csv(panel_b_csv, axes_data, OUTPUT_DIR, tag="paper_b")
        
        print("  Creating Panel B shape state...")
        plot_panel_b_shape_from_csv(panel_b_csv, axes_data, OUTPUT_DIR, tag="paper_b")
        
        print("  Creating Panel B colormap...")
        create_panel_b_colormap(OUTPUT_DIR)
        
        print("  Creating Panel B legends...")
        create_panel_b_legend(OUTPUT_DIR, "color")
        create_panel_b_legend(OUTPUT_DIR, "shape")
        
        print("  ✓ Panel B complete")
    else:
        print(f"  ✗ Panel B data not found!")
    
    # ===== Panel C =====
    print("\n[2/3] Processing Panel C...")
    panel_c_csv = DATA_DIR / "panel_c_gopt_data.csv"
    
    if panel_c_csv.exists():
        print(f"  Reading: {panel_c_csv}")
        print("  Creating Panel C (gopt)...")
        plot_panel_c_gopt_from_csv(panel_c_csv, OUTPUT_DIR, tag="paper_c")
        
        print("  ✓ Panel C complete")
    else:
        print(f"  ✗ Panel C data not found!")
    
    # ===== Panel D =====
    print("\n[3/3] Processing Panel D...")
    panel_d_csv = DATA_DIR / "panel_d_data.csv"
    
    if panel_d_csv.exists():
        print(f"  Reading: {panel_d_csv}")
        print("  Creating Panel D...")
        plot_panel_d_from_csv(panel_d_csv, OUTPUT_DIR, tag="paper_panel_d")
        
        print("  ✓ Panel D complete")
    else:
        print(f"  ✗ Panel D data not found!")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("All figures have been saved to:")
    print(f"  {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  Panel B:")
    print("    - panel_b_color.{png,pdf,svg}")
    print("    - panel_b_shape.{png,pdf,svg}")
    print("    - panel_b_colormap.{png,pdf,svg}")
    print("    - panel_b_color_legend.{png,pdf,svg}")
    print("    - panel_b_shape_legend.{png,pdf,svg}")
    print("  Panel C:")
    print("    - panel_c_gopt.{png,pdf,svg}")
    print("  Panel D:")
    print("    - panel_d.{png,pdf,svg}")
    print(f"\nData files are in: {DATA_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
