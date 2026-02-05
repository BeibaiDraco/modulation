#!/usr/bin/env python3
"""
Reproduce paper figures using saved optimized gains from figs_data.

This script bypasses the optimization step and uses pre-saved gains
to ensure exact reproducibility regardless of scipy version or numerical precision.

The key insight: figs_data/panel_c_gopt_data.csv contains the original optimized
gains (g_color, g_shape) that produced the paper figures. We use these directly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import colorsys

# Import the alignlab library
from src.alignlab.network import LinearRNN
from src.alignlab.config import NetworkConfig, GridConfig, ObjectiveConfig, WRNormalization
from src.alignlab.objectives import target_in_neuron_space

# ========== Configuration ==========
SCRIPT_DIR = Path(__file__).parent.resolve()
FIGS_DATA_DIR = SCRIPT_DIR / "figs_data"
OUTPUT_DIR = SCRIPT_DIR / "outputs_reproduced"

# Network config matching paper_panel_b.yaml
NETWORK_CONFIG = NetworkConfig(
    N=100,
    K=2,
    seed=21,
    desired_radius=0.8,
    p_high=0.2,
    p_low=0.2,
    zero_sum=WRNormalization.ROW,
    wr_tuned=False,
    weight_scale=0.1,
    baseline_equalize=False,
    gain_mode="both",
)

GRID_CONFIG = GridConfig(
    shape_vals=[0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
    color_vals=[0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
)

OBJECTIVE_CONFIG = ObjectiveConfig(
    target_type="custom_pc",
    theta_deg=80.0,
    phi_deg=5.0,
    axis_of_interest="color",
    shape_for_color_line=0.3,
    color_for_shape_line=0.6,
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def bivariate_colors(shape_list, color_list):
    """Generate bivariate colors for shape/color encoding."""
    shape = np.asarray(shape_list, dtype=float)
    color = np.asarray(color_list, dtype=float)
    L = np.clip(0.35 + 0.45 * shape, 0.0, 1.0)
    S = 0.90 * np.ones_like(L)
    H = 0.67 - 0.5 * color
    rgb = [colorsys.hls_to_rgb(h, l, s) for h, l, s in zip(H, L, S)]
    return np.asarray(rgb)


def load_saved_gains(csv_path: Path):
    """Load saved gains from panel_c_gopt_data.csv"""
    df = pd.read_csv(csv_path)
    g_color = df['g_color'].values
    g_shape = df['g_shape'].values
    selectivity = df['selectivity_shape_minus_color'].values
    return g_color, g_shape, selectivity


def verify_network_matches():
    """Verify that the generated network matches the saved weights."""
    net = LinearRNN(NETWORK_CONFIG)
    
    # Load saved weights
    W_R_saved = np.load(FIGS_DATA_DIR / "paper_b_W_R.npy")
    W_F_saved = np.loadtxt(FIGS_DATA_DIR / "paper_b_W_F.csv", delimiter=",")
    readout_saved = np.loadtxt(FIGS_DATA_DIR / "paper_b_readout.csv", delimiter=",")
    
    # Compare
    W_R_match = np.allclose(net.W_R, W_R_saved, rtol=1e-10, atol=1e-14)
    W_F_match = np.allclose(net.W_F, W_F_saved, rtol=1e-10, atol=1e-14)
    
    print(f"W_R matches saved: {W_R_match}")
    print(f"W_F matches saved: {W_F_match}")
    
    if not W_R_match:
        print(f"  W_R max diff: {np.max(np.abs(net.W_R - W_R_saved))}")
    if not W_F_match:
        print(f"  W_F max diff: {np.max(np.abs(net.W_F - W_F_saved))}")
    
    return W_R_match and W_F_match, net, readout_saved


def generate_figures_with_saved_gains():
    """Main function to reproduce all figures using saved gains."""
    print("=" * 70)
    print("Reproducing figures using saved optimized gains")
    print("=" * 70)
    
    ensure_dir(OUTPUT_DIR)
    
    # Step 1: Verify network reproducibility
    print("\n[1/5] Verifying network generation is deterministic...")
    network_ok, net, target = verify_network_matches()
    if not network_ok:
        print("WARNING: Network weights don't match exactly. Results may differ.")
    
    # Step 2: Load saved gains
    print("\n[2/5] Loading saved optimized gains from figs_data...")
    g_color, g_shape, selectivity = load_saved_gains(FIGS_DATA_DIR / "panel_c_gopt_data.csv")
    print(f"  Loaded {len(g_color)} neurons' gains")
    print(f"  g_color range: [{g_color.min():.4f}, {g_color.max():.4f}]")
    print(f"  g_shape range: [{g_shape.min():.4f}, {g_shape.max():.4f}]")
    
    # Step 3: Compute PCA (fit on unmodulated responses)
    print("\n[3/5] Computing PCA on unmodulated responses...")
    X0 = net.grid_responses(GRID_CONFIG, g=None)  # Unmodulated
    pca, Z0 = net.pca3(X0)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
    
    # Step 4: Compute responses with saved gains
    print("\n[4/5] Computing responses with saved gains...")
    X_color = net.grid_responses(GRID_CONFIG, g=g_color)
    X_shape = net.grid_responses(GRID_CONFIG, g=g_shape)
    
    # Project into PCA space (same basis fitted on unmodulated)
    Z_color = pca.transform(X_color)
    Z_shape = pca.transform(X_shape)
    
    # Compute axes
    color_axis_color = net.color_axis(OBJECTIVE_CONFIG.shape_for_color_line, g=g_color)
    color_axis_shape = net.color_axis(OBJECTIVE_CONFIG.shape_for_color_line, g=g_shape)
    shape_axis_color = net.shape_axis(OBJECTIVE_CONFIG.color_for_shape_line, g=g_color)
    shape_axis_shape = net.shape_axis(OBJECTIVE_CONFIG.color_for_shape_line, g=g_shape)
    
    # Project axes to PC space
    comps = pca.components_[:3, :]
    target_pc = comps @ target
    col_axis_color_pc = comps @ color_axis_color
    col_axis_shape_pc = comps @ color_axis_shape
    shp_axis_color_pc = comps @ shape_axis_color
    shp_axis_shape_pc = comps @ shape_axis_shape
    
    # Step 5: Verify against saved data
    print("\n[5/5] Verifying against saved panel_b data...")
    saved_points = pd.read_csv(FIGS_DATA_DIR / "panel_b_points.csv")
    with open(FIGS_DATA_DIR / "panel_b_axes.json") as f:
        saved_axes = json.load(f)
    
    # Extract saved color and shape state points
    saved_color = saved_points[saved_points['state'] == 'color'][['pc1', 'pc2', 'pc3']].values
    saved_shape = saved_points[saved_points['state'] == 'shape'][['pc1', 'pc2', 'pc3']].values
    
    # Compare
    color_match = np.allclose(Z_color, saved_color, rtol=1e-6)
    shape_match = np.allclose(Z_shape, saved_shape, rtol=1e-6)
    
    print(f"  Color state points match: {color_match}")
    print(f"  Shape state points match: {shape_match}")
    
    if not color_match:
        print(f"    Max diff: {np.max(np.abs(Z_color - saved_color)):.6e}")
    if not shape_match:
        print(f"    Max diff: {np.max(np.abs(Z_shape - saved_shape)):.6e}")
    
    # Compare axes
    target_match = np.allclose(target_pc, saved_axes['target_pc'], rtol=1e-6)
    print(f"  Target axis matches: {target_match}")
    
    if color_match and shape_match:
        print("\n✓ SUCCESS: Reproduced data matches figs_data exactly!")
    else:
        print("\n✗ WARNING: Some discrepancy detected. Check above for details.")
    
    # Save reproduced data
    print("\n[Saving] Writing reproduced data to outputs_reproduced/...")
    
    # Save points CSV
    shape_vals = GRID_CONFIG.shape_vals
    color_vals = GRID_CONFIG.color_vals
    rows = []
    idx = 0
    for sv in shape_vals:
        for cv in color_vals:
            rows.append({
                'state': 'color',
                'pc1': Z_color[idx, 0],
                'pc2': Z_color[idx, 1],
                'pc3': Z_color[idx, 2],
                'shape': sv,
                'color': cv,
            })
            idx += 1
    idx = 0
    for sv in shape_vals:
        for cv in color_vals:
            rows.append({
                'state': 'shape',
                'pc1': Z_shape[idx, 0],
                'pc2': Z_shape[idx, 1],
                'pc3': Z_shape[idx, 2],
                'shape': sv,
                'color': cv,
            })
            idx += 1
    
    df_points = pd.DataFrame(rows)
    df_points.to_csv(OUTPUT_DIR / "panel_b_points.csv", index=False)
    
    # Save axes JSON
    axes_data = {
        'target_pc': target_pc.tolist(),
        'color_axis_color_pc': col_axis_color_pc.tolist(),
        'color_axis_shape_pc': col_axis_shape_pc.tolist(),
        'shape_axis_shape_pc': shp_axis_shape_pc.tolist(),
        'shape_axis_color_pc': shp_axis_color_pc.tolist(),
    }
    with open(OUTPUT_DIR / "panel_b_axes.json", 'w') as f:
        json.dump(axes_data, f, indent=2)
    
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    return {
        'network_ok': network_ok,
        'color_match': color_match,
        'shape_match': shape_match,
        'g_color': g_color,
        'g_shape': g_shape,
        'Z_color': Z_color,
        'Z_shape': Z_shape,
        'pca': pca,
        'net': net,
    }


if __name__ == "__main__":
    result = generate_figures_with_saved_gains()
