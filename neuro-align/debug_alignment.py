#!/usr/bin/env python3
"""
Diagnostic script to identify why there's no improvement between attention states.
Run this after installing the package: pip install -e .
"""

import numpy as np
import yaml
from pathlib import Path
from alignlab.config import ExperimentConfig
from alignlab.network import LinearRNN
from alignlab.objectives import target_in_neuron_space
from alignlab.optimize import optimize_triad

def load_config_from_yaml(config_path: Path) -> ExperimentConfig:
    """Load config using the same logic as CLI"""
    from alignlab.cli import load_config
    return load_config(config_path)

def diagnose_alignment(config_path: str = "configs/paper.yaml"):
    """Diagnose why improvements might be zero or small"""
    
    print("=" * 80)
    print("ALIGNMENT DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # Load configuration
    cfg = load_config_from_yaml(Path(config_path))
    
    # Build network
    net = LinearRNN(cfg.network)
    
    # Get unmodulated responses and fit PCA
    X0 = net.grid_responses(cfg.grid, g=None)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, svd_solver="full", random_state=cfg.network.seed)
    Z0 = pca.fit_transform(X0)
    
    # Get target in neuron space
    target = target_in_neuron_space(pca, cfg.objective)
    
    # Define axes at baseline (g=1)
    g_ones = np.ones(cfg.network.N)
    color_axis_base = net.color_axis(cfg.objective.shape_for_color_line, g=g_ones)
    shape_axis_base = net.shape_axis(cfg.objective.color_for_shape_line, g=g_ones)
    
    # Calculate initial angles to target
    from alignlab.network import angle_deg
    color_to_target_base = angle_deg(color_axis_base, target)
    shape_to_target_base = angle_deg(shape_axis_base, target)
    axes_angle_base = angle_deg(color_axis_base, shape_axis_base)
    
    print("\n1. BASELINE CONFIGURATION")
    print("-" * 40)
    print(f"Network size: {cfg.network.N} neurons")
    print(f"Zero-sum mode: {cfg.network.zero_sum}")
    print(f"Baseline equalize: {cfg.network.baseline_equalize}")
    print(f"Weight scale: {cfg.network.weight_scale}")
    
    print("\n2. INITIAL GEOMETRY (Unmodulated)")
    print("-" * 40)
    print(f"Color axis → Target: {color_to_target_base:.2f}°")
    print(f"Shape axis → Target: {shape_to_target_base:.2f}°")
    print(f"Color axis ↔ Shape axis: {axes_angle_base:.2f}°")
    print(f"||Color axis||: {np.linalg.norm(color_axis_base):.4f}")
    print(f"||Shape axis||: {np.linalg.norm(shape_axis_base):.4f}")
    print(f"||Target||: {np.linalg.norm(target):.4f}")
    
    # Check if axes are too similar
    cos_sim = np.dot(color_axis_base, shape_axis_base) / (
        np.linalg.norm(color_axis_base) * np.linalg.norm(shape_axis_base) + 1e-12
    )
    print(f"Cosine similarity between axes: {cos_sim:.4f}")
    
    print("\n3. TARGET CONFIGURATION")
    print("-" * 40)
    print(f"Target type: {cfg.objective.target_type}")
    print(f"Theta (azimuth): {cfg.objective.theta_deg}°")
    print(f"Phi (elevation): {cfg.objective.phi_deg}°")
    
    # Project target to PC space
    target_pc = pca.components_[:3, :] @ target
    target_pc_norm = target_pc / (np.linalg.norm(target_pc) + 1e-12)
    print(f"Target in PC space (normalized): [{target_pc_norm[0]:.3f}, {target_pc_norm[1]:.3f}, {target_pc_norm[2]:.3f}]")
    
    print("\n4. CONSTRAINTS")
    print("-" * 40)
    print(f"Type: {cfg.constraints.type}")
    if cfg.constraints.type == "ball":
        print(f"Radius: {cfg.constraints.radius} (fraction of √N)")
        print(f"Effective radius: {cfg.constraints.radius * np.sqrt(cfg.network.N):.3f}")
    else:
        print(f"Box half-width: {cfg.constraints.box_half_width}")
    print(f"Hard norm constraint: {cfg.constraints.hard_norm}")
    print(f"Positive gains: {cfg.constraints.positive_gains}")
    
    print("\n5. AXIS DEFINITION POINTS")
    print("-" * 40)
    print(f"Shape value for color axis: {cfg.objective.shape_for_color_line}")
    print(f"Color value for shape axis: {cfg.objective.color_for_shape_line}")
    
    # Check responses at these points
    r_base = net.response(cfg.objective.shape_for_color_line, 0.5, g=None)
    print(f"Response magnitude at color axis center: {np.linalg.norm(r_base):.4f}")
    
    print("\n6. RUNNING TRIAD OPTIMIZATION...")
    print("-" * 40)
    
    # Run the actual optimization
    (summary, net, pca, Z0, Zc, Zs,
     dcol0, dcol1, dshp0, dshp1, target, gcol, gshp) = optimize_triad(cfg)
    
    # Extract results
    color_results = summary.get("color_alignment", {})
    shape_results = summary.get("shape_alignment", {})
    to_target = summary.get("to_target", {})
    
    print("\n7. OPTIMIZATION RESULTS")
    print("-" * 40)
    
    # Color optimization
    print("Color axis optimization:")
    print(f"  Success: {color_results.get('success', False)}")
    print(f"  Initial angle: {color_results.get('angles_deg', {}).get('unmod_to_target', 0):.2f}°")
    print(f"  Final angle: {color_results.get('angles_deg', {}).get('opt_to_target', 0):.2f}°")
    print(f"  Improvement: {color_results.get('angles_deg', {}).get('improvement', 0):.2f}°")
    
    # Shape optimization  
    print("\nShape axis optimization:")
    print(f"  Success: {shape_results.get('success', False)}")
    print(f"  Initial angle: {shape_results.get('angles_deg', {}).get('unmod_to_target', 0):.2f}°")
    print(f"  Final angle: {shape_results.get('angles_deg', {}).get('opt_to_target', 0):.2f}°")
    print(f"  Improvement: {shape_results.get('angles_deg', {}).get('improvement', 0):.2f}°")
    
    print("\n8. CROSS-ATTENTION ANALYSIS")
    print("-" * 40)
    
    color_to_target_data = to_target.get("color_axis", {})
    shape_to_target_data = to_target.get("shape_axis", {})
    
    print("Color axis under different gains:")
    print(f"  Under color gains → Target: {color_to_target_data.get('under_color_deg', 0):.2f}°")
    print(f"  Under shape gains → Target: {color_to_target_data.get('under_shape_deg', 0):.2f}°")
    print(f"  Improvement (shape - color): {color_to_target_data.get('improvement_deg', 0):.2f}°")
    
    print("\nShape axis under different gains:")
    print(f"  Under shape gains → Target: {shape_to_target_data.get('under_shape_deg', 0):.2f}°")
    print(f"  Under color gains → Target: {shape_to_target_data.get('under_color_deg', 0):.2f}°")
    print(f"  Improvement (color - shape): {shape_to_target_data.get('improvement_deg', 0):.2f}°")
    
    print("\n9. GAIN STATISTICS")
    print("-" * 40)
    
    # Analyze the optimized gains
    print(f"Color gains: min={gcol.min():.3f}, max={gcol.max():.3f}, mean={gcol.mean():.3f}, std={gcol.std():.3f}")
    print(f"Shape gains: min={gshp.min():.3f}, max={gshp.max():.3f}, mean={gshp.mean():.3f}, std={gshp.std():.3f}")
    
    # Check if gains are hitting bounds
    if cfg.constraints.type == "box":
        w = cfg.constraints.box_half_width
        lo, hi = 1.0 - w, 1.0 + w
        color_at_bounds = np.sum(np.isclose(gcol, lo, atol=1e-6)) + np.sum(np.isclose(gcol, hi, atol=1e-6))
        shape_at_bounds = np.sum(np.isclose(gshp, lo, atol=1e-6)) + np.sum(np.isclose(gshp, hi, atol=1e-6))
        print(f"Color gains at bounds: {color_at_bounds}/{cfg.network.N}")
        print(f"Shape gains at bounds: {shape_at_bounds}/{cfg.network.N}")
    
    print("\n10. POTENTIAL ISSUES IDENTIFIED")
    print("-" * 40)
    
    issues = []
    
    # Check if initial angles are too small
    if color_to_target_base < 10 and shape_to_target_base < 10:
        issues.append("✗ Both axes are already well-aligned to target (<10°)")
    
    # Check if axes are too similar
    if axes_angle_base < 20:
        issues.append(f"✗ Color and shape axes are too similar (angle={axes_angle_base:.1f}°)")
    
    # Check if improvements are suspiciously small
    if abs(color_to_target_data.get('improvement_deg', 0)) < 0.5:
        issues.append("✗ Color axis improvement is negligible (<0.5°)")
    if abs(shape_to_target_data.get('improvement_deg', 0)) < 0.5:
        issues.append("✗ Shape axis improvement is negligible (<0.5°)")
    
    # Check if constraints might be too tight
    if cfg.constraints.hard_norm:
        issues.append("⚠ Hard norm constraint is active (limits to pure rotations)")
    
    if cfg.constraints.type == "ball" and cfg.constraints.radius < 0.05:
        issues.append("⚠ Ball constraint might be too tight (radius < 0.05)")
    
    # Check baseline equalization
    if cfg.network.baseline_equalize:
        issues.append("⚠ Baseline equalization is ON (may reduce differentiation)")
    
    if not issues:
        issues.append("✓ No obvious issues detected")
    
    for issue in issues:
        print(f"  {issue}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    return summary, net, pca, gcol, gshp

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnostic script to understand alignment behavior."
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to YAML config (defaults to configs/paper_panel_bc.yaml).",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        help="Explicit path to YAML config.",
    )
    args = parser.parse_args()

    config_file = args.config_flag or args.config or "configs/paper_panel_bc.yaml"
    diagnose_alignment(config_file)
