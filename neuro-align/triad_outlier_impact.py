#!/usr/bin/env python3
# Compare impact of Panel‑C outlier neurons on Panel‑D improvements.
#
# Usage:
#   python triad_outlier_impact.py --config configs/paper_panel_bc.yaml --thresh 20 --outdir outputs/outlier_impact
#   # Optional: sweep across ranges (slower)
#   python triad_outlier_impact.py --config configs/paper_panel_bc.yaml --thresh 20 --outdir outputs/outlier_impact --sweep
#
# What it does:
# 1) Runs optimize_triad on your YAML config.
# 2) Recomputes 'Panel C (activity‑derived)' robust ratio = gact_color / gact_shape and finds ALL outliers > thresh.
# 3) Patches ALL outlier neurons' gains: sets g_color[i] = g_shape[i] = 1.0 (baseline, i.e., 'no modulation').
# 4) Computes the Panel‑D 'Δ angle‑to‑target' improvements for color and shape, before vs after patch.
# 5) Saves simple figures + CSV with the numbers. If --sweep, repeats across your sweep ranges.
import argparse
from pathlib import Path
import json
import math

import numpy as np
import matplotlib.pyplot as plt

# import your package
import alignlab.config as C
import alignlab.optimize as O
import alignlab.plotting as P  # only for potential future reuse; not strictly required here


def load_yaml_config(path: Path) -> C.ExperimentConfig:
    'Lightweight loader to map YAML -> dataclasses found in alignlab.config.'
    import yaml
    raw = yaml.safe_load(open(path, 'r', encoding='utf-8')) or {}

    # ---- Network ----
    n = raw.get('network', {})
    zero = n.get('zero_sum', 'row')
    if isinstance(zero, str):
        zero = C.WRNormalization(zero)
    gain_mode = n.get('gain_mode', 'both')
    if isinstance(gain_mode, str):
        gain_mode = C.GainMode(gain_mode)
    net = C.NetworkConfig(
        N=int(n.get('N', C.NetworkConfig.N)),
        K=int(n.get('K', C.NetworkConfig.K)),
        seed=int(n.get('seed', C.NetworkConfig.seed)),
        desired_radius=float(n.get('desired_radius', C.NetworkConfig.desired_radius)),
        p_high=float(n.get('p_high', C.NetworkConfig.p_high)),
        p_low=float(n.get('p_low', C.NetworkConfig.p_low)),
        zero_sum=zero,
        wr_tuned=bool(n.get('wr_tuned', C.NetworkConfig.wr_tuned)),
        weight_scale=float(n.get('weight_scale', C.NetworkConfig.weight_scale)),
        baseline_equalize=bool(n.get('baseline_equalize', C.NetworkConfig.baseline_equalize)),
        gain_mode=gain_mode,
    )

    # ---- Objective ----
    o = raw.get('objective', {})
    tgt = o.get('target_type', 'custom_pc')
    if isinstance(tgt, str):
        tgt = C.TargetType(tgt)
    aoi = o.get('axis_of_interest', 'color')
    if isinstance(aoi, str):
        aoi = C.AxisOfInterest(aoi)
    obj = C.ObjectiveConfig(
        target_type=tgt,
        theta_deg=float(o.get('theta_deg', C.ObjectiveConfig.theta_deg)),
        phi_deg=float(o.get('phi_deg', C.ObjectiveConfig.phi_deg)),
        axis_of_interest=aoi,
        shape_for_color_line=float(o.get('shape_for_color_line', C.ObjectiveConfig.shape_for_color_line)),
        color_for_shape_line=float(o.get('color_for_shape_line', C.ObjectiveConfig.color_for_shape_line)),
        custom_stim_line_start=tuple(o['custom_stim_line_start']) if o.get('custom_stim_line_start') else None,
        custom_stim_line_end=tuple(o['custom_stim_line_end']) if o.get('custom_stim_line_end') else None,
    )

    # ---- Constraints ----
    c = raw.get('constraints', {})
    ctype = c.get('type', 'ball')
    if isinstance(ctype, str):
        ctype = C.ConstraintType(ctype)
    cons = C.ConstraintConfig(
        type=ctype,
        radius=float(c.get('radius', C.ConstraintConfig.radius)),
        box_half_width=float(c.get('box_half_width', C.ConstraintConfig.box_half_width)),
        hard_norm=bool(c.get('hard_norm', C.ConstraintConfig.hard_norm)),
        positive_gains=bool(c.get('positive_gains', C.ConstraintConfig.positive_gains)),
    )

    # ---- Grid ----
    g = raw.get('grid', {})
    grid = C.GridConfig(
        shape_vals=tuple(g.get('shape_vals', C.GridConfig.shape_vals)),
        color_vals=tuple(g.get('color_vals', C.GridConfig.color_vals)),
    )

    # ---- Optimization ----
    z = raw.get('optimization', {})
    opt = C.OptimizationConfig(
        mode=C.OptimizationMode(z.get('mode', C.OptimizationConfig.mode)),
        activity_floor_frac=float(z.get('activity_floor_frac', C.OptimizationConfig.activity_floor_frac)),
        activity_floor_axes=str(z.get('activity_floor_axes', C.OptimizationConfig.activity_floor_axes)),
        activity_floor_baseline_percentile=float(z.get('activity_floor_baseline_percentile', C.OptimizationConfig.activity_floor_baseline_percentile)),
        activity_huber_delta=float(z.get('activity_huber_delta', C.OptimizationConfig.activity_huber_delta)),
        lambda_activity_modesty=float(z.get('lambda_activity_modesty', C.OptimizationConfig.lambda_activity_modesty)),
        lambda_resolvent=float(z.get('lambda_resolvent', C.OptimizationConfig.lambda_resolvent)),
    )

    # ---- Shuffle ----
    sh = raw.get('shuffle', {})
    shuf = C.ShuffleConfig(
        enabled=bool(sh.get('enabled', C.ShuffleConfig.enabled)),
        num_bins=int(sh.get('num_bins', C.ShuffleConfig.num_bins)),
        binning=str(sh.get('binning', C.ShuffleConfig.binning)),
        mode=str(sh.get('mode', C.ShuffleConfig.mode)),
        seed=sh.get('seed', C.ShuffleConfig.seed),
        repeats=int(sh.get('repeats', C.ShuffleConfig.repeats)),
    )

    # ---- Sweep ----
    sw = raw.get('sweep', {})
    sweep = C.SweepConfig(
        ranges_ball=tuple(sw.get('ranges_ball', C.SweepConfig.ranges_ball)),
        ranges_box=tuple(sw.get('ranges_box', C.SweepConfig.ranges_box)),
    )

    cfg = C.ExperimentConfig(
        network=net, objective=obj, constraints=cons, grid=grid,
        save_dir=str(raw.get('save_dir', 'outputs')),
        tag=str(raw.get('tag', 'outlier_test')),
        shuffle=shuf, optimization=opt, sweep=sweep
    )
    return cfg


def compute_panel_c_robust_ratio(net, cfg, g_color, g_shape):
    '''
    Reproduce Panel C (activity-derived) robust ratio per neuron.
    Returns dict with arrays and floors.
    '''
    shape_const = cfg.objective.shape_for_color_line
    color_const = cfg.objective.color_for_shape_line

    # baselines
    r_c0_base = net.response(shape_const, 0.0, g=None)
    r_c1_base = net.response(shape_const, 1.0, g=None)
    d_color_base = r_c1_base - r_c0_base

    r_s0_base = net.response(0.0, color_const, g=None)
    r_s1_base = net.response(1.0, color_const, g=None)
    d_shape_base = r_s1_base - r_s0_base

    # modulated
    r_c0_mod = net.response(shape_const, 0.0, g=g_color)
    r_c1_mod = net.response(shape_const, 1.0, g=g_color)
    d_color_mod = r_c1_mod - r_c0_mod

    r_s0_mod = net.response(0.0, color_const, g=g_shape)
    r_s1_mod = net.response(1.0, color_const, g=g_shape)
    d_shape_mod = r_s1_mod - r_s0_mod

    # robust floors
    perc = float(getattr(cfg.optimization, 'activity_floor_baseline_percentile', 10.0) or 10.0)
    floor_c = float(np.percentile(np.abs(d_color_base), perc))
    floor_s = float(np.percentile(np.abs(d_shape_base), perc))

    den_c = np.maximum(np.abs(d_color_base), floor_c)
    den_s = np.maximum(np.abs(d_shape_base), floor_s)

    eps = 1e-12
    gact_c = np.abs(d_color_mod) / (den_c + 0.0)
    gact_s = np.abs(d_shape_mod) / (den_s + 0.0)

    robust_ratio = gact_c / (gact_s + eps)
    sym_change = (gact_c - gact_s) / (gact_c + gact_s + eps)
    sel_sc = net.S[:, 0] - net.S[:, 1]

    return {
        'robust_ratio': robust_ratio,
        'sym_change': sym_change,
        'selectivity': sel_sc,
        'gact_color': gact_c,
        'gact_shape': gact_s,
        'floor_color': floor_c,
        'floor_shape': floor_s,
    }


def improvements_to_target(net, cfg, target, g_color, g_shape):
    '''
    Panel D metric (per triad_sweep docstring):
      Δ angle-to-target for each axis under 'own' vs 'other' gains.
    '''
    # axes under each attention
    col_axis_color = net.color_axis(cfg.objective.shape_for_color_line, g=g_color)  # own
    col_axis_shape = net.color_axis(cfg.objective.shape_for_color_line, g=g_shape)  # other

    shp_axis_shape = net.shape_axis(cfg.objective.color_for_shape_line, g=g_shape)  # own
    shp_axis_color = net.shape_axis(cfg.objective.color_for_shape_line, g=g_color)  # other

    # undirectional angles
    def undir_angle(a, b):
        th = O.angle_deg(a, b)
        return th if th <= 90.0 else 180.0 - th

    a_cc = undir_angle(col_axis_color, target)
    a_cs = undir_angle(col_axis_shape, target)
    a_ss = undir_angle(shp_axis_shape, target)
    a_sc = undir_angle(shp_axis_color, target)

    return {
        'impr_color_deg': float(a_cs - a_cc),
        'impr_shape_deg': float(a_sc - a_ss),
        'angles': {
            'color_under_color': float(a_cc),
            'color_under_shape': float(a_cs),
            'shape_under_shape': float(a_ss),
            'shape_under_color': float(a_sc),
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, default=Path('outputs/outlier_impact'))
    ap.add_argument('--thresh', type=float, default=20.0, help='Outlier threshold on robust ratio')
    ap.add_argument('--sweep', action='store_true', help='Also recompute improvements across sweep ranges (per-range outlier)')
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Run triad optimization once
    (summary, net, pca, Z0, Z_color, Z_shape,
     d_color0, d_color_opt, d_shape0, d_shape_opt,
     target, g_color, g_shape) = O.optimize_triad(cfg)

    # Panel C robust ratio and outlier detection
    C0 = compute_panel_c_robust_ratio(net, cfg, g_color, g_shape)
    rr = C0['robust_ratio']
    
    # Find ALL outliers above threshold
    outlier_indices = np.where(rr > args.thresh)[0]
    
    if len(outlier_indices) == 0:
        print(f'[WARN] No neurons with robust_ratio > {args.thresh:.3f}. Max is {np.max(rr):.3f}.')
        print(f'[INFO] Using the top neuron (index={np.argmax(rr)}) as fallback.')
        outlier_indices = np.array([np.argmax(rr)])
    else:
        print(f'[OK] Found {len(outlier_indices)} outlier(s) with robust_ratio > {args.thresh:.3f}')
        for idx in outlier_indices:
            print(f'     index={idx}, robust_ratio={rr[idx]:.3f}')

    # Patch gains: push ALL outlier neurons back to baseline (g=1.0) for both attention conditions
    g_color_fix = np.array(g_color, dtype=float)
    g_shape_fix = np.array(g_shape, dtype=float)
    for idx in outlier_indices:
        g_color_fix[idx] = 1.0
        g_shape_fix[idx] = 1.0

    # Recompute improvements (Panel D metric) before/after
    imp_orig = improvements_to_target(net, cfg, target, g_color, g_shape)
    imp_fix  = improvements_to_target(net, cfg, target, g_color_fix, g_shape_fix)

    # Save CSV with the numbers
    import csv
    with open(args.outdir / 'outlier_impact_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'color_improvement_deg', 'shape_improvement_deg', 'num_outliers_patched'])
        w.writerow(['original', f"{imp_orig['impr_color_deg']:.6f}", f"{imp_orig['impr_shape_deg']:.6f}", '0'])
        w.writerow(['patched',  f"{imp_fix['impr_color_deg']:.6f}",  f"{imp_fix['impr_shape_deg']:.6f}", f"{len(outlier_indices)}"])
    print('[saved]', args.outdir / 'outlier_impact_summary.csv')
    
    # Save list of patched neuron indices
    with open(args.outdir / 'outlier_indices.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['neuron_index', 'robust_ratio'])
        for idx in outlier_indices:
            w.writerow([idx, f"{rr[idx]:.6f}"])
    print('[saved]', args.outdir / 'outlier_indices.csv')

    # Simple text printout
    print('\n=== Outlier summary ===')
    print(f'Number of outliers patched: {len(outlier_indices)}')
    print(f'Outlier indices: {list(outlier_indices)}')
    if len(outlier_indices) > 0:
        print(f'Robust ratio range: {rr[outlier_indices].min():.3f} to {rr[outlier_indices].max():.3f}')
    print('Panel‑D Δ angle‑to‑target (deg):')
    print(f"  COLOR:  original={imp_orig['impr_color_deg']:.4f}   patched={imp_fix['impr_color_deg']:.4f}")
    print(f"  SHAPE:  original={imp_orig['impr_shape_deg']:.4f}   patched={imp_fix['impr_shape_deg']:.4f}")

    # --- Plots (one chart per figure; no explicit colors) ---

    # 1) Robust ratio scatter with outlier annotation
    plt.figure()
    x = C0['selectivity']
    y = rr
    plt.axhline(1.0, linestyle='--', alpha=0.6)
    plt.axhline(args.thresh, linestyle=':', alpha=0.6, label=f'threshold={args.thresh}')
    plt.scatter(x, y, s=20, alpha=0.85)
    # annotate all outliers
    for idx in outlier_indices:
        plt.annotate(f'{idx}', (x[idx], y[idx]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Selectivity (shape - color)')
    plt.ylabel('Robust relative change (color/shape)')
    plt.title(f'Panel C: robust ratio with {len(outlier_indices)} outlier(s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir / 'panel_c_robust_ratio_outlier.png', dpi=200)
    plt.close()

    # Also show the robust ratio after patching (sanity check)
    C1 = compute_panel_c_robust_ratio(net, cfg, g_color_fix, g_shape_fix)
    plt.figure()
    plt.axhline(1.0, linestyle='--', alpha=0.6)
    plt.scatter(C1['selectivity'], C1['robust_ratio'], s=20, alpha=0.85)
    plt.xlabel('Selectivity (shape - color)')
    plt.ylabel('Robust relative change (color/shape)')
    plt.title('Panel C: robust ratio after patch')
    plt.tight_layout()
    plt.savefig(args.outdir / 'panel_c_robust_ratio_after_patch.png', dpi=200)
    plt.close()

    # 2) Bar chart: COLOR improvement before/after
    plt.figure()
    plt.bar([0, 1], [imp_orig['impr_color_deg'], imp_fix['impr_color_deg']])
    plt.xticks([0, 1], ['original', 'patched'])
    plt.ylabel('Δ angle‑to‑target (deg)')
    plt.title('Panel D metric — COLOR axis')
    plt.tight_layout()
    plt.savefig(args.outdir / 'panel_d_metric_color_bar.png', dpi=200)
    plt.close()

    # 3) Bar chart: SHAPE improvement before/after
    plt.figure()
    plt.bar([0, 1], [imp_orig['impr_shape_deg'], imp_fix['impr_shape_deg']])
    plt.xticks([0, 1], ['original', 'patched'])
    plt.ylabel('Δ angle‑to‑target (deg)')
    plt.title('Panel D metric — SHAPE axis')
    plt.tight_layout()
    plt.savefig(args.outdir / 'panel_d_metric_shape_bar.png', dpi=200)
    plt.close()

    # --- Optional: sweep across ranges with per-range outlier re‑patching ---
    if args.sweep:
        if cfg.constraints.type == C.ConstraintType.BALL:
            ranges = list(cfg.sweep.ranges_ball)
        else:
            ranges = list(cfg.sweep.ranges_box)

        rows_orig = []
        rows_patched = []

        for r in ranges:
            local = C.ExperimentConfig(
                network=cfg.network, objective=cfg.objective, constraints=cfg.constraints,
                grid=cfg.grid, save_dir=cfg.save_dir, tag=f"{cfg.tag}_r{r}",
                shuffle=cfg.shuffle, optimization=cfg.optimization, sweep=cfg.sweep
            )
            # mutate constraint range
            if local.constraints.type == C.ConstraintType.BALL:
                local.constraints.radius = r
            else:
                local.constraints.box_half_width = r

            (summary_r, net_r, pca_r, Z0_r, Zc_r, Zs_r,
             dc0_r, dcopt_r, ds0_r, dsopt_r,
             target_r, gcol_r, gshp_r) = O.optimize_triad(local)

            # Panel D metrics for original at this r
            imp_r = improvements_to_target(net_r, local, target_r, gcol_r, gshp_r)
            rows_orig.append({
                'range': r,
                'impr_color_deg': imp_r['impr_color_deg'],
                'impr_shape_deg': imp_r['impr_shape_deg'],
            })

            # detect ALL outliers at this r
            C_r = compute_panel_c_robust_ratio(net_r, local, gcol_r, gshp_r)
            rr_r = C_r['robust_ratio']
            outlier_indices_r = np.where(rr_r > args.thresh)[0]
            
            if len(outlier_indices_r) == 0:
                # fallback to top neuron if no outliers found
                outlier_indices_r = np.array([np.argmax(rr_r)])

            # patch ALL outliers and recompute
            gcol_fix_r = np.array(gcol_r, dtype=float)
            gshp_fix_r = np.array(gshp_r, dtype=float)
            for idx_r in outlier_indices_r:
                gcol_fix_r[idx_r] = 1.0
                gshp_fix_r[idx_r] = 1.0
            imp_fix_r = improvements_to_target(net_r, local, target_r, gcol_fix_r, gshp_fix_r)

            rows_patched.append({
                'range': r,
                'impr_color_deg': imp_fix_r['impr_color_deg'],
                'impr_shape_deg': imp_fix_r['impr_shape_deg'],
                'num_outliers': len(outlier_indices_r),
            })

        # Save CSVs
        import csv
        with open(args.outdir / 'sweep_panel_d_original.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['range','impr_color_deg','impr_shape_deg'])
            for r in rows_orig: w.writerow([r['range'], r['impr_color_deg'], r['impr_shape_deg']])
        with open(args.outdir / 'sweep_panel_d_patched.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['range','impr_color_deg','impr_shape_deg','num_outliers'])
            for r in rows_patched: w.writerow([r['range'], r['impr_color_deg'], r['impr_shape_deg'], r['num_outliers']])
        print('[saved]', args.outdir / 'sweep_panel_d_original.csv')
        print('[saved]', args.outdir / 'sweep_panel_d_patched.csv')

        # Plot: COLOR improvement vs range, original vs patched
        xs = [r['range'] for r in rows_orig]
        yc = [r['impr_color_deg'] for r in rows_orig]
        yc_fix = [r['impr_color_deg'] for r in rows_patched]
        x_plot = [math.sqrt(x + 1.0) for x in xs]  # same transform used in panel_d_full

        plt.figure()
        plt.plot(x_plot, yc, marker='o')
        plt.plot(x_plot, yc_fix, marker='s')
        plt.axhline(0.0, linestyle='--')
        plt.xlabel('Constraint range (sqrt-scaled)')
        plt.ylabel('Δ angle‑to‑target (deg)')
        plt.title('Panel D sweep — COLOR (orig vs patched)')
        plt.tight_layout()
        plt.savefig(args.outdir / 'sweep_panel_d_color.png', dpi=200)
        plt.close()

        # Plot: SHAPE improvement vs range, original vs patched
        ys = [r['impr_shape_deg'] for r in rows_orig]
        ys_fix = [r['impr_shape_deg'] for r in rows_patched]

        plt.figure()
        plt.plot(x_plot, ys, marker='o')
        plt.plot(x_plot, ys_fix, marker='s')
        plt.axhline(0.0, linestyle='--')
        plt.xlabel('Constraint range (sqrt-scaled)')
        plt.ylabel('Δ angle‑to‑target (deg)')
        plt.title('Panel D sweep — SHAPE (orig vs patched)')
        plt.tight_layout()
        plt.savefig(args.outdir / 'sweep_panel_d_shape.png', dpi=200)
        plt.close()


if __name__ == '__main__':
    main()
