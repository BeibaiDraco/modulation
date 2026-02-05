from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from .config import (
    ExperimentConfig, NetworkConfig, ObjectiveConfig, ConstraintConfig,
    GridConfig, ConstraintType, OptimizationMode
)
from .optimize import optimize_once, sweep_range_vs_degree, save_weight_matrices, save_json, optimize_triad, triad_sweep, baseline_axis_pc_angles
from .plotting import (
    plot_original_two_panel, plot_embedding_3d,
    plot_range_vs_degree, plot_gains_vs_selectivity,
    plot_triad_three_panel,
    plot_triad_cross_sweep,
    plot_panel_b_color_state, plot_panel_b_shape_state,
    plot_panel_c_activity, plot_panel_c_gopt,
    plot_panel_d_full, plot_panel_d_shape_only,
)




def load_config(path: Path) -> ExperimentConfig:
    import yaml
    from .config import (
        ExperimentConfig, NetworkConfig, ObjectiveConfig, ConstraintConfig, GridConfig,
        TargetType, AxisOfInterest, ConstraintType, WRNormalization,
        ShuffleConfig, SweepConfig, OptimizationConfig, OptimizationMode
    )

    def _enum(enum_cls, v, *, tolower=True):
        if isinstance(v, enum_cls):
            return v
        if isinstance(v, str):
            s = v.lower() if tolower else v
            # allow synonyms
            aliases = {
                WRNormalization: {"row_and_col": "row_and_col", "row+col": "row_and_col"},
            }
            if enum_cls in aliases and s in aliases[enum_cls]:
                s = aliases[enum_cls][s]
            return enum_cls(s)
        return v  # fall back (lets dataclass defaults handle None)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) | {}

    # --- Network ---
    nraw = dict(raw.get("network", {}))
    if "zero_sum" in nraw:
        nraw["zero_sum"] = _enum(WRNormalization, nraw["zero_sum"])
    net = NetworkConfig(**nraw)

    # --- Objective ---
    oraw = dict(raw.get("objective", {}))
    if "target_type" in oraw:
        oraw["target_type"] = _enum(TargetType, oraw["target_type"])
    if "axis_of_interest" in oraw:
        oraw["axis_of_interest"] = _enum(AxisOfInterest, oraw["axis_of_interest"])
    obj = ObjectiveConfig(**oraw)

    # --- Constraints ---
    craw = dict(raw.get("constraints", {}))
    if "type" in craw:
        craw["type"] = _enum(ConstraintType, craw["type"])
    con = ConstraintConfig(**craw)

    
    # --- Grid ---
    grd = GridConfig(**raw.get("grid", {}))
    

    # --- Shuffle ---
    shraw = dict(raw.get("shuffle", {}))
    shuf = ShuffleConfig(**shraw)

    # --- Sweep ---
    sweraw = dict(raw.get("sweep", {}))
    swe = SweepConfig(**sweraw)

    # --- Optimization ---
    orz = dict(raw.get("optimization", {}))
    if "mode" in orz:
        orz["mode"] = _enum(OptimizationMode, orz["mode"])
    optim = OptimizationConfig(**orz)

    tag = raw.get("tag", "experiment")
    save_dir = raw.get("save_dir", "outputs")
    return ExperimentConfig(
        network=net,
        objective=obj,
        constraints=con,
        grid=grd,
        tag=tag,
        save_dir=save_dir,
        shuffle=shuf,
        optimization=optim,
        sweep=swe,
    ) 

def _angdiff(a: float, b: float) -> float:
    """Shortest angular difference in degrees on [0, 360)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _preview_axes_and_confirm(cfg: ExperimentConfig, *, auto_yes: bool = False) -> bool:
    """
    Print unmodulated color/shape axes in PC space alongside current custom_pc target.
    Returns True if the user confirms (or auto_yes), False otherwise.
    """
    info = baseline_axis_pc_angles(cfg)
    color = info.get("color", {})
    shape = info.get("shape", {})
    target_theta = float(cfg.objective.theta_deg)
    target_phi = float(cfg.objective.phi_deg)

    print("\n=== Triad Sweep: Target Preview ===")
    print(
        f"COLOR axis (unmodulated) in PC space:  θ={color.get('theta_deg', float('nan')):.2f}°, "
        f"φ={color.get('phi_deg', float('nan')):.2f}°"
    )
    print(
        f"SHAPE axis (unmodulated) in PC space:  θ={shape.get('theta_deg', float('nan')):.2f}°, "
        f"φ={shape.get('phi_deg', float('nan')):.2f}°"
    )
    print(
        f"Current YAML target (custom_pc):       θ={target_theta:.2f}°, φ={target_phi:.2f}°"
    )
    print(
        f"Δθ to COLOR: {_angdiff(target_theta, color.get('theta_deg', target_theta)):.2f}°,   "
        f"Δθ to SHAPE: {_angdiff(target_theta, shape.get('theta_deg', target_theta)):.2f}°\n"
    )

    if auto_yes:
        print("--yes provided; proceeding without prompt.\n")
        return True

    ans = input("Proceed with this target? [y/N]: ").strip().lower()
    if ans in ("y", "yes"):
        print("OK — running sweep...\n")
        return True

    print("Aborted by user. Edit your YAML target and re-run.\n")
    return False


def _apply_optimization_overrides(cfg, args) -> None:
    if getattr(args, "optim_mode", None):
        cfg.optimization.mode = OptimizationMode(args.optim_mode)
    if getattr(args, "eff_floor_frac", None) is not None:
        cfg.optimization.activity_floor_frac = float(args.eff_floor_frac)
    if getattr(args, "eff_floor_axes", None):
        cfg.optimization.activity_floor_axes = args.eff_floor_axes
    if getattr(args, "eff_floor_baseline_pct", None) is not None:
        cfg.optimization.activity_floor_baseline_percentile = float(args.eff_floor_baseline_pct)

def cmd_run(args):
    cfg = load_config(Path(args.config))
    _apply_optimization_overrides(cfg, args)
    out, net, pca, Z0, Zopt, d0, dopt, target, gopt = optimize_once(cfg)
    outdir = Path(cfg.save_dir) / cfg.tag
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(out, outdir / f"{cfg.tag}_summary.json")

    if getattr(args, "print_eff_diagnostics", False):
        diag = out.get("optimization", {}).get("activity_floor", {}).get("diagnostics")
        if diag:
            print(
                "[effective_mod] axis margin stats: "
                f"min={diag['min_margin']:.3f}, "
                f"p5={diag['p5_margin']:.3f}, "
                f"median={diag['median_margin']:.3f}"
            )
        else:
            print("[effective_mod] diagnostics unavailable (floor inactive or disabled).")

    if args.style == "original":
        plot_original_two_panel(
            Z_unmod=Z0, Z_mod=Zopt, pca_components=pca.components_,
            target_vec_neuron=target, d_unmod_neuron=d0, d_mod_neuron=dopt,
            shape_vals=cfg.grid.shape_vals, color_vals=cfg.grid.color_vals,
            outdir=outdir, tag=cfg.tag,
            zlim=tuple(args.zlim) if args.zlim else None,
            elev=args.elev, azim=args.azim, show=args.show
        )
        plot_gains_vs_selectivity(net.S, gopt, outdir, cfg.tag)
    else:
        plot_embedding_3d(Z0, Zopt, pca.components_, target, d0, dopt, outdir, cfg.tag)

def _resolve_ranges(cfg, arg_ranges, preset=None):
    if arg_ranges:  # CLI provided
        return [float(x) for x in arg_ranges]
    # preset handled in Option B below; for now just YAML defaults:
    if cfg.constraints.type == ConstraintType.BALL:
        return list(cfg.sweep.ranges_ball)
    else:
        return list(cfg.sweep.ranges_box)


def cmd_sweep(args):
    cfg = load_config(Path(args.config))
    if args.use_box:
        cfg.constraints.type = ConstraintType.BOX
    if args.use_ball:
        cfg.constraints.type = ConstraintType.BALL
    ranges = _resolve_ranges(cfg, getattr(args, "ranges", None), getattr(args,"preset",None))
    res = sweep_range_vs_degree(cfg, ranges)
    outdir = Path(cfg.save_dir) / (cfg.tag + "_sweep")
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(res, outdir / f"{cfg.tag}_sweep.json")
    plot_range_vs_degree(res["rows"], outdir, cfg.tag)

def cmd_triad(args):
    cfg = load_config(Path(args.config))
    _apply_optimization_overrides(cfg, args)
    (summary, net, pca, Z0, Zc, Zs,
     dcol0, dcol1, dshp0, dshp1, target, gcol, gshp) = optimize_triad(cfg)

    outdir = Path(cfg.save_dir) / cfg.tag
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(summary, outdir / f"{cfg.tag}_triad_summary.json")
    save_weight_matrices(net, target, outdir, cfg.tag)
    
    # 3-panel plot
    plot_triad_three_panel(
        Z_unmod=Z0, Z_color=Zc, Z_shape=Zs,
        pca_components=pca.components_, target_vec_neuron=target,
        d_color_unmod=dcol0, d_color_opt=dcol1,
        d_shape_unmod=dshp0, d_shape_opt=dshp1,
        shape_vals=cfg.grid.shape_vals, color_vals=cfg.grid.color_vals,
        outdir=outdir, tag=cfg.tag,
        zlim=tuple(args.zlim) if args.zlim else None,
        elev=args.elev, azim=args.azim, show=args.show
    )

    # --- Paper panels B & C ---
    if getattr(args, "paper_panels", False):
        # cross-axes under both attention states
        d_col_c = net.color_axis(cfg.objective.shape_for_color_line, g=gcol)
        d_col_s = net.color_axis(cfg.objective.shape_for_color_line, g=gshp)
        d_shp_c = net.shape_axis(cfg.objective.color_for_shape_line, g=gcol)
        d_shp_s = net.shape_axis(cfg.objective.color_for_shape_line, g=gshp)

        zlim = tuple(args.zlim) if args.zlim else None
        plot_panel_b_color_state(
            Z_color=Zc, Z_shape=Zs, pca_components=pca.components_,
            target_vec_neuron=target,
            color_axis_color=d_col_c, color_axis_shape=d_col_s,
            shape_axis_color=d_shp_c, shape_axis_shape=d_shp_s,
            shape_vals=cfg.grid.shape_vals, color_vals=cfg.grid.color_vals,
            outdir=outdir, tag=cfg.tag, zlim=zlim,
            elev=args.elev, azim=args.azim, show=args.show, save_data=True
        )
        plot_panel_b_shape_state(
            Z_color=Zc, Z_shape=Zs, pca_components=pca.components_,
            target_vec_neuron=target,
            color_axis_color=d_col_c, color_axis_shape=d_col_s,
            shape_axis_color=d_shp_c, shape_axis_shape=d_shp_s,
            shape_vals=cfg.grid.shape_vals, color_vals=cfg.grid.color_vals,
            outdir=outdir, tag=cfg.tag, zlim=zlim,
            elev=args.elev, azim=args.azim, show=args.show
        )
        plot_panel_c_activity(net, cfg, gcol, gshp, outdir, cfg.tag, show=False)
        plot_panel_c_gopt(net.S, gcol, gshp, outdir, cfg.tag, show=False)
        
        # Print Panel D metric: improvement to target
        to_target = summary.get("to_target", {})
        color_ax = to_target.get("color_axis", {})
        shape_ax = to_target.get("shape_axis", {})
        print("\n=== Panel D Metric: Δ angle-to-target (improvement) ===")
        print("(Positive = 'own' attention helps axis align better to target)")
        print(f"\nColor axis:")
        print(f"  Angle to target under color gains (own):   {color_ax.get('under_color_deg', float('nan')):.4f}°")
        print(f"  Angle to target under shape gains (other): {color_ax.get('under_shape_deg', float('nan')):.4f}°")
        print(f"  → Improvement (other - own):               {color_ax.get('improvement_deg', float('nan')):.4f}°")
        print(f"\nShape axis:")
        print(f"  Angle to target under shape gains (own):   {shape_ax.get('under_shape_deg', float('nan')):.4f}°")
        print(f"  Angle to target under color gains (other): {shape_ax.get('under_color_deg', float('nan')):.4f}°")
        print(f"  → Improvement (other - own):               {shape_ax.get('improvement_deg', float('nan')):.4f}°")
        
        # Compute and print optimized axes in PC space (theta, phi)
        def _pc_angles(vec):
            """Convert PC vector to spherical coordinates (theta, phi in degrees)."""
            import numpy as np
            x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
            theta = float(np.degrees(np.arctan2(y, x)))  # [-180, 180]
            rho_xy = float(np.hypot(x, y))
            phi = float(np.degrees(np.arctan2(z, rho_xy)))  # [-90, 90]
            theta = (theta + 360.0) % 360.0  # [0, 360)
            return theta, phi
        
        comps = pca.components_[:3, :]  # 3 x N
        col_c_pc = comps @ d_col_c
        col_s_pc = comps @ d_col_s
        shp_c_pc = comps @ d_shp_c
        shp_s_pc = comps @ d_shp_s
        
        theta_col_c, phi_col_c = _pc_angles(col_c_pc)
        theta_col_s, phi_col_s = _pc_angles(col_s_pc)
        theta_shp_c, phi_shp_c = _pc_angles(shp_c_pc)
        theta_shp_s, phi_shp_s = _pc_angles(shp_s_pc)
        
        print("\n=== Optimized Axes in PC Space (θ, φ) ===")
        print(f"\nColor axis under color gains (own):")
        print(f"  θ = {theta_col_c:.2f}°,  φ = {phi_col_c:.2f}°")
        print(f"\nColor axis under shape gains (other):")
        print(f"  θ = {theta_col_s:.2f}°,  φ = {phi_col_s:.2f}°")
        print(f"\nShape axis under shape gains (own):")
        print(f"  θ = {theta_shp_s:.2f}°,  φ = {phi_shp_s:.2f}°")
        print(f"\nShape axis under color gains (other):")
        print(f"  θ = {theta_shp_c:.2f}°,  φ = {phi_shp_c:.2f}°")
        
        # Also print target for reference
        target_pc = comps @ target
        theta_tgt, phi_tgt = _pc_angles(target_pc)
        print(f"\nTarget (for reference):")
        print(f"  θ = {theta_tgt:.2f}°,  φ = {phi_tgt:.2f}°")
        print()
    if getattr(args, "print_eff_diagnostics", False):
        axes_outputs = [
            ("color", summary.get("color_alignment")),
            ("shape", summary.get("shape_alignment")),
        ]
        for name, out_axis in axes_outputs:
            if not out_axis:
                continue
            diag = out_axis.get("optimization", {}).get("activity_floor", {}).get("diagnostics")
            if diag:
                print(
                    f"[effective_mod] {name} axis margin stats: "
                    f"min={diag['min_margin']:.3f}, "
                    f"p5={diag['p5_margin']:.3f}, "
                    f"median={diag['median_margin']:.3f}"
                )
            else:
                print(f"[effective_mod] {name} axis diagnostics unavailable (floor inactive or disabled).")

    
def cmd_triad_sweep(args):
    cfg = load_config(Path(args.config))
    _apply_optimization_overrides(cfg, args)
    if not _preview_axes_and_confirm(cfg, auto_yes=getattr(args, "yes", False)):
        return

    ranges = _resolve_ranges(cfg, getattr(args, "ranges", None), getattr(args,"preset",None))
    
    # optional CLI overrides
    if args.no_shuffle:
        cfg.shuffle.enabled = False
    if args.shuffle_bins is not None:
        cfg.shuffle.num_bins = int(args.shuffle_bins)
    if args.shuffle_mode is not None:
        cfg.shuffle.mode = args.shuffle_mode
    if args.shuffle_seed is not None:
        cfg.shuffle.seed = int(args.shuffle_seed)

    res = triad_sweep(cfg, ranges)
    if getattr(args, "print_eff_diagnostics", False):
        for row in res.get("rows", []):
            rng = row.get("range")
            color_diag = row.get("color_eff_diag")
            shape_diag = row.get("shape_eff_diag")
            parts = []
            if color_diag:
                parts.append(
                    "color"
                    + f" min={color_diag['min_margin']:.3f}"
                    + f" p5={color_diag['p5_margin']:.3f}"
                    + f" median={color_diag['median_margin']:.3f}"
                )
            else:
                parts.append("color=NA")
            if shape_diag:
                parts.append(
                    "shape"
                    + f" min={shape_diag['min_margin']:.3f}"
                    + f" p5={shape_diag['p5_margin']:.3f}"
                    + f" median={shape_diag['median_margin']:.3f}"
                )
            else:
                parts.append("shape=NA")
            print(f"[effective_mod] range={rng}: " + ", ".join(parts))

    outdir = Path(cfg.save_dir) / (cfg.tag + "_triad_sweep")
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(res, outdir / f"{cfg.tag}_triad_sweep.json")
    plot_triad_cross_sweep(res["rows"], outdir, cfg.tag)
    if getattr(args, "paper_panels", False):
        plot_panel_d_full(res["rows"], outdir, cfg.tag)
        # Color-axis only variant (panel_d.*)
        plot_panel_d_shape_only(res["rows"], outdir, cfg.tag)
   



def main():
    p = argparse.ArgumentParser(prog="alignlab", description="PC alignment experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run a single optimization and plot results")
    pr.add_argument("--config", required=True)
    pr.add_argument("--style", choices=["original", "combined"], default="original",
                    help="Plot style: 'original' (two panels, colorbars, lines) or 'combined' (single axes).")
    # NEW interactive & zlim knobs:
    pr.add_argument("--zlim", nargs=2, type=float, metavar=("ZMIN","ZMAX"),
                    help="Fix z-axis limits, e.g., --zlim -1 1")
    pr.add_argument("--elev", type=float, default=None, help="Initial elevation angle (deg)")
    pr.add_argument("--azim", type=float, default=None, help="Initial azimuth angle (deg)")
    pr.add_argument("--show", action="store_true", help="Open interactive window so you can drag to rotate")
    pr.add_argument("--optim-mode", choices=["standard", "effective_mod"], help="Override optimization mode.")
    pr.add_argument("--eff-floor-frac", type=float, help="Effective modulation floor fraction (alpha).")
    pr.add_argument("--eff-floor-axes", choices=["shape", "color", "both"], help="Axes receiving the effective-mod floor.")
    pr.add_argument("--eff-floor-baseline-pct", type=float,
                    help="Percentile threshold on baseline |d(1)| used to select neurons for the floor.")
    pr.add_argument("--print-eff-diagnostics", action="store_true",
                    help="Print effective-modulation diagnostics (min/p5/median margins) if available.")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("sweep", help="Sweep constraint range and plot Δangle")
    ps.add_argument("--config", required=True)
    ps.add_argument("--ranges", nargs="+", help="List of range values")
    g = ps.add_mutually_exclusive_group()
    g.add_argument("--use-box", action="store_true", dest="use_box")
    g.add_argument("--use-ball", action="store_true", dest="use_ball")
    ps.set_defaults(func=cmd_sweep)
    
    pt = sub.add_parser("triad", help="Three-panel analysis: default, color-aligned, shape-aligned")
    pt.add_argument("--config", required=True)
    pt.add_argument("--zlim", nargs=2, type=float, metavar=("ZMIN","ZMAX"),
                    help="Fix z-axis limits, e.g., --zlim -1 1")
    pt.add_argument("--elev", type=float, default=None, help="Initial elevation angle (deg)")
    pt.add_argument("--azim", type=float, default=None, help="Initial azimuth angle (deg)")
    pt.add_argument("--show", action="store_true", help="Open interactive windows so you can drag to rotate")
    pt.add_argument("--gcompare", choices=["ratio","diff"], default="ratio",
                    help="Compare gains as ratio (color/shape) or difference (color-shape)")
    pt.add_argument("--logratio", action="store_true",
                    help="Use log2 for the ratio plot (only sensible if gains are positive)")
    pt.add_argument("--paper-panels", action="store_true",help="Also produce panel_b (3D bivariate dots + axes) and panel_c (gain vs selectivity) with data")
    pt.add_argument("--optim-mode", choices=["standard", "effective_mod"], help="Override optimization mode.")
    pt.add_argument("--eff-floor-frac", type=float, help="Effective modulation floor fraction (alpha).")
    pt.add_argument("--eff-floor-axes", choices=["shape", "color", "both"], help="Axes receiving the effective-mod floor.")
    pt.add_argument("--eff-floor-baseline-pct", type=float,
                    help="Percentile threshold on baseline |d(1)| used to select neurons for the floor.")
    pt.add_argument("--print-eff-diagnostics", action="store_true",
                    help="Print effective-modulation diagnostics (min/p5/median margins) if available.")
    pt.set_defaults(func=cmd_triad)

    

    pts = sub.add_parser("triad-sweep", help="Sweep range and plot cross-attend angles (with shuffled null)")
    pts.add_argument("--config", required=True)
    pts.add_argument("--ranges", nargs="+", help="List of range values")
    pts.add_argument("--no-shuffle", action="store_true", help="Disable shuffle test")
    pts.add_argument("--shuffle-bins", type=int, help="Number of selectivity bins (default: 10)")
    pts.add_argument("--shuffle-mode", choices=["independent","paired"], help="Shuffle independently or as pairs")
    pts.add_argument("--shuffle-seed", type=int, help="Seed for shuffle RNG")
    pts.add_argument("--paper-panels", action="store_true",help="Also produce panel_d (cross-attend vs range with shuffled mean ± CI) with data")
    pts.add_argument("--optim-mode", choices=["standard", "effective_mod"], help="Override optimization mode.")
    pts.add_argument("--eff-floor-frac", type=float, help="Effective modulation floor fraction (alpha).")
    pts.add_argument("--eff-floor-axes", choices=["shape", "color", "both"], help="Axes receiving the effective-mod floor.")
    pts.add_argument("--eff-floor-baseline-pct", type=float,
                     help="Percentile threshold on baseline |d(1)| used to select neurons for the floor.")
    pts.add_argument("--print-eff-diagnostics", action="store_true",
                     help="Print effective-modulation diagnostics (min/p5/median margins) if available.")
    pts.add_argument("-y", "--yes", action="store_true",
                     help="Skip interactive target preview and proceed immediately.")
    pts.set_defaults(func=cmd_triad_sweep)




    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
