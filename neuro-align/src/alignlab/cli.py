from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from .config import ExperimentConfig, NetworkConfig, ObjectiveConfig, ConstraintConfig, GridConfig, ConstraintType
from .optimize import optimize_once, sweep_range_vs_degree, save_json
from .plotting import (
    plot_original_two_panel, plot_embedding_3d,
    plot_range_vs_degree, plot_gains_vs_selectivity
)
from .optimize import optimize_once, sweep_range_vs_degree, save_json, optimize_triad
from .plotting import (
    plot_original_two_panel, plot_embedding_3d,
    plot_range_vs_degree, plot_gains_vs_selectivity,  # existing
    plot_triad_three_panel
)

from .optimize import optimize_once, sweep_range_vs_degree, save_json, optimize_triad, triad_sweep
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
        TargetType, AxisOfInterest, ConstraintType, WRNormalization, ShuffleConfig, SweepConfig
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

    tag = raw.get("tag", "experiment")
    save_dir = raw.get("save_dir", "outputs")
    return ExperimentConfig(network=net, objective=obj, constraints=con, grid=grd,
                            tag=tag, save_dir=save_dir, shuffle=shuf, sweep=swe) 

def cmd_run(args):
    cfg = load_config(Path(args.config))
    out, net, pca, Z0, Zopt, d0, dopt, target, gopt = optimize_once(cfg)
    outdir = Path(cfg.save_dir) / cfg.tag
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(out, outdir / f"{cfg.tag}_summary.json")

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
    (summary, net, pca, Z0, Zc, Zs,
     dcol0, dcol1, dshp0, dshp1, target, gcol, gshp) = optimize_triad(cfg)

    outdir = Path(cfg.save_dir) / cfg.tag
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(summary, outdir / f"{cfg.tag}_triad_summary.json")

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
    
    
def cmd_triad_sweep(args):
    cfg = load_config(Path(args.config))
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
    pt.set_defaults(func=cmd_triad)

    

    pts = sub.add_parser("triad-sweep", help="Sweep range and plot cross-attend angles (with shuffled null)")
    pts.add_argument("--config", required=True)
    pts.add_argument("--ranges", nargs="+", help="List of range values")
    pts.add_argument("--no-shuffle", action="store_true", help="Disable shuffle test")
    pts.add_argument("--shuffle-bins", type=int, help="Number of selectivity bins (default: 10)")
    pts.add_argument("--shuffle-mode", choices=["independent","paired"], help="Shuffle independently or as pairs")
    pts.add_argument("--shuffle-seed", type=int, help="Seed for shuffle RNG")
    pts.add_argument("--paper-panels", action="store_true",help="Also produce panel_d (cross-attend vs range with shuffled mean ± CI) with data")
    pts.set_defaults(func=cmd_triad_sweep)




    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
