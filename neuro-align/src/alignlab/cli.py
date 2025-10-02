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

def load_config(path: Path) -> ExperimentConfig:
    import yaml
    from .config import (
        ExperimentConfig, NetworkConfig, ObjectiveConfig, ConstraintConfig, GridConfig,
        TargetType, AxisOfInterest, ConstraintType, WRNormalization
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
        raw = yaml.safe_load(f) or {}

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

    tag = raw.get("tag", "experiment")
    save_dir = raw.get("save_dir", "outputs")
    return ExperimentConfig(network=net, objective=obj, constraints=con, grid=grd, tag=tag, save_dir=save_dir)

def cmd_run(args):
    cfg = load_config(Path(args.config))
    out, net, pca, Z0, Zopt, d0, dopt, target, gopt = optimize_once(cfg)
    outdir = Path(cfg.save_dir) / cfg.tag
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(out, outdir / f"{cfg.tag}_summary.json")

    if args.style == "original":
        # Your two-panel plot + gains vs selectivity
        plot_original_two_panel(
            Z_unmod=Z0, Z_mod=Zopt, pca_components=pca.components_,
            target_vec_neuron=target, d_unmod_neuron=d0, d_mod_neuron=dopt,
            shape_vals=cfg.grid.shape_vals, color_vals=cfg.grid.color_vals,
            outdir=outdir, tag=cfg.tag
        )
        plot_gains_vs_selectivity(net.S, gopt, outdir, cfg.tag)
    else:
        # Keep the single-axes combined plot
        plot_embedding_3d(Z0, Zopt, pca.components_, target, d0, dopt, outdir, cfg.tag)

def cmd_sweep(args):
    cfg = load_config(Path(args.config))
    if args.use_box:
        cfg.constraints.type = ConstraintType.BOX
    if args.use_ball:
        cfg.constraints.type = ConstraintType.BALL
    ranges = [float(x) for x in args.ranges]
    res = sweep_range_vs_degree(cfg, ranges)
    outdir = Path(cfg.save_dir) / (cfg.tag + "_sweep")
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(res, outdir / f"{cfg.tag}_sweep.json")
    plot_range_vs_degree(res["rows"], outdir, cfg.tag)

def main():
    p = argparse.ArgumentParser(prog="alignlab", description="PC alignment experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run a single optimization and plot results")
    pr.add_argument("--config", required=True)
    pr.add_argument("--style", choices=["original", "combined"], default="original",
                    help="Plot style: 'original' (two panels, colorbars, lines) or 'combined' (single axes).")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("sweep", help="Sweep constraint range and plot Δangle")
    ps.add_argument("--config", required=True)
    ps.add_argument("--ranges", nargs="+", required=True, help="List of range values")
    g = ps.add_mutually_exclusive_group()
    g.add_argument("--use-box", action="store_true", dest="use_box")
    g.add_argument("--use-ball", action="store_true", dest="use_ball")
    ps.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
