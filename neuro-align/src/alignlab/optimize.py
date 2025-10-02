from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from .config import ExperimentConfig, ConstraintType
from .network import LinearRNN, angle_deg
from .objectives import target_in_neuron_space, axis_of_interest_vec, angle_to_target
from .constraints import build_bounds_and_constraints
from .config import ExperimentConfig, ConstraintType, AxisOfInterest
from .config import ExperimentConfig, ConstraintType, AxisOfInterest
from .network import LinearRNN, angle_deg
from .objectives import target_in_neuron_space, axis_of_interest_vec, angle_to_target
from .constraints import build_bounds_and_constraints

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _enum_val(x):
    return getattr(x, "value", x)

def _cos2(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return float(np.clip(c, -1.0, 1.0) ** 2)

def _axis_fn_for(net, cfg, choice: AxisOfInterest):
    if choice == AxisOfInterest.COLOR:
        return lambda g: net.color_axis(cfg.objective.shape_for_color_line, g)
    if choice == AxisOfInterest.SHAPE:
        return lambda g: net.shape_axis(cfg.objective.color_for_shape_line, g)
    raise ValueError(f"AxisOfInterest {choice} not supported for triad mode")

def optimize_once(cfg: ExperimentConfig) -> Dict:
    # ---- model & PCA (fit on UNMOD only) ----
    net = LinearRNN(cfg.network)
    X0 = net.grid_responses(cfg.grid, g=None)
    pca, Z0 = net.pca3(X0)
    target = target_in_neuron_space(pca, cfg.objective)

    # ---- axis at g=1 (unmod) ----
    g0 = np.ones(cfg.network.N)
    d0 = axis_of_interest_vec(net, cfg.objective, g=g0)

    # ---- objective (maximize cos^2 -> minimize negative) ----
    def obj(g):
        d = axis_of_interest_vec(net, cfg.objective, g=g)
        return -_cos2(d, target)

    bounds, constraints = build_bounds_and_constraints(
        axis_fn=lambda g: axis_of_interest_vec(net, cfg.objective, g=g),
        g0=g0, cfg=cfg.constraints,
        positive_gains=cfg.constraints.positive_gains
    )

    # ---- optimize ----
    res = minimize(
        obj, x0=g0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False}
    )

    g_opt = res.x
    d_opt = axis_of_interest_vec(net, cfg.objective, g=g_opt)

    # Project MOD into the SAME PCA basis (do NOT refit PCA)
    X_opt = net.grid_responses(cfg.grid, g=g_opt)
    Z_opt = pca.transform(X_opt)

    # ---- metrics ----
    angle_pre  = angle_to_target(d0,   target)
    angle_post = angle_to_target(d_opt, target)
    improvement = angle_pre - angle_post

    norm_pre  = float(np.linalg.norm(d0))
    norm_post = float(np.linalg.norm(d_opt))
    eq_residual = float(norm_post**2 - norm_pre**2) if cfg.constraints.hard_norm else None

    cos2_pre  = _cos2(d0,   target)
    cos2_post = _cos2(d_opt, target)

    # gains stats
    gm1 = g_opt - 1.0
    g_stats = {
        "min": float(np.min(g_opt)),
        "max": float(np.max(g_opt)),
        "mean": float(np.mean(g_opt)),
        "std": float(np.std(g_opt)),
        "l2_norm_g_minus_1": float(np.linalg.norm(gm1)),
        "linf_norm_g_minus_1": float(np.max(np.abs(gm1))),
    }

    # constraint residuals
    constraint_residuals = {}
    if cfg.constraints.type == ConstraintType.BALL:
        R = cfg.constraints.radius * np.sqrt(cfg.network.N)
        l2_residual = float(R - np.linalg.norm(gm1))                 # like your print
        sq_residual = float(R**2 - float(np.dot(gm1, gm1)))          # what SLSQP sees
        constraint_residuals["ball"] = {
            "R_fraction": float(cfg.constraints.radius),
            "R_absolute": float(R),
            "residual_l2": l2_residual,
            "residual_sq": sq_residual,
        }
    else:
        w = float(cfg.constraints.box_half_width)
        lo, hi = 1.0 - w, 1.0 + w
        below = np.maximum(lo - g_opt, 0.0)
        above = np.maximum(g_opt - hi, 0.0)
        max_violation = float(max(below.max(initial=0.0), above.max(initial=0.0)))
        at_lower = int(np.sum(np.isclose(g_opt, lo, atol=1e-9)))
        at_upper = int(np.sum(np.isclose(g_opt, hi, atol=1e-9)))
        inside_frac = float(np.mean((g_opt >= lo) & (g_opt <= hi)))
        constraint_residuals["box"] = {
            "half_width": w,
            "lower": lo, "upper": hi,
            "max_abs_violation": max_violation,
            "n_at_lower_bound": at_lower,
            "n_at_upper_bound": at_upper,
            "fraction_inside_bounds": inside_frac,
        }

    # ---- assemble output ----
    out = {
        # keep existing high-level fields:
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),

        # richer optimizer info (matches your printouts)
        "solver": "SLSQP",
        "final_objective": float(res.fun),
        "iterations": int(getattr(res, "nit", -1)),
        "func_evals": int(getattr(res, "nfev", -1)),
        "grad_evals": int(getattr(res, "njev", -1)),

        # angles and norms
        "angles_deg": {
            "unmod_to_target": float(angle_pre),
            "opt_to_target": float(angle_post),
            "delta_opt_vs_unmod": float(angle_deg(d0, d_opt)),
            "improvement": float(improvement),          # pre - post
        },
        "alignment_cos2": {
            "pre": float(cos2_pre),
            "post": float(cos2_post),
        },
        "axis_norms": {
            "pre": float(norm_pre),
            "post": float(norm_post),
            "equality_residual": eq_residual,          # None if hard_norm=False
        },

        "g_stats": g_stats,
        "constraint_type": _enum_val(cfg.constraints.type),
        "hard_norm": bool(cfg.constraints.hard_norm),
        "constraint_residuals": constraint_residuals,

        "pca": {
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_.tolist()],
        },

        # config snapshot (enum values flattened for readability)
        "cfg": {
            "network": {**vars(cfg.network), "zero_sum": _enum_val(cfg.network.zero_sum)},
            "objective": {
                **vars(cfg.objective),
                "target_type": _enum_val(cfg.objective.target_type),
                "axis_of_interest": _enum_val(cfg.objective.axis_of_interest),
            },
            "constraints": {**vars(cfg.constraints), "type": _enum_val(cfg.constraints.type)},
            "grid": {"shape_vals": tuple(cfg.grid.shape_vals), "color_vals": tuple(cfg.grid.color_vals)},
            "tag": cfg.tag,
        },
    }

    return out, net, pca, Z0, Z_opt, d0, d_opt, target, g_opt

def _optimize_axis_with_shared_pca(net, pca, target, cfg, axis_choice: AxisOfInterest):
    """
    Optimize gains for one axis (COLOR or SHAPE) using the PCA basis fitted on UNMOD.
    Returns a dict with metrics, and arrays for plotting.
    """
    import numpy as np
    from scipy.optimize import minimize

    N = cfg.network.N
    g0 = np.ones(N)

    axis_fn = _axis_fn_for(net, cfg, axis_choice)
    d0 = axis_fn(g0)

    # objective: maximize cos^2(d(g), target) -> minimize negative
    def obj(g):
        d = axis_fn(g)
        na = float(np.linalg.norm(d)); nb = float(np.linalg.norm(target))
        if na < 1e-15 or nb < 1e-15:
            return 0.0
        c = float(np.dot(d, target) / (na * nb))
        return -(np.clip(c, -1.0, 1.0) ** 2)

    bounds, constraints = build_bounds_and_constraints(
        axis_fn=axis_fn, g0=g0, cfg=cfg.constraints,
        positive_gains=cfg.constraints.positive_gains
    )

    res = minimize(
        obj, x0=g0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False}
    )

    g_opt = res.x
    d_opt = axis_fn(g_opt)

    # grid responses & projection in the SAME PCA basis
    X_opt = net.grid_responses(cfg.grid, g=g_opt)
    Z_opt = pca.transform(X_opt)

    # angles/norms
    def ang(a, b):
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na < 1e-15 or nb < 1e-15: return 0.0
        c = float(np.dot(a, b) / (na * nb))
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    angle_pre  = ang(d0, target)
    angle_post = ang(d_opt, target)
    improvement = angle_pre - angle_post
    norm_pre  = float(np.linalg.norm(d0))
    norm_post = float(np.linalg.norm(d_opt))
    eq_residual = float(norm_post**2 - norm_pre**2) if cfg.constraints.hard_norm else None

    gm1 = g_opt - 1.0
    g_stats = {
        "min": float(np.min(g_opt)),
        "max": float(np.max(g_opt)),
        "mean": float(np.mean(g_opt)),
        "std": float(np.std(g_opt)),
        "l2_norm_g_minus_1": float(np.linalg.norm(gm1)),
        "linf_norm_g_minus_1": float(np.max(np.abs(gm1))),
    }

    constraint_residuals = {}
    if cfg.constraints.type == ConstraintType.BALL:
        R = cfg.constraints.radius * np.sqrt(N)
        constraint_residuals["ball"] = {
            "R_fraction": float(cfg.constraints.radius),
            "R_absolute": float(R),
            "residual_l2": float(R - np.linalg.norm(gm1)),
            "residual_sq": float(R**2 - float(np.dot(gm1, gm1))),
        }
    else:
        w = float(cfg.constraints.box_half_width)
        lo, hi = 1.0 - w, 1.0 + w
        below = np.maximum(lo - g_opt, 0.0)
        above = np.maximum(g_opt - hi, 0.0)
        constraint_residuals["box"] = {
            "half_width": w, "lower": lo, "upper": hi,
            "max_abs_violation": float(max(below.max(initial=0.0), above.max(initial=0.0))),
            "n_at_lower_bound": int(np.sum(np.isclose(g_opt, lo, atol=1e-9))),
            "n_at_upper_bound": int(np.sum(np.isclose(g_opt, hi, atol=1e-9))),
            "fraction_inside_bounds": float(np.mean((g_opt >= lo) & (g_opt <= hi))),
        }

    out = {
        "solver": "SLSQP",
        "final_objective": float(res.fun),
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "iterations": int(getattr(res, "nit", -1)),
        "func_evals": int(getattr(res, "nfev", -1)),
        "grad_evals": int(getattr(res, "njev", -1)),
        "angles_deg": {
            "unmod_to_target": float(angle_pre),
            "opt_to_target": float(angle_post),
            "delta_opt_vs_unmod": float(ang(d0, d_opt)),
            "improvement": float(improvement),
        },
        "axis_norms": {
            "pre": norm_pre,
            "post": norm_post,
            "equality_residual": eq_residual,
        },
        "g_stats": g_stats,
        "constraint_residuals": constraint_residuals,
    }
    return out, g_opt, d0, d_opt, Z_opt

def optimize_triad(cfg):
    """
    Build network and PCA on UNMOD; optimize COLOR and SHAPE axes against the same target.
    Returns a triad summary + arrays for plotting.
    """
    net = LinearRNN(cfg.network)
    X0 = net.grid_responses(cfg.grid, g=None)
    pca, Z0 = net.pca3(X0)                    # fit ONCE on unmod
    target = target_in_neuron_space(pca, cfg.objective)

    # COLOR
    out_color, g_color, d_color0, d_color_opt, Z_color = _optimize_axis_with_shared_pca(
        net, pca, target, cfg, AxisOfInterest.COLOR
    )
    # SHAPE
    out_shape, g_shape, d_shape0, d_shape_opt, Z_shape = _optimize_axis_with_shared_pca(
        net, pca, target, cfg, AxisOfInterest.SHAPE
    )
    
    # cross-attend axes: evaluate the same axis under the other attention's gains
    color_axis_color_g = net.color_axis(cfg.objective.shape_for_color_line, g=g_color)  # color axis under color gains
    color_axis_shape_g = net.color_axis(cfg.objective.shape_for_color_line, g=g_shape)  # color axis under shape gains

    shape_axis_color_g = net.shape_axis(cfg.objective.color_for_shape_line, g=g_color)  # shape axis under color gains
    shape_axis_shape_g = net.shape_axis(cfg.objective.color_for_shape_line, g=g_shape)  # shape axis under shape gains

    # cross-attend angles
    color_cross_attend_deg = angle_deg(color_axis_color_g, color_axis_shape_g)
    shape_cross_attend_deg = angle_deg(shape_axis_color_g, shape_axis_shape_g)


    # summarized comparison of gains (safe if positive_gains=True; otherwise ratio may be negative)
    import numpy as np
    eps = 1e-12
    gain_ratio = (g_color + eps) / (g_shape + eps)
    gain_diff  = g_color - g_shape
    def stats(x):
        return {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "std": float(np.std(x)),
        }

    summary = {
        "pca": {
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_.tolist()],
        },
        "unmod": {
            "color_axis_norm": float(np.linalg.norm(d_color0)),
            "shape_axis_norm": float(np.linalg.norm(d_shape0)),
        },
        "color_alignment": out_color,
        "shape_alignment": out_shape,
        "gain_comparison": {
            "ratio_color_over_shape": stats(gain_ratio),
            "diff_color_minus_shape": stats(gain_diff),
        },
        "cfg": {
            "network": {**vars(cfg.network), "zero_sum": _enum_val(cfg.network.zero_sum)},
            "objective": {
                **vars(cfg.objective),
                "target_type": _enum_val(cfg.objective.target_type),
                "axis_of_interest": _enum_val(cfg.objective.axis_of_interest),
            },
            "constraints": {**vars(cfg.constraints), "type": _enum_val(cfg.constraints.type)},
            "grid": {"shape_vals": tuple(cfg.grid.shape_vals), "color_vals": tuple(cfg.grid.color_vals)},
            "tag": cfg.tag,
        },
        "cross_attend": {
        "color_axis_angle_deg": float(color_cross_attend_deg),
        "shape_axis_angle_deg": float(shape_cross_attend_deg),
        "color_axis_norms": {
            "under_color_attend": float(np.linalg.norm(color_axis_color_g)),
            "under_shape_attend": float(np.linalg.norm(color_axis_shape_g)),
        },
        "shape_axis_norms": {
            "under_color_attend": float(np.linalg.norm(shape_axis_color_g)),
            "under_shape_attend": float(np.linalg.norm(shape_axis_shape_g)),
        },
    },
    }

    return (
        summary, net, pca, Z0, Z_color, Z_shape,
        d_color0, d_color_opt, d_shape0, d_shape_opt,
        target, g_color, g_shape
    )

def sweep_range_vs_degree(cfg: ExperimentConfig, ranges: List[float]) -> Dict:
    # unchanged...
    table = []
    for r in ranges:
        local = cfg
        if cfg.constraints.type == ConstraintType.BALL:
            local.constraints.radius = r
        else:
            local.constraints.box_half_width = r
        out, *_ = optimize_once(local)
        row = {
            "range": r,
            "unmod_to_target_deg": out["angles_deg"]["unmod_to_target"],
            "opt_to_target_deg": out["angles_deg"]["opt_to_target"],
            "delta_opt_vs_unmod_deg": out["angles_deg"]["delta_opt_vs_unmod"],
            "success": out["success"]
        }
        table.append(row)
    return {
        "constraint_type": _enum_val(cfg.constraints.type),
        "hard_norm": bool(cfg.constraints.hard_norm),
        "ranges": ranges,
        "rows": table,
        "tag": cfg.tag
    }

def save_json(obj: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def triad_sweep(cfg: ExperimentConfig, ranges: list[float]) -> dict:
    """
    Sweep constraint magnitude and record cross-attend angles:
      - color_cross_attend_deg: angle between color axis under (color gains) vs (shape gains)
      - shape_cross_attend_deg: angle between shape axis under (color gains) vs (shape gains)
    """
    rows = []
    for r in ranges:
        local = cfg  # shallow copy ok
        if cfg.constraints.type == ConstraintType.BALL:
            local.constraints.radius = r
        else:
            local.constraints.box_half_width = r

        (summary, net, pca, Z0, Zc, Zs,
         dcol0, dcol1, dshp0, dshp1, target, gcol, gshp) = optimize_triad(local)

        rows.append({
            "range": r,
            "color_cross_attend_deg": summary["cross_attend"]["color_axis_angle_deg"],
            "shape_cross_attend_deg": summary["cross_attend"]["shape_axis_angle_deg"],
            "success_color": summary["color_alignment"]["success"],
            "success_shape": summary["shape_alignment"]["success"],
        })

    return {
        "constraint_type": getattr(cfg.constraints.type, "value", cfg.constraints.type),
        "hard_norm": bool(cfg.constraints.hard_norm),
        "ranges": ranges,
        "rows": rows,
        "tag": cfg.tag
    }
