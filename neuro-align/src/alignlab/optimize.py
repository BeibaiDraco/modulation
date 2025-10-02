from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from .config import ExperimentConfig, ConstraintType
from .network import LinearRNN, angle_deg
from .objectives import target_in_neuron_space, axis_of_interest_vec, angle_to_target
from .constraints import build_bounds_and_constraints

def _enum_val(x):
    return getattr(x, "value", x)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def optimize_once(cfg: ExperimentConfig) -> Dict:
    # build network
    net = LinearRNN(cfg.network)

    # baseline grid + PCA
    X0 = net.grid_responses(cfg.grid, g=None)
    pca, Z0 = net.pca3(X0)
    # target vector (neuron space)
    target = target_in_neuron_space(pca, cfg.objective)

    # axis of interest
    g0 = np.ones(cfg.network.N)
    d0 = axis_of_interest_vec(net, cfg.objective, g=g0)

    # objective: maximize squared cosine alignment with target (minimize negative)
    def obj(g):
        d = axis_of_interest_vec(net, cfg.objective, g=g)
        num = float(np.dot(d, target))
        den = float(np.linalg.norm(d) * np.linalg.norm(target) + 1e-15)
        cos2 = (num / den) ** 2
        return -cos2

    # constraints & bounds
    bounds, constraints = build_bounds_and_constraints(
        axis_fn=lambda g: axis_of_interest_vec(net, cfg.objective, g=g),
        g0=g0, cfg=cfg.constraints,
        positive_gains=cfg.constraints.positive_gains
    )

    # choose SLSQP, reasonable defaults
    res = minimize(
        obj, x0=g0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False}
    )

    g_opt = res.x
    d_opt = axis_of_interest_vec(net, cfg.objective, g=g_opt)

    # post grid for plotting
    X_opt = net.grid_responses(cfg.grid, g=g_opt)
    _, Z_opt = net.pca3(X_opt)  # reuse random_state for consistency

    # angles & metrics
    ang_unmod_to_target = angle_to_target(d0, target)
    ang_opt_to_target   = angle_to_target(d_opt, target)
    delta_angle_from_unmod = angle_deg(d0, d_opt)

    out = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "g_stats": {
            "min": float(np.min(g_opt)),
            "max": float(np.max(g_opt)),
            "l2_norm_g_minus_1": float(np.linalg.norm(g_opt - 1.0))
        },
        "angles_deg": {
            "unmod_to_target": float(ang_unmod_to_target),
            "opt_to_target": float(ang_opt_to_target),
            "delta_opt_vs_unmod": float(delta_angle_from_unmod),
        },
        "target_type": _enum_val(cfg.objective.target_type),
        "constraint_type": _enum_val(cfg.constraints.type),
        "hard_norm": bool(cfg.constraints.hard_norm),
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
        # raw arrays are saved separately by caller if needed
    }
    return out, net, pca, Z0, Z_opt, d0, d_opt, target, g_opt

def sweep_range_vs_degree(cfg: ExperimentConfig, ranges: List[float]) -> Dict:
    """
    Sweep constraint magnitude and record angle metrics.
    - For BALL: 'range' = radius (fraction of sqrt(N))
    - For BOX:  'range' = box_half_width
    Returns dict with table + metadata.
    """
    table = []
    for r in ranges:
        local = cfg  # shallow copy of nested dataclasses is fine for scalar changes
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
        "constraint_type": cfg.constraints.type.value,
        "hard_norm": bool(cfg.constraints.hard_norm),
        "ranges": ranges,
        "rows": table,
        "tag": cfg.tag
    }

def save_json(obj: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
