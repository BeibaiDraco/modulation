from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
import copy

from .config import ExperimentConfig, ConstraintType, AxisOfInterest
from .network import LinearRNN, angle_deg
from .objectives import target_in_neuron_space, axis_of_interest_vec, angle_to_target
from .constraints import build_bounds_and_constraints
from .shuffle import assign_bins, shuffle_within_bins, shuffle_pair_within_bins


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

def _pc_angles(vec: NDArray[np.float64]) -> tuple[float, float]:
    x, y, z = map(float, np.asarray(vec, dtype=float))
    theta = float(np.degrees(np.arctan2(y, x)))  # [-180, 180]
    rho_xy = float(np.hypot(x, y))
    phi = float(np.degrees(np.arctan2(z, rho_xy)))  # [-90, 90]
    theta = (theta + 360.0) % 360.0
    return theta, phi

def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm

def _slerp(u: NDArray[np.float64], v: NDArray[np.float64], delta_deg: float) -> NDArray[np.float64]:
    u = _unit(u)
    v = _unit(v)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    total = float(np.degrees(np.arccos(dot))) if -1.0 < dot < 1.0 else 0.0
    if total < 1e-6 or delta_deg <= 0.0:
        return u
    t = min(1.0, delta_deg / total)
    omega = np.arccos(dot)
    sin_omega = float(np.sin(omega))
    if sin_omega < 1e-9:
        return _unit((1.0 - t) * u + t * v)
    w = (np.sin((1.0 - t) * omega) / sin_omega) * u + (np.sin(t * omega) / sin_omega) * v
    return _unit(w)

def _axis_fn_for(net, cfg, choice: AxisOfInterest):
    if choice == AxisOfInterest.COLOR:
        return lambda g: net.color_axis(cfg.objective.shape_for_color_line, g)
    if choice == AxisOfInterest.SHAPE:
        return lambda g: net.shape_axis(cfg.objective.color_for_shape_line, g)
    raise ValueError(f"AxisOfInterest {choice} not supported for triad mode")

def baseline_axis_pc_angles(
    cfg: ExperimentConfig,
    *,
    delta_deg: Optional[float] = None,
    direction: str = "color_to_shape",
) -> Dict[str, Dict[str, float]]:
    """
    Compute unmodulated color/shape axes in PC space and optionally propose a nearby target.

    Args:
        cfg: Experiment configuration.
        delta_deg: If provided, take a spherical-linear step of this many degrees from the source axis
                   toward the destination axis when suggesting a target.
        direction: "color_to_shape", "shape_to_color", or "midpoint".
    """
    net = LinearRNN(cfg.network)
    X0 = net.grid_responses(cfg.grid, g=None)
    pca, _ = net.pca3(X0)

    d_color = net.color_axis(cfg.objective.shape_for_color_line, g=None)
    d_shape = net.shape_axis(cfg.objective.color_for_shape_line, g=None)

    comps = pca.components_[:3, :]
    v_color_pc = _unit(comps @ d_color)
    v_shape_pc = _unit(comps @ d_shape)

    theta_c, phi_c = _pc_angles(v_color_pc)
    theta_s, phi_s = _pc_angles(v_shape_pc)

    result: Dict[str, Dict[str, float]] = {
        "color": {
            "theta_deg": theta_c,
            "phi_deg": phi_c,
            "vector_pc1_pc2_pc3": v_color_pc.tolist(),
        },
        "shape": {
            "theta_deg": theta_s,
            "phi_deg": phi_s,
            "vector_pc1_pc2_pc3": v_shape_pc.tolist(),
        },
        "geodesic_angle_deg": float(np.degrees(np.arccos(np.clip(np.dot(v_color_pc, v_shape_pc), -1.0, 1.0)))),
    }

    step = float(delta_deg) if delta_deg is not None else None
    if step is not None:
        direction = direction.lower()
        if direction == "color_to_shape":
            target_vec = _slerp(v_color_pc, v_shape_pc, step)
        elif direction == "shape_to_color":
            target_vec = _slerp(v_shape_pc, v_color_pc, step)
        elif direction == "midpoint":
            target_vec = _unit(v_color_pc + v_shape_pc)
        else:
            raise ValueError("direction must be 'color_to_shape', 'shape_to_color', or 'midpoint'")
        theta_t, phi_t = _pc_angles(target_vec)
        result["suggested_target"] = {
            "theta_deg": theta_t,
            "phi_deg": phi_t,
            "vector_pc1_pc2_pc3": target_vec.tolist(),
            "delta_deg": step,
            "direction": direction,
        }

    return result

def optimize_once(cfg: ExperimentConfig) -> Dict:
    # ---- model & PCA (fit on UNMOD only) ----
    net = LinearRNN(cfg.network)
    X0 = net.grid_responses(cfg.grid, g=None)
    pca, Z0 = net.pca3(X0)
    target = target_in_neuron_space(pca, cfg.objective)

    # ---- axis at g=1 (unmod) ----
    g0 = np.ones(cfg.network.N)
    d0 = axis_of_interest_vec(net, cfg.objective, g=g0)
    d0_abs = np.abs(d0)
    opt_cfg = cfg.optimization
    perc = float(getattr(opt_cfg, "activity_floor_baseline_percentile", 10.0))
    floor_val = float(np.percentile(d0_abs, perc)) if d0_abs.size else 0.0
    denom_modesty = np.maximum(d0_abs, floor_val)
    denom_modesty = np.where(denom_modesty > 0.0, denom_modesty, 1.0)
    delta = float(getattr(opt_cfg, "activity_huber_delta", 0.25))
    lam_modesty = float(getattr(opt_cfg, "lambda_activity_modesty", 0.0))
    lam_cond = float(getattr(opt_cfg, "lambda_resolvent", 0.0))

    # ---- objective (maximize cos^2 -> minimize negative) ----
    def obj(g):
        g_arr = np.asarray(g, dtype=float)
        d = axis_of_interest_vec(net, cfg.objective, g=g_arr)
        na = float(np.linalg.norm(d))
        nb = float(np.linalg.norm(target))
        if na < 1e-15 or nb < 1e-15:
            return 0.0
        c = float(np.dot(d, target) / (na * nb))
        align_term = -(np.clip(c, -1.0, 1.0) ** 2)

        loss = align_term

        if lam_modesty > 0.0:
            g_eff = np.abs(d) / denom_modesty
            x = g_eff - 1.0
            huber = np.where(
                np.abs(x) <= delta,
                0.5 * x**2,
                delta * (np.abs(x) - 0.5 * delta),
            )
            loss_modest = float(np.mean(huber))
            loss += lam_modesty * loss_modest

        if lam_cond > 0.0:
            G_rows = g_arr[:, None]
            M = net.I - (G_rows * net.W_R)
            smin = float(np.linalg.svd(M, compute_uv=False)[-1])
            inv_smin = 1.0 / max(smin, 1e-6)
            loss_cond = float(inv_smin**2)
            loss += lam_cond * loss_cond

        return loss

    bounds, constraints = build_bounds_and_constraints(
        axis_fn=lambda g: axis_of_interest_vec(net, cfg.objective, g=g),
        g0=g0, cfg=cfg.constraints,
        positive_gains=cfg.constraints.positive_gains
    )

    # --- NEW: hard stability margin, freeze risky, hard modesty cap ---
    # 1) Hard stability: σ_min(I - diag(g) W_R) >= smin_floor
    tau = float(getattr(cfg.optimization, "smin_floor", 0.0))
    if tau > 0.0:
        def _ineq_smin(g, W=net.W_R, I=net.I, t=tau):
            M = I - (np.asarray(g, dtype=float)[:, None] * W)
            smin = float(np.linalg.svd(M, compute_uv=False)[-1])
            return smin - t  # >= 0 when feasible
        constraints.append({"type": "ineq", "fun": _ineq_smin})

    # 2) Freeze top-q risky neurons at g_i = 1 (by equality constraints)
    #    Risk = ||H0[:,i]|| * (1 / denom_modesty_i), where H0 = (I - W_R)^{-1}, g = 1 baseline
    q = float(getattr(cfg.optimization, "freeze_topq_risky", 0.0))
    protected_idx = np.array([], dtype=int)
    if q > 0.0:
        H0 = np.linalg.inv(net.I - net.W_R)           # N x N
        colnorm = np.linalg.norm(H0, axis=0)          # (N,)
        small_base = 1.0 / denom_modesty              # bigger when baseline tiny
        risk = colnorm * small_base
        cutoff = np.quantile(risk, 1.0 - q)
        protected_idx = np.flatnonzero(risk >= cutoff)
        for i in protected_idx.tolist():
            constraints.append({"type": "eq", "fun": (lambda g, i=i: float(np.asarray(g, dtype=float)[i] - 1.0))})

    # 3) Hard cap on activity-derived gain g_eff = |d(g)| / denom_modesty
    #    Applies to protected set if non-empty; otherwise global (all neurons).
    eps_cap = float(getattr(cfg.optimization, "protect_activity_eps", 0.0))
    qtile = float(getattr(cfg.optimization, "protect_activity_quantile", 1.0))
    if eps_cap > 0.0:
        def _ineq_modesty(g, idx=protected_idx, qtile=qtile, eps_cap=eps_cap):
            g_arr = np.asarray(g, dtype=float)
            d = axis_of_interest_vec(net, cfg.objective, g=g_arr)   # N,
            g_eff = np.abs(d) / denom_modesty
            vals = g_eff[idx] if idx.size else g_eff
            if vals.size == 0:
                return 1.0  # nothing to constrain
            k = max(0, min(vals.size - 1, int(np.ceil(qtile * vals.size) - 1)))
            thresh = float(np.partition(vals, k)[k])  # q-quantile (1.0 = max)
            return (1.0 + eps_cap) - thresh  # >= 0 when feasible
        constraints.append({"type": "ineq", "fun": _ineq_modesty})

    # ---- optimize ----
    res = minimize(
        obj, x0=g0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False, "eps": 1e-4}
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
    d0_abs = np.abs(d0)
    opt_cfg = cfg.optimization
    perc = float(getattr(opt_cfg, "activity_floor_baseline_percentile", 10.0))
    floor_val = float(np.percentile(d0_abs, perc)) if d0_abs.size else 0.0
    denom_modesty = np.maximum(d0_abs, floor_val)
    denom_modesty = np.where(denom_modesty > 0.0, denom_modesty, 1.0)
    delta = float(getattr(opt_cfg, "activity_huber_delta", 0.25))
    lam_modesty = float(getattr(opt_cfg, "lambda_activity_modesty", 0.0))
    lam_cond = float(getattr(opt_cfg, "lambda_resolvent", 0.0))

    # objective: maximize cos^2(d(g), target) -> minimize negative
    def obj(g):
        g_arr = np.asarray(g, dtype=float)
        d = axis_fn(g_arr)
        na = float(np.linalg.norm(d)); nb = float(np.linalg.norm(target))
        if na < 1e-15 or nb < 1e-15:
            return 0.0
        c = float(np.dot(d, target) / (na * nb))
        align_term = -(np.clip(c, -1.0, 1.0) ** 2)

        loss = align_term

        if lam_modesty > 0.0:
            g_eff = np.abs(d) / denom_modesty
            x = g_eff - 1.0
            huber = np.where(
                np.abs(x) <= delta,
                0.5 * x**2,
                delta * (np.abs(x) - 0.5 * delta),
            )
            loss_modest = float(np.mean(huber))
            loss += lam_modesty * loss_modest

        if lam_cond > 0.0:
            G_rows = g_arr[:, None]
            M = net.I - (G_rows * net.W_R)
            smin = float(np.linalg.svd(M, compute_uv=False)[-1])
            inv_smin = 1.0 / max(smin, 1e-6)
            loss_cond = float(inv_smin**2)
            loss += lam_cond * loss_cond

        return loss

    bounds, constraints = build_bounds_and_constraints(
        axis_fn=axis_fn, g0=g0, cfg=cfg.constraints,
        positive_gains=cfg.constraints.positive_gains
    )

    # --- NEW: hard stability margin, freeze risky, hard modesty cap ---
    tau = float(getattr(cfg.optimization, "smin_floor", 0.0))
    if tau > 0.0:
        def _ineq_smin(g, W=net.W_R, I=net.I, t=tau):
            M = I - (np.asarray(g, dtype=float)[:, None] * W)
            smin = float(np.linalg.svd(M, compute_uv=False)[-1])
            return smin - t
        constraints.append({"type": "ineq", "fun": _ineq_smin})

    q = float(getattr(cfg.optimization, "freeze_topq_risky", 0.0))
    protected_idx = np.array([], dtype=int)
    if q > 0.0:
        H0 = np.linalg.inv(net.I - net.W_R)
        colnorm = np.linalg.norm(H0, axis=0)
        small_base = 1.0 / denom_modesty
        risk = colnorm * small_base
        cutoff = np.quantile(risk, 1.0 - q)
        protected_idx = np.flatnonzero(risk >= cutoff)
        for i in protected_idx.tolist():
            constraints.append({"type": "eq", "fun": (lambda g, i=i: float(np.asarray(g, dtype=float)[i] - 1.0))})

    eps_cap = float(getattr(cfg.optimization, "protect_activity_eps", 0.0))
    qtile = float(getattr(cfg.optimization, "protect_activity_quantile", 1.0))
    if eps_cap > 0.0:
        def _ineq_modesty(g, idx=protected_idx, qtile=qtile, eps_cap=eps_cap):
            g_arr = np.asarray(g, dtype=float)
            d = axis_fn(g_arr)                         # N,
            g_eff = np.abs(d) / denom_modesty
            vals = g_eff[idx] if idx.size else g_eff
            if vals.size == 0:
                return 1.0
            k = max(0, min(vals.size - 1, int(np.ceil(qtile * vals.size) - 1)))
            thresh = float(np.partition(vals, k)[k])
            return (1.0 + eps_cap) - thresh
        constraints.append({"type": "ineq", "fun": _ineq_modesty})

    res = minimize(
        obj, x0=g0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False, "eps": 1e-4}
    )

    g_opt = res.x
    d_opt = axis_fn(g_opt)

    # grid responses & projection in the SAME PCA basis
    X_opt = net.grid_responses(cfg.grid, g=g_opt)
    Z_opt = pca.transform(X_opt)

    # angles/norms
    angle_pre  = angle_deg(d0,   target)
    angle_post = angle_deg(d_opt, target)
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
            "delta_opt_vs_unmod": float(angle_deg(d0, d_opt)),
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

    # angles to target (un-directional)
    a_cc = angle_deg(color_axis_color_g, target)
    a_cs = angle_deg(color_axis_shape_g, target)
    a_ss = angle_deg(shape_axis_shape_g, target)
    a_sc = angle_deg(shape_axis_color_g, target)

    to_target = {
        "color_axis": {
            "under_color_deg": float(a_cc),
            "under_shape_deg": float(a_cs),
            "improvement_deg": float(a_cs - a_cc)
        },
        "shape_axis": {
            "under_shape_deg": float(a_ss),
            "under_color_deg": float(a_sc),
            "improvement_deg": float(a_sc - a_ss)
        }
    }

    # cross-attend angles
    color_cross_attend_deg = angle_deg(color_axis_color_g, color_axis_shape_g)
    shape_cross_attend_deg = angle_deg(shape_axis_color_g, shape_axis_shape_g)

    # ---------- Shuffle test ----------
    shuffle_cfg = cfg.shuffle
    shuffle_summary = None
    if shuffle_cfg.enabled:
        rng = np.random.default_rng(shuffle_cfg.seed if shuffle_cfg.seed is not None else (cfg.network.seed + 101))
        sel = net.S[:, 1] - net.S[:, 0]                   # color - shape selectivity
        bins = assign_bins(sel, shuffle_cfg.num_bins, shuffle_cfg.binning)

        if shuffle_cfg.mode == "paired":
            g_color_shuf, g_shape_shuf = shuffle_pair_within_bins(g_color, g_shape, bins, rng)
        else:  # independent (default)
            g_color_shuf = shuffle_within_bins(g_color, bins, rng)
            g_shape_shuf = shuffle_within_bins(g_shape, bins, rng)

        # axes under shuffled gains
        col_axis_color_g_shuf = net.color_axis(cfg.objective.shape_for_color_line, g=g_color_shuf)
        col_axis_shape_g_shuf = net.color_axis(cfg.objective.shape_for_color_line, g=g_shape_shuf)
        shp_axis_color_g_shuf = net.shape_axis(cfg.objective.color_for_shape_line, g=g_color_shuf)
        shp_axis_shape_g_shuf = net.shape_axis(cfg.objective.color_for_shape_line, g=g_shape_shuf)
        a_cc_sh = angle_deg(col_axis_color_g_shuf, target)
        a_cs_sh = angle_deg(col_axis_shape_g_shuf, target)
        a_ss_sh = angle_deg(shp_axis_shape_g_shuf, target)
        a_sc_sh = angle_deg(shp_axis_color_g_shuf, target)

        color_cross_attend_deg_shuf = angle_deg(col_axis_color_g_shuf, col_axis_shape_g_shuf)
        shape_cross_attend_deg_shuf = angle_deg(shp_axis_color_g_shuf, shp_axis_shape_g_shuf)

        shuffle_summary = {
            "mode": shuffle_cfg.mode,
            "num_bins": int(shuffle_cfg.num_bins),
            "binning": shuffle_cfg.binning,
            "seed": shuffle_cfg.seed if shuffle_cfg.seed is not None else (cfg.network.seed + 101),
            "cross_attend": {
                "color_axis_angle_deg": float(color_cross_attend_deg_shuf),
                "shape_axis_angle_deg": float(shape_cross_attend_deg_shuf),
            },
            "axis_norms": {
                "color_axis": {
                    "under_color_attend_shuf": float(np.linalg.norm(col_axis_color_g_shuf)),
                    "under_shape_attend_shuf": float(np.linalg.norm(col_axis_shape_g_shuf)),
                },
                "shape_axis": {
                    "under_color_attend_shuf": float(np.linalg.norm(shp_axis_color_g_shuf)),
                    "under_shape_attend_shuf": float(np.linalg.norm(shp_axis_shape_g_shuf)),
                }
            },
            "to_target": {
                "color_axis": {
                    "under_color_deg": float(a_cc_sh),
                    "under_shape_deg": float(a_cs_sh),
                    "improvement_deg": float(a_cs_sh - a_cc_sh)
                },
                "shape_axis": {
                    "under_shape_deg": float(a_ss_sh),
                    "under_color_deg": float(a_sc_sh),
                    "improvement_deg": float(a_sc_sh - a_ss_sh)
                }
            }
                        
        }



    # summarized comparison of gains (safe if positive_gains=True; otherwise ratio may be negative)

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
        "to_target": to_target,
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
        "shuffle": shuffle_summary,
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
        local = copy.deepcopy(cfg)
        if local.constraints.type == ConstraintType.BALL:
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


def save_weight_matrices(net, target_vec_neuron, outdir: Path, tag: str) -> dict:
    """
    Save W_F (N x K), W_R (N x N), and the readout/target axis t (N,)
    into <outdir>/ as NPZ, plus individual NPY/CSV files for convenience.
    """
    _ensure_dir(outdir)
    paths = {}

    # Bundle everything
    np.savez(outdir / f"{tag}_weights_readout.npz",
             W_F=net.W_F, W_R=net.W_R, readout=target_vec_neuron)
    paths["npz"] = str(outdir / f"{tag}_weights_readout.npz")

    # Also save separate files (easy to inspect)
    np.save(outdir / f"{tag}_W_F.npy", net.W_F)
    np.save(outdir / f"{tag}_W_R.npy", net.W_R)
    np.save(outdir / f"{tag}_readout.npy", target_vec_neuron)

    np.savetxt(outdir / f"{tag}_W_F.csv", net.W_F, delimiter=",", fmt="%.6g")
    np.savetxt(outdir / f"{tag}_W_R.csv", net.W_R, delimiter=",", fmt="%.6g")
    np.savetxt(outdir / f"{tag}_readout.csv", target_vec_neuron.reshape(1, -1), delimiter=",", fmt="%.6g")

    meta = {
        "W_F_shape": list(net.W_F.shape),
        "W_R_shape": list(net.W_R.shape),
        "readout_shape": [int(target_vec_neuron.size)],
        "tag": tag,
    }
    save_json(meta, outdir / f"{tag}_weights_readout_meta.json")
    return paths


def save_json(obj: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def triad_sweep(cfg: ExperimentConfig, ranges: list[float]) -> dict:
    """
    Sweep constraint magnitude and record:
      - cross-attend angles (undirectional): angle between the same axis under color vs shape gains
      - improvement-to-target (undirectional): for each axis, how much 'own' attention helps vs 'other'
        * impr_color_deg = angle(d_color(g_shape), target) - angle(d_color(g_color), target)
        * impr_shape_deg = angle(d_shape(g_color), target) - angle(d_shape(g_shape), target)
      - If shuffle.enabled and repeats>1: mean ± SEM for both cross-attend and improvement metrics
    """
    def undir_angle(a, b):
        # min(theta, 180 - theta) to treat axes as unoriented
        th = angle_deg(a, b)
        return th if th <= 90.0 else 180.0 - th

    rows = []
    for r in ranges:
        # (1) set constraint magnitude on a local copy (shallow is OK for these scalars)
        local = copy.deepcopy(cfg)
        if local.constraints.type == ConstraintType.BALL:
            local.constraints.radius = r
        else:
            local.constraints.box_half_width = r

        # (2) run triad once to get gains and target etc.
        (summary, net, pca, Z0, Zc, Zs,
         dcol0, dcol1, dshp0, dshp1, target, gcol, gshp) = optimize_triad(local)

        # (3) build axes under each attention state
        col_axis_color = net.color_axis(local.objective.shape_for_color_line, g=gcol)  # color gains
        col_axis_shape = net.color_axis(local.objective.shape_for_color_line, g=gshp)  # shape gains
        shp_axis_shape = net.shape_axis(local.objective.color_for_shape_line, g=gshp)  # shape gains
        shp_axis_color = net.shape_axis(local.objective.color_for_shape_line, g=gcol)  # color gains

        # (4) undirectional angles to target
        a_cc = undir_angle(col_axis_color, target)  # color axis under color
        a_cs = undir_angle(col_axis_shape, target)  # color axis under shape
        a_ss = undir_angle(shp_axis_shape, target)  # shape axis under shape
        a_sc = undir_angle(shp_axis_color, target)  # shape axis under color

        # improvements (positive means "own" attention helps that axis align to target)
        impr_color = float(a_cs - a_cc)  # color axis: (shape) - (color)
        impr_shape = float(a_sc - a_ss)  # shape axis: (color) - (shape)

        # (5) undirectional cross-attend angles (same axis under the two gain states)
        color_cross = undir_angle(col_axis_color, col_axis_shape)
        shape_cross = undir_angle(shp_axis_color, shp_axis_shape)

        row = {
            "range": r,
            # new improvement-to-target metrics
            "impr_color_deg": impr_color,
            "impr_shape_deg": impr_shape,
            # keep cross-attend for reference (now undirectional)
            "color_cross_attend_deg": float(color_cross),
            "shape_cross_attend_deg": float(shape_cross),
            "success_color": summary["color_alignment"]["success"],
            "success_shape": summary["shape_alignment"]["success"],
        }

        # (6) include single-run shuffled stats if optimize_triad produced them
        shuf = summary.get("shuffle") or summary.get("cross_attend", {}).get("shuffle")
        if shuf:
            if shuf.get("cross_attend"):
                row["color_cross_attend_deg_shuf"] = shuf["cross_attend"]["color_axis_angle_deg"]
                row["shape_cross_attend_deg_shuf"] = shuf["cross_attend"]["shape_axis_angle_deg"]
            # improvement-to-target from single-run shuffle, if present
            if shuf.get("to_target"):
                cax = shuf["to_target"].get("color_axis", {})
                sax = shuf["to_target"].get("shape_axis", {})
                if "improvement_deg" in cax:
                    row["impr_color_deg_shuf"] = float(cax["improvement_deg"])
                if "improvement_deg" in sax:
                    row["impr_shape_deg_shuf"] = float(sax["improvement_deg"])

        # (7) repeated shuffles for mean ± SEM (panel_d)
        if local.shuffle.enabled and int(local.shuffle.repeats) > 1:
            repeats   = int(local.shuffle.repeats)
            base_seed = local.shuffle.seed if local.shuffle.seed is not None else (local.network.seed + 101)

            sel  = net.S[:, 1] - net.S[:, 0]
            bins = assign_bins(sel, local.shuffle.num_bins, local.shuffle.binning)

            col_cross_list, shp_cross_list = [], []
            impr_color_list, impr_shape_list = [], []

            for i in range(repeats):
                rng = np.random.default_rng(base_seed + 10007 * i)
                if local.shuffle.mode == "paired":
                    g_color_shuf, g_shape_shuf = shuffle_pair_within_bins(gcol, gshp, bins, rng)
                else:
                    g_color_shuf = shuffle_within_bins(gcol, bins, rng)
                    g_shape_shuf = shuffle_within_bins(gshp, bins, rng)

                # axes under shuffled gains
                col_axis_color_sh = net.color_axis(local.objective.shape_for_color_line, g=g_color_shuf)
                col_axis_shape_sh = net.color_axis(local.objective.shape_for_color_line, g=g_shape_shuf)
                shp_axis_color_sh = net.shape_axis(local.objective.color_for_shape_line, g=g_color_shuf)
                shp_axis_shape_sh = net.shape_axis(local.objective.color_for_shape_line, g=g_shape_shuf)

                # cross-attend (undirectional)
                col_cross_list.append(undir_angle(col_axis_color_sh, col_axis_shape_sh))
                shp_cross_list.append(undir_angle(shp_axis_color_sh, shp_axis_shape_sh))

                # improvements to target (undirectional)
                a_cc_sh = undir_angle(col_axis_color_sh, target)
                a_cs_sh = undir_angle(col_axis_shape_sh, target)
                a_ss_sh = undir_angle(shp_axis_shape_sh, target)
                a_sc_sh = undir_angle(shp_axis_color_sh, target)
                impr_color_list.append(a_cs_sh - a_cc_sh)
                impr_shape_list.append(a_sc_sh - a_ss_sh)

            # means & SEM
            col_mean = float(np.mean(col_cross_list))
            shp_mean = float(np.mean(shp_cross_list))
            col_sem  = float(np.std(col_cross_list, ddof=1) / np.sqrt(repeats)) if repeats > 1 else 0.0
            shp_sem  = float(np.std(shp_cross_list, ddof=1) / np.sqrt(repeats)) if repeats > 1 else 0.0

            row["color_cross_attend_deg_shuf_mean"] = col_mean
            row["shape_cross_attend_deg_shuf_mean"] = shp_mean
            row["color_cross_attend_deg_shuf_sem"]  = col_sem
            row["shape_cross_attend_deg_shuf_sem"]  = shp_sem

            impr_c_mean = float(np.mean(impr_color_list))
            impr_s_mean = float(np.mean(impr_shape_list))
            impr_c_sem  = float(np.std(impr_color_list, ddof=1) / np.sqrt(repeats)) if repeats > 1 else 0.0
            impr_s_sem  = float(np.std(impr_shape_list, ddof=1) / np.sqrt(repeats)) if repeats > 1 else 0.0

            row["impr_color_deg_shuf_mean"] = impr_c_mean
            row["impr_shape_deg_shuf_mean"] = impr_s_mean
            row["impr_color_deg_shuf_sem"]  = impr_c_sem
            row["impr_shape_deg_shuf_sem"]  = impr_s_sem

        rows.append(row)

    return {
        "constraint_type": getattr(cfg.constraints.type, "value", cfg.constraints.type),
        "hard_norm": bool(cfg.constraints.hard_norm),
        "ranges": ranges,
        "rows": rows,
        "tag": cfg.tag,
        "shuffle": {
            "enabled": bool(cfg.shuffle.enabled),
            "num_bins": int(cfg.shuffle.num_bins),
            "binning": cfg.shuffle.binning,
            "mode": cfg.shuffle.mode,
            "seed": cfg.shuffle.seed,
            "repeats": int(getattr(cfg.shuffle, "repeats", 1)),
        },
        "metric": "improvement_to_target",  # explicit: this sweep uses Δ to-target
    }
