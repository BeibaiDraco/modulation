from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from .config import ConstraintConfig, ConstraintType

def build_bounds_and_constraints(
    axis_fn,
    g0: NDArray[np.float64],
    cfg: ConstraintConfig,
    positive_gains: bool,
) -> Tuple[Optional[List[Tuple[float, float]]], list]:
    """
    Returns (bounds, constraints) for scipy.optimize.minimize (SLSQP).
    - If cfg.hard_norm: equality constraint ||axis(g)||^2 == ||axis(g0)||^2
    - If BALL: inequality R^2 - ||g-1||^2 >= 0
    - If BOX: bounds [1-w, 1+w]
    - positive_gains: g >= 0 (merged into bounds)
    """
    constraints = []

    # equality on axis norm
    if cfg.hard_norm:
        d0 = axis_fn(g0)
        norm0_sq = float(np.dot(d0, d0))
        constraints.append({
            "type": "eq",
            "fun": lambda g, n0=norm0_sq: float(np.dot(axis_fn(g), axis_fn(g)) - n0),
        })

    bounds = None
    if cfg.type == ConstraintType.BALL:
        R = cfg.radius * np.sqrt(g0.size)
        constraints.append({
            "type": "ineq",
            "fun": lambda g, R=R: float(R**2 - np.sum((g - 1.0)**2)),
        })
        if positive_gains:
            # bounds only enforce positivity (no box)
            bounds = [(0.0, None)] * g0.size
    else:
        # BOX: bounds around 1
        w = cfg.box_half_width
        lo, hi = 1.0 - w, 1.0 + w
        if positive_gains:
            lo = max(0.0, lo)
        bounds = [(lo, hi)] * g0.size

    return bounds, constraints
