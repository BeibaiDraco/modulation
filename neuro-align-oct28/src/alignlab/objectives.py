from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from .config import ObjectiveConfig, TargetType, AxisOfInterest
from .network import angle_deg

def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _spherical_dir(theta_deg: float, phi_deg: float) -> NDArray[np.float64]:
    """
    theta: azimuth in XY plane [0,360), phi: elevation from XY plane [-90,90].
    """
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    x = np.cos(th) * np.cos(ph)
    y = np.sin(th) * np.cos(ph)
    z = np.sin(ph)
    u = np.array([x, y, z], dtype=float)
    return _unit(u)

def target_in_neuron_space(pca: PCA, obj: ObjectiveConfig) -> NDArray[np.float64]:
    comps = pca.components_[:3, :]  # 3 x N
    if obj.target_type == TargetType.PC1:
        return _unit(comps[0, :])
    if obj.target_type == TargetType.PC2:
        return _unit(comps[1, :])
    # custom vector in PC space mapped back to neuron space
    u3 = _spherical_dir(obj.theta_deg, obj.phi_deg)   # (3,)
    t = u3 @ comps                                    # (N,)
    return _unit(t)

def axis_of_interest_vec(net, obj: ObjectiveConfig, g=None) -> NDArray[np.float64]:
    if obj.axis_of_interest == AxisOfInterest.COLOR:
        return net.color_axis(obj.shape_for_color_line, g)
    if obj.axis_of_interest == AxisOfInterest.SHAPE:
        return net.shape_axis(obj.color_for_shape_line, g)
    # CUSTOM_LINE
    assert obj.custom_stim_line_start is not None and obj.custom_stim_line_end is not None,         "CUSTOM_LINE requires custom_stim_line_start/end"
    (s0, c0) = obj.custom_stim_line_start
    (s1, c1) = obj.custom_stim_line_end
    r0 = net.response(s0, c0, g); r1 = net.response(s1, c1, g)
    return r1 - r0

def angle_to_target(vec: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    return angle_deg(vec, target)
