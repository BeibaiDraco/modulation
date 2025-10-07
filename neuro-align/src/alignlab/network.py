from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve
from sklearn.decomposition import PCA
from .config import NetworkConfig, GridConfig, WRNormalization

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

class LinearRNN:
    """
    Linear recurrent population model with feedforward drive (W_F) and recurrent weights (W_R).
    r = (I - W_R)^(-1) (diag(g) @ W_F @ s), s = [shape, color]^T.
    """
    def __init__(self, cfg: NetworkConfig) -> None:
        self.cfg = cfg
        self.rng = _rng(cfg.seed)
        self.N = cfg.N
        self.K = cfg.K

        self.S = self._init_selectivity_matrix(self.N, self.K)
        self.W_F = self._init_W_F(self.S)
        self.W_R = self._init_W_R(
            self.N, cfg.p_high, cfg.p_low, self.S,
            wr_tuned=cfg.wr_tuned, weight_scale=cfg.weight_scale
        )
        if cfg.zero_sum != WRNormalization.NONE:
            self.W_R = self._enforce_zero_sums(self.W_R, mode=cfg.zero_sum)
        self.W_R = self._spectral_normalize(self.W_R, cfg.desired_radius)

        self.I = np.eye(self.N)
        self._lu = lu_factor(self.I - self.W_R)  # pre-factor
        self._wf_scales = np.ones(self.K, dtype=float)
        if getattr(self.cfg, "baseline_equalize", False):
            try:
                self.equalize_feature_columns(target_norm=1.0)
            except Exception as exc:
                self._baseline_equalize_error = str(exc)

    # ---------- initialization ----------
    def _init_selectivity_matrix(self, N:int, K:int) -> NDArray[np.float64]:
        assert K == 2, "Implementation assumes K=2 (shape,color)"
        half = N // 2
        S = np.zeros((N, K), dtype=float)
        # First half biased to shape, second half to color
        S[:half, 0] = self.rng.random(half) * 1.0
        S[:half, 1] = self.rng.random(half) * 0.5
        S[half:, 0] = self.rng.random(N - half) * 0.5
        S[half:, 1] = self.rng.random(N - half) * 1.0
        # small noise & clamp
        S += 0.02 * self.rng.standard_normal(S.shape)
        S = np.clip(S, 0.0, None)
        return S

    def _init_W_F(self, S: NDArray[np.float64]) -> NDArray[np.float64]:
        W_F = np.zeros_like(S)
        rowsums = S.sum(axis=1, keepdims=True)
        mask = rowsums.squeeze() > 0
        W_F[mask] = S[mask] / rowsums[mask]
        return W_F

    def _init_W_R(self, N:int, p_high:float, p_low:float, S:NDArray[np.float64],
                  wr_tuned:bool=False, weight_scale:float=0.1) -> NDArray[np.float64]:
        half = N // 2
        W = np.zeros((N, N), dtype=float)

        # within-group (shape-shape, color-color)
        mask_ss = self.rng.random((half, half)) < p_high
        W[:half, :half][mask_ss] = self.rng.random(np.sum(mask_ss)) * weight_scale
        mask_cc = self.rng.random((N - half, N - half)) < p_high
        W[half:, half:][mask_cc] = self.rng.random(np.sum(mask_cc)) * weight_scale

        # cross-group
        mask_sc = self.rng.random((half, N - half)) < p_low
        W[:half, half:][mask_sc] = self.rng.random(np.sum(mask_sc)) * weight_scale
        mask_cs = self.rng.random((N - half, half)) < p_low
        W[half:, :half][mask_cs] = self.rng.random(np.sum(mask_cs)) * weight_scale

        np.fill_diagonal(W, 0.0)

        if wr_tuned:
            # boost connections for similar selectivity (cosine similarity)
            norms = np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
            Sn = S / norms
            sim = Sn @ Sn.T
            W *= 1.0 + 0.5 * np.clip(sim, 0, None)
            np.fill_diagonal(W, 0.0)

        return W

    def _enforce_zero_sums(self, W: NDArray[np.float64], mode: WRNormalization) -> NDArray[np.float64]:
        W = W.copy()
        np.fill_diagonal(W, 0.0)

        # Row-sum zero on off-diagonals
        row_sums = W.sum(axis=1, keepdims=True)
        mu = row_sums / (self.N - 1)
        offdiag = ~np.eye(self.N, dtype=bool)
        W[offdiag] -= np.repeat(mu.ravel(), self.N - 1)
        np.fill_diagonal(W, 0.0)

        if mode == WRNormalization.ROW_AND_COL:
            # approximate column-sum zeroing while keeping diag zero
            col_sums = W.sum(axis=0, keepdims=True)
            mu_c = col_sums / (self.N - 1)
            W[offdiag] -= np.tile(mu_c.ravel(), self.N).reshape(self.N, self.N)[offdiag]
            np.fill_diagonal(W, 0.0)

        return W

    def _spectral_normalize(self, W: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals))
        if rho > 0:
            W = W * (radius / rho)
        return W

    def equalize_feature_columns(self, target_norm: float = 1.0) -> None:
        """
        Rescale W_F columns so baseline effective columns
        M = (I - W_R)^(-1) W_F have comparable L2 norms.
        """
        M = lu_solve(self._lu, self.W_F)
        norms = np.linalg.norm(M, axis=0) + 1e-12
        scales = (target_norm / norms).astype(float)
        self.W_F = self.W_F * scales[np.newaxis, :]
        self._wf_scales = scales

    # ---------- core computations ----------
    def response(self, shape_val: float, color_val: float,
                 g: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        if g is None:
            g = np.ones(self.N)
        stim = np.array([shape_val, color_val], dtype=float)
        WFeff = (g[:, None] * self.W_F)
        drive = WFeff @ stim
        r = lu_solve(self._lu, drive)
        return r

    def grid_responses(self, grid: GridConfig, g: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        Rs = []
        for sv in grid.shape_vals:
            for cv in grid.color_vals:
                Rs.append(self.response(sv, cv, g))
        return np.stack(Rs, axis=0)  # T x N

    def pca3(self, X: NDArray[np.float64]) -> Tuple[PCA, NDArray[np.float64]]:
        p = PCA(n_components=3, svd_solver="full", random_state=self.cfg.seed)
        Z = p.fit_transform(X)  # T x 3
        return p, Z

    # Axes in neuron space
    def color_axis(self, shape_const: float, g: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        r0 = self.response(shape_const, 0.0, g)
        r1 = self.response(shape_const, 1.0, g)
        return r1 - r0

    def shape_axis(self, color_const: float, g: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        r0 = self.response(0.0, color_const, g)
        r1 = self.response(1.0, color_const, g)
        return r1 - r0

def angle_deg(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """
    Un-directional angle between two axes in neuron space.
    Returns a value in [0, 90] degrees: min(theta, 180 - theta).
    """
    a = np.asarray(a); b = np.asarray(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cosv = float(np.dot(a, b) / (na * nb))
    cosv = np.clip(cosv, -1.0, 1.0)
    theta = float(np.degrees(np.arccos(cosv)))
    return float(min(theta, 180.0 - theta))
