from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def assign_bins(sel: NDArray[np.float64], num_bins: int, binning: str = "quantile") -> NDArray[np.int32]:
    sel = np.asarray(sel).ravel()
    N = sel.size
    if num_bins <= 1:
        return np.zeros(N, dtype=np.int32)

    if binning == "quantile":
        qs = np.linspace(0, 1, num_bins + 1)
        edges = np.quantile(sel, qs)
        edges[0] = -np.inf
        edges[-1] = np.inf
    elif binning == "equal":
        lo, hi = float(sel.min()), float(sel.max())
        edges = np.linspace(lo, hi, num_bins + 1)
        edges[0] = -np.inf; edges[-1] = np.inf
    else:
        raise ValueError("binning must be 'quantile' or 'equal'")

    bins = np.digitize(sel, edges[1:-1], right=False)  # 0..num_bins-1
    return bins.astype(np.int32)

def shuffle_within_bins(g: NDArray[np.float64], bins: NDArray[np.int32], rng: np.random.Generator) -> NDArray[np.float64]:
    g = np.asarray(g).ravel()
    out = g.copy()
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if idx.size >= 2:
            perm = rng.permutation(idx.size)
            out[idx] = g[idx][perm]
    return out

def shuffle_pair_within_bins(
    g1: NDArray[np.float64], g2: NDArray[np.float64],
    bins: NDArray[np.int32], rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    g1 = np.asarray(g1).ravel(); g2 = np.asarray(g2).ravel()
    out1 = g1.copy(); out2 = g2.copy()
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if idx.size >= 2:
            perm = rng.permutation(idx.size)
            out1[idx] = g1[idx][perm]
            out2[idx] = g2[idx][perm]
    return out1, out2
