# shuffle.py
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def assign_bins(sel: NDArray[np.float64], num_bins: int, binning: str = "quantile") -> NDArray[np.int32]:
    """
    Return an int32 array of bin indices in [0, num_bins-1], partitioning neurons by selectivity.
    Supports 'quantile' and 'equal' (equal-width) binning. Always covers the full range.
    """
    sel = np.asarray(sel).ravel()
    N = sel.size
    if num_bins <= 1:
        return np.zeros(N, dtype=np.int32)

    if binning == "quantile":
        qs = np.linspace(0, 1, num_bins + 1)
        edges = np.quantile(sel, qs)
    elif binning == "equal":
        lo, hi = float(np.min(sel)), float(np.max(sel))
        # if all equal, just one bin
        if np.isclose(hi, lo):
            return np.zeros(N, dtype=np.int32)
        edges = np.linspace(lo, hi, num_bins + 1)
    else:
        raise ValueError(f"Unknown binning: {binning}")

    # ensure everything falls into a bin
    edges = edges.astype(float)
    edges[0]  = -np.inf
    edges[-1] =  np.inf

    # bins in [0, num_bins-1]
    b = np.digitize(sel, edges, right=False) - 1
    b = np.clip(b, 0, num_bins - 1).astype(np.int32)
    return b

def shuffle_within_bins(g: NDArray[np.float64], bins: NDArray[np.int32], rng: np.random.Generator) -> NDArray[np.float64]:
    """
    Independently shuffle a single gain vector within each bin (preserves per-bin histogram).
    """
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
    """
    Shuffle two gain vectors by applying the SAME permutation within each bin
    (preserves pairing structure between g1 and g2 within strata).
    """
    g1 = np.asarray(g1).ravel(); g2 = np.asarray(g2).ravel()
    out1 = g1.copy(); out2 = g2.copy()
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if idx.size >= 2:
            perm = rng.permutation(idx.size)
            out1[idx] = g1[idx][perm]
            out2[idx] = g2[idx][perm]
    return out1, out2
