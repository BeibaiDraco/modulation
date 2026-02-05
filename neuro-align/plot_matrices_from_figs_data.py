#!/usr/bin/env python3
"""
plot_matrices_from_figs_data.py

Read W_F, W_R, and readout/target vectors from figs_data/ and export publication-quality
plots (PNG/SVG/PDF) to figs_paper/.

Typical filenames expected in figs_data/ (example tag = "paper_b"):
    paper_b_weights_readout.npz   # preferred bundle: contains W_F, W_R, readout
    paper_b_W_F.npy               # fallback
    paper_b_W_R.npy               # fallback
    paper_b_readout.npy           # fallback

Usage
-----
# With defaults (no flags needed):
python plot_matrices_from_figs_data.py

# Or explicitly:
python plot_matrices_from_figs_data.py \
    --tag paper_b \
    --data-dir figs_data \
    --out-dir figs_paper \
    --formats png svg pdf \
    --dpi 300

Options
-------
--vmin/--vmax       : numeric range for heatmaps (both W_F & W_R). If omitted, auto.
--center 0          : center diverging normalization at 0. Requires matplotlib>=3.2.
--wf-collabels ...  : comma-separated labels for W_F columns (default for K=2 is shape,color).
--readout-style ... : "heatmap" (default) | "line" | "stem".
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from matplotlib.colors import TwoSlopeNorm
except Exception:
    TwoSlopeNorm = None  # gracefully handle old matplotlib


# ------------------------- I/O helpers ------------------------- #

def _maybe_load(path: Path):
    if path.suffix == ".npy" and path.exists():
        return np.load(path)
    if path.suffix == ".csv" and path.exists():
        return np.loadtxt(path, delimiter=",")
    return None


def load_mats(data_dir: Path, tag: str):
    """
    Load W_F, W_R, readout (t) from figs_data.
    Prefer the NPZ bundle; fall back to .npy (then .csv).
    """
    # 1) NPZ bundle
    npz_path = data_dir / f"{tag}_weights_readout.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        W_F = d["W_F"]
        W_R = d["W_R"]
        readout = d["readout"]
        return W_F, W_R, readout

    # 2) Separate files (.npy preferred, then .csv)
    WF = _maybe_load(data_dir / f"{tag}_W_F.npy")
    if WF is None:
        WF = _maybe_load(data_dir / f"{tag}_W_F.csv")
    if WF is None:
        raise FileNotFoundError(f"Missing {tag}_W_F.(npy|csv) in {data_dir}")

    WR = _maybe_load(data_dir / f"{tag}_W_R.npy")
    if WR is None:
        WR = _maybe_load(data_dir / f"{tag}_W_R.csv")
    if WR is None:
        raise FileNotFoundError(f"Missing {tag}_W_R.(npy|csv) in {data_dir}")

    t = _maybe_load(data_dir / f"{tag}_readout.npy")
    if t is None:
        t = _maybe_load(data_dir / f"{tag}_readout.csv")
    if t is None:
        # Some repos name it "paper_b_readout.npy" etc.; try that just in case.
        t = _maybe_load(data_dir / f"{tag}_readout_vector.npy")
    if t is None:
        raise FileNotFoundError(f"Missing {tag}_readout.(npy|csv) in {data_dir}")

    return WF, WR, t


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_all_formats(fig, outprefix: Path, formats, dpi=300):
    ensure_dir(outprefix.parent)
    if "png" in formats:
        fig.savefig(outprefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if "svg" in formats:
        fig.savefig(outprefix.with_suffix(".svg"), bbox_inches="tight")
    if "pdf" in formats:
        fig.savefig(outprefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------- Plotting ------------------------- #

def _add_colorbar(ax, im, pad=0.04, fraction=0.046):
    cbar = ax.figure.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    return cbar


def plot_WF(WF: np.ndarray, outprefix: Path, formats, dpi=300, vmin=None, vmax=None,
            center=None, collabels=None):
    N, K = WF.shape
    fig, ax = plt.subplots(figsize=(max(5, K * 1.8), max(4.5, N * 0.06)))
    norm = None
    if center is not None and TwoSlopeNorm is not None:
        vmin = vmin if vmin is not None else np.nanmin(WF)
        vmax = vmax if vmax is not None else np.nanmax(WF)
        norm = TwoSlopeNorm(vcenter=float(center), vmin=float(vmin), vmax=float(vmax))

    im = ax.imshow(WF, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, norm=norm)
    ax.set_title(r"Feedforward matrix $W_F$  (N×K)")
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Feature")
    if collabels is None and K == 2:
        collabels = ["shape", "color"]
    if collabels is not None and len(collabels) == K:
        ax.set_xticks(list(range(K)))
        ax.set_xticklabels(collabels)
    _add_colorbar(ax, im)
    save_all_formats(fig, outprefix, formats, dpi=dpi)


def plot_WR(WR: np.ndarray, outprefix: Path, formats, dpi=300, vmin=None, vmax=None,
            center=None):
    N, N2 = WR.shape
    assert N == N2, f"W_R must be square, got {WR.shape}"
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    norm = None
    if center is not None and TwoSlopeNorm is not None:
        vmin = vmin if vmin is not None else np.nanmin(WR)
        vmax = vmax if vmax is not None else np.nanmax(WR)
        norm = TwoSlopeNorm(vcenter=float(center), vmin=float(vmin), vmax=float(vmax))

    im = ax.imshow(WR, aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax, norm=norm)
    ax.set_title(r"Recurrent matrix $W_R$  (N×N)")
    ax.set_xlabel("Presynaptic neuron")
    ax.set_ylabel("Postsynaptic neuron")
    _add_colorbar(ax, im)
    save_all_formats(fig, outprefix, formats, dpi=dpi)


def plot_readout(t: np.ndarray, outprefix: Path, formats, dpi=300, style="heatmap"):
    t = t.ravel()
    N = t.size

    if style == "line":
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        ax.plot(np.arange(N), t, linewidth=1.6)
        ax.set_xlim(0, N - 1)
        ax.set.xlabel("Neuron index")
        ax.set_ylabel("Weight")
        ax.set_title(r"Readout / target axis $t$ (N)")
        ax.grid(alpha=0.25)
        save_all_formats(fig, outprefix, formats, dpi=dpi)
        return

    if style == "stem":
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        markerline, stemlines, baseline = ax.stem(np.arange(N), t, use_line_collection=True)
        ax.set_xlim(0, N - 1)
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Weight")
        ax.set_title(r"Readout / target axis $t$ (N)")
        ax.grid(alpha=0.25)
        save_all_formats(fig, outprefix, formats, dpi=dpi)
        return

    # default: heatmap (1 x N) for compactness
    fig, ax = plt.subplots(figsize=(10, 1.8))
    im = ax.imshow(t[np.newaxis, :], aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("Neuron index")
    ax.set_title(r"Readout / target axis $t$ (N)")
    _add_colorbar(ax, im, pad=0.28)
    save_all_formats(fig, outprefix, formats, dpi=dpi)


# ------------------------- Main CLI ------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description="Plot W_F, W_R, and readout from figs_data and export to figs_paper.")
    p.add_argument("--tag", default="paper_b", help="Tag prefix (default: paper_b)")
    p.add_argument("--data-dir", default="figs_data", help="Directory containing <tag>_*.npz/npy/csv")
    p.add_argument("--out-dir", default="figs_paper", help="Output directory for figures")
    p.add_argument("--formats", nargs="+", default=["png", "svg", "pdf"], help="Any of: png svg pdf")
    p.add_argument("--dpi", type=int, default=300, help="DPI for PNG export")

    # heatmap options
    p.add_argument("--vmin", type=float, default=None, help="Global vmin for heatmaps")
    p.add_argument("--vmax", type=float, default=None, help="Global vmax for heatmaps")
    p.add_argument("--center", type=float, default=None, help="Center value for diverging norm (requires TwoSlopeNorm)")

    # labels & style
    p.add_argument("--wf-collabels", type=str, default=None, help='Comma-separated labels for W_F columns (e.g., "shape,color")')
    p.add_argument("--readout-style", type=str, default="heatmap", choices=["heatmap", "line", "stem"],
                   help="Visualization style for readout vector")

    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    tag = args.tag

    try:
        WF, WR, t = load_mats(data_dir, tag)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    collabels = None
    if args.wf_collabels:
        collabels = [s.strip() for s in args.wf_collabels.split(",")]

    # Construct output prefixes
    fig_root = out_dir / tag
    outprefix_WF = fig_root / f"{tag}__WF_matrix"
    outprefix_WR = fig_root / f"{tag}__WR_matrix"
    outprefix_t  = fig_root / f"{tag}__readout_vector"

    # Plot
    plot_WF(WF, outprefix_WF, args.formats, dpi=args.dpi,
            vmin=args.vmin, vmax=args.vmax, center=args.center, collabels=collabels)
    plot_WR(WR, outprefix_WR, args.formats, dpi=args.dpi,
            vmin=args.vmin, vmax=args.vmax, center=args.center)
    plot_readout(t, outprefix_t, args.formats, dpi=args.dpi, style=args.readout_style)

    print(f"[OK] Saved to {fig_root} as: " + ", ".join(args.formats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
