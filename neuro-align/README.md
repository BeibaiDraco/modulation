

# alignlab — Alignment Experiments for Population Codes

**alignlab** is a small, reproducible framework for running alignment experiments on a linear recurrent population model. It supports:

- **Single-axis alignment** (align Color/Shape axis to a target direction)
- **Triad analysis** (Unmodulated | Color-aligned | Shape-aligned)
- **Cross-attend metrics** (compare the *same* feature axis under two attention states)
- **Bin-based shuffle tests** (nulls that destroy gain–selectivity mapping within selectivity bins)
- **Sweeps** over constraint strength with publication-quality plots

All projections use a **single PCA basis** fitted on the **unmodulated** grid to avoid basis-mismatch artifacts.

------

## 0) Requirements

- Python 3.9+ (tested with 3.10/3.11/3.12)
- Packages (installed automatically via editable install):
  - `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `PyYAML`

> For interactive 3D rotation you need a GUI backend:
>
> - macOS: `export MPLBACKEND=MacOSX`
> - Cross-platform: install PyQt (`conda install pyqt`) then `export MPLBACKEND=QtAgg`
> - Or Tk: `conda install tk` then `export MPLBACKEND=TkAgg`

------

## 1) Install

```bash
# from the repo root
pip install -e .
```

This makes the `alignlab` package importable and enables the `python -m alignlab.cli ...` commands.

------

## 2) Project Layout

```
neuro-align/
├─ configs/                 # experiment configurations (YAML)
├─ outputs/                 # results (created automatically)
├─ src/alignlab/
│  ├─ __init__.py
│  ├─ cli.py               # command-line interface
│  ├─ config.py            # dataclasses & enums
│  ├─ constraints.py       # SLSQP bounds/constraints builders
│  ├─ network.py           # LinearRNN, responses, PCA, axes
│  ├─ objectives.py        # target vectors & axis helpers
│  ├─ optimize.py          # optimizers, triad & sweep logic
│  ├─ plotting.py          # 3D plots, sweep plots
│  └─ shuffle.py           # binning & within-bin shuffle
└─ pyproject.toml / README.md
```

------

## 3) Configuration (YAML)

Create a config in `configs/`. Example:

```yaml
# configs/smallpc1_ball.yaml
tag: smallpc1_ball
save_dir: outputs

network:
  N: 120
  K: 2
  seed: 21
  desired_radius: 0.9
  p_high: 0.2
  p_low: 0.2
  zero_sum: row         # {row, none, row_and_col} → 'row' ~ small PC1 regime
  wr_tuned: false
  weight_scale: 0.1

objective:
  target_type: custom_pc  # {pc1, pc2, custom_pc}
  theta_deg: 30.0         # for custom_pc: azimuth in PC1-PC2
  phi_deg: 25.0           # elevation towards PC3
  axis_of_interest: color # {color, shape, custom_line}
  shape_for_color_line: 0.3
  color_for_shape_line: 0.3
  # custom_stim_line_start/end required only if axis_of_interest: custom_line

constraints:
  type: ball              # {ball, box}
  radius: 0.12            # for ball: L2 radius fraction (absolute R = radius * sqrt(N))
  box_half_width: 0.2     # for box: half-width around g=1
  hard_norm: true         # enforce ||axis(g)|| == ||axis(1)||
  positive_gains: false   # set true to constrain g_i >= 0

grid:
  shape_vals: [0.0, 0.33, 0.66, 1.0]   # or use 11 points: [0.0, 0.1, ..., 1.0]
  color_vals: [0.0, 0.33, 0.66, 1.0]

shuffle:
  enabled: true
  num_bins: 10             # number of selectivity bins
  binning: quantile        # {quantile, equal}
  mode: independent        # {independent, paired}
  seed: 1234               # if omitted, uses network.seed + 101
```

### Key concepts

- **Network** (`network.py`):
  - Linear recurrent model with feedforward `W_F` built from a selectivity matrix `S`.
  - Recurrent `W_R` block structure; optional row/column zero-sum enforcement (`zero_sum`).
  - Spectral normalization to set a desired spectral radius.
- **Target** (`objectives.py`):
  - `pc1`, `pc2`, or `custom_pc` (spherical angles in 3D PC space).
  - Mapped back to neuron space using the unmodulated PCA components.
- **Axis of interest** (`color` / `shape` / `custom_line`):
  - `color`: response difference along color at fixed shape.
  - `shape`: response difference along shape at fixed color.
  - `custom_line`: response difference between two custom stimulus points.
- **Constraints** (`constraints.py`):
  - **Ball**: ( |g-1|_2 \le R), with (R = \text{radius}\cdot\sqrt{N}).
  - **Box**: ( g_i \in [1-w, 1+w]).
  - `hard_norm: true` keeps the feature-axis norm constant.
  - Optional `positive_gains` to force (g_i \ge 0).
- **Shuffle** (`shuffle.py`):
  - Bin neurons by selectivity ( \text{sel} = S_{:,1} - S_{:,0}).
  - Shuffle gains **within bins**:
    - `independent`: shuffle color and shape gains independently within each bin.
    - `paired`: apply the **same permutation** within each bin to ((g_\text{color}, g_\text{shape})) pairs.
  - Compute **cross-attend** angles again under shuffled gains.

> PCA is **always** fit on the **unmodulated** grid and reused (we never refit PCA on modulated data).

------

## 4) Commands (CLI)

All commands run as a module:

```bash
python -m alignlab.cli <subcommand> [flags]
```

### A) Single alignment run

```bash
python -m alignlab.cli run \
  --config configs/smallpc1_ball.yaml \
  --style original \
  --zlim -1 1 \
  --show --elev 25 --azim 135
```

- `--style original` → two-panel 3D scatter (Unmod | Mod) with your line overlays and colorbar.
- `--zlim ZMIN ZMAX` → fix z-axis limits (e.g., `-1 1`) without rescaling data.
- `--show` + `--elev/--azim` → open interactive window with initial camera angles.
- Outputs in `outputs/<tag>/`:
  - `<tag>_embedding_twopanel.(png|pdf)`
  - `<tag>_gains_vs_selectivity.(png|pdf)`
  - `<tag>_summary.json` (rich metrics; see **Outputs** below)

### B) Single-axis sweep (legacy)

```bash
python -m alignlab.cli sweep \
  --config configs/smallpc1_ball.yaml \
  --ranges 0.02 0.04 0.08 0.12 0.16
```

- Varies **constraint strength** (ball radius fraction or box half-width).
- Plots **Δ angle (axis moved)** vs range:
  - `<tag>_range_vs_degree.(png|pdf)`
  - `<tag>_sweep.json`

### C) Triad run (Color-attend & Shape-attend)

```bash
python -m alignlab.cli triad \
  --config configs/smallpc1_ball.yaml \
  --zlim -1 1 --show --elev 25 --azim 135 \
  --gcompare ratio        # or --gcompare diff ; add --logratio if gains are positive
```

- Fits PCA on **unmodulated**, then runs **two** optimizations:
  - Color-axis aligned → gains (g_\text{color})
  - Shape-axis aligned → gains (g_\text{shape})
- Three-panel plot: Unmod | Color-aligned | Shape-aligned
- Gains comparison: color vs shape (ratio or difference)
- If `shuffle.enabled: true`, the triad JSON **also** includes shuffled cross-attend angles.
- Outputs in `outputs/<tag>/`:
  - `<tag>_triad_threepanel.(png|pdf)`
  - `<tag>_gains_ratio_color_over_shape.(png|pdf)` (or `_diff_...` if `--gcompare diff`)
  - `<tag>_triad_summary.json`

### D) Triad sweep (Cross-attend angles vs range)

```bash
python -m alignlab.cli triad-sweep \
  --config configs/smallpc1_ball.yaml \
  --ranges 0.02 0.04 0.08 0.12 0.16
```

- For each range, runs the full triad and records:
  - **Color cross-attend angle**: angle between the **color axis** under color gains vs under shape gains.
  - **Shape cross-attend angle**: angle between the **shape axis** under color gains vs under shape gains.
  - If shuffle is enabled, also the **shuffled** versions of both angles.
- Outputs in `outputs/<tag>_triad_sweep/`:
  - `<tag>_triad_cross_sweep.(png|pdf)` (2 lines; **4 lines** if shuffle is enabled)
  - `<tag>_triad_sweep.json`

> **Note:** `triad-sweep` uses the **shuffle settings from YAML** (`shuffle:` block). Change them there.

------

## 5) Outputs & JSON fields

### Single run (`<tag>_summary.json`)

Key fields:

- `final_objective`, `iterations`, `func_evals`, `grad_evals`
- `angles_deg`:
  - `unmod_to_target`, `opt_to_target`, `delta_opt_vs_unmod`, `improvement` (pre − post)
- `axis_norms`:
  - `pre`, `post`, `equality_residual` (≈0 if `hard_norm: true`)
- `constraint_residuals`:
  - **ball**: `R_fraction`, `R_absolute`, `residual_l2`, `residual_sq`
  - **box**: `half_width`, `max_abs_violation`, counts at bounds
- `g_stats`: min/max/mean/std, ‖g−1‖₂, ‖g−1‖∞
- `pca.explained_variance_ratio` (top-3)
- Full `cfg` snapshot (enums flattened)

### Triad run (`<tag>_triad_summary.json`)

Adds:

- `color_alignment` and `shape_alignment` blocks (each has objective, iterations, angles, norms, constraint residuals, g-stats)
- `cross_attend`:
  - `color_axis_angle_deg` (angle between color axis under color gains vs under shape gains)
  - `shape_axis_angle_deg` (angle between shape axis under color gains vs under shape gains)
  - Norms under each attention state
- `gain_comparison`:
  - `ratio_color_over_shape` or `diff_color_minus_shape` (min/max/mean/median/std)
- `shuffle` (if enabled):
  - `mode`, `num_bins`, `binning`, `seed`
  - `cross_attend.color_axis_angle_deg` / `shape_axis_angle_deg` under shuffled gains
  - Axis norms under shuffled gains

### Triad sweep (`<tag>_triad_sweep.json`)

- One row per range:
  - `color_cross_attend_deg`, `shape_cross_attend_deg`
  - `color_cross_attend_deg_shuf`, `shape_cross_attend_deg_shuf` (if shuffle enabled)
  - `success_color`, `success_shape`
- Top-level `shuffle` block echoes the sweep settings used.

------

## 6) Practical Tips

- **PC basis sanity**: We always do `Z_opt = pca.transform(X_opt)` to project modulated responses into the **same** PCA basis; we never refit on modulated data.
- **Choosing ranges**:
  - Ball: start with `[0.02, 0.04, 0.08, 0.12, 0.16]`.
  - Box: start with `[0.05, 0.10, 0.15, 0.20]`.
- **Hard norm**: Keep it `true` to ensure improvements come from rotation, not scaling.
- **Positive gains**: Use `positive_gains: true` if you plan to use log-ratios for gains plots.
- **Grid density**: For publication figures, consider 11×11 grids (set both `shape_vals` and `color_vals` to `[0.0, 0.1, …, 1.0]`).
- **Dominant vs small PC1**: `network.zero_sum: none` usually yields dominant PC1; `row` (or `row_and_col`) tends to make PC1 & PC2 comparable.

------

## 7) Troubleshooting

- **`ModuleNotFoundError: alignlab`**
   Run `pip install -e .` from the repo root.

- **No interactive 3D window pops up**
   Set a GUI backend:

  ```bash
  export MPLBACKEND=MacOSX   # macOS
  # or:
  conda install pyqt
  export MPLBACKEND=QtAgg
  ```

  Then add `--show` to your CLI command.

- **Axis out of plane / looks wrong**
   Ensure the code path uses `pca.transform(...)` for modulated data (this repo does).

- **Enums from YAML**
   Accepted strings:

  - `network.zero_sum`: `row`, `none`, `row_and_col`
  - `objective.target_type`: `pc1`, `pc2`, `custom_pc`
  - `objective.axis_of_interest`: `color`, `shape`, `custom_line`
  - `constraints.type`: `ball`, `box`
  - `shuffle.binning`: `quantile`, `equal`
  - `shuffle.mode`: `independent`, `paired`

- **Optimizer stuck / not improving**
   Try loosening the constraint (bigger radius / box width), or disable `hard_norm` to check if scaling is the limiter. Confirm feasibility via `constraint_residuals`.

------

## 8) Reproducibility

- **Seeds**: The network, PCA, and (by default) the shuffle RNG are deterministic:
  - network/PCA: `network.seed`
  - shuffle: `shuffle.seed` (if omitted: `network.seed + 101`)
- Runs are fully determined by the YAML + command args.

------

## 9) Minimal Examples

**Color alignment (ball) with interactive plot**

```bash
python -m alignlab.cli run \
  --config configs/smallpc1_ball.yaml \
  --style original \
  --zlim -1 1 --show --elev 25 --azim 135
```

**Triad (3-panel) and gains ratio**

```bash
python -m alignlab.cli triad \
  --config configs/smallpc1_ball.yaml \
  --zlim -1 1 --show --elev 25 --azim 135 \
  --gcompare ratio
```

**Triad sweep (4 lines if shuffle enabled)**

```bash
python -m alignlab.cli triad-sweep \
  --config configs/smallpc1_ball.yaml \
  --ranges 0.02 0.04 0.08 0.12 0.16
```

------

