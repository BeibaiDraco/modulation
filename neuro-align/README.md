# neuro‑align — Alignment experiments for population codes

**neuro‑align** is a small, reproducible framework for studying how multiplicative **gain modulation** aligns population **feature axes** (e.g., color/shape) to a **target direction** in a low‑dimensional embedding (top 3 PCs from the unmodulated responses).

The package provides:

* **Single‑axis alignment** (align an axis of interest to a target)
* **Triad analysis**: *Unmodulated* | *Color‑attended* | *Shape‑attended*
* **Improvement‑to‑target metric** (undirectional angles, in [0°, 90°])
* **Shuffle tests** (bin‑wise shuffling of gains by selectivity, with mean ± 95% CI)
* Publication‑quality **figures** + **per‑figure CSV/JSON** for full reproducibility

> **Angles are orientation‑invariant everywhere**: axes are treated as *undirected* (angle θ is reported as min(θ, 180°−θ)).

---

## 1) Installation

**Requirements**

* Python ≥ 3.9 (tested with 3.10–3.12)
* `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `PyYAML`

**Install (editable):**

```bash
pip install -e .
```

This installs the `alignlab` package and enables commands like `python -m alignlab.cli ...`.

---

## 2) Repository layout

```
neuro-align/
├─ configs/                    # YAML configs (e.g., paper.yaml)
├─ outputs/                    # results (created automatically)
├─ reproduce_panels_from_csv.py# standalone reproducer (no model code needed)
├─ src/alignlab/
│  ├─ cli.py                   # command-line interface
│  ├─ config.py                # dataclasses & enums (network, objective, constraints, grid, shuffle)
│  ├─ constraints.py           # SLSQP bounds/constraints builders
│  ├─ network.py               # LinearRNN, responses, PCA, angle utilities
│  ├─ objectives.py            # target vectors & axis-of-interest helpers
│  ├─ optimize.py              # single-run, triad, and sweeps (incl. shuffle means/CI)
│  └─ plotting.py              # paper figures + data saving
└─ README.md
```

---

## 3) Quickstart (paper replication)

We provide a ready‑to‑run paper config: **`configs/paper.yaml`**. It defines the network, target, constraints, grid, shuffle, and default sweep ranges. 

> **No `--zlim` needed by default**. You can still pass `--elev/--azim` if you want a specific view.

### A) Triad (produces the 3‑panel figure + **panel_b** and **panel_c**)

```bash
python -m alignlab.cli triad \
  --config configs/paper.yaml \
  --paper-panels \
  --elev 25 --azim 135 \
  --show
```

**Outputs** (in `outputs/<tag>/`):

* `*_triad_threepanel.(png|pdf)` — Unmod | Color‑attend | Shape‑attend (3D)
* `panel_b_color.(png|pdf|svg)` — color-attend cloud (bold) + axes under color gains
* `panel_b_shape.(png|pdf|svg)` — shape-attend cloud (bold) + axes under shape gains
* `panel_c.(png|pdf|svg)` — activity-derived gain ratio (color ÷ shape)
* `panel_c_gopt.(png|pdf|svg)` — optimized gain ratio (color ÷ shape)
* `panel_b_points.csv`, `panel_b_axes.json`
* `panel_c_activity_data.csv`, `panel_c_gopt_data.csv`
* `*_triad_summary.json` (angles, norms, gains, constraints, shuffle details, etc.)
*Default view for panel B figures is elev=30°, azim=166° (the same look you get from `--elev 30 --azim 166`).*

### B) Triad‑sweep (produces **panel_d**)

```bash
python -m alignlab.cli triad-sweep \
  --config configs/paper.yaml \
  --paper-panels
```

**Outputs** (in `outputs/<tag>_triad_sweep/`):

* `panel_d_full.(png|pdf|svg)` — color + shape improvements vs range (with shuffled 95% CI if available)
* `panel_d.(png|pdf|svg)` — color-axis improvement vs range (with shuffled 95% CI if available)
* `panel_d_data.csv` — source data
* `*_triad_sweep.json` — rows for each range (includes improvements and shuffle stats)

---

## 4) What the figures show

### Panel B — Embedding & axes (3D, unmodulated PCA basis)

* **Dots** (two clouds): responses projected into the SAME PCA basis fitted on **unmodulated** responses.

  * ○ **Color‑attended** grid
  * △ **Shape‑attended** grid
    Dot color is **bivariate**:
  * **Hue** encodes *color* stimulus (0→1 wraps hue)
  * **Lightness** encodes *shape* stimulus (darker→lighter as shape increases)
* **Lines from origin**:

  * Target (crimson)
  * Color axis under color gains (darkorange, solid)
  * Color axis under shape gains (darkorange, dashed)
  * Shape axis under shape gains (seagreen, solid)
  * Shape axis under color gains (seagreen, dashed)

### Panel C — Gains vs selectivity

* **Activity-derived**: gain ratio from responses = |Δr_color| / |Δr_shape|
* **Optimized g**: gain ratio from parameters = g_color / g_shape
* **x**: neuron selectivity = (*shape − color*)
* **y**: gain ratio (color / shape), unity line at 1.0 for reference

### Panel D — Improvement‑to‑target vs constraint range

For each axis (color, shape), we compare alignment to the target under its **own** attention vs the **other** attention:

* Color‑axis improvement
  [
  \Delta_\text{color} =
  \angle(d_\text{color}(g_\text{shape}),\ t) - \angle(d_\text{color}(g_\text{color}),\ t)
  ]
* Shape‑axis improvement
  [
  \Delta_\text{shape} =
  \angle(d_\text{shape}(g_\text{color}),\ t) - \angle(d_\text{shape}(g_\text{shape}),\ t)
  ]

Angles are **undirectional** (in [0°, 90°]). **Positive** values mean “own attention helps that axis align better to the target.”
If shuffles are enabled with `repeats > 1`, we overlay dashed **shuffle mean** curves with shaded **95% CI**.

---

## 5) Standalone reproduction from CSV/JSON (no model code)

After you’ve generated the outputs once, readers can **reproduce the figures without running the model** by using the standalone script:

```bash
python reproduce_panels_from_csv.py \
  --triad-dir outputs/paper_main \
  --sweep-dir outputs/paper_main_triad_sweep \
  --out-dir outputs/paper_main/repro_from_csv \
  --elev 25 --azim 135 \
  --dpi 300 --transparent
```

This reads `panel_b_points.csv`, `panel_b_axes.json`, `panel_c_activity_data.csv`, `panel_c_gopt_data.csv`, and `panel_d_data.csv` and recreates **panel_b_color_repro.***, **panel_b_shape_repro.***, **panel_c_repro.***, **panel_c_gopt_repro.***, **panel_d_full_repro.***, and **panel_d_repro.***.
The script includes descriptions of what each panel shows for readers.

---

## 6) Configuration guide

Use `configs/paper.yaml` as a template for your experiments. Highlights: 

```yaml
tag: paper
save_dir: outputs

network:
  N: 120
  K: 2
  seed: 21
  desired_radius: 0.9
  p_high: 0.2
  p_low: 0.2
  zero_sum: row         # small PC1 regime (use "none" for dominant PC1)
  wr_tuned: false
  weight_scale: 0.1
  baseline_equalize: true  # rescale feedforward columns so baseline shape/color modulations match

objective:
  target_type: custom_pc # or pc1/pc2
  theta_deg: 45.0        # azimuth in PC1–PC2 plane
  phi_deg: 60.0          # elevation toward PC3
  axis_of_interest: color
  shape_for_color_line: 0.3  # slice for color axis (fix shape and vary color)
  color_for_shape_line: 0.6  # slice for shape axis (fix color and vary shape)

constraints:
  type: ball             # or box
  radius: 0.05           # L2 ball: ||g−1||₂ ≤ radius * sqrt(N)
  box_half_width: 0.2    # if type: box, each g_i ∈ [1−w, 1+w]
  hard_norm: true        # keep ||axis(g)|| equal to baseline ||axis(1)||
  positive_gains: false  # set true to force g_i ≥ 0

shuffle:
  enabled: true
  num_bins: 8
  binning: quantile      # or equal
  mode: independent      # or paired
  seed: 2025             # optional; default = network.seed + 101
  repeats: 50            # for panel D mean ± 95% CI

grid:
  shape_vals: [0.0, 0.1, ..., 1.0]
  color_vals: [0.0, 0.1, ..., 1.0]

sweep:
  ranges_ball: [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30]
  ranges_box:  [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
```

**Common customizations**

* **Dominant vs small PC1**: `network.zero_sum: none` (dominant PC1) vs `row` or `row_and_col` (small PC1).
* **Target direction**:

  * Use `target_type: pc1|pc2` for the principal axes directly.
  * Use `custom_pc` with `theta_deg` and `phi_deg` to set an arbitrary 3D direction.
* **Axis definition points**:

  * `shape_for_color_line`: fix **shape** here and vary **color** to define the **color** axis.
  * `color_for_shape_line`: fix **color** here and vary **shape** to define the **shape** axis.
* **Constraints**:

  * **Ball**: tune `radius` (fraction of √N).
  * **Box**: tune `box_half_width` (per‑neuron bounds around 1).
  * `hard_norm: true` enforces axis‑norm equality (pure rotations).
* **Shuffle test**:

  * `num_bins`, `binning` control binning of selectivity = (color − shape).
  * `mode: independent` shuffles each gain vector independently within bins; `paired` applies the same permutation to the (g_color, g_shape) pairs per bin.
  * `repeats` controls how many shuffled replicas to average (panel D mean ± 95% CI).

---

## 7) CLI reference (most relevant flags)

```bash
# Triad (3-panel; add paper panels)
python -m alignlab.cli triad --config <cfg.yaml> [--paper-panels] [--show] [--elev <deg>] [--azim <deg>]

# Triad sweep (range vs improvement; add panel_d)
python -m alignlab.cli triad-sweep --config <cfg.yaml> [--paper-panels]
```

* `--paper-panels` → write **panel_b/panel_c** (triad) or **panel_d** (sweep) + per‑figure CSV.
* `--show` + `--elev`/`--azim` → open an interactive 3D window and set a nice initial view.
  (No `--zlim` needed by default; pass it only if you want to fix PC3 scale.)

---

## 8) Reproducibility

* **Determinism**:

  * `network.seed` controls network initialization, PCA, etc.
  * `shuffle.seed` (or default = `network.seed + 101`) controls shuffling.
* PCA is **always fitted on unmodulated responses** and reused for all projections in a run (no basis mismatch).
* All figures are saved as **PNG (300 dpi)** and **vector PDF/SVG**.

---

## 9) Troubleshooting

* **`ModuleNotFoundError: alignlab`** → run `pip install -e .` from the repo root.
* **No interactive window** → set a GUI backend (`export MPLBACKEND=MacOSX`, or install `pyqt5` and use `QtAgg`).
* **Optimization feels stuck** → try relaxing constraints (larger ball radius / box width), or set `hard_norm: false` to diagnose feasibility.
* **Angles seem flipped** → all angles are **undirectional** by design (min(θ, 180°−θ)).
  If you need oriented angles for diagnostics only, compute them ad‑hoc (not used in paper).

---

## 10) Recreate figures from CSV only

See **`reproduce_panels_from_csv.py`** for a fully standalone reproducer that renders **panel_b/panel_c/panel_d** from the saved CSV/JSON without any model code.

```bash
python reproduce_panels_from_csv.py \
  --triad-dir outputs/<tag> \
  --sweep-dir outputs/<tag>_triad_sweep \
  --out-dir outputs/<tag>/repro_from_csv \
  --elev 25 --azim 135 --dpi 300 --transparent
```
