# Figure Data Files

This directory contains all the data files needed to reproduce paper figures using the `reproduce_all_paper_figures.py` script.

## Data Files

### Panel B - 3D Neural State Representations
- **`panel_b_points.csv`** (99 rows) - PC coordinates for all data points
  - Columns: `state, pc1, pc2, pc3, shape, color`
  - Contains both color-attend and shape-attend states (49 points each)
  
- **`panel_b_axes.json`** - Axis vectors in PC space
  - Contains: target axis, color axes (color/shape states), shape axes (color/shape states)

### Panel C - Gain Modulation Analysis
- **`panel_c_gopt_data.csv`** (101 rows: 100 neurons + header)
  - Columns: `neuron, selectivity_shape_minus_color, g_color, g_shape, ratio_color_over_shape_gopt_adjusted`
  - Per-neuron feature selectivity and optimized gain ratios

### Panel D - Constraint Range Sweep
- **`panel_d_data.csv`** (14 rows: 13 range values + header)
  - Columns: `range, impr_color_deg, impr_shape_deg, impr_color_shuf_mean_or_single, impr_shape_shuf_mean_or_single, impr_color_shuf_sem, impr_shape_shuf_sem`
  - Range sweep results with shuffle control statistics

## Usage

Simply run the reproduction script from the parent directory:

```bash
python reproduce_all_paper_figures.py
```

The script will automatically find these data files and generate all figures in the `figs_paper/` directory.

## Standalone Package

This directory, along with `reproduce_all_paper_figures.py`, forms a complete standalone package. To share with collaborators:

1. Copy both:
   - `figs_data/` (this directory)
   - `reproduce_all_paper_figures.py`

2. Recipient runs: `python reproduce_all_paper_figures.py`

3. Figures are generated in: `figs_paper/`

No other dependencies or configuration files are needed!

## Data Origin

These files were generated from optimization results in:
- `outputs/paper_b/`
- `outputs/paper_c/`
- `outputs/paper_panel_d_triad_sweep/`

They represent the final optimized network states and analysis results used in the paper.
