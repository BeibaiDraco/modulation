# Paper Figures - Reproduced from CSV Data

This directory contains all paper figures that have been reproduced from saved CSV data files.

## Directory Contents

### Panel B (3D scatter plots with bivariate color encoding)
- `panel_b_color.{png,pdf,svg}` - Color-attend state visualization
- `panel_b_shape.{png,pdf,svg}` - Shape-attend state visualization
- `panel_b_colormap.{png,pdf,svg}` - Bivariate color legend (color × shape)
- `panel_b_color_legend.{png,pdf,svg}` - Legend for color-attend state
- `panel_b_shape_legend.{png,pdf,svg}` - Legend for shape-attend state
- **Data files:**
  - `panel_b_points.csv` - PC coordinates for all data points (color and shape states)
  - `panel_b_axes.json` - Axis vectors in PC space (target, color, shape)

### Panel C (Gain modulation vs selectivity)
- `panel_c_gopt.{png,pdf,svg}` - Feature selectivity index vs gain modulation
- **Data files:**
  - `panel_c_gopt_data.csv` - Per-neuron selectivity and gain ratios

### Panel D (Constraint range sweep)
- `panel_d.{png,pdf,svg}` - Δ angle-to-target vs constraint range (color-axis)
- **Data files:**
  - `panel_d_data.csv` - Range sweep results with shuffle statistics

## Reproducing Figures

To reproduce all figures from the CSV data, run:

```bash
python reproduce_all_paper_figures.py
```

This script:
1. Reads CSV data from `outputs/paper_b`, `outputs/paper_c`, and `outputs/paper_panel_d_triad_sweep`
2. Recreates all figures with the same styling as the original
3. Saves figures in PNG, PDF, and SVG formats
4. Copies CSV data files to this directory for future reference

## Data Format

### panel_b_points.csv
Columns: `state,pc1,pc2,pc3,shape,color`
- `state`: "color" or "shape" (attention state)
- `pc1,pc2,pc3`: Principal component coordinates
- `shape,color`: Stimulus feature values (0.0 to 1.0)

### panel_b_axes.json
Contains PC-space vectors for:
- `target_pc`: Target axis
- `color_axis_color_pc`: Color axis in color-attend state
- `color_axis_shape_pc`: Color axis in shape-attend state
- `shape_axis_shape_pc`: Shape axis in shape-attend state
- `shape_axis_color_pc`: Shape axis in color-attend state

### panel_c_gopt_data.csv
Columns: `neuron,selectivity_shape_minus_color,g_color,g_shape,ratio_color_over_shape_gopt_adjusted`
- `neuron`: Neuron index
- `selectivity_shape_minus_color`: Feature selectivity (color - shape, note: column name is misleading)
- `g_color`: Optimized gain for color-attend
- `g_shape`: Optimized gain for shape-attend
- `ratio_color_over_shape_gopt_adjusted`: (g_color / g_shape) - 1.0

### panel_d_data.csv
Columns: `range,impr_color_deg,impr_shape_deg,impr_color_shuf_mean_or_single,impr_shape_shuf_mean_or_single,impr_color_shuf_sem,impr_shape_shuf_sem`
- `range`: Constraint range value
- `impr_color_deg`: Color-axis improvement (degrees)
- `impr_shape_deg`: Shape-axis improvement (degrees)
- `impr_color_shuf_mean_or_single`: Shuffle control mean
- `impr_shape_shuf_mean_or_single`: Shuffle control mean
- `impr_color_shuf_sem`: Standard error of shuffle control
- `impr_shape_shuf_sem`: Standard error of shuffle control

## Notes

- All figures use consistent styling with bold labels and appropriate font sizes
- Panel B 3D plots use **elevation=8°, azimuth=160°** viewing angle
- Color scheme follows the bivariate encoding: lightness for shape, hue for color
- Panel C x-axis: feature selectivity = color - shape (S[:, 1] - S[:, 0])
- Panel C y-axis: gain modulation = (g_color / g_shape) - 1.0
- Panel D uses a square-root transformation: `x_plot = sqrt(range + 1.0) - 1.0`

## Generated

This directory was created by `reproduce_all_paper_figures.py`.
All figures can be regenerated at any time by running the script again.
