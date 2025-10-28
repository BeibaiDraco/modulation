# Quick Start Guide - Paper Figures

## Summary

A **standalone** script has been created to reproduce all paper figures from CSV data files. The setup consists of:
- `figs_data/` - All source data files (CSV + JSON)
- `reproduce_all_paper_figures.py` - Standalone plotting script
- `figs_paper/` - Generated figures (created by the script)

## Quick Reproduce

To regenerate all figures at any time:

```bash
python reproduce_all_paper_figures.py
```

This will create/update the `figs_paper/` directory with all figures in PNG, PDF, and SVG formats.

## Standalone Package

To share with collaborators, simply provide:
1. `figs_data/` directory (4 data files)
2. `reproduce_all_paper_figures.py` script

No other dependencies or configuration needed! The script automatically locates data files relative to its location.

## Generated Files

### Panel B - 3D Neural State Representations
**Figures:**
- `panel_b_color.{png,pdf,svg}` - Color-attend state (circles)
- `panel_b_shape.{png,pdf,svg}` - Shape-attend state (triangles)  
- `panel_b_colormap.{png,pdf,svg}` - Bivariate color legend
- `panel_b_color_legend.{png,pdf,svg}` - Color state legend
- `panel_b_shape_legend.{png,pdf,svg}` - Shape state legend

**Data:**
- `panel_b_points.csv` (99 rows: 49 color + 49 shape state points)
- `panel_b_axes.json` (axis vectors in PC space)

### Panel C - Gain Modulation Analysis
**Figures:**
- `panel_c_gopt.{png,pdf,svg}` - Feature selectivity vs gain modulation

**Data:**
- `panel_c_gopt_data.csv` (101 rows: 100 neurons + header)

### Panel D - Constraint Range Sweep
**Figures:**
- `panel_d.{png,pdf,svg}` - Angle improvement vs constraint range

**Data:**
- `panel_d_data.csv` (14 rows: 13 range values + header)

## File Locations

### Standalone Package
```
neuro-align/
├── figs_data/              # Source data (standalone)
│   ├── panel_b_points.csv
│   ├── panel_b_axes.json
│   ├── panel_c_gopt_data.csv
│   └── panel_d_data.csv
├── reproduce_all_paper_figures.py  # Plotting script
└── figs_paper/             # Generated figures (output)
    ├── panel_b_*.{png,pdf,svg}
    ├── panel_c_*.{png,pdf,svg}
    └── panel_d.{png,pdf,svg}
```

### Original Source (for reference)
Data files in `figs_data/` were copied from:
- Panel B: `outputs/paper_b/`
- Panel C: `outputs/paper_c/`
- Panel D: `outputs/paper_panel_d_triad_sweep/`

## Key Features

✓ **Standalone**: Only needs `figs_data/` + script (no other files or configs)
✓ **Portable**: Share 2 items with collaborators - they can regenerate everything
✓ **Multiple formats**: PNG (raster, 300dpi), PDF (vector), SVG (vector)
✓ **Reproducible**: Run script anytime to regenerate from CSV
✓ **No dependencies**: Uses only numpy, matplotlib, and standard library (no pandas!)
✓ **Consistent styling**: Matches original plotting functions with correct view angles

## Technical Details

- Panel B uses bivariate color encoding: hue → color value, lightness → shape value
- Panel B 3D plots use viewing angle: **elevation=8°, azimuth=160°**
- Panel C x-axis: feature selectivity = `S[:, 1] - S[:, 0]` (color - shape)
- Panel C y-axis: gain modulation = `(g_color / g_shape) - 1.0` (centered at zero)
- Panel D uses square-root x-axis transformation: `sqrt(range + 1.0) - 1.0`

## CSV Data Format

**panel_b_points.csv:**
```
state,pc1,pc2,pc3,shape,color
color,-4.66,-1.81,2.37e-17,0.0,0.0
shape,1.92,-5.27,-0.11,0.0,0.0
...
```

**panel_c_gopt_data.csv:**
```
neuron,selectivity_shape_minus_color,g_color,g_shape,ratio_color_over_shape_gopt_adjusted
0,0.492,0.991,1.002,-0.011
...
```

**panel_d_data.csv:**
```
range,impr_color_deg,impr_shape_deg,impr_color_shuf_mean_or_single,impr_shape_shuf_mean_or_single,impr_color_shuf_sem,impr_shape_shuf_sem
0.02,1.42,2.30,0.31,0.75,0.028,0.052
...
```

## Next Steps

### For Collaborators
1. Receive `figs_data/` directory and `reproduce_all_paper_figures.py`
2. Run: `python reproduce_all_paper_figures.py`
3. Find all figures in the generated `figs_paper/` directory

### For Additional Analysis
- Data files are in `figs_data/` - use them for custom plots or further analysis
- Regenerate figures anytime by running the script
- All figures are publication-ready with proper styling and multiple format options

