#!/usr/bin/env python3
"""
Standalone script to plot Panel D from CSV data with custom styling.
Usage: python plot_panel_d_from_csv.py [input_csv] [output_dir]
"""

import sys
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def plot_panel_d_styled(csv_path: str, output_dir: str = "outputs/panel_d_custom"):
    """
    Plot Panel D with custom styling:
    - Color-axis line in orange
    - Shuffle mean line in grey
    - Y-axis range: -2 to 30 (allowing small negative space)
    - Large text for all labels
    - No title
    - Legend: "Color axis change" and "Mean of shuffle"
    """
    # Read CSV data
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for key in reader.fieldnames:
            data[key] = []
        for row in reader:
            for key in reader.fieldnames:
                try:
                    data[key].append(float(row[key]))
                except (ValueError, KeyError):
                    data[key].append(0.0)
    
    # Extract data
    xs = data['range']
    yc = data['impr_color_deg']
    
    # Check if we have shuffle data
    has_shuffle = 'impr_color_shuf_mean_or_single' in data
    
    # Create output directory
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with white background
    fig = plt.figure(figsize=(6.5, 5.5), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Plot color-axis line in orange
    plt.plot(xs, yc, marker="o", lw=3.0, color='darkorange', 
             markersize=8, label="Color axis change")
    
    # Plot shuffle data if available
    if has_shuffle:
        yc_m = data['impr_color_shuf_mean_or_single']
        yc_se = data.get('impr_color_shuf_sem', [0.0] * len(xs))
        
        # Plot shuffle mean in grey
        plt.plot(xs, yc_m, marker="^", lw=2.5, ls="--", color='grey', 
                 markersize=6, label="Mean of shuffle")
        
        # Add confidence interval if available
        if sum(yc_se) > 0:
            yc_lower = [m - 1.96 * se for m, se in zip(yc_m, yc_se)]
            yc_upper = [m + 1.96 * se for m, se in zip(yc_m, yc_se)]
            plt.fill_between(xs, yc_lower, yc_upper, 
                            alpha=0.15, color='grey', linewidth=0)
    
    # Reference line at y=0
    plt.axhline(0.0, color='k', lw=2.0, ls='--', alpha=0.7)
    
    # Set axis labels with large text
    plt.xlabel("Constraint range", fontsize=20, fontweight='bold')
    plt.ylabel("Δ angle-to-target (deg)", fontsize=20, fontweight='bold')
    
    # No title as requested
    
    # Set Y range from -2 to 30 (allowing small negative space)
    plt.ylim(-2.0, 30.0)
    
    # Customize tick labels with large font
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Add grid
    plt.grid(True, linestyle=':', alpha=0.5, color='lightgray')
    
    # Legend with larger font, no border
    plt.legend(fontsize=16, frameon=True, facecolor='white', edgecolor='none')
    
    # Adjust layout to prevent text cutoff
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    # Save in multiple formats
    for ext in ("png", "pdf", "svg"):
        output_path = outdir / f"panel_d_custom.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", 
                   transparent=True, facecolor='white', pad_inches=0.2)
        print(f"Saved: {output_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    # Default paths
    default_csv = "outputs/panel_d_data_good.csv"
    default_output = "outputs/panel_d_custom"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_csv
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output
    
    print(f"Reading data from: {csv_path}")
    print(f"Saving plots to: {output_dir}")
    
    plot_panel_d_styled(csv_path, output_dir)
    
    print("\nDone! Panel D has been generated with the custom styling.")

