import numpy as np

def simulate_decoding_accuracy(pc1_range, sigma=0.3, n_colors=100, n_samples=10000):
    """
    Simulate decoding accuracy when a linear decoder only has access to PC1.
    
    Args:
      pc1_range (float): The total PC1 difference between color=0 and color=1.
      sigma (float)    : Standard deviation of additive Gaussian noise.
      n_colors (int)   : Number of discrete color levels in [0..1).
      n_samples (int)  : Number of noise samples per color level to test.
    
    Returns:
      accuracy (float) : Average fraction of correct classifications.
    """
    step = pc1_range / (n_colors - 1)  # PC1 increment per color step
    correct_count = 0
    total_count = n_colors * n_samples

    for true_idx in range(n_colors):
        # Mean PC1 value for this color index:
        mean_pc1 = true_idx * step
        
        # Generate noisy samples for this color:
        noisy_samples = mean_pc1 + np.random.normal(0.0, sigma, size=n_samples)
        
        # Decode by rounding to nearest index in 0..(n_colors-1):
        decoded_idx = np.round(noisy_samples / step).astype(int)
        decoded_idx = np.clip(decoded_idx, 0, n_colors - 1)
        
        # Count how many samples are correct
        correct_count += np.sum(decoded_idx == true_idx)
    
    accuracy = correct_count / total_count
    return accuracy

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # From your data:
    pc1_unmod = 5.1751   # Unmodulated color-axis projection for color=1
    pc1_mod   = 5.7784   # Modulated   color-axis projection for color=1

    # Noise level:
    sigma_noise = 0.02

    # Simulation parameters:
    n_colors = 100
    n_samples_per_color = 10000

    # Run simulation for unmodulated scenario:
    acc_unmod = simulate_decoding_accuracy(
        pc1_range=pc1_unmod,
        sigma=sigma_noise,
        n_colors=n_colors,
        n_samples=n_samples_per_color
    )

    # Run simulation for modulated scenario:
    acc_mod = simulate_decoding_accuracy(
        pc1_range=pc1_mod,
        sigma=sigma_noise,
        n_colors=n_colors,
        n_samples=n_samples_per_color
    )

    # Print results
    print("=== Simulation Results ===")
    print(f"Unmodulated PC1 range = {pc1_unmod:.4f}")
    print(f"Modulated   PC1 range = {pc1_mod:.4f}")
    print(f"Noise sigma = {sigma_noise:.2f}")
    print(f"Decoding accuracy (unmod) = {acc_unmod*100:.2f}%")
    print(f"Decoding accuracy (mod)   = {acc_mod*100:.2f}%")
    
    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Bar chart comparing accuracies
    conditions = ['Unmodulated', 'Modulated']
    accuracies = [acc_unmod*100, acc_mod*100]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(conditions, accuracies, color=colors, width=0.6)
    ax1.set_ylabel('Decoding Accuracy (%)')
    ax1.set_title('Decoding Accuracy Comparison')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom')
    
    # Plot 2: Visualization of PC1 ranges and noise
    x_range = np.linspace(0, max(pc1_mod, pc1_unmod) + 3*sigma_noise, 1000)
    
    # Plot distributions for a few color points
    color_indices = [0, 49, 99]  # First, middle, and last color
    
    for i, condition in enumerate([pc1_unmod, pc1_mod]):
        color = colors[i]
        label = conditions[i]
        step = condition / (n_colors - 1)
        
        for idx in color_indices:
            mean = idx * step
            y = np.exp(-0.5 * ((x_range - mean) / sigma_noise)**2) / (sigma_noise * np.sqrt(2 * np.pi))
            
            # Scale for visibility
            y = y * 0.8 if i == 0 else y * 0.6
            
            # Offset the second condition's distributions for clarity
            offset = 1.0 if i == 0 else 2.0
            ax2.plot(x_range, y + offset, color=color, alpha=0.7 if idx == 49 else 0.4)
            
            # Only add legend for the middle distribution
            if idx == 49:
                ax2.plot(x_range, y + offset, color=color, label=label)
    
    # Add annotations
    ax2.annotate('PC1 Range\nUnmodulated', xy=(pc1_unmod/2, 3.3), 
                xytext=(pc1_unmod/2, 3.5), ha='center',
                arrowprops=dict(arrowstyle='->x'))
    
    ax2.annotate('PC1 Range\nModulated', xy=(pc1_mod/2, 2.3), 
                xytext=(pc1_mod/2, 2.5), ha='center',
                arrowprops=dict(arrowstyle='->x'))
    
    # Add rectangles showing the ranges
    ax2.add_patch(Rectangle((0, 0.9), pc1_unmod, 0.2, color=colors[0], alpha=0.3))
    ax2.add_patch(Rectangle((0, 0.6), pc1_mod, 0.2, color=colors[1], alpha=0.3))
    
    ax2.set_xlabel('PC1 Value')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('PC1 Distributions with Noise')
    ax2.legend()
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.show()
