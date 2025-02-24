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

    print("=== Simulation Results ===")
    print(f"Unmodulated PC1 range = {pc1_unmod:.4f}")
    print(f"Modulated   PC1 range = {pc1_mod:.4f}")
    print(f"Noise sigma = {sigma_noise:.2f}")
    print(f"Decoding accuracy (unmod) = {acc_unmod*100:.2f}%")
    print(f"Decoding accuracy (mod)   = {acc_mod*100:.2f}%")
