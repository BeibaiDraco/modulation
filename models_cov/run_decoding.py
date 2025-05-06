import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import os # Added for path handling

# --- Input File ---
input_filename = "simulation_results.npz"

# -------------------------------------------------
# Response Computation Function (Required for Calibration)
# -------------------------------------------------
# NOTE: This function MUST be consistent with the one used in the setup script
def compute_response(W_R, W_F, feature_val, g_vector=None):
    """
    Computes steady-state response for a single NOISELESS feature value.
    r = (I - G @ W_R)^-1 @ (G @ W_F @ feature_val)
    """
    N = W_R.shape[0]
    I = np.eye(N)
    if g_vector is None: g_vector = np.ones(N)
    G = np.diag(g_vector)
    inv_mat = None
    try:
        mat_to_invert = I - G @ W_R
        inv_mat = np.linalg.inv(mat_to_invert)
    except np.linalg.LinAlgError:
        # print(f"Warning: Matrix inversion failed. Returning zeros.") # Keep minimal
        return np.zeros(N)

    ff_input = G @ W_F * feature_val # Shape (N, 1)

    if inv_mat is None or ff_input.shape[0] != N:
        # print("Error: Problem with input shapes for final calculation.")
        return np.zeros(N)

    response = inv_mat @ ff_input # Expected shape (N, 1)

    expected_shape = (N, 1)
    if response.shape != expected_shape:
        # print(f"Warning: Unexpected response shape {response.shape}. Attempting reshape.")
        try: response = response.reshape(expected_shape)
        except ValueError:
            # print(f"Error: Cannot reshape. Returning zeros.")
            return np.zeros(N)
    final_response = response.flatten()
    if final_response.shape != (N,):
        # print(f"Error: Flattening failed. Returning zeros.")
        return np.zeros(N)
    return final_response

# -------------------------------------------------
# Decoding Simulation Function (Based on User Example)
# -------------------------------------------------
def simulate_classification_accuracy(pc1_range_total, sigma_pc1, n_features=20, n_samples_per_feature=100):
    """
    Simulate classification accuracy based on noisy PC1 projection.

    Args:
      pc1_range_total (float): Total difference in mean PC1 projection between feature=0 and feature=1.
                               (This corresponds to the 'slope' from linear regression).
      sigma_pc1 (float): Standard deviation of Gaussian noise *in the PC1 projection dimension*.
      n_features (int): Number of discrete feature levels to classify (e.g., 10, 20, 100).
      n_samples_per_feature (int): Number of noisy samples per feature level to test.

    Returns:
      accuracy (float): Average fraction of correct classifications (0 to 1).
    """
    if n_features <= 1:
        print("Warning: n_features must be > 1 for classification.")
        return 0.0
    if abs(pc1_range_total) < 1e-9:
        # If range is zero, performance is chance level
        # print("Warning: PC1 range is near zero. Cannot distinguish features.")
        return 1.0 / n_features # Chance level

    # Calculate the mean PC1 projection for each discrete feature level
    # Assume feature levels are evenly spaced from 0 to 1
    feature_levels = np.linspace(0, 1, n_features)
    # Calculate mean PC1 projection based on the total range (slope)
    # Assuming projection at feature=0 is the reference (intercept doesn't affect spacing)
    mean_pc1_values = feature_levels * pc1_range_total

    # Calculate the step size between mean PC1 values for adjacent features
    # pc1_step = pc1_range_total / (n_features - 1) # Not needed for nearest-mean

    correct_count = 0
    total_count = n_features * n_samples_per_feature

    for i, mean_pc1 in enumerate(mean_pc1_values):
        # Generate noisy PC1 projection samples for this feature level
        noisy_pc1_samples = mean_pc1 + np.random.normal(0.0, sigma_pc1, size=n_samples_per_feature)

        # --- Simple Nearest-Mean Decoder ---
        # Find the index of the mean PC1 value closest to each noisy sample
        # Expand dims for broadcasting: (n_samples, 1) vs (n_features,)
        distances = np.abs(noisy_pc1_samples[:, np.newaxis] - mean_pc1_values)
        decoded_indices = np.argmin(distances, axis=1)

        # Count how many samples are correctly classified (decoded index matches true index i)
        correct_count += np.sum(decoded_indices == i)

    accuracy = correct_count / total_count
    return accuracy

# -------------------------------------------------
# Main Decoding Logic
# -------------------------------------------------
if __name__ == "__main__":
    print(f"--- Loading Results from {input_filename} ---")
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        print("Please run 'run_setup_and_optimization.py' first.")
        exit()

    try:
        data = np.load(input_filename)
        # Load necessary variables
        W_R = data['W_R']
        W_F = data['W_F']
        noise_mean = data['noise_mean']
        noise_pcs_basis = data['noise_pcs_basis']
        g_aligned = data['g_aligned']
        g_misaligned = data['g_misaligned']
        N = int(data['N']) # Ensure N is integer
        feature_range = data['feature_range']
        noise_std_original = data['noise_std_original'] # Input noise std from setup
        num_stim_steps = int(data['num_stim_steps'])
        print("Results loaded successfully.")
    except Exception as e:
        print(f"Error loading data from {input_filename}: {e}")
        exit()

    # Extract noise PC1 for decoding
    noise_pc1 = noise_pcs_basis[0, :]

    print("\n--- 9. Setting up Decoding Task ---")

    # --- Recalculate Noiseless Projections for Calibration ---
    # This is needed to get the effective PC1 range (slope) for each condition
    print("Recalculating noiseless projections for decoder calibration...")
    feature_vals_stim = np.linspace(feature_range[0], feature_range[1], num_stim_steps)
    # Use the correct compute_response (only takes feature_val, g_vector)
    stim_responses_aligned_nl = np.array([compute_response(W_R, W_F, f, g_aligned) for f in feature_vals_stim])
    stim_responses_misaligned_nl = np.array([compute_response(W_R, W_F, f, g_misaligned) for f in feature_vals_stim])
    # Center using loaded noise_mean
    stim_responses_aligned_centered_nl = stim_responses_aligned_nl - noise_mean
    stim_responses_misaligned_centered_nl = stim_responses_misaligned_nl - noise_mean
    # Project onto loaded noise PC basis
    stim_proj_aligned_nl = stim_responses_aligned_centered_nl @ noise_pcs_basis.T
    stim_proj_misaligned_nl = stim_responses_misaligned_centered_nl @ noise_pcs_basis.T

    # --- Decoder Calibration (Get PC1 Range/Slope) ---
    X_feature = feature_vals_stim.reshape(-1, 1)
    y_proj_aligned = stim_proj_aligned_nl[:, 0].reshape(-1, 1) # Use only PC1 projection
    y_proj_misaligned = stim_proj_misaligned_nl[:, 0].reshape(-1, 1) # Use only PC1 projection

    decoder_aligned = LinearRegression()
    decoder_aligned.fit(X_feature, y_proj_aligned)
    pc1_range_aligned = decoder_aligned.coef_[0, 0] # Slope IS the range for feature [0,1]
    print(f"Decoder (Aligned):   PC1 Range (Slope) = {pc1_range_aligned:.3f}")

    decoder_misaligned = LinearRegression()
    decoder_misaligned.fit(X_feature, y_proj_misaligned)
    pc1_range_misaligned = decoder_misaligned.coef_[0, 0] # Slope IS the range
    print(f"Decoder (Misaligned): PC1 Range (Slope) = {pc1_range_misaligned:.3f}")

    # --- Simulation Parameters ---
    # **** MODIFIED NOISE LEVEL GENERATION ****
    num_noise_levels = 300 # User request
    min_sigma = 0.5      # User request: minimum noise std dev
    # Determine max sigma dynamically but ensure it's reasonably large
    min_pc1_range_abs = min(abs(pc1_range_aligned), abs(pc1_range_misaligned))
    if min_pc1_range_abs < 1e-6: min_pc1_range_abs = 1.0 # Avoid zero range if slopes are tiny
    max_sigma = max(0.75 * min_pc1_range_abs, 2.0)-1 # Max related to signal range, but at least 2.0

    # Define points to split the range for concentration
    # Split into 3 sections: low (25%), mid (50%), high (25%) of the points
    n_low = int(num_noise_levels * 0.25)
    n_high = int(num_noise_levels * 0.25)
    n_mid = num_noise_levels - n_low - n_high # Remaining points in the middle

    # Define range boundaries for concentration
    # e.g., concentrate points between 40% and 70% of the full range
    mid_range_low_boundary = 0.5#min_sigma + (max_sigma - min_sigma) * 0.05
    mid_range_high_boundary = 1.4#min_sigma + (max_sigma - min_sigma) * 0.95

    # Generate points in each section
    low_sigmas = np.linspace(min_sigma, mid_range_low_boundary, n_low, endpoint=False)
    mid_sigmas = np.linspace(mid_range_low_boundary, mid_range_high_boundary, n_mid, endpoint=False)
    high_sigmas = np.linspace(mid_range_high_boundary, max_sigma, n_high, endpoint=True) # Include max sigma

    # Combine and ensure uniqueness (in case boundaries overlap due to float precision)
    sigma_pc1_levels = np.unique(np.concatenate((low_sigmas, mid_sigmas, high_sigmas)))
    print(f"Generated {len(sigma_pc1_levels)} unique sigma_pc1 levels from {sigma_pc1_levels.min():.3f} to {sigma_pc1_levels.max():.3f}")
    # **** END MODIFIED SECTION ****

    n_features_classify = 10 # Number of discrete feature levels for classification task
    n_trials_per_feature = 200 # Trials per feature level per noise level

    accuracy_results_aligned = []
    accuracy_results_misaligned = []

    print(f"\nRunning decoding simulation for {len(sigma_pc1_levels)} PC1 noise levels...")
    print(f"Classifying into {n_features_classify} feature bins.")
    start_decoding_time = time.time()

    # --- Simulation Loops ---
    for i, current_sigma_pc1 in enumerate(sigma_pc1_levels):
        # Calculate accuracy for aligned condition
        acc_aligned = simulate_classification_accuracy(
            pc1_range_total=pc1_range_aligned,
            sigma_pc1=current_sigma_pc1,
            n_features=n_features_classify,
            n_samples_per_feature=n_trials_per_feature
        )
        accuracy_results_aligned.append(acc_aligned)

        # Calculate accuracy for misaligned condition
        acc_misaligned = simulate_classification_accuracy(
            pc1_range_total=pc1_range_misaligned,
            sigma_pc1=current_sigma_pc1,
            n_features=n_features_classify,
            n_samples_per_feature=n_trials_per_feature
        )
        accuracy_results_misaligned.append(acc_misaligned)

        # Update print to show progress
        print(f"  PC1 Noise Std: {current_sigma_pc1:.4f} -> Acc Aligned: {acc_aligned*100:.1f}%, Acc Misaligned: {acc_misaligned*100:.1f}%")

    print(f"Decoding simulation took {time.time() - start_decoding_time:.2f}s")

    # Convert results to numpy arrays for plotting
    accuracy_results_aligned = np.array(accuracy_results_aligned)
    accuracy_results_misaligned = np.array(accuracy_results_misaligned)

    # --- Visualize Decoding Performance ---
    print("\n--- 10. Visualizing Decoding Performance (Accuracy) ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_decoding, ax_decoding = plt.subplots(figsize=(8, 7))

    # Scatter plot: Misaligned Accuracy vs Aligned Accuracy
    scatter = ax_decoding.scatter(accuracy_results_misaligned * 100, accuracy_results_aligned * 100,
                                  c=sigma_pc1_levels, cmap='viridis_r', # Reversed viridis: high noise = purple
                                  s=60, zorder=3, ec='black', lw=0.5,
                                  norm=plt.matplotlib.colors.LogNorm()) # Use LogNorm for colorbar

    # Add diagonal line y=x for reference (chance level to 100%)
    chance_level = 100.0 / n_features_classify
    ax_decoding.plot([chance_level, 100], [chance_level, 100], color='gray', linestyle='--', label='Aligned Acc = Misaligned Acc', zorder=1)

    ax_decoding.set_xlabel("Decoding Accuracy (%) (Misaligned Gain)")
    ax_decoding.set_ylabel("Decoding Accuracy (%) (Aligned Gain)")
    ax_decoding.set_title(f"Decoding Accuracy Comparison ({n_features_classify} Features)")
    # Set limits from slightly below chance to 100
    ax_decoding.set_xlim(left=chance_level - 5, right=102)
    ax_decoding.set_ylim(bottom=chance_level - 5, top=102)
    ax_decoding.grid(True, linestyle=':', alpha=0.7, zorder=0)
    ax_decoding.legend()

    # Add colorbar
    cbar = fig_decoding.colorbar(scatter)
    cbar.set_label('Noise Std Dev in PC1 Projection ($\sigma_{PC1}$)')

    plt.tight_layout()
    plt.show()

    print("\n--- Decoding Script Finished ---")
