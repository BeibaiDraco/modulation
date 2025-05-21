import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import os

# --- Input File ---
input_filename = "simulation_results.npz"

# -------------------------------------------------
# Response Computation Function (Required for Calibration)
# -------------------------------------------------
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
        return np.zeros(N)

    ff_input = G @ W_F * feature_val

    if inv_mat is None or ff_input.shape[0] != N:
        return np.zeros(N)

    response = inv_mat @ ff_input

    expected_shape = (N, 1)
    if response.shape != expected_shape:
        try: response = response.reshape(expected_shape)
        except ValueError:
            return np.zeros(N)
    final_response = response.flatten()
    if final_response.shape != (N,):
        return np.zeros(N)
    return final_response

# -------------------------------------------------
# Decoding Simulation Function
# -------------------------------------------------
def simulate_classification_accuracy(pc1_range_total, sigma_pc1, n_features=20, n_samples_per_feature=100):
    """
    Simulate classification accuracy based on noisy PC1 projection.
    """
    if n_features <= 1:
        print("Warning: n_features must be > 1 for classification.")
        return 0.0
    if abs(pc1_range_total) < 1e-9:
        return 1.0 / n_features

    feature_levels = np.linspace(0, 1, n_features)
    mean_pc1_values = feature_levels * pc1_range_total

    correct_count = 0
    total_count = n_features * n_samples_per_feature

    for i, mean_pc1 in enumerate(mean_pc1_values):
        noisy_pc1_samples = mean_pc1 + np.random.normal(0.0, sigma_pc1, size=n_samples_per_feature)
        distances = np.abs(noisy_pc1_samples[:, np.newaxis] - mean_pc1_values)
        decoded_indices = np.argmin(distances, axis=1)
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
        W_R = data['W_R']
        W_F = data['W_F']
        noise_mean = data['noise_mean']
        noise_pcs_basis = data['noise_pcs_basis']
        g_aligned = data['g_aligned']
        g_misaligned = data['g_misaligned']
        N = int(data['N'])
        feature_range = data['feature_range']
        noise_std_original = data['noise_std_original']
        num_stim_steps = int(data['num_stim_steps'])
        print("Results loaded successfully.")
    except Exception as e:
        print(f"Error loading data from {input_filename}: {e}")
        exit()

    # Define gain scaling factors and their corresponding colormaps
    gain_scales = [ 1.0, 1.5, 2.0, 2.5]
    colormaps = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges']
    
    # Extract noise PC1 for decoding
    noise_pc1 = noise_pcs_basis[0, :]

    print("\n--- Setting up Decoding Task with Multiple Gain Scales ---")

    # Create figure for plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_decoding, ax_decoding = plt.subplots(figsize=(10, 8))

    # Simulation parameters
    num_noise_levels = 300
    min_sigma = 0.5
    n_features_classify = 10
    n_trials_per_feature = 200

    # Generate base noise levels
    n_low = int(num_noise_levels * 0.25)
    n_high = int(num_noise_levels * 0.25)
    n_mid = num_noise_levels - n_low - n_high

    mid_range_low_boundary = 0.5
    mid_range_high_boundary = 2.0

    low_sigmas = np.linspace(min_sigma, mid_range_low_boundary, n_low, endpoint=False)
    mid_sigmas = np.linspace(mid_range_low_boundary, mid_range_high_boundary, n_mid, endpoint=False)
    high_sigmas = np.linspace(mid_range_high_boundary, 3.0, n_high, endpoint=True)

    base_sigma_pc1_levels = np.unique(np.concatenate((low_sigmas, mid_sigmas, high_sigmas)))

    # Run simulation for each gain scale
    for scale_idx, (gain_scale, cmap_name) in enumerate(zip(gain_scales, colormaps)):
        print(f"\nProcessing gain scale: {gain_scale}")
        
        # Scale the gains
        g_aligned_scaled = g_aligned * gain_scale
        g_misaligned_scaled = g_misaligned * gain_scale

        # Recalculate noiseless projections
        feature_vals_stim = np.linspace(feature_range[0], feature_range[1], num_stim_steps)
        stim_responses_aligned_nl = np.array([compute_response(W_R, W_F, f, g_aligned_scaled) for f in feature_vals_stim])
        stim_responses_misaligned_nl = np.array([compute_response(W_R, W_F, f, g_misaligned_scaled) for f in feature_vals_stim])

        # Center and project
        stim_responses_aligned_centered_nl = stim_responses_aligned_nl - noise_mean
        stim_responses_misaligned_centered_nl = stim_responses_misaligned_nl - noise_mean
        stim_proj_aligned_nl = stim_responses_aligned_centered_nl @ noise_pcs_basis.T
        stim_proj_misaligned_nl = stim_responses_misaligned_centered_nl @ noise_pcs_basis.T

        # Calibrate decoders
        X_feature = feature_vals_stim.reshape(-1, 1)
        y_proj_aligned = stim_proj_aligned_nl[:, 0].reshape(-1, 1)
        y_proj_misaligned = stim_proj_misaligned_nl[:, 0].reshape(-1, 1)

        decoder_aligned = LinearRegression()
        decoder_aligned.fit(X_feature, y_proj_aligned)
        pc1_range_aligned = decoder_aligned.coef_[0, 0]

        decoder_misaligned = LinearRegression()
        decoder_misaligned.fit(X_feature, y_proj_misaligned)
        pc1_range_misaligned = decoder_misaligned.coef_[0, 0]

        # Scale noise levels based on the gain scale
        sigma_pc1_levels = base_sigma_pc1_levels * gain_scale

        # Run decoding simulation
        accuracy_results_aligned = []
        accuracy_results_misaligned = []

        for current_sigma_pc1 in sigma_pc1_levels:
            acc_aligned = simulate_classification_accuracy(
                pc1_range_total=pc1_range_aligned,
                sigma_pc1=current_sigma_pc1,
                n_features=n_features_classify,
                n_samples_per_feature=n_trials_per_feature
            )
            accuracy_results_aligned.append(acc_aligned)

            acc_misaligned = simulate_classification_accuracy(
                pc1_range_total=pc1_range_misaligned,
                sigma_pc1=current_sigma_pc1,
                n_features=n_features_classify,
                n_samples_per_feature=n_trials_per_feature
            )
            accuracy_results_misaligned.append(acc_misaligned)

        # Plot results for this gain scale
        scatter = ax_decoding.scatter(
            np.array(accuracy_results_misaligned) * 100,
            np.array(accuracy_results_aligned) * 100,
            c=sigma_pc1_levels,
            cmap=cmap_name,
            s=60,
            zorder=3,
            ec='black',
            lw=0.5,
            alpha=0.6,
            norm=plt.matplotlib.colors.LogNorm(),
            label=f'Gain Scale: {gain_scale}'
        )

    # Add diagonal line
    chance_level = 100.0 / n_features_classify
    ax_decoding.plot([chance_level, 100], [chance_level, 100], color='gray', linestyle='--', label='y=x', zorder=1)

    ax_decoding.set_xlabel("Least Aligned Decoding Accuracy (%)")
    ax_decoding.set_ylabel("Most Aligned Decoding Accuracy (%)")
    ax_decoding.set_title("Decoding Accuracy (%) Comparison Across Gain Scales")
    ax_decoding.set_xlim(left=chance_level - 5, right=102)
    ax_decoding.set_ylim(bottom=chance_level - 5, top=102)
    ax_decoding.grid(True, linestyle=':', alpha=0.7, zorder=0)
    ax_decoding.legend()

    # Add colorbar
    cbar = fig_decoding.colorbar(scatter)
    cbar.set_label('Noise Std Dev in PC1 Projection')

    plt.tight_layout()
    plt.show()

    print("\n--- Decoding Script Finished ---") 