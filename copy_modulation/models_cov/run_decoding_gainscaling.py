import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import os

# --- Input File ---
input_filename = "simulation_results.npz" # Make sure this exists

# -------------------------------------------------
# Response Computation Functions
# -------------------------------------------------
def compute_response_noiseless(W_R, W_F, feature_val, g_vector):
    """Computes steady-state response for a single NOISELESS feature value."""
    N = W_R.shape[0]
    I = np.eye(N)
    if g_vector is None: g_vector = np.ones(N) # Should always be provided here
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
        except ValueError: return np.zeros(N)
    final_response = response.flatten()
    if final_response.shape != (N,): return np.zeros(N)
    return final_response

def compute_response_with_internal_noise(W_R, W_F, feature_val, g_vector, noise_internal_vec):
    """
    Computes steady-state response with internal neuronal noise.
    noise_internal_vec should be an N-dimensional vector.
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

    ff_input = G @ W_F * feature_val # Shape (N,1)
    
    # Add internal noise (must be reshaped to (N,1) for addition)
    if noise_internal_vec is None:
        noise_internal_reshaped = np.zeros((N,1))
    else:
        noise_internal_reshaped = noise_internal_vec.reshape(-1,1)

    total_input = ff_input + noise_internal_reshaped

    if inv_mat is None or total_input.shape[0] != N:
        return np.zeros(N)
    response = inv_mat @ total_input

    expected_shape = (N, 1)
    if response.shape != expected_shape:
        try: response = response.reshape(expected_shape)
        except ValueError: return np.zeros(N)
    final_response = response.flatten()
    if final_response.shape != (N,): return np.zeros(N)
    return final_response


# -------------------------------------------------
# Decoding Simulation Function (for Marlene's plots)
# -------------------------------------------------
def simulate_classification_accuracy_pc1noise(pc1_range_total, sigma_pc1, n_features=20, n_samples_per_feature=100):
    """
    Simulate classification accuracy based on noisy PC1 projection.
    sigma_pc1 is noise STDEV *in the PC1 projection dimension*.
    """
    if n_features <= 1: return 0.0
    if abs(pc1_range_total) < 1e-9: return 1.0 / n_features

    feature_levels = np.linspace(0, 1, n_features)
    mean_pc1_values_for_features = feature_levels * pc1_range_total

    correct_count = 0
    total_count = n_features * n_samples_per_feature

    for i, mean_pc1_target in enumerate(mean_pc1_values_for_features):
        noisy_pc1_samples = mean_pc1_target + np.random.normal(0.0, sigma_pc1, size=n_samples_per_feature)
        distances = np.abs(noisy_pc1_samples[:, np.newaxis] - mean_pc1_values_for_features)
        decoded_indices = np.argmin(distances, axis=1)
        correct_count += np.sum(decoded_indices == i)
    return correct_count / total_count

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
        data = np.load(input_filename, allow_pickle=True)
        W_R = data['W_R']
        W_F = data['W_F']
        noise_mean = data['noise_mean'] 
        noise_pcs_basis = data['noise_pcs_basis']
        g_aligned_loaded = data['g_aligned']       
        g_misaligned_loaded = data['g_misaligned'] 
        N = int(data['N'])
        feature_range = data['feature_range']
        num_stim_steps = int(data['num_stim_steps'])
        internal_noise_std_for_pca_setup = data['noise_std_original'] 
        print("Results loaded successfully.")
    except Exception as e:
        print(f"Error loading data from {input_filename}: {e}")
        exit()

    noise_pc1 = noise_pcs_basis[0, :] 

    # === Section for Marlene's Figure (Accuracy vs PC1 Noise & SNR) ===
    print("\n--- Setting up for Marlene's Figure (Accuracy vs PC1 Noise & SNR) ---")
    feature_vals_calib = np.linspace(feature_range[0], feature_range[1], num_stim_steps)
    X_calib = feature_vals_calib.reshape(-1, 1)
    global_gain_scales_marlene = [0.5, 1.0, 1.5, 2.0]
    pc1_ranges_marlene = {}
    g_baseline_ones = np.ones(N)

    print("Calibrating PC1 signal range for different global gain scales (Marlene's figure)...")
    for scale in global_gain_scales_marlene:
        current_g = g_baseline_ones * scale
        stim_responses_nl = np.array([compute_response_noiseless(W_R, W_F, f, current_g) for f in feature_vals_calib])
        stim_responses_centered_nl = stim_responses_nl - noise_mean
        stim_proj_pc1_nl = stim_responses_centered_nl @ noise_pc1
        decoder_fit = LinearRegression()
        decoder_fit.fit(X_calib, stim_proj_pc1_nl.reshape(-1, 1))
        pc1_ranges_marlene[scale] = decoder_fit.coef_[0, 0]
        print(f"  Global Gain Scale: {scale:.1f} -> PC1 Range (Slope): {pc1_ranges_marlene[scale]:.3f}")

    num_sigma_levels_marlene = 20
    ref_pc1_range_marlene = abs(pc1_ranges_marlene.get(1.0, 1.0))
    if ref_pc1_range_marlene < 1e-3: ref_pc1_range_marlene = 1.0
    sigma_pc1_test_levels_marlene = np.logspace(-2.5, 0, num_sigma_levels_marlene) * (ref_pc1_range_marlene / 2.0)
    sigma_pc1_test_levels_marlene = np.clip(sigma_pc1_test_levels_marlene, 0.005, ref_pc1_range_marlene * 2)
    sigma_pc1_test_levels_marlene = np.unique(sigma_pc1_test_levels_marlene)

    n_features_classify_marlene = 10
    n_trials_per_feature_marlene = 200
    all_accuracies_marlene = {}

    print(f"\nRunning decoding simulation for Marlene's figure across {len(sigma_pc1_test_levels_marlene)} PC1 noise levels...")
    for scale in global_gain_scales_marlene:
        current_pc1_range = pc1_ranges_marlene[scale]
        accuracies_for_current_gain = []
        for _, current_sigma_pc1 in enumerate(sigma_pc1_test_levels_marlene):
            acc = simulate_classification_accuracy_pc1noise(
                current_pc1_range, current_sigma_pc1, n_features_classify_marlene, n_trials_per_feature_marlene
            )
            accuracies_for_current_gain.append(acc)
        all_accuracies_marlene[scale] = np.array(accuracies_for_current_gain)
    
    # Plot 1 for Marlene: Accuracy vs. Sigma_PC1
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_marlene1, ax_marlene1 = plt.subplots(figsize=(8, 6))
    colors_marlene = plt.cm.coolwarm(np.linspace(0, 1, len(global_gain_scales_marlene)))
    for i, scale in enumerate(global_gain_scales_marlene):
        ax_marlene1.plot(sigma_pc1_test_levels_marlene, all_accuracies_marlene[scale] * 100,
                        marker='o', linestyle='-', markersize=5,
                        label=f'Global Gain Scale = {scale:.1f}', color=colors_marlene[i])
    ax_marlene1.set_xlabel('Noise Std Dev in PC1 Projection ($\sigma_{PC1}$)')
    ax_marlene1.set_ylabel('Decoding Accuracy (%)')
    ax_marlene1.set_title('Decoding Along Noise PC1 vs. PC1 Noise (Different Global Gains)')
    ax_marlene1.set_xscale('log'); ax_marlene1.grid(True, which="both", ls=":", alpha=0.7)
    ax_marlene1.legend(title="Global Gain Scaling"); ax_marlene1.set_ylim(bottom=0, top=105)
    plt.tight_layout(); plt.show()

    # Plot 2 for Marlene: Accuracy vs. SNR
    fig_marlene2, ax_marlene2 = plt.subplots(figsize=(8,6))
    for i, scale in enumerate(global_gain_scales_marlene):
        current_pc1_range = pc1_ranges_marlene[scale]
        valid_snr_indices = sigma_pc1_test_levels_marlene > 1e-9
        snr_values = np.abs(current_pc1_range) / sigma_pc1_test_levels_marlene[valid_snr_indices]
        sorted_indices = np.argsort(snr_values)
        ax_marlene2.plot(snr_values[sorted_indices], all_accuracies_marlene[scale][valid_snr_indices][sorted_indices] * 100,
                    marker='o', linestyle='-', markersize=5,
                    label=f'Global Gain Scale = {scale:.1f}', color=colors_marlene[i])
    ax_marlene2.set_xlabel('Effective SNR in PC1 ($\|PC1_{range}\| / \sigma_{PC1}$)')
    ax_marlene2.set_ylabel('Decoding Accuracy (%)'); ax_marlene2.set_title('Decoding Accuracy vs. SNR in PC1 Projection')
    ax_marlene2.set_xscale('log'); ax_marlene2.grid(True, which="both", ls=":", alpha=0.7)
    ax_marlene2.legend(title="Global Gain Scaling"); ax_marlene2.set_ylim(bottom=0, top=105)
    plt.tight_layout(); plt.show()


    # === Section for New Scatter Plot (Original Aligned vs. Misaligned with Global Gain Scaling) ===
    print("\n--- Setting up for New Scatter Plot (Optimized Aligned vs. Misaligned with Global Scaling) ---")
    
    global_gain_scales_scatter = [0.75, 1.0, 1.25] 
    n_features_classify_scatter = 10
    # Define feature_levels_discrete for this section
    feature_levels_discrete = np.linspace(feature_range[0], feature_range[1], n_features_classify_scatter) # <<< CORRECTION HERE

    n_samples_per_feature_scatter = 100 

    num_internal_noise_levels = 20 
    min_internal_noise = 0.01
    max_internal_noise = max(0.5, 2 * internal_noise_std_for_pca_setup) 
    if max_internal_noise <= min_internal_noise : max_internal_noise = min_internal_noise * 20
    
    internal_noise_std_levels_scatter = np.logspace(np.log10(min_internal_noise), np.log10(max_internal_noise), num_internal_noise_levels)
    print(f"Generated {len(internal_noise_std_levels_scatter)} internal neuronal noise std levels for scatter plot.")

    scatter_plot_data = []

    print(f"Running decoding simulation for new scatter plot...")
    start_scatter_time = time.time()

    for scale_idx, current_global_scale in enumerate(global_gain_scales_scatter):
        print(f"  Testing Global Gain Scale for Scatter: {current_global_scale:.2f}")
        
        g_aligned_scaled = g_aligned_loaded * current_global_scale
        g_misaligned_scaled = g_misaligned_loaded * current_global_scale

        mean_proj_scaled_aligned_per_feature = []
        mean_proj_scaled_misaligned_per_feature = []
        for f_val in feature_levels_discrete: # Now defined for this scope
            r_nl_al_sc = compute_response_noiseless(W_R, W_F, f_val, g_aligned_scaled)
            mean_proj_scaled_aligned_per_feature.append((r_nl_al_sc - noise_mean) @ noise_pc1)
            
            r_nl_mis_sc = compute_response_noiseless(W_R, W_F, f_val, g_misaligned_scaled)
            mean_proj_scaled_misaligned_per_feature.append((r_nl_mis_sc - noise_mean) @ noise_pc1)

        mean_proj_scaled_aligned_per_feature = np.array(mean_proj_scaled_aligned_per_feature)
        mean_proj_scaled_misaligned_per_feature = np.array(mean_proj_scaled_misaligned_per_feature)

        for internal_noise_std in internal_noise_std_levels_scatter:
            correct_scaled_aligned = 0
            correct_scaled_misaligned = 0
            total_trials_for_level = n_features_classify_scatter * n_samples_per_feature_scatter

            for i_feat, true_feature_val in enumerate(feature_levels_discrete): # Use defined feature_levels_discrete
                for _ in range(n_samples_per_feature_scatter):
                    internal_noise_vector = np.random.normal(0, internal_noise_std, size=N)

                    r_al_sc_noisy = compute_response_with_internal_noise(W_R, W_F, true_feature_val, g_aligned_scaled, internal_noise_vector)
                    proj_al_sc_noisy = (r_al_sc_noisy - noise_mean) @ noise_pc1
                    distances_al = np.abs(proj_al_sc_noisy - mean_proj_scaled_aligned_per_feature)
                    if len(distances_al)>0: 
                        decoded_idx_al = np.argmin(distances_al)
                        if decoded_idx_al == i_feat: correct_scaled_aligned += 1

                    r_mis_sc_noisy = compute_response_with_internal_noise(W_R, W_F, true_feature_val, g_misaligned_scaled, internal_noise_vector)
                    proj_mis_sc_noisy = (r_mis_sc_noisy - noise_mean) @ noise_pc1
                    distances_mis = np.abs(proj_mis_sc_noisy - mean_proj_scaled_misaligned_per_feature)
                    if len(distances_mis)>0:
                        decoded_idx_mis = np.argmin(distances_mis)
                        if decoded_idx_mis == i_feat: correct_scaled_misaligned += 1
            
            acc_scaled_aligned = correct_scaled_aligned / total_trials_for_level if total_trials_for_level > 0 else 0
            acc_scaled_misaligned = correct_scaled_misaligned / total_trials_for_level if total_trials_for_level > 0 else 0
            
            scatter_plot_data.append({
                'acc_misaligned': acc_scaled_misaligned * 100,
                'acc_aligned': acc_scaled_aligned * 100,
                'global_scale': current_global_scale,
                'internal_noise_std': internal_noise_std
            })

    print(f"Scatter plot simulation took {time.time() - start_scatter_time:.2f}s")

    # --- Visualize New Scatter Plot ---
    print("\n--- Visualizing New Scatter Plot (Scaled Aligned vs. Misaligned Accuracies) ---")
    fig_scatter_new, ax_scatter_new = plt.subplots(figsize=(9, 8))
    
    misaligned_accs_all = [d['acc_misaligned'] for d in scatter_plot_data]
    aligned_accs_all = [d['acc_aligned'] for d in scatter_plot_data]
    global_scales_all = [d['global_scale'] for d in scatter_plot_data]
    
    unique_scales = sorted(list(set(global_scales_all)))
    scale_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_scales)))
    scale_to_color_map = {scale_val: color for scale_val, color in zip(unique_scales, scale_colors)}
    point_colors = [scale_to_color_map[s] for s in global_scales_all]

    scatter_new = ax_scatter_new.scatter(misaligned_accs_all, aligned_accs_all,
                                         c=point_colors, 
                                         s=40, alpha=0.7, ec='black', lw=0.3, zorder=3)

    chance_level_scatter = 100.0 / n_features_classify_scatter
    ax_scatter_new.plot([chance_level_scatter, 100], [chance_level_scatter, 100],
                        color='gray', linestyle='--', label='Aligned Acc = Misaligned Acc', zorder=1)

    ax_scatter_new.set_xlabel("Decoding Accuracy (%) (Scaled Misaligned Gain)")
    ax_scatter_new.set_ylabel("Decoding Accuracy (%) (Scaled Aligned Gain)")
    ax_scatter_new.set_title(f"Decoding Accuracy: Scaled Optimized Gains ({n_features_classify_scatter} Features)")
    ax_scatter_new.set_xlim(left=min(chance_level_scatter - 5, np.min(misaligned_accs_all)-5 if misaligned_accs_all else chance_level_scatter-5), right=102)
    ax_scatter_new.set_ylim(bottom=min(chance_level_scatter - 5, np.min(aligned_accs_all)-5 if aligned_accs_all else chance_level_scatter-5), top=102)
    ax_scatter_new.grid(True, linestyle=':', alpha=0.7, zorder=0)
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Global Gain {s:.2f}',
                                  markerfacecolor=scale_to_color_map[s], markersize=8) for s in unique_scales]
    ax_scatter_new.legend(handles=legend_elements, title="Global Gain Scaling")

    plt.tight_layout()
    plt.show()

    print("\n--- Full Decoding Script Finished ---")
