import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize, Bounds
import time
import os # Added for path handling

# -------------------------------------------------
# Configuration (Keep relevant parts)
# -------------------------------------------------
np.random.seed(42) # Seed for reproducibility
N = 100           # Total number of neurons
K = 1             # Dimensionality of feature space (now 1D)
frac_E = 0.8       # Fraction of excitatory neurons
N_E = int(N * frac_E)
N_I = N - N_E

# --- Feature Tuning Params ---
feature_range = (0, 1) # Range of the 1D feature
tuning_sigma_W_R = 0.15  # Width of Gaussian tuning for W_R connections
tuning_sigma_W_F = 0.1   # Width of Gaussian tuning for W_F (how broadly neuron responds)

# --- W_R Connection Params ---
# Base connection strengths (before tuning) - adjust these as needed
J_EE_base = 0.8 / N_E # Scaled by number of presynaptic E cells
J_IE_base = 0.6 / N_E # E -> I
J_EI_base = -1.0 / N_I # I -> E (Inhibitory)
J_II_base = -0.8 / N_I # I -> I (Inhibitory)
desired_radius_W_R = 0.9 # Spectral radius for stability

# --- Noise Params ---
num_noise_trials = 500
noise_std = 0.2 # Standard deviation of internal noise added

# --- Optimization Params ---
gain_bounds = (0.5, 1.5) # Bounds for individual gain elements in g
max_opt_iter = 150      # Max iterations for optimization

# --- Visualization Params ---
num_stim_steps = 21 # Number of feature steps to show in stimulus trajectory

# --- Output File ---
output_filename = "simulation_results.npz"

# -------------------------------------------------
# 1) Initialization Functions (Copied from original)
# -------------------------------------------------
def initialize_tuning_and_types(N, N_E, feature_range):
    """Initializes neuron types (E/I) and feature preferences."""
    neuron_types = np.array(['E'] * N_E + ['I'] * (N - N_E))
    np.random.shuffle(neuron_types) # Randomly assign E/I status
    prefs = np.linspace(feature_range[0], feature_range[1], N)
    np.random.shuffle(prefs) # Randomly assign preferences
    is_E = (neuron_types == 'E')
    is_I = (neuron_types == 'I')
    return prefs, is_E, is_I

def initialize_W_F(N, K, prefs, tuning_sigma_W_F, feature_range):
    """Initializes feedforward weights W_F."""
    W_F = np.zeros((N, K))  # Shape (N, 1)
    feature_center = np.mean(feature_range)
    for i in range(N):
        # Create Gaussian tuning curve centered at neuron's preferred feature
        pref_distance = prefs[i] - feature_center
        tuning_factor = np.exp(-(pref_distance**2) / (2 * tuning_sigma_W_F**2))
        W_F[i, :] = tuning_factor
    return W_F

def initialize_W_R_EI(N, prefs, is_E, is_I, J_base, sigma, desired_radius):
    """Initializes feature-tuned W_R with E/I structure."""
    W_R = np.zeros((N, N))
    J_EE, J_IE, J_EI, J_II = J_base # Unpack base strengths
    for i in range(N):
        for j in range(N):
            if i == j: continue
            pref_diff = prefs[i] - prefs[j]
            # Using Gaussian tuning factor
            tuning_factor = pref_diff
            # Assign weight based on types
            if is_E[j] and is_E[i]:    W_R[i, j] = J_EE * tuning_factor
            elif is_E[j] and is_I[i]:  W_R[i, j] = J_IE * tuning_factor
            elif is_I[j] and is_E[i]:  W_R[i, j] = J_EI * tuning_factor
            elif is_I[j] and is_I[i]:  W_R[i, j] = J_II * tuning_factor
    # Rescale W_R
    try:
        eivals = np.linalg.eigvals(W_R)
        max_ev = np.max(np.abs(eivals))
        if max_ev > 1e-9:
            W_R *= (desired_radius / max_ev)
            print(f"W_R scaled. Initial max eigenvalue abs: {max_ev:.4f}, Target radius: {desired_radius}")
        else:
            print("W_R max eigenvalue is zero or close to zero. No scaling applied.")
    except np.linalg.LinAlgError:
        print("Warning: Eigendecomposition failed for W_R. Skipping scaling.")
    return W_R

# -------------------------------------------------
# 2) Response Computation Functions (Copied)
# -------------------------------------------------
def compute_response(W_R, W_F, feature_val, g_vector=None, noise_internal=None):
    """Computes steady-state response for a single feature value."""
    N = W_R.shape[0]
    I = np.eye(N)
    if g_vector is None: g_vector = np.ones(N)
    G = np.diag(g_vector)
    inv_mat = None
    try:
        mat_to_invert = I - G @ W_R
        inv_mat = np.linalg.inv(mat_to_invert)
    except np.linalg.LinAlgError:
        print(f"Warning: Matrix inversion failed. Returning zeros.")
        return np.zeros(N)
    ff_input = G @ W_F * feature_val
    if noise_internal is None: noise_internal = np.zeros((N, 1))
    else: noise_internal = noise_internal.reshape(-1, 1)
    total_input = ff_input + noise_internal
    if inv_mat is None or total_input.shape[0] != N:
         print("Error: Problem with input shapes for final calculation.")
         return np.zeros(N)
    response = inv_mat @ total_input
    expected_shape = (N, 1)
    if response.shape != expected_shape:
         print(f"Warning: Unexpected response shape {response.shape}. Attempting reshape.")
         try: response = response.reshape(expected_shape)
         except ValueError:
             print(f"Error: Cannot reshape. Returning zeros.")
             return np.zeros(N)
    final_response = response.flatten()
    if final_response.shape != (N,):
        print(f"Error: Flattening failed. Returning zeros.")
        return np.zeros(N)
    return final_response

# -------------------------------------------------
# 3) Noise Response & PCA (Copied)
# -------------------------------------------------
def get_noise_responses_and_pcs(W_R, W_F, num_trials, noise_std, g_vector=None):
    """Generates noise responses and performs PCA."""
    N = W_R.shape[0]
    if g_vector is None: g_vector = np.ones(N)
    noise_responses = np.zeros((num_trials, N))
    print(f"Computing {num_trials} noise responses...")
    for i in range(num_trials):
        noise_internal = np.random.normal(0, noise_std, size=N)
        noise_responses[i, :] = compute_response(W_R, W_F, 0.0, g_vector, noise_internal)
    print("Performing PCA on noise responses...")
    pca_noise = PCA(n_components=2)
    noise_responses_centered = noise_responses - np.mean(noise_responses, axis=0)
    pca_noise.fit(noise_responses_centered)
    noise_pcs_basis = pca_noise.components_
    explained_variance = pca_noise.explained_variance_ratio_
    print(f"Noise variance explained by PC1: {explained_variance[0]:.3f}")
    print(f"Noise variance explained by PC2: {explained_variance[1]:.3f}")
    # Return the mean as well, needed for centering later
    noise_mean = np.mean(noise_responses, axis=0)
    return noise_responses_centered, noise_pcs_basis, noise_mean

# -------------------------------------------------
# 4) Stimulus Axis & Optimization Functions (Copied)
# -------------------------------------------------
def stimulus_axis_direction(g, W_R, W_F, feature_range):
    """Defines stimulus axis as response difference across feature range."""
    N = W_R.shape[0]
    resp_f0 = compute_response(W_R, W_F, feature_range[0], g, noise_internal=None)
    resp_f1 = compute_response(W_R, W_F, feature_range[1], g, noise_internal=None)
    if resp_f0.shape != (N,) or resp_f1.shape != (N,):
        print(f"Error: Unexpected shapes in stimulus_axis_direction.")
        return np.zeros(N)
    return resp_f1 - resp_f0

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-15 or norm2 < 1e-15: return 0.0
    dot_prod = np.dot(vec1, vec2)
    return np.clip(dot_prod / (norm1 * norm2), -1.0, 1.0)

def objective_align_noise_pc1(g, W_R, W_F, feature_range, noise_pc1):
    """Minimize negative squared cosine similarity with noise_pc1."""
    stim_axis = stimulus_axis_direction(g, W_R, W_F, feature_range)
    cos_sim = cosine_similarity(stim_axis, noise_pc1)
    return -(cos_sim**2)

def objective_misalign_noise_pc1(g, W_R, W_F, feature_range, noise_pc1):
    """Minimize squared cosine similarity with noise_pc1."""
    stim_axis = stimulus_axis_direction(g, W_R, W_F, feature_range)
    cos_sim = cosine_similarity(stim_axis, noise_pc1)
    return cos_sim**2

def stim_axis_norm_constraint(g, W_R, W_F, feature_range, target_norm_sq):
    """Constraint: ||stim_axis(g)||^2 == target_norm_sq."""
    stim_axis = stimulus_axis_direction(g, W_R, W_F, feature_range)
    return np.linalg.norm(stim_axis)**2 - target_norm_sq

# -------------------------------------------------
# 5) Main Setup and Optimization Logic
# -------------------------------------------------
if __name__ == "__main__":
    print("--- 1. Initializing Network ---")
    prefs, is_E, is_I = initialize_tuning_and_types(N, N_E, feature_range)
    W_F = initialize_W_F(N, K, prefs, tuning_sigma_W_F, feature_range)
    base_Js = (J_EE_base, J_IE_base, J_EI_base, J_II_base)
    # Use the Gaussian tuning factor here
    W_R = initialize_W_R_EI(N, prefs, is_E, is_I, base_Js, tuning_sigma_W_R, desired_radius_W_R)
    print(f"Initialized: Prefs({prefs.shape}), W_F({W_F.shape}), W_R({W_R.shape})")
    print(f"Neuron types: {N_E} E, {N_I} I")

    print("\n--- 2. Computing Noise Responses and PCs ---")
    g_baseline = np.ones(N)
    # Get noise_mean back from the function
    noise_responses_centered, noise_pcs_basis, noise_mean = get_noise_responses_and_pcs(
        W_R, W_F, num_noise_trials, noise_std, g_baseline
    )
    noise_pc1 = noise_pcs_basis[0, :]
    noise_pc2 = noise_pcs_basis[1, :]

    print("\n--- 3. Defining Baseline Stimulus Axis ---")
    stim_axis_baseline = stimulus_axis_direction(g_baseline, W_R, W_F, feature_range)
    norm_stim_axis_baseline_sq = np.linalg.norm(stim_axis_baseline)**2
    print(f"Baseline stimulus axis norm: {np.sqrt(norm_stim_axis_baseline_sq):.4f}")
    angle_baseline = np.degrees(np.arccos(cosine_similarity(stim_axis_baseline, noise_pc1)))
    print(f"Baseline angle with Noise PC1: {angle_baseline:.2f} deg")

    print("\n--- 4. Optimizing for ALIGNED Gain Vector ---")
    bounds = Bounds([gain_bounds[0]]*N, [gain_bounds[1]]*N)
    constraints = ({'type': 'eq',
                    'fun': stim_axis_norm_constraint,
                    'args': (W_R, W_F, feature_range, norm_stim_axis_baseline_sq)})
    opt_args_align = (W_R, W_F, feature_range, noise_pc1)
    start_time = time.time()
    res_aligned = minimize(objective_align_noise_pc1, g_baseline, args=opt_args_align,
                           method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': max_opt_iter, 'disp': False, 'ftol': 1e-7}) # Changed disp to False
    print(f"Alignment optimization took {time.time() - start_time:.2f}s")
    g_aligned = g_baseline
    if res_aligned.success:
        g_aligned = res_aligned.x
        print("Alignment optimization SUCCEEDED.")
    else:
        print(f"Alignment optimization FAILED: {res_aligned.message}. Using baseline gain.")
    stim_axis_aligned = stimulus_axis_direction(g_aligned, W_R, W_F, feature_range)
    angle_aligned = np.degrees(np.arccos(cosine_similarity(stim_axis_aligned, noise_pc1)))
    print(f"Aligned angle with Noise PC1: {angle_aligned:.2f} deg")

    print("\n--- 5. Optimizing for MISALIGNED Gain Vector ---")
    opt_args_misalign = (W_R, W_F, feature_range, noise_pc1)
    start_time = time.time()
    res_misaligned = minimize(objective_misalign_noise_pc1, g_baseline, args=opt_args_misalign,
                              method='SLSQP', bounds=bounds, constraints=constraints,
                              options={'maxiter': max_opt_iter, 'disp': False, 'ftol': 1e-7}) # Changed disp to False
    print(f"Misalignment optimization took {time.time() - start_time:.2f}s")
    g_misaligned = g_baseline
    if res_misaligned.success:
        g_misaligned = res_misaligned.x
        print("Misalignment optimization SUCCEEDED.")
    else:
        print(f"Misalignment optimization FAILED: {res_misaligned.message}. Using baseline gain.")
    stim_axis_misaligned = stimulus_axis_direction(g_misaligned, W_R, W_F, feature_range)
    angle_misaligned = np.degrees(np.arccos(cosine_similarity(stim_axis_misaligned, noise_pc1)))
    print(f"Misaligned angle with Noise PC1: {angle_misaligned:.2f} deg")

    # --- (Optional) Visualize Trajectories and Gains ---
    # You might comment these out if you only want to save data
    print("\n--- Visualizing Trajectories and Gains (Optional) ---")

    # Trajectory Plot (Section 7 from previous script)
    noise_proj = noise_responses_centered @ noise_pcs_basis.T
    feature_vals_stim = np.linspace(feature_range[0], feature_range[1], num_stim_steps)
    stim_responses_baseline = np.array([compute_response(W_R, W_F, f, g_baseline) for f in feature_vals_stim])
    stim_responses_aligned = np.array([compute_response(W_R, W_F, f, g_aligned) for f in feature_vals_stim])
    stim_responses_misaligned = np.array([compute_response(W_R, W_F, f, g_misaligned) for f in feature_vals_stim])
    stim_responses_baseline_centered = stim_responses_baseline - noise_mean
    stim_responses_aligned_centered = stim_responses_aligned - noise_mean
    stim_responses_misaligned_centered = stim_responses_misaligned - noise_mean
    stim_proj_baseline = stim_responses_baseline_centered @ noise_pcs_basis.T
    stim_proj_aligned = stim_responses_aligned_centered @ noise_pcs_basis.T
    stim_proj_misaligned = stim_responses_misaligned_centered @ noise_pcs_basis.T

    plt.style.use('seaborn-v0_8-whitegrid')
    fig_traj, ax_traj = plt.subplots(figsize=(9, 8))
    ax_traj.scatter(noise_proj[:, 0], noise_proj[:, 1], alpha=0.2, s=15, color='gray', label=f'Noise Trials')
    ax_traj.plot(stim_proj_baseline[:, 0], stim_proj_baseline[:, 1], marker='.', linestyle='-', color='black', alpha=0.7, label=f'Baseline G=1 ({angle_baseline:.1f}°)')
    ax_traj.plot(stim_proj_aligned[:, 0], stim_proj_aligned[:, 1], marker='.', linestyle='-', color='blue', alpha=0.8, label=f'Most Aligned ({angle_aligned:.1f}°)')
    ax_traj.plot(stim_proj_misaligned[:, 0], stim_proj_misaligned[:, 1], marker='.', linestyle='-', color='red', alpha=0.8, label=f'Least Aligned ({angle_misaligned:.1f}°)')
    arrow_scale = np.max(np.abs(noise_proj)) * 0.6
    ax_traj.arrow(0, 0, arrow_scale, 0, head_width=arrow_scale*0.08, head_length=arrow_scale*0.1, fc='darkgreen', ec='darkgreen', lw=1.5, label='Noise PC1')
    ax_traj.arrow(0, 0, 0, arrow_scale, head_width=arrow_scale*0.08, head_length=arrow_scale*0.1, fc='purple', ec='purple', lw=1.5, label='Noise PC2')
    ax_traj.set_xlabel("Projection onto Noise PC1"); ax_traj.set_ylabel("Projection onto Noise PC2")
    ax_traj.set_title(f"Stimulus Trajectories vs Noise in Noise PC Space (N={N})"); ax_traj.legend(fontsize=9); ax_traj.grid(True); ax_traj.axhline(0, color='gray', lw=0.5); ax_traj.axvline(0, color='gray', lw=0.5); ax_traj.set_aspect('equal', adjustable='box')
    plt.tight_layout(); plt.show()

    # Gains Plot (Section 8 from previous script)
    fig_gains, (ax_aligned_g, ax_misaligned_g) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig_gains.suptitle("Optimized Neuron Gains vs. Feature Preference", fontsize=14)
    e_indices = np.where(is_E)[0]; i_indices = np.where(is_I)[0]
    ax_aligned_g.scatter(prefs[e_indices], g_aligned[e_indices], alpha=0.7, s=30, label=f'E Neurons', c='royalblue')
    ax_aligned_g.scatter(prefs[i_indices], g_aligned[i_indices], alpha=0.7, s=40, label=f'I Neurons', c='crimson', marker='x')
    ax_aligned_g.axhline(1.0, color='gray', linestyle='--', label='Baseline G=1'); ax_aligned_g.set_xlabel("Neuron Feature Preference"); ax_aligned_g.set_ylabel("Optimized Gain Value")
    ax_aligned_g.set_title(f"Aligned Gain: {res_aligned.success}"); ax_aligned_g.legend(); ax_aligned_g.grid(True, linestyle=':', alpha=0.7); ax_aligned_g.set_ylim(gain_bounds[0] - 0.1, gain_bounds[1] + 0.1)
    ax_misaligned_g.scatter(prefs[e_indices], g_misaligned[e_indices], alpha=0.7, s=30, label=f'E Neurons', c='royalblue')
    ax_misaligned_g.scatter(prefs[i_indices], g_misaligned[i_indices], alpha=0.7, s=40, label=f'I Neurons', c='crimson', marker='x')
    ax_misaligned_g.axhline(1.0, color='gray', linestyle='--', label='Baseline G=1'); ax_misaligned_g.set_xlabel("Neuron Feature Preference")
    ax_misaligned_g.set_title(f"Misaligned Gain: {res_misaligned.success}"); ax_misaligned_g.legend(); ax_misaligned_g.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # -------------------------------------------------
    # 6) Save Results to File
    # -------------------------------------------------
    print(f"\n--- Saving Results to {output_filename} ---")
    try:
        np.savez(output_filename,
                 # Network Structure
                 W_R=W_R,
                 W_F=W_F,
                 prefs=prefs,
                 is_E=is_E,
                 is_I=is_I,
                 # Noise Analysis
                 noise_mean=noise_mean,
                 noise_pcs_basis=noise_pcs_basis, # Save the full basis (PC1 and PC2)
                 # Optimized Gains
                 g_aligned=g_aligned,
                 g_misaligned=g_misaligned,
                 res_aligned_success=res_aligned.success,
                 res_misaligned_success=res_misaligned.success,
                 # Baseline Info
                 angle_baseline=angle_baseline,
                 angle_aligned=angle_aligned,
                 angle_misaligned=angle_misaligned,
                 # Config Params needed later
                 N=N,
                 feature_range=np.array(feature_range), # Ensure array for saving
                 noise_std_original=noise_std, # Save the original noise std used
                 num_stim_steps=num_stim_steps
                 )
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

    print("\n--- Setup and Optimization Script Finished ---")

