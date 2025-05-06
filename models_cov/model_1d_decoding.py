import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize, Bounds
import time
from sklearn.linear_model import LinearRegression

# -------------------------------------------------
# Configuration
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
noise_std = 0.1 # Standard deviation of internal noise added

# --- Optimization Params ---
gain_bounds = (0.5, 1.5) # Bounds for individual gain elements in g
max_opt_iter = 150      # Max iterations for optimization

# --- Visualization Params ---
num_stim_steps = 21 # Number of feature steps to show in stimulus trajectory


# -------------------------------------------------
# 1) Initialization Functions
# -------------------------------------------------

def initialize_tuning_and_types(N, N_E, feature_range):
    """Initializes neuron types (E/I) and feature preferences."""
    neuron_types = np.array(['E'] * N_E + ['I'] * (N - N_E))
    np.random.shuffle(neuron_types) # Randomly assign E/I status

    # Assign preferred feature value uniformly
    prefs = np.linspace(feature_range[0], feature_range[1], N)
    np.random.shuffle(prefs) # Randomly assign preferences

    is_E = (neuron_types == 'E')
    is_I = (neuron_types == 'I')
    return prefs, is_E, is_I

def initialize_W_F(N, K, prefs, tuning_sigma_W_F, feature_range):
    """Initializes feedforward weights W_F based on tuning preferences."""
    # Here, W_F represents the sensitivity of each neuron to the input feature.
    # Let's model it as the peak of the tuning curve (max sensitivity at pref).
    # For simplicity, let W_F be a column vector where each element is 1
    # (assuming input feature 'F' directly drives neurons based on their gain/recurrence).
    # A more complex W_F could model the shape of the tuning curve itself.
    # Let's use a simple approach: W_F element i is 1.
    W_F = np.ones((N, K)) # Shape (N, 1)
    # Alternative: Gaussian tuning curve value at a reference point? Let's stick to simple=1 for now.
    # This means the input F is scaled by G and then drives the network dynamics.
    return W_F


def initialize_W_R_EI(N, prefs, is_E, is_I, J_base, sigma, desired_radius):
    """Initializes feature-tuned W_R with E/I structure."""
    W_R = np.zeros((N, N))
    J_EE, J_IE, J_EI, J_II = J_base # Unpack base strengths

    for i in range(N): # Post-synaptic neuron
        for j in range(N): # Pre-synaptic neuron
            if i == j: continue # No self-connections

            # Calculate tuning factor based on preference difference
            pref_diff = prefs[i] - prefs[j]
            # Handle wrap-around if feature space was circular (not needed here)
            #tuning_factor = np.exp(- (pref_diff**2) / (2 * sigma**2))
            tuning_factor = pref_diff
            #tuning_factor = np.exp(- (pref_diff**2) / (2 * sigma**2))

            # Assign weight based on pre- and post-synaptic types
            if is_E[j] and is_E[i]:    # E -> E
                W_R[i, j] = J_EE * tuning_factor
            elif is_E[j] and is_I[i]:  # E -> I
                W_R[i, j] = J_IE * tuning_factor
            elif is_I[j] and is_E[i]:  # I -> E
                W_R[i, j] = J_EI * tuning_factor # Base J is already negative
            elif is_I[j] and is_I[i]:  # I -> I
                W_R[i, j] = J_II * tuning_factor # Base J is already negative

    # Rescale W_R so spectral radius = desired_radius
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
# 2) Response Computation Functions
# -------------------------------------------------
def compute_response(W_R, W_F, feature_val, g_vector=None, noise_internal=None):
    """
    Computes steady-state response for a single feature value.
    r = (I - G @ W_R)^-1 @ (G @ W_F @ F + noise_internal)
    Assumes F is a scalar feature_val, W_F is (N, 1).
    noise_internal should be of shape (N,).
    """
    N = W_R.shape[0]
    I = np.eye(N)

    if g_vector is None:
        g_vector = np.ones(N)

    G = np.diag(g_vector)
    inv_mat = None
    try:
        mat_to_invert = I - G @ W_R
        inv_mat = np.linalg.inv(mat_to_invert)
    except np.linalg.LinAlgError:
        print(f"Warning: Matrix inversion failed. Check stability (spectral radius?). Returning zeros.")
        return np.zeros(N) # Ensure return shape is (N,)

    ff_input = G @ W_F * feature_val # Shape (N, 1)

    if noise_internal is None:
        noise_internal = np.zeros((N, 1)) # Ensure shape (N, 1)
    else:
        noise_internal = noise_internal.reshape(-1, 1) # Make it (N, 1)

    total_input = ff_input + noise_internal # Shape (N, 1)

    # --- Explicit Check ---
    if inv_mat is None or total_input.shape[0] != N: # Should not happen if inverse succeeded
         print("Error: Problem with input shapes for final calculation.")
         return np.zeros(N)

    response = inv_mat @ total_input # Expected shape (N, 1)

    # --- Assertion and Reshape ---
    expected_shape = (N, 1)
    if response.shape != expected_shape:
         print(f"Warning: Unexpected response shape {response.shape} inside compute_response. Expected {expected_shape}. Attempting reshape.")
         # Attempt to reshape defensively, although the cause is unclear.
         # This might hide the real issue, but aims to prevent the downstream crash.
         try:
             response = response.reshape(expected_shape)
         except ValueError:
             print(f"Error: Cannot reshape response from {response.shape} to {expected_shape}. Returning zeros.")
             return np.zeros(N) # Return correct shape on failure

    # Ensure final output is (N,)
    final_response = response.flatten()
    if final_response.shape != (N,):
        print(f"Error: Flattening response did not result in shape ({N},). Got {final_response.shape}. Returning zeros.")
        return np.zeros(N)

    return final_response
# -------------------------------------------------
# 3) Noise Response & PCA
# -------------------------------------------------
def get_noise_responses_and_pcs(W_R, W_F, num_trials, noise_std, g_vector=None):
    """Generates noise responses and performs PCA."""
    N = W_R.shape[0]
    if g_vector is None:
        g_vector = np.ones(N)

    noise_responses = np.zeros((num_trials, N))
    print(f"Computing {num_trials} noise responses...")
    for i in range(num_trials):
        # Generate independent Gaussian noise for each neuron
        noise_internal = np.random.normal(0, noise_std, size=N)
        # Compute response to noise only (feature_val = 0)
        noise_responses[i, :] = compute_response(W_R, W_F, 0.0, g_vector, noise_internal)

    print("Performing PCA on noise responses...")
    pca_noise = PCA(n_components=2)
    # Center the data before PCA
    noise_responses_centered = noise_responses - np.mean(noise_responses, axis=0)
    pca_noise.fit(noise_responses_centered)
    noise_pcs_basis = pca_noise.components_ # Shape (2, N)
    explained_variance = pca_noise.explained_variance_ratio_

    print(f"Noise variance explained by PC1: {explained_variance[0]:.3f}")
    print(f"Noise variance explained by PC2: {explained_variance[1]:.3f}")

    return noise_responses_centered, noise_pcs_basis

# -------------------------------------------------
# 4) Stimulus Axis & Optimization Functions
# -------------------------------------------------
def stimulus_axis_direction(g, W_R, W_F, feature_range):
    """Defines stimulus axis as response difference across feature range."""
    N = W_R.shape[0]
    resp_f0 = compute_response(W_R, W_F, feature_range[0], g, noise_internal=None)
    resp_f1 = compute_response(W_R, W_F, feature_range[1], g, noise_internal=None)

    # --- Add Shape Checks ---
    if resp_f0.shape != (N,) or resp_f1.shape != (N,):
        print(f"Error: Unexpected shapes in stimulus_axis_direction. resp_f0: {resp_f0.shape}, resp_f1: {resp_f1.shape}. Expected ({N},)")
        # Return zeros of correct shape to allow script continuation if possible
        return np.zeros(N)

    return resp_f1 - resp_f0

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-15 or norm2 < 1e-15:
        return 0.0
    dot_prod = np.dot(vec1, vec2)
    return np.clip(dot_prod / (norm1 * norm2), -1.0, 1.0)

# --- Optimization Objectives ---
def objective_align_noise_pc1(g, W_R, W_F, feature_range, noise_pc1):
    """Minimize negative squared cosine similarity with noise_pc1."""
    stim_axis = stimulus_axis_direction(g, W_R, W_F, feature_range)
    cos_sim = cosine_similarity(stim_axis, noise_pc1)
    return -(cos_sim**2) # Minimize negative to maximize alignment

def objective_misalign_noise_pc1(g, W_R, W_F, feature_range, noise_pc1):
    """Minimize squared cosine similarity with noise_pc1 (push towards 90 deg)."""
    stim_axis = stimulus_axis_direction(g, W_R, W_F, feature_range)
    cos_sim = cosine_similarity(stim_axis, noise_pc1)
    return cos_sim**2 # Minimize to push towards zero (orthogonal)

# --- Optimization Constraint ---
def stim_axis_norm_constraint(g, W_R, W_F, feature_range, target_norm_sq):
    """Constraint: ||stim_axis(g)||^2 == target_norm_sq."""
    stim_axis = stimulus_axis_direction(g, W_R, W_F, feature_range)
    return np.linalg.norm(stim_axis)**2 - target_norm_sq

# -------------------------------------------------
# 5) Main Script Logic
# -------------------------------------------------
if __name__ == "__main__":
    print("--- 1. Initializing Network ---")
    prefs, is_E, is_I = initialize_tuning_and_types(N, N_E, feature_range)
    W_F = initialize_W_F(N, K, prefs, tuning_sigma_W_F, feature_range)
    base_Js = (J_EE_base, J_IE_base, J_EI_base, J_II_base)
    W_R = initialize_W_R_EI(N, prefs, is_E, is_I, base_Js, tuning_sigma_W_R, desired_radius_W_R)
    print(f"Initialized: Prefs({prefs.shape}), W_F({W_F.shape}), W_R({W_R.shape})")
    print(f"Neuron types: {N_E} E, {N_I} I")

    print("\n--- 2. Computing Noise Responses and PCs ---")
    # Use baseline gain g=1 for noise computation
    g_baseline = np.ones(N)
    noise_responses, noise_pcs_basis = get_noise_responses_and_pcs(
        W_R, W_F, num_noise_trials, noise_std, g_baseline
    )
    noise_pc1 = noise_pcs_basis[0, :] # Shape (N,)
    noise_pc2 = noise_pcs_basis[1, :] # Shape (N,)

    print("\n--- 3. Defining Baseline Stimulus Axis ---")
    stim_axis_baseline = stimulus_axis_direction(g_baseline, W_R, W_F, feature_range)
    norm_stim_axis_baseline_sq = np.linalg.norm(stim_axis_baseline)**2
    print(f"Baseline stimulus axis norm: {np.sqrt(norm_stim_axis_baseline_sq):.4f}")

    # Calculate baseline angle with noise PC1
    angle_baseline = np.degrees(np.arccos(cosine_similarity(stim_axis_baseline, noise_pc1)))
    print(f"Baseline angle with Noise PC1: {angle_baseline:.2f} deg")


    print("\n--- 4. Optimizing for ALIGNED Gain Vector ---")
    bounds = Bounds([gain_bounds[0]]*N, [gain_bounds[1]]*N)
    constraints = ({'type': 'eq',
                    'fun': stim_axis_norm_constraint,
                    'args': (W_R, W_F, feature_range, norm_stim_axis_baseline_sq)})
    opt_args_align = (W_R, W_F, feature_range, noise_pc1)

    start_time = time.time()
    res_aligned = minimize(objective_align_noise_pc1,
                           g_baseline, # Start from baseline
                           args=opt_args_align,
                           method='SLSQP',
                           bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': max_opt_iter, 'disp': True, 'ftol': 1e-7})
    print(f"Alignment optimization took {time.time() - start_time:.2f}s")

    g_aligned = g_baseline
    if res_aligned.success:
        g_aligned = res_aligned.x
        print("Alignment optimization SUCCEEDED.")
    else:
        print("Alignment optimization FAILED. Using baseline gain.")

    stim_axis_aligned = stimulus_axis_direction(g_aligned, W_R, W_F, feature_range)
    angle_aligned = np.degrees(np.arccos(cosine_similarity(stim_axis_aligned, noise_pc1)))
    print(f"Aligned stimulus axis norm: {np.linalg.norm(stim_axis_aligned):.4f} (Target: {np.sqrt(norm_stim_axis_baseline_sq):.4f})")
    print(f"Aligned angle with Noise PC1: {angle_aligned:.2f} deg")


    print("\n--- 5. Optimizing for MISALIGNED Gain Vector ---")
    # Re-use bounds, constraints. Change objective and args.
    opt_args_misalign = (W_R, W_F, feature_range, noise_pc1)

    start_time = time.time()
    res_misaligned = minimize(objective_misalign_noise_pc1,
                              g_baseline, # Start from baseline
                              args=opt_args_misalign,
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': max_opt_iter, 'disp': True, 'ftol': 1e-7})
    print(f"Misalignment optimization took {time.time() - start_time:.2f}s")

    g_misaligned = g_baseline
    if res_misaligned.success:
        g_misaligned = res_misaligned.x
        print("Misalignment optimization SUCCEEDED.")
    else:
        print("Misalignment optimization FAILED. Using baseline gain.")

    stim_axis_misaligned = stimulus_axis_direction(g_misaligned, W_R, W_F, feature_range)
    angle_misaligned = np.degrees(np.arccos(cosine_similarity(stim_axis_misaligned, noise_pc1)))
    print(f"Misaligned stimulus axis norm: {np.linalg.norm(stim_axis_misaligned):.4f} (Target: {np.sqrt(norm_stim_axis_baseline_sq):.4f})")
    print(f"Misaligned angle with Noise PC1: {angle_misaligned:.2f} deg")


    print("\n--- 6. Generating Data for Visualization ---")
    # Project noise responses onto noise PC basis
    noise_proj = noise_responses @ noise_pcs_basis.T # Shape (num_trials, 2)

    # Generate stimulus responses for different feature values under each gain condition
    feature_vals_stim = np.linspace(feature_range[0], feature_range[1], num_stim_steps)
    stim_responses_baseline = np.array([compute_response(W_R, W_F, f, g_baseline) for f in feature_vals_stim])
    stim_responses_aligned = np.array([compute_response(W_R, W_F, f, g_aligned) for f in feature_vals_stim])
    stim_responses_misaligned = np.array([compute_response(W_R, W_F, f, g_misaligned) for f in feature_vals_stim])

    # Center stimulus responses using the mean of the *noise* responses (for consistent projection)
    noise_mean = np.mean(noise_responses, axis=0)
    stim_responses_baseline_centered = stim_responses_baseline - noise_mean
    stim_responses_aligned_centered = stim_responses_aligned - noise_mean
    stim_responses_misaligned_centered = stim_responses_misaligned - noise_mean

    # Project stimulus responses onto noise PC basis
    stim_proj_baseline = stim_responses_baseline_centered @ noise_pcs_basis.T
    stim_proj_aligned = stim_responses_aligned_centered @ noise_pcs_basis.T
    stim_proj_misaligned = stim_responses_misaligned_centered @ noise_pcs_basis.T


    print("\n--- 7. Creating Visualization ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 8))

    # Plot noise trials
    ax.scatter(noise_proj[:, 0], noise_proj[:, 1], alpha=0.2, s=15, color='gray', label=f'Noise Trials ({num_noise_trials})')

    # Plot stimulus trajectories
    ax.plot(stim_proj_baseline[:, 0], stim_proj_baseline[:, 1], marker='.', linestyle='-', color='black', alpha=0.7,
            label=f'Stimulus (Baseline G=1) Angle={angle_baseline:.1f}°')
    ax.plot(stim_proj_aligned[:, 0], stim_proj_aligned[:, 1], marker='.', linestyle='-', color='blue', alpha=0.8,
            label=f'Stimulus (Aligned G) Angle={angle_aligned:.1f}° (Success: {res_aligned.success})')
    ax.plot(stim_proj_misaligned[:, 0], stim_proj_misaligned[:, 1], marker='.', linestyle='-', color='red', alpha=0.8,
            label=f'Stimulus (Misaligned G) Angle={angle_misaligned:.1f}° (Success: {res_misaligned.success})')

    # Add markers for start (feature=0) and end (feature=1) of stimulus trajectories
    ax.scatter(stim_proj_baseline[0, 0], stim_proj_baseline[0, 1], marker='o', s=50, color='black', alpha=0.7, label='_nolegend_') # Start
    ax.scatter(stim_proj_baseline[-1, 0], stim_proj_baseline[-1, 1], marker='s', s=50, color='black', alpha=0.7, label='_nolegend_') # End
    ax.scatter(stim_proj_aligned[0, 0], stim_proj_aligned[0, 1], marker='o', s=50, color='blue', alpha=0.8, label='_nolegend_')
    ax.scatter(stim_proj_aligned[-1, 0], stim_proj_aligned[-1, 1], marker='s', s=50, color='blue', alpha=0.8, label='_nolegend_')
    ax.scatter(stim_proj_misaligned[0, 0], stim_proj_misaligned[0, 1], marker='o', s=50, color='red', alpha=0.8, label='_nolegend_')
    ax.scatter(stim_proj_misaligned[-1, 0], stim_proj_misaligned[-1, 1], marker='s', s=50, color='red', alpha=0.8, label='_nolegend_')


    # Add arrows for Noise PCs (scaled for visibility)
    arrow_scale = np.max(np.abs(noise_proj)) * 0.6 # Scale factor for arrows
    # PC1 vector in PC basis is [1, 0], PC2 is [0, 1]
    ax.arrow(0, 0, arrow_scale, 0, head_width=arrow_scale*0.08, head_length=arrow_scale*0.1, fc='darkgreen', ec='darkgreen', lw=1.5, label='Noise PC1 Axis')
    ax.arrow(0, 0, 0, arrow_scale, head_width=arrow_scale*0.08, head_length=arrow_scale*0.1, fc='purple', ec='purple', lw=1.5, label='Noise PC2 Axis')

    ax.set_xlabel("Projection onto Noise PC1")
    ax.set_ylabel("Projection onto Noise PC2")
    ax.set_title(f"Stimulus Trajectories vs Noise Distribution in Noise PC Space (N={N})")
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal', adjustable='box') # Make axes visually orthogonal

    plt.tight_layout()
    plt.show()
    
    # -------------------------------------------------
    # 8) Visualize Optimized Gains vs. Preference
    # -------------------------------------------------
    print("\n--- 8. Visualizing Optimized Gains ---")

    fig_gains, (ax_aligned, ax_misaligned) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig_gains.suptitle("Optimized Neuron Gains vs. Feature Preference", fontsize=14)

    # Indices for E and I neurons
    e_indices = np.where(is_E)[0]
    i_indices = np.where(is_I)[0]

    # --- Plot Aligned Gains ---
    ax_aligned.scatter(prefs[e_indices], g_aligned[e_indices], alpha=0.7, s=30,
                       label=f'E Neurons ({len(e_indices)})', c='royalblue')
    ax_aligned.scatter(prefs[i_indices], g_aligned[i_indices], alpha=0.7, s=40,
                       label=f'I Neurons ({len(i_indices)})', c='crimson', marker='x')
    ax_aligned.axhline(1.0, color='gray', linestyle='--', label='Baseline Gain=1')
    ax_aligned.set_xlabel("Neuron Feature Preference")
    ax_aligned.set_ylabel("Optimized Gain Value")
    ax_aligned.set_title(f"Aligned Gain (g_aligned) - Success: {res_aligned.success}")
    ax_aligned.legend()
    ax_aligned.grid(True, linestyle=':', alpha=0.7)
    ax_aligned.set_ylim(gain_bounds[0] - 0.1, gain_bounds[1] + 0.1) # Set Y limits based on bounds

    # --- Plot Misaligned Gains ---
    ax_misaligned.scatter(prefs[e_indices], g_misaligned[e_indices], alpha=0.7, s=30,
                          label=f'E Neurons ({len(e_indices)})', c='royalblue')
    ax_misaligned.scatter(prefs[i_indices], g_misaligned[i_indices], alpha=0.7, s=40,
                          label=f'I Neurons ({len(i_indices)})', c='crimson', marker='x')
    ax_misaligned.axhline(1.0, color='gray', linestyle='--', label='Baseline Gain=1')
    ax_misaligned.set_xlabel("Neuron Feature Preference")
    #ax_misaligned.set_ylabel("Optimized Gain Value") # Y-label shared
    ax_misaligned.set_title(f"Misaligned Gain (g_misaligned) - Success: {res_misaligned.success}")
    ax_misaligned.legend()
    ax_misaligned.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

    # -------------------------------------------------
    # 9) Decoding Task Simulation
    # -------------------------------------------------
    print("\n--- 9. Setting up Decoding Task ---")

    # --- Decoder Calibration ---
    # Use the noiseless stimulus projections calculated earlier (Section 6)
    # X: feature values, y: projection onto noise PC1
    X_feature = feature_vals_stim.reshape(-1, 1)
    y_proj_aligned = stim_proj_aligned[:, 0].reshape(-1, 1) # Use only PC1 projection
    y_proj_misaligned = stim_proj_misaligned[:, 0].reshape(-1, 1) # Use only PC1 projection

    # Fit linear decoders (find mapping: feature -> mean_proj_pc1)
    decoder_aligned = LinearRegression()
    decoder_aligned.fit(X_feature, y_proj_aligned)
    slope_aligned = decoder_aligned.coef_[0, 0]
    intercept_aligned = decoder_aligned.intercept_[0]
    print(f"Decoder (Aligned):   Proj_PC1 = {slope_aligned:.3f} * Feature + {intercept_aligned:.3f}")

    decoder_misaligned = LinearRegression()
    decoder_misaligned.fit(X_feature, y_proj_misaligned)
    slope_misaligned = decoder_misaligned.coef_[0, 0]
    intercept_misaligned = decoder_misaligned.intercept_[0]
    print(f"Decoder (Misaligned): Proj_PC1 = {slope_misaligned:.3f} * Feature + {intercept_misaligned:.3f}")

    # Define the decoding function (invert the linear map)
    def decode_feature(projection, slope, intercept):
        if abs(slope) < 1e-9: # Avoid division by zero if projection doesn't change
            return feature_range[0] + (feature_range[1] - feature_range[0]) / 2 # Return midpoint
        decoded = (projection - intercept) / slope
        # Clip to valid feature range
        return np.clip(decoded, feature_range[0], feature_range[1])

    # --- Simulation Parameters ---
    num_noise_levels = 20
    # Create noise levels (e.g., logarithmically spaced around original noise_std)
    noise_std_levels = np.logspace(np.log10(noise_std / 5), np.log10(noise_std * 5), num_noise_levels)
    num_decoding_trials = 1000 # Number of trials per noise level (more than 100 for stable MAE)

    mae_results_aligned = []
    mae_results_misaligned = []

    print(f"\nRunning decoding simulation for {num_noise_levels} noise levels...")
    start_decoding_time = time.time()

    # --- Simulation Loops ---
    for current_noise_std in noise_std_levels:
        errors_aligned = []
        errors_misaligned = []
        for _ in range(num_decoding_trials):
            # Choose a random true feature value
            true_feature = np.random.uniform(feature_range[0], feature_range[1])

            # Generate internal noise for this trial
            noise_vec = np.random.normal(0, current_noise_std, size=N)

            # Compute noisy responses (remember compute_response adds noise internally)
            r_aligned = compute_response(W_R, W_F, true_feature, g_aligned, noise_vec)
            r_misaligned = compute_response(W_R, W_F, true_feature, g_misaligned, noise_vec)

            # Center responses using the baseline noise mean calculated earlier
            r_aligned_centered = r_aligned - noise_mean
            r_misaligned_centered = r_misaligned - noise_mean

            # Project onto Noise PC1 (the fixed decoding axis)
            proj_aligned = np.dot(r_aligned_centered, noise_pc1)
            proj_misaligned = np.dot(r_misaligned_centered, noise_pc1)

            # Decode the feature
            decoded_aligned = decode_feature(proj_aligned, slope_aligned, intercept_aligned)
            decoded_misaligned = decode_feature(proj_misaligned, slope_misaligned, intercept_misaligned)

            # Calculate absolute error
            errors_aligned.append(np.abs(decoded_aligned - true_feature))
            errors_misaligned.append(np.abs(decoded_misaligned - true_feature))

        # Calculate Mean Absolute Error (MAE) for this noise level
        mae_aligned = np.mean(errors_aligned)
        mae_misaligned = np.mean(errors_misaligned)
        mae_results_aligned.append(mae_aligned)
        mae_results_misaligned.append(mae_misaligned)
        print(f"  Noise Std: {current_noise_std:.4f} -> MAE Aligned: {mae_aligned:.4f}, MAE Misaligned: {mae_misaligned:.4f}")

    print(f"Decoding simulation took {time.time() - start_decoding_time:.2f}s")

    # --- Visualize Decoding Performance ---
    print("\n--- 10. Visualizing Decoding Performance ---")
    fig_decoding, ax_decoding = plt.subplots(figsize=(8, 7))

    # Scatter plot: Misaligned MAE vs Aligned MAE
    scatter = ax_decoding.scatter(mae_results_misaligned, mae_results_aligned,
                                  c=noise_std_levels, cmap='viridis', s=50, zorder=3)

    # Add diagonal line y=x for reference
    lim_min = 0
    lim_max = max(np.max(mae_results_aligned), np.max(mae_results_misaligned)) * 1.1
    ax_decoding.plot([lim_min, lim_max], [lim_min, lim_max], color='gray', linestyle='--', label='Aligned = Misaligned')

    ax_decoding.set_xlabel("Decoding MAE (Misaligned Gain)")
    ax_decoding.set_ylabel("Decoding MAE (Aligned Gain)")
    ax_decoding.set_title("Decoding Performance Comparison (Lower MAE is Better)")
    ax_decoding.set_xlim(left=lim_min)
    ax_decoding.set_ylim(bottom=lim_min)
    ax_decoding.grid(True, linestyle=':', alpha=0.7)
    ax_decoding.legend()

    # Add colorbar
    cbar = fig_decoding.colorbar(scatter)
    cbar.set_label('Internal Noise Std Dev')

    plt.tight_layout()
    plt.show()


    print("\n--- Script Finished ---") # Move this line after all plots
