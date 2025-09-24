import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Functions from my_network.py
def initialize_selectivity_matrix(N, K):
    """
    Half are shape-based, half are color-based, with a random distribution.
    The first half has shape > color (if needed, we reassign),
    and the second half is mirrored (swap shape/color).
    """
    S = np.zeros((N, K))
    # 1) Random shape (first half)
    S[:N//2, 0] = np.random.rand(N//2)
    # 2) Assign color as (0.5 - shape/2)
    S[:N//2, 1] = 0.5 - S[:N//2, 0] / 2

    # 3) If shape < color, reassign shape and recalc color
    neg_idx = (S[:N//2, 0] - S[:N//2, 1]) < 0
    if np.any(neg_idx):
        S[:N//2, 0][neg_idx] = np.random.uniform(0, 0.5, size=np.sum(neg_idx))
        # Recompute the color to match the new shape
        S[:N//2, 1][neg_idx] = 0.5 - S[:N//2, 0][neg_idx] / 2

    # 4) Mirror for the second half: color = shape, shape = color
    S[N//2:, 1] = S[:N//2, 0]
    S[N//2:, 0] = S[:N//2, 1]

    return S

def initialize_W_F(S):
    """
    Initialize W_F based on the selectivity matrix S.
    """
    W_F = np.zeros_like(S)
    for i in range(S.shape[0]):
        row_sum = np.sum(S[i])
        if row_sum > 0:
            W_F[i] = S[i] / row_sum
        else:
            W_F[i] = S[i]
    return W_F

def initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=0.9):
    """
    Initialize the recurrent connectivity matrix W_R.
    """
    W_R = np.zeros((N, N))
    half_N = N // 2

    # Shape-shape block
    shape_shape_mask = np.random.rand(half_N, half_N) < p_high
    W_R[:half_N, :half_N][shape_shape_mask] = np.random.rand(np.sum(shape_shape_mask)) * 0.1

    # Shape-color block
    shape_color_mask = np.random.rand(half_N, N - half_N) < p_low
    W_R[:half_N, half_N:][shape_color_mask] = np.random.rand(np.sum(shape_color_mask)) * 0.1

    # Color-shape block
    color_shape_mask = np.random.rand(N - half_N, half_N) < p_low
    W_R[half_N:, :half_N][color_shape_mask] = np.random.rand(np.sum(color_shape_mask)) * 0.1

    # Color-color block
    color_color_mask = np.random.rand(N - half_N, N - half_N) < p_high
    W_R[half_N:, half_N:][color_color_mask] = np.random.rand(np.sum(color_color_mask)) * 0.1

    # No self-connections
    np.fill_diagonal(W_R, 0)

    # Optional tuning
    if WR_tuned:
        threshold = 0.2
        for i in range(N):
            for j in range(N):
                if i != j:
                    distance = np.linalg.norm(S[i] - S[j])
                    if distance < threshold:
                        W_R[i, j] *= (2 - distance / threshold)

    # Scale W_R to ensure stability (spectral radius <= desired_radius)
    eigenvalues = np.linalg.eigvals(W_R)
    max_eval = np.max(np.abs(eigenvalues))
    if max_eval > 0:
        W_R *= (desired_radius / max_eval)

    return W_R

def compute_responses(W_F, W_R, shape_stimuli, color_stimuli):
    """
    Compute the network responses for a grid of stimuli.
    Returns:
      responses: (num_stimuli * num_stimuli, N)
      stimuli_grid: (num_stimuli * num_stimuli, 2)
    """
    N = W_F.shape[0]
    stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)
    inv_I_minus_WR = np.linalg.inv(np.eye(N) - W_R)
    responses = np.zeros((len(stimuli_grid), N))

    for idx, (shape_val, color_val) in enumerate(stimuli_grid):
        F = np.array([shape_val, color_val])
        adjusted_F = W_F @ F
        responses[idx] = inv_I_minus_WR @ adjusted_F

    return responses, stimuli_grid

def generate_noisy_responses(W_R, noise_level, stimuli_grid, num_noise_trials):
    """
    Generate noise-only responses in unmodulated form.
    Returns shape = (num_stimuli_grid * num_noise_trials, N)
    """
    N = W_R.shape[0]
    inv_I_minus_WR = np.linalg.inv(np.eye(N) - W_R)
    noisy_responses = []

    for (shape_val, color_val) in stimuli_grid:
        for _ in range(num_noise_trials):
            noise_input = np.random.randn(N) * noise_level
            noisy_responses.append(inv_I_minus_WR @ noise_input)

    return np.array(noisy_responses)

def compute_modulated_responses_pc3(W_R, W_F, S, stimuli_grid, pc3, alpha=0.3, beta=0.5):
    """
    Boost color neurons along PC3, optionally suppress shape neurons along PC3.
    """
    N = W_R.shape[0]
    I = np.eye(N)

    # 1) Normalize pc3
    pc3_unit = pc3 / np.linalg.norm(pc3)

    # 2) Build "color ratio"
    color_part = np.maximum(S[:, 1], 0.0)
    shape_part = np.maximum(S[:, 0], 0.0)
    denom = color_part + shape_part + 1e-9
    color_ratio = color_part / denom  # in [0,1]

    # 3) Overlap with PC3
    overlap_pc3 = np.abs(pc3_unit)     # or keep the sign if you want
    overlap_pc3 /= np.max(overlap_pc3) # optional normalization to [0,1]

    # Gains: 
    #    + alpha for color-pref neurons that load on PC3
    #    - beta for shape-pref neurons that load on PC3
    g_vector = 1 \
               + alpha * (color_ratio * overlap_pc3) \
               - beta * ((1 - color_ratio) * overlap_pc3)

    # Clip gains if desired, so we don't get negative or huge
    g_vector = np.clip(g_vector, 0.1, 10.0)

    # Build G
    G = np.diag(g_vector)

    I_minus_GWR = I - G @ W_R
    cond_number = np.linalg.cond(I_minus_GWR)
    if cond_number > 1 / np.finfo(I_minus_GWR.dtype).eps:
        raise ValueError("(I - G W_R) is nearly singular.")

    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)
    G_WF = G @ W_F

    # 4) Compute final modulated responses
    mod_responses = np.zeros((len(stimuli_grid), N))
    for idx, (shape_val, color_val) in enumerate(stimuli_grid):
        F = np.array([shape_val, color_val])
        mod_responses[idx] = inv_I_minus_GWR @ (G_WF @ F)

    return mod_responses, g_vector

def generate_external_noise_in_third_pc(W_R, pc3, stimuli_grid, noise_level, num_noise_trials):
    """
    Force the FINAL network response to lie in PC3, by 
    x = (I - W_R)*[alpha * pc3].
    """
    N = W_R.shape[0]
    I = np.eye(N)
    M = I - W_R
    inv_M = np.linalg.inv(M)

    pc3_unit = pc3 / np.linalg.norm(pc3)
    external_noise_responses = []

    for (shape_val, color_val) in stimuli_grid:
        for _ in range(num_noise_trials):
            alpha = np.random.randn() * noise_level
            final_desired = alpha * pc3_unit
            noise_input = M @ final_desired
            response = inv_M @ noise_input  # should equal final_desired
            external_noise_responses.append(response)

    return np.array(external_noise_responses)

def set_axes_equal(ax):
    """
    Ensure 3D axes have equal scale (so spheres look like spheres).
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = (x_limits[0] + x_limits[1]) / 2
    y_middle = (y_limits[0] + y_limits[1]) / 2
    z_middle = (z_limits[0] + z_limits[1]) / 2

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def integrated_visualization():
    # 1) Hyperparameters
    np.random.seed(15)
    N = 100           
    K = 2             
    num_stimuli = 10  
    num_noise_trials = 50  
    noise_level = 0.01       
    desired_radius = 0.9    
    p_high = 0.25
    p_low = 0.25

    # 2) Build S, W_F
    S = initialize_selectivity_matrix(N, K)
    W_F = initialize_W_F(S)

    # Stimuli
    shape_stimuli = np.linspace(0, 1, num_stimuli)
    color_stimuli = np.linspace(0, 1, num_stimuli)

    # 3) Initialize UNTUNED W_R
    W_R_untuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)
    
    # 4) Compute responses - unmodulated
    responses_grid_unmod, stimuli_grid = compute_responses(W_F, W_R_untuned, shape_stimuli, color_stimuli)
    
    # 5) Fit PCA on unmodulated responses to get PC3
    pca_3 = PCA(n_components=3)
    pca_3.fit(responses_grid_unmod)
    pc3_untuned = pca_3.components_[2]  # shape (N,)
    
    # 6) Compute modulated responses using PC3-based modulation
    responses_grid_mod_pc3, g_vector = compute_modulated_responses_pc3(
        W_R_untuned,
        W_F,
        S,
        stimuli_grid,
        pc3_untuned
    )
    
    # 7) Generate external noise in the 3rd PC
    external_noise_responses_3rd_pc = generate_external_noise_in_third_pc(
        W_R_untuned,
        pc3_untuned,
        stimuli_grid,
        noise_level=0.08,
        num_noise_trials=num_noise_trials
    )
    
    # 8) Create integrated visualization
    # Transform all data using the same PCA (fit on unmodulated responses)
    grid_unmod_3d = pca_3.transform(responses_grid_unmod)
    grid_mod_3d = pca_3.transform(responses_grid_mod_pc3)
    noise_3d = pca_3.transform(external_noise_responses_3rd_pc)
    
    # Create the integrated 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot unmodulated grid responses
    sc1 = ax.scatter(
        grid_unmod_3d[:, 0], grid_unmod_3d[:, 1], grid_unmod_3d[:, 2],
        c=stimuli_grid[:, 1], cmap='winter', s=40, alpha=0.3, label='Unmodulated Grid'
    )
    
    # Plot modulated grid responses
    sc2 = ax.scatter(
        grid_mod_3d[:, 0], grid_mod_3d[:, 1], grid_mod_3d[:, 2],
        c=stimuli_grid[:, 1], cmap='winter', s=40, alpha=0.95, label='PC3-Modulated Grid'
    )
    
    # Plot external noise in 3rd PC
    sc3 = ax.scatter(
        noise_3d[:, 0], noise_3d[:, 1], noise_3d[:, 2],
        c='gray', s=15, alpha=0.3, label='External Noise (3rd PC)'
    )
    
    # Add labels and title
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)
    ax.set_title("Integrated Visualization: Unmodulated, Modulated, and PC3 Noise", fontsize=14)
    
    # Add colorbar
    cbar = fig.colorbar(sc1, ax=ax, shrink=0.7, aspect=20)
    cbar.set_label('Color Stimulus Value', fontsize=12)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Set axes to equal scale
    #set_axes_equal(ax)
    
    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    integrated_visualization()