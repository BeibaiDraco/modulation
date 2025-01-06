# my_network.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# Parameters (optional defaults)
# -------------------------
# (You can store global parameters here, or pass them around explicitly.)
# N = 300
# ...

# -------------------------
# 1) Initialization Functions
# -------------------------
def initialize_selectivity_matrix(N, K):
    """
    Initialize the selectivity matrix S with constraints.
    First half of neurons selective mainly to shape,
    the second half mirrors this for color.
    """
    S = np.zeros((N, K))
    # First half selectivity
    S[:N//2, 0] = np.random.rand(N//2)
    S[:N//2, 1] = 0.5 - S[:N//2, 0] / 2

    # Adjust negative indices
    negative_indices = (S[:N//2, 0] - S[:N//2, 1]) < 0
    S[:N//2, 0][negative_indices] = np.random.uniform(0, 0.5, size=np.sum(negative_indices))

    # Mirror for second half
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

def initialize_W_R_cycle(N, desired_radius=0.9):
    """
    Cycle W_R: each neuron connects to the next in a ring with weight = desired_radius.
    """
    W_R = np.zeros((N, N))
    for i in range(N):
        W_R[i, (i + 1) % N] = desired_radius
    return W_R

# -------------------------------------------------
# 2) Response Computations (Unmodulated)
# -------------------------------------------------
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

# -------------------------------------------------
# 3) Modulated Responses
# -------------------------------------------------
def compute_modulated_responses(W_R, W_F, S, stimuli_grid):
    """
    Original modulation: G = diag(1 + 0.15*(color - shape)).
    """
    N = W_R.shape[0]
    selectivity_diff = S[:, 1] - S[:, 0]
    g_vector = 1.0 + 0.2 * selectivity_diff  # can adjust factor

    G = np.diag(g_vector)
    I = np.eye(N)
    I_minus_GWR = I - G @ W_R

    # Check invertibility
    cond_number = np.linalg.cond(I_minus_GWR)
    if cond_number > 1 / np.finfo(I_minus_GWR.dtype).eps:
        raise ValueError("(I - G W_R) is nearly singular.")

    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)
    G_WF = G @ W_F

    mod_responses = np.zeros((len(stimuli_grid), N))
    for idx, (shape_val, color_val) in enumerate(stimuli_grid):
        F = np.array([shape_val, color_val])
        mod_responses[idx] = inv_I_minus_GWR @ (G_WF @ F)
        
    return np.array(mod_responses)


def compute_modulated_responses_pc3(W_R, W_F, S, stimuli_grid, pc3, alpha=0.3):
    """
    Compute modulated grid responses, specifically amplifying the color dimension
    in the direction of PC3. 
    """
    N = W_R.shape[0]
    I = np.eye(N)
    pc3_unit = pc3 / np.linalg.norm(pc3)

    color_pref = S[:, 1]-S[:,0]  # how much each neuron prefers color
    overlap_pc3 = np.abs(pc3_unit)
    
    #normalize to max 1
    overlap_pc3 = overlap_pc3/np.max(overlap_pc3)
    g_vector = 1.0 + alpha * color_pref * overlap_pc3

    G = np.diag(g_vector)
    I_minus_GWR = I - G @ W_R
    cond_number = np.linalg.cond(I_minus_GWR)
    if cond_number > 1 / np.finfo(I_minus_GWR.dtype).eps:
        raise ValueError("(I - G W_R) is nearly singular.")

    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)
    G_WF = G @ W_F

    mod_responses = np.zeros((len(stimuli_grid), N))
    for idx, (shape_val, color_val) in enumerate(stimuli_grid):
        F = np.array([shape_val, color_val])
        mod_responses[idx] = inv_I_minus_GWR @ (G_WF @ F)

    return mod_responses

def compute_modulated_responses_pc3(
    W_R,
    W_F,
    S,
    stimuli_grid,
    pc3,
    alpha=0.3,   # how strongly to boost color in PC3
    beta=0.5    # how strongly to suppress shape in PC3
):
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
    g_vector = 1.0 \
               + alpha * (color_ratio * overlap_pc3) \
               - beta * ((1 - color_ratio) * overlap_pc3)

    # Clip gains if desired, so we don't get negative or huge
    g_vector = np.clip(g_vector, 0.1, 10.0)
    print(g_vector)

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

    return mod_responses


def compute_modulated_noise(W_R, W_F, S, stimuli_grid, noise_level, num_noise_trials):
    """
    Similar to generate_noisy_responses, but with the original gain modulation G.
    """
    N = W_R.shape[0]
    selectivity_diff = S[:, 1] - S[:, 0]
    g_vector = 1.0 + 0.15 * selectivity_diff  # can adjust factor
    G = np.diag(g_vector)
    

    I = np.eye(N)
    I_minus_GWR = I - G @ W_R
    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)

    noisy_responses = []
    for (shape_val, color_val) in stimuli_grid:
        for _ in range(num_noise_trials):
            noise_input = np.random.randn(N) * noise_level
            modded_noise = G @ noise_input
            noisy_responses.append(inv_I_minus_GWR @ modded_noise)

    return np.array(noisy_responses)

# -------------------------------------------------
# 4) 3D Visualization
# -------------------------------------------------
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

def visualize_four_subplots_3d(
    responses_grid_unmod,
    responses_noise_unmod,
    responses_grid_mod,
    responses_noise_mod,
    stimuli_grid,
    title_main
):
    """
    Show 4 subplots in 3D PCA space, similarly to your original code but in 3D.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title_main, fontsize=16)

    # PCA from unmod GRID
    pca_grid = PCA(n_components=3)
    pca_grid.fit(responses_grid_unmod)
    grid_unmod_3d = pca_grid.transform(responses_grid_unmod)
    noise_unmod_3d_in_grid = pca_grid.transform(responses_noise_unmod)

    # PCA from unmod NOISE
    pca_noise = PCA(n_components=3)
    pca_noise.fit(responses_noise_unmod)
    noise_unmod_3d = pca_noise.transform(responses_noise_unmod)
    grid_unmod_3d_in_noise = pca_noise.transform(responses_grid_unmod)

    # Subplot 1: Unmod – PCA from Grid
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    sc1 = ax1.scatter(
        grid_unmod_3d[:, 0], grid_unmod_3d[:, 1], grid_unmod_3d[:, 2],
        c=stimuli_grid[:, 1], cmap='winter', s=30, alpha=0.8, label='Unmod Grid'
    )
    ax1.scatter(
        noise_unmod_3d_in_grid[:, 0], noise_unmod_3d_in_grid[:, 1], noise_unmod_3d_in_grid[:, 2],
        c='gray', alpha=0.3, s=10, label='Unmod Noise'
    )
    ax1.set_title("Unmod – PCA from Grid")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.legend()
    set_axes_equal(ax1)

    # Subplot 2: Unmod – PCA from Noise
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    sc2 = ax2.scatter(
        noise_unmod_3d[:, 0], noise_unmod_3d[:, 1], noise_unmod_3d[:, 2],
        c='gray', alpha=0.3, s=10, label='Unmod Noise'
    )
    ax2.scatter(
        grid_unmod_3d_in_noise[:, 0], grid_unmod_3d_in_noise[:, 1], grid_unmod_3d_in_noise[:, 2],
        c=stimuli_grid[:, 1], cmap='winter', s=30, alpha=0.8, label='Unmod Grid'
    )
    ax2.set_title("Unmod – PCA from Noise")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.legend()
    set_axes_equal(ax2)

    # Modded in Grid PCA
    grid_mod_3d = pca_grid.transform(responses_grid_mod)
    noise_mod_3d_in_grid = pca_grid.transform(responses_noise_mod)

    # Subplot 3: Mod – PCA from Grid
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    sc3 = ax3.scatter(
        grid_mod_3d[:, 0], grid_mod_3d[:, 1], grid_mod_3d[:, 2],
        c=stimuli_grid[:, 1], cmap='spring', s=30, alpha=0.8, label='Mod Grid'
    )
    ax3.scatter(
        noise_mod_3d_in_grid[:, 0], noise_mod_3d_in_grid[:, 1], noise_mod_3d_in_grid[:, 2],
        c='gray', alpha=0.3, s=10, label='Mod Noise'
    )
    ax3.set_title("Mod – PCA from Grid")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.legend()
    set_axes_equal(ax3)

    # Modded in Noise PCA
    grid_mod_3d_in_noise = pca_noise.transform(responses_grid_mod)
    noise_mod_3d_in_noise = pca_noise.transform(responses_noise_mod)

    # Subplot 4: Mod – PCA from Noise
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    sc4 = ax4.scatter(
        noise_mod_3d_in_noise[:, 0], noise_mod_3d_in_noise[:, 1], noise_mod_3d_in_noise[:, 2],
        c='gray', alpha=0.3, s=10, label='Mod Noise'
    )
    ax4.scatter(
        grid_mod_3d_in_noise[:, 0], grid_mod_3d_in_noise[:, 1], grid_mod_3d_in_noise[:, 2],
        c=stimuli_grid[:, 1], cmap='spring', s=30, alpha=0.8, label='Mod Grid'
    )
    ax4.set_title("Mod – PCA from Noise")
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_zlabel("PC3")
    ax4.legend()
    set_axes_equal(ax4)

    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# 5) External Noise in 3rd PC
# -------------------------------------------------
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

def plot_3d_pca_grid_and_external_noise(responses_grid, external_noise_responses, stimuli_grid, title_main="3D PCA"):
    """
    Plot the original grid responses plus the new external noise in 3D PCA space.
    """
    pca_3 = PCA(n_components=3)
    pca_3.fit(responses_grid)

    grid_3d = pca_3.transform(responses_grid)
    noise_3d = pca_3.transform(external_noise_responses)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc1 = ax.scatter(
        grid_3d[:, 0],
        grid_3d[:, 1],
        grid_3d[:, 2],
        c=stimuli_grid[:, 1],
        cmap='coolwarm',
        s=40,
        alpha=0.8,
        label='Grid Responses'
    )

    sc2 = ax.scatter(
        noise_3d[:, 0],
        noise_3d[:, 1],
        noise_3d[:, 2],
        c='gray',
        s=10,
        alpha=0.3,
        label='External Noise (3rd PC)'
    )

    ax.set_title(title_main, fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    set_axes_equal(ax)
    cbar = fig.colorbar(sc1, ax=ax, label="Color Stimulus Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
