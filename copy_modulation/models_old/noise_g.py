import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# Parameters
# -------------------------

np.random.seed(15)
N = 300           # Number of neurons
K = 2             # Two features: shape (0) and color (1)
num_stimuli = 10  # Number of stimuli per feature dimension
num_noise_trials = 50  # Number of noisy trials per stimulus
noise_level = 0.05       # Noise magnitude
desired_radius = 0.9    # Desired spectral radius for stability scaling
p_high = 0.25
p_low = 0.05

# -------------------------------------------------
# 1) Initialization Functions
# -------------------------------------------------
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
    Compute modulated grid responses using G = diag(1 + 0.15*(color-shape)).
    """
    selectivity_diff = S[:, 1] - S[:, 0]
    g_vector = 1.0 + 0.2 * selectivity_diff

    G = np.diag(g_vector)

    I = np.eye(N)
    I_minus_GWR = I - G @ W_R

    # Check invertibility
    cond_number = np.linalg.cond(I_minus_GWR)
    if cond_number > 1 / np.finfo(I_minus_GWR.dtype).eps:
        raise ValueError("(I - G W_R) is nearly singular.")

    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)
    G_WF = G @ W_F

    # Compute modulated grid responses
    #mod_responses = []
    mod_responses = np.zeros((len(stimuli_grid), N))

    for idx, (shape_val, color_val) in enumerate(stimuli_grid):
        F = np.array([shape_val, color_val])
        #mod_responses.append(inv_I_minus_GWR @ (G_WF @ F))
        mod_responses[idx] = inv_I_minus_GWR @ (G_WF @ F)
        
    return np.array(mod_responses)

def compute_modulated_noise(W_R, W_F, S, stimuli_grid, noise_level, num_noise_trials):
    """
    Similar to generate_noisy_responses, but with gain modulation G.
    For each stimulus, we create noise input and pass it through (I - G W_R)^(-1) * G.
    Returns shape = (num_stimuli_grid * num_noise_trials, N)
    """
    selectivity_diff = S[:, 1] - S[:, 0]
    g_vector = 1.0 + 0.15 * selectivity_diff
    G = np.diag(g_vector)

    I = np.eye(N)
    I_minus_GWR = I - G @ W_R
    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)

    noisy_responses = []
    for (shape_val, color_val) in stimuli_grid:
        for _ in range(num_noise_trials):
            noise_input = np.random.randn(N) * noise_level
            # Apply gain to the noise as well:
            modded_noise = G @ noise_input
            noisy_responses.append(inv_I_minus_GWR @ modded_noise)

    return np.array(noisy_responses)

# -------------------------------------------------
# 4) Visualization: 4 Subplots, Each Shows Both Grid & Noise
# -------------------------------------------------
def visualize_four_subplots(
    responses_grid_unmod,    # shape (grid_size, N)
    responses_noise_unmod,   # shape (grid_size * num_noise_trials, N)
    responses_grid_mod,      # shape (grid_size, N)
    responses_noise_mod,     # shape (grid_size * num_noise_trials, N)
    stimuli_grid,
    title_main
):
    """
    4 subplots, each with BOTH grid and noise:
      Top-left: PCA from unmod GRID; plot unmod GRID (colored by color dimension) + unmod NOISE (gray).
      Top-right: PCA from unmod NOISE; plot unmod NOISE (gray) + unmod GRID (colored).
      Bottom-left: Re-use PCA from unmod GRID, plot mod GRID (colored) + mod NOISE (gray).
      Bottom-right: Re-use PCA from unmod NOISE, plot mod GRID (colored) + mod NOISE (gray).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title_main, fontsize=16)

    # ========== 1) PCA from Unmod GRID ==========
    pca_grid = PCA(n_components=2)
    pca_grid.fit(responses_grid_unmod)  # fit on unmod grid only

    # Transform both unmod grid & noise
    grid_unmod_pca = pca_grid.transform(responses_grid_unmod)
    noise_unmod_pca_in_grid = pca_grid.transform(responses_noise_unmod)

    # Plot (grid in color, noise in gray)
    ax1 = axes[0, 0]
    ax1.scatter(
        grid_unmod_pca[:, 0],
        grid_unmod_pca[:, 1],
        c=stimuli_grid[:, 1],  # color dimension
        cmap='winter',
        s=30,
        alpha=0.8,
        label='Unmod Grid'
    )
    ax1.scatter(
        noise_unmod_pca_in_grid[:, 0],
        noise_unmod_pca_in_grid[:, 1],
        c='gray',
        alpha=0.3,
        s=10,
        label='Unmod Noise'
    )
    ax1.set_title("Unmodulated – PCA from Grid")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()

    # ========== 2) PCA from Unmod NOISE ==========
    pca_noise = PCA(n_components=2)
    pca_noise.fit(responses_noise_unmod)  # fit on unmod noise

    # Transform both unmod noise & grid
    noise_unmod_pca = pca_noise.transform(responses_noise_unmod)
    grid_unmod_pca_in_noise = pca_noise.transform(responses_grid_unmod)

    ax2 = axes[0, 1]
    ax2.scatter(
        noise_unmod_pca[:, 0],
        noise_unmod_pca[:, 1],
        c='gray',
        alpha=0.3,
        s=10,
        label='Unmod Noise'
    )
    ax2.scatter(
        grid_unmod_pca_in_noise[:, 0],
        grid_unmod_pca_in_noise[:, 1],
        c=stimuli_grid[:, 1],
        cmap='winter',
        s=30,
        alpha=0.8,
        label='Unmod Grid'
    )
    ax2.set_title("Unmodulated – PCA from Noise")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.grid(True)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()

    # ========== 3) Modulated in Grid PCA-Space ==========
    # Re-use pca_grid from above
    grid_mod_pca = pca_grid.transform(responses_grid_mod)
    noise_mod_pca_in_grid = pca_grid.transform(responses_noise_mod)

    ax3 = axes[1, 0]
    ax3.scatter(
        grid_mod_pca[:, 0],
        grid_mod_pca[:, 1],
        c=stimuli_grid[:, 1],
        cmap='spring',
        s=30,
        alpha=0.8,
        label='Mod Grid'
    )
    ax3.scatter(
        noise_mod_pca_in_grid[:, 0],
        noise_mod_pca_in_grid[:, 1],
        c='gray',
        alpha=0.3,
        s=10,
        label='Mod Noise'
    )
    ax3.set_title("Modulated – PCA from Grid")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.grid(True)
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend()

    # ========== 4) Modulated in Noise PCA-Space ==========
    # Re-use pca_noise from top-right
    grid_mod_pca_in_noise = pca_noise.transform(responses_grid_mod)
    noise_mod_pca_in_noise = pca_noise.transform(responses_noise_mod)

    ax4 = axes[1, 1]
    ax4.scatter(
        noise_mod_pca_in_noise[:, 0],
        noise_mod_pca_in_noise[:, 1],
        c='gray',
        alpha=0.3,
        s=10,
        label='Mod Noise'
    )
    ax4.scatter(
        grid_mod_pca_in_noise[:, 0],
        grid_mod_pca_in_noise[:, 1],
        c=stimuli_grid[:, 1],
        cmap='spring',
        s=30,
        alpha=0.8,
        label='Mod Grid'
    )
    ax4.set_title("Modulated – PCA from Noise")
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.grid(True)
    ax4.set_aspect('equal', adjustable='box')
    ax4.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# 5) Main Code
# -------------------------------------------------
if __name__ == "__main__":

    # 1) Selectivity and Feedforward
    S = initialize_selectivity_matrix(N, K)
    W_F = initialize_W_F(S)

    # 2) Stimuli
    shape_stimuli = np.linspace(0, 1, num_stimuli)
    color_stimuli = np.linspace(0, 1, num_stimuli)

    # -------------------------------------------------
    # Example: UNTUNED W_R
    # -------------------------------------------------
    W_R_untuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)
    responses_grid_unmod, stimuli_grid = compute_responses(W_F, W_R_untuned, shape_stimuli, color_stimuli)
    responses_noise_unmod = generate_noisy_responses(W_R_untuned, noise_level, stimuli_grid, num_noise_trials)
    responses_grid_mod = compute_modulated_responses(W_R_untuned, W_F, S, stimuli_grid)
    responses_noise_mod = compute_modulated_noise(W_R_untuned, W_F, S, stimuli_grid, noise_level, num_noise_trials)
    
    
    visualize_four_subplots(
        responses_grid_unmod,
        responses_noise_unmod,
        responses_grid_mod,
        responses_noise_mod,
        stimuli_grid,
        "UNTUNED W_R"
    )

    # -------------------------------------------------
    # Example: TUNED W_R
    # -------------------------------------------------
    W_R_tuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=True, desired_radius=desired_radius)
    responses_grid_unmod_tuned, stimuli_grid_tuned = compute_responses(W_F, W_R_tuned, shape_stimuli, color_stimuli)
    responses_noise_unmod_tuned = generate_noisy_responses(W_R_tuned, noise_level, stimuli_grid_tuned, num_noise_trials)
    responses_grid_mod_tuned = compute_modulated_responses(W_R_tuned, W_F, S, stimuli_grid_tuned)
    responses_noise_mod_tuned = compute_modulated_noise(W_R_tuned, W_F, S, stimuli_grid_tuned, noise_level, num_noise_trials)

    visualize_four_subplots(
        responses_grid_unmod_tuned,
        responses_noise_unmod_tuned,
        responses_grid_mod_tuned,
        responses_noise_mod_tuned,
        stimuli_grid_tuned,
        "TUNED W_R"
    )

    # -------------------------------------------------
    # Example: CYCLE W_R
    # -------------------------------------------------
    W_R_cycle = initialize_W_R_cycle(N, desired_radius=desired_radius)
    responses_grid_unmod_cycle, stimuli_grid_cycle = compute_responses(W_F, W_R_cycle, shape_stimuli, color_stimuli)
    responses_noise_unmod_cycle = generate_noisy_responses(W_R_cycle, noise_level, stimuli_grid_cycle, num_noise_trials)
    responses_grid_mod_cycle = compute_modulated_responses(W_R_cycle, W_F, S, stimuli_grid_cycle)
    responses_noise_mod_cycle = compute_modulated_noise(W_R_cycle, W_F, S, stimuli_grid_cycle, noise_level, num_noise_trials)

    visualize_four_subplots(
        responses_grid_unmod_cycle,
        responses_noise_unmod_cycle,
        responses_grid_mod_cycle,
        responses_noise_mod_cycle,
        stimuli_grid_cycle,
        "CYCLE W_R"
    )

    print("All done! Good!")
