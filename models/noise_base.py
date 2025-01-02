import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# Parameters
# -------------------------
np.random.seed(10)
N = 300           # Number of neurons
K = 2             # Two features: shape (0) and color (1)
num_stimuli = 10  # Number of stimuli per feature dimension
num_noise_trials = 50  # Number of noisy trials per stimulus
noise_level = 0.1       # Noise magnitude
desired_radius = 0.9    # Desired spectral radius for stability scaling
WR_tuned = False
p_high = 0.3
p_low = 0.3

# -------------------------
# Functions
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
            W_F[i] = S[i]  # remains all zeros if the row sum is 0
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

    # Optional tuning: boost connections if selectivities are similar
    if WR_tuned:
        threshold = 1
        for i in range(N):
            for j in range(N):
                if i != j:
                    distance = np.linalg.norm(S[i] - S[j])
                    if distance < threshold:
                        # Scale by some factor that depends on distance
                        W_R[i, j] *= (2 - distance / threshold)

    # Scale W_R to ensure stability (spectral radius < desired_radius)
    eigenvalues = np.linalg.eigvals(W_R)
    scaling_factor = np.max(np.abs(eigenvalues))
    if scaling_factor > 0:
        W_R = W_R * (desired_radius / scaling_factor)

    return W_R

def compute_responses(W_F, W_R, shape_stimuli, color_stimuli):
    """
    Compute the network responses for a grid of stimuli defined by shape_stimuli and color_stimuli.
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
    Generate noise-only responses using the loop-based method to preserve stimulus-specificity.
    """
    inv_I_minus_WR = np.linalg.inv(np.eye(N) - W_R)
    noisy_responses = np.zeros((len(stimuli_grid), num_noise_trials, N))

    for i, (shape_val, color_val) in enumerate(stimuli_grid):
        for t in range(num_noise_trials):
            adjusted_input = np.random.randn(N) * noise_level
            noisy_responses[i, t, :] = inv_I_minus_WR @ adjusted_input

    # Shape is (num_stimuli_grid, num_noise_trials, N) -> flatten trials
    return noisy_responses.reshape(num_noise_trials * len(stimuli_grid), N)

# New function to produce two subplots:
# (1) PCA derived from the grid responses
# (2) PCA derived from the noisy responses
def visualize_in_two_pca_spaces(responses, noisy_responses, stimuli_grid, fig_title):
    """
    Create a figure with 2 subplots:
      Left subplot : PCA is computed from the deterministic grid responses.
      Right subplot: PCA is computed from the noisy responses.
    In both subplots, we plot the grid responses and noisy responses in the derived PCA space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(fig_title, fontsize=16)

    # 1) PCA based on grid responses
    pca_grid = PCA(n_components=2)
    pca_grid.fit(responses)

    # Transform both grid and noisy responses
    responses_pca_grid = pca_grid.transform(responses)
    noisy_pca_grid = pca_grid.transform(noisy_responses)

    axes[0].scatter(
        responses_pca_grid[:, 0], 
        responses_pca_grid[:, 1],
        c=stimuli_grid[:, 1], 
        cmap='winter',
        s=30, 
        alpha=0.7, 
        label='Grid Responses (Mean)'
    )
    axes[0].scatter(
        noisy_pca_grid[:, 0], 
        noisy_pca_grid[:, 1],
        c='gray',
        alpha=0.3, 
        s=10, 
        label='Noise Trials'
    )
    axes[0].set_title('PCA from Grid Responses')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_aspect('equal', adjustable='box')
    #increase the ylim to make the plot more readable
    axes[0].set_ylim(-50, 50)

    # 2) PCA based on noisy responses
    pca_noise = PCA(n_components=2)
    pca_noise.fit(noisy_responses)

    # Transform both grid and noisy responses
    responses_pca_noise = pca_noise.transform(responses)
    noisy_pca_noise = pca_noise.transform(noisy_responses)

    axes[1].scatter(
        responses_pca_noise[:, 0], 
        responses_pca_noise[:, 1],
        c=stimuli_grid[:, 1],
        cmap='winter',
        s=30, 
        alpha=0.7, 
        label='Grid Responses (Mean)'
    )
    axes[1].scatter(
        noisy_pca_noise[:, 0], 
        noisy_pca_noise[:, 1],
        c='gray',
        alpha=0.3, 
        s=10, 
        label='Noise Trials'
    )
    axes[1].set_title('PCA from Noise Responses')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].legend()
    axes[1].grid(True)
    #axes[1].set_aspect('equal', adjustable='box')
    #axes[1].set_ylim(-50, 50)

    plt.tight_layout()
    plt.show()

# -------------------------
# Main Code
# -------------------------
# Initialize selectivity
S = initialize_selectivity_matrix(N, K)

# Initialize W_F
W_F = initialize_W_F(S)

# Define stimuli
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# --- 1) Untuned W_R ---
W_R_untuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)
responses_untuned, _ = compute_responses(W_F, W_R_untuned, shape_stimuli, color_stimuli)
noisy_responses_untuned = generate_noisy_responses(W_R_untuned, noise_level, stimuli_grid, num_noise_trials)
visualize_in_two_pca_spaces(responses_untuned, noisy_responses_untuned, stimuli_grid, "Untuned W_R")

# --- 2) Tuned W_R ---
W_R_tuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=True, desired_radius=desired_radius)
responses_tuned, _ = compute_responses(W_F, W_R_tuned, shape_stimuli, color_stimuli)
noisy_responses_tuned = generate_noisy_responses(W_R_tuned, noise_level, stimuli_grid, num_noise_trials)
visualize_in_two_pca_spaces(responses_tuned, noisy_responses_tuned, stimuli_grid, "Tuned W_R")

# --- 3) Cycle W_R ---
def initialize_W_R_cycle(N, desired_radius):
    W_R = np.zeros((N, N))
    for i in range(N):
        # Each neuron connects to the next in a ring with weight = desired_radius
        W_R[i, (i + 1) % N] = desired_radius
    return W_R

W_R_cycle = initialize_W_R_cycle(N, desired_radius)
responses_cycle, _ = compute_responses(W_F, W_R_cycle, shape_stimuli, color_stimuli)
noisy_responses_cycle = generate_noisy_responses(W_R_cycle, noise_level, stimuli_grid, num_noise_trials)
visualize_in_two_pca_spaces(responses_cycle, noisy_responses_cycle, stimuli_grid, "Cycle W_R")
