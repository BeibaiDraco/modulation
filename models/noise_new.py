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
noise_level = 0.5       # Noise magnitude
desired_radius = 0.9     # Desired spectral radius for stability scaling
WR_tuned = False
p_high = 0.25
p_low = 0.25

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
        W_F[i] = S[i] / np.sum(S[i]) if np.sum(S[i]) > 0 else S[i]
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

    # Scale W_R to ensure stability (spectral radius < desired_radius)
    eigenvalues = np.linalg.eigvals(W_R)
    scaling_factor = np.max(np.abs(eigenvalues))
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

    return noisy_responses.reshape(num_noise_trials * len(stimuli_grid), N)

def visualize_responses(responses, noisy_responses, stimuli_grid, original_pca, title):
    """
    Visualize deterministic and noisy responses in PCA space.
    """
    responses_pca = original_pca.transform(responses)
    noisy_pca = original_pca.transform(noisy_responses)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the deterministic responses in PC space
    scatter_clean = ax.scatter(
        responses_pca[:, 0], responses_pca[:, 1],
        c=stimuli_grid[:, 1], cmap='winter', s=20, alpha=0.7, label='Grid Responses (Mean)'
    )

    # Plot the noise trials in PC space
    scatter_noisy = ax.scatter(
        noisy_pca[:, 0], noisy_pca[:, 1],
        c='gray', alpha=0.3, s=10, label='Noise Trials'
    )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True)
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

# PCA setup
original_pca = PCA(n_components=2)

# --- Plot 1: Untuned W_R ---
W_R_untuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)
responses_untuned, stimuli_grid = compute_responses(W_F, W_R_untuned, shape_stimuli, color_stimuli)
noisy_responses_untuned = generate_noisy_responses(W_R_untuned, noise_level, stimuli_grid, num_noise_trials)
original_pca.fit(responses_untuned)
visualize_responses(responses_untuned, noisy_responses_untuned, stimuli_grid, original_pca, "Untuned W_R")

# --- Plot 2: Tuned W_R ---
W_R_tuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=True, desired_radius=desired_radius)
responses_tuned, _ = compute_responses(W_F, W_R_tuned, shape_stimuli, color_stimuli)
noisy_responses_tuned = generate_noisy_responses(W_R_tuned, noise_level, stimuli_grid, num_noise_trials)
original_pca.fit(responses_tuned)
visualize_responses(responses_tuned, noisy_responses_tuned, stimuli_grid, original_pca, "Tuned W_R")

# --- Plot 3: Cycle W_R ---
def initialize_W_R_cycle(N, desired_radius):
    W_R = np.zeros((N, N))
    for i in range(N):
        W_R[i, (i + 1) % N] = desired_radius
    return W_R

W_R_cycle = initialize_W_R_cycle(N, desired_radius)
responses_cycle, _ = compute_responses(W_F, W_R_cycle, shape_stimuli, color_stimuli)
noisy_responses_cycle = generate_noisy_responses(W_R_cycle, noise_level, stimuli_grid, num_noise_trials)
original_pca.fit(responses_cycle)
visualize_responses(responses_cycle, noisy_responses_cycle, stimuli_grid, original_pca, "Cycle W_R")
