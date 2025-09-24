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
noise_level = 0.1     # Noise magnitude
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
    Neurons strongly selective (>0.5) for one dimension
    are assigned weights primarily to that dimension.
    Otherwise, both dimensions share weights according to S.
    """
    N, K = S.shape
    W_F = np.zeros((N, K))
    for i in range(N):
        if S[i, 0] > 0.5:
           W_F[i, 0] = S[i, 0]
        elif S[i, 1] > 0.5:
           W_F[i, 1] = S[i, 1]
        else:
           W_F[i, 0] = S[i, 0]
           W_F[i, 1] = S[i, 1]

    # Normalize rows of W_F
    row_sums = W_F.sum(axis=1, keepdims=True)
    nonzero_mask = (row_sums != 0).ravel()
    W_F_normalized = np.zeros_like(W_F)
    W_F_normalized[nonzero_mask] = W_F[nonzero_mask] / row_sums[nonzero_mask]
    return W_F_normalized

def initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=0.9):
    """
    Initialize the recurrent connectivity matrix W_R.
    The matrix has structured connectivity such that neurons selective
    for shape connect strongly among themselves and similarly for color.
    Cross-feature connectivity is weaker.
    If WR_tuned is True, adjust weights based on similarity in S.
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
                if i == j:
                    continue
                distance = np.linalg.norm(S[i] - S[j])
                if distance < threshold:
                    W_R[i, j] *= (2 - distance / threshold)

    # Scale W_R to ensure stability (spectral radius < 1)
    eigenvalues = np.linalg.eigvals(W_R)
    scaling_factor = np.max(np.abs(eigenvalues))
    W_R = W_R * (desired_radius / scaling_factor)
    W_R = np.zeros((N, N))
    return W_R

def compute_responses(W_F, W_R, shape_stimuli, color_stimuli):
    """
    Compute the network responses for a grid of stimuli defined by shape_stimuli and color_stimuli.
    Returns:
        responses: array of shape (#stimuli, N) containing steady-state responses.
        stimuli_grid: array of shape (#stimuli, 2) with (shape, color) pairs.
    """
    stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)
    inv_I_minus_WR = np.linalg.inv(np.eye(N) - W_R)
    responses = np.zeros((len(stimuli_grid), N))

    for idx, (shape_val, color_val) in enumerate(stimuli_grid):
        F = np.array([shape_val, color_val])
        adjusted_F = W_F @ F
        responses[idx] = inv_I_minus_WR @ adjusted_F
    return responses, stimuli_grid

# -------------------------
# Main Code
# -------------------------
# Initialize selectivity
S = initialize_selectivity_matrix(N, K)

# Initialize W_F
W_F = initialize_W_F(S)

# Initialize W_R
W_R = initialize_W_R(N, p_high, p_low, S, WR_tuned=WR_tuned, desired_radius=desired_radius)

# Define stimuli
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)

# Compute deterministic (mean) responses for the stimuli grid
responses, stimuli_grid = compute_responses(W_F, W_R, shape_stimuli, color_stimuli)

# Fit PCA on the deterministic grid responses to define the PC space
original_pca = PCA(n_components=2)
original_pca.fit(responses)
responses_pca = original_pca.transform(responses)

# ---------------------------------------------
# Compute Noise-Only Responses (No External Input)
# ---------------------------------------------
inv_I_minus_WR = np.linalg.inv(np.eye(N) - W_R)
num_stimuli_total = len(stimuli_grid)

# Prepare array for noisy responses (no external input)
noisy_responses = np.zeros((len(stimuli_grid), num_noise_trials, N))
for i, (shape_val, color_val) in enumerate(stimuli_grid):
    for t in range(num_noise_trials):
        # No deterministic input, only noise:
        a= np.random.randn() * noise_level
        F = np.array([np.random.randn() * noise_level, np.random.randn() * noise_level])
        adjusted_F = W_F @ F
        # Compute steady-state response with noise-only input
        noisy_responses[i, t, :] = inv_I_minus_WR @ adjusted_F

# Compute mean responses across trials (per stimulus)
mean_response_per_stim = np.mean(noisy_responses, axis=1, keepdims=True)
noise_only = noisy_responses - mean_response_per_stim

# Reshape noise data: (num_stimuli_total * num_noise_trials, N)
noise_data = noise_only.reshape(num_stimuli_total * num_noise_trials, N)

# Compute noise covariance
noise_cov = np.cov(noise_data, rowvar=False)
eigvals, eigvecs = np.linalg.eig(noise_cov)
idx_max = np.argmax(eigvals)
noise_max_axis = eigvecs[:, idx_max].real

# Project the noise-dominant axis into the stimulus PCA space
pc_projection = original_pca.components_ @ noise_max_axis
print("Projection of noise-dominant axis onto PC1 and PC2:")
print("PC1 projection:", pc_projection[0])
print("PC2 projection:", pc_projection[1])

# For visualization, let's also show a subset of the noise data in the stimulus PCA space
subset_size = 10000
subset_size = min(subset_size, noise_data.shape[0])
indices = np.random.choice(noise_data.shape[0], subset_size, replace=False)
noisy_pca = original_pca.transform(noise_data[indices])
#noisy_pca = original_pca.transform(noise_data)
# -------------------------
# Visualization
# -------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the deterministic responses in PC space
scatter_clean = ax.scatter(
    responses_pca[:, 0], responses_pca[:, 1],
    c=stimuli_grid[:, 1], cmap='winter', s=20, alpha=0.7, label='Grid Responses '
)

# Plot the noise trials in the same PC space
scatter_noisy = ax.scatter(
    noisy_pca[:, 0], noisy_pca[:, 1],
    c='gray', alpha=0.3, s=10, label='Noise Trials'
)

# Plot the noise-dominant axis
arrow_scale = 3.0
ax.arrow(0, 0, pc_projection[0]*arrow_scale, pc_projection[1]*arrow_scale,    width=0.005, head_width=0.05, head_length=0.1,   fc='red', ec='red', label='Noise-Dominant Axis')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
#ax.set_ylim(-50, 50)
ax.set_title('Stimulus-Driven PC Space and Noise-Only Responses')
ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()
