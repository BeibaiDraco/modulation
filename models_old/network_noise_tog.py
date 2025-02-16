import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(15)

# Parameters
N = 300  # Number of neurons
K = 2     # Two features
num_stimuli = 10  # Number of stimuli per feature dimension

# Initialize selectivity matrix
S = np.zeros((N, K))

# =======================
# First Half of Neurons
S[:N//2, 0] = np.random.rand(N//2)
S[:N//2, 1] = (1/2 - S[:N//2, 0]/2)

negative_indices = S[:N//2, 0] - S[:N//2, 1] < 0
S[:N//2, 0][negative_indices] = np.random.uniform(0, 0.5, size=np.sum(negative_indices))

S[N//2:, 1] = S[:N//2, 0]
S[N//2:, 0] = S[:N//2, 1]

# Initialize and normalize W_F
W_F = np.zeros((N, K))
for i in range(N):
    if S[i, 0]>0.5:
       W_F[i, 0] = S[i, 0]
    elif S[i, 1]>0.5:
       W_F[i, 1] = S[i, 1]
    else:
         W_F[i, 0] = S[i, 0]
         W_F[i, 1] = S[i, 1]

row_sums = W_F.sum(axis=1, keepdims=True)
nonzero_mask = row_sums != 0
W_F_normalized = np.zeros_like(W_F)
W_F_normalized[nonzero_mask[:, 0], :] = W_F[nonzero_mask[:, 0], :] / row_sums[nonzero_mask].reshape(-1, 1)
W_F = W_F_normalized

# Initialize W_R with structured connectivity
W_R = np.zeros((N, N))  # Start with all connections as zero
half_N = N // 2
p_high = 0.25
p_low = 0.25

shape_shape_mask = np.random.rand(half_N, half_N) < p_high
W_R[:half_N, :half_N][shape_shape_mask] = np.random.rand(np.sum(shape_shape_mask)) * 0.1

shape_color_mask = np.random.rand(half_N, N - half_N) < p_low
W_R[:half_N, half_N:][shape_color_mask] = np.random.rand(np.sum(shape_color_mask)) * 0.1

color_shape_mask = np.random.rand(N - half_N, half_N) < p_low
W_R[half_N:, :half_N][color_shape_mask] = np.random.rand(np.sum(color_shape_mask)) * 0.1

color_color_mask = np.random.rand(N - half_N, N - half_N) < p_high
W_R[half_N:, half_N:][color_color_mask] = np.random.rand(np.sum(color_color_mask)) * 0.1

# Remove self-connections
np.fill_diagonal(W_R, 0)

# Optional: Distance-based tuning (off by default)
WR_tuned = False
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
desired_radius = 0.9
W_R = W_R * (desired_radius / scaling_factor)

# ---------------------------------------------
# Compute Grid Responses (Deterministic: W^F * x + recurrent)
# ---------------------------------------------
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

responses = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

# Fit PCA on the grid responses
original_pca = PCA(n_components=2)
original_pca.fit(responses)
responses_pca = original_pca.transform(responses)

# ---------------------------------------------
# Compute Noise-Only Responses (x=0, just noise)
# ---------------------------------------------
num_noise_trials = 100
noise_level = 0.05
I = np.eye(N)
inv_I_minus_WR = np.linalg.inv(I - W_R)

# For simplicity, let's just consider noise-only trials (no stimulus) as a separate batch
# We'll generate a set of noise-only responses:
noise_only_responses = np.zeros((num_noise_trials, N))
for t in range(num_noise_trials):
    adjusted_input = np.random.randn(N) * noise_level
    noise_only_responses[t, :] = inv_I_minus_WR @ adjusted_input

# Project noise-only responses onto the same PC space
noise_only_pca = original_pca.transform(noise_only_responses)

# ---------------------------------------------
# Visualization
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(8,6))

# Plot deterministic grid responses (W^F * x + recurrent)
scatter_clean = ax.scatter(responses_pca[:, 0], responses_pca[:, 1],
                           c=stimuli_grid[:, 1], cmap='winter', s=20, alpha=0.7, label='Grid Responses (W^F*x + recurrent)')

# Plot noise-only responses
scatter_noisy = ax.scatter(noise_only_pca[:, 0], noise_only_pca[:, 1],
                           c='gray', alpha=0.5, s=20, label='Noise-Only Responses (x=0)')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Grid-based (W^F*x + recurrent) vs Noise-Only Responses in PCA Space')
ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()
