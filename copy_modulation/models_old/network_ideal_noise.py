import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(15)

# Parameters
N = 300  # Number of neurons
K = 2     # Two features: shape (index=0) and color (index=1)
num_stimuli = 10  # Number of stimuli per feature dimension

# Initialize selectivity matrix
S = np.zeros((N, K))

# =======================
# First Half of Neurons
S[:N//2, 0] = np.random.rand(N//2)
S[:N//2, 1] = (1/2 - S[:N//2, 0]/2)

negative_indices = S[:N//2, 0] - S[:N//2, 1] < 0
S[:N//2, 0][negative_indices] = np.random.uniform(0, 0.5, size=np.sum(negative_indices))

# Mirror for second half
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
W_R = np.zeros((N, N))
half_N = N // 2
p_high = 0.25
p_low = 0.25

shape_shape_mask = np.random.rand(half_N, half_N) < p_high
W_R[:half_N, :half_N][shape_shape_mask] = (np.random.rand(np.sum(shape_shape_mask)) )* 0.1

shape_color_mask = np.random.rand(half_N, N - half_N) < p_low
W_R[:half_N, half_N:][shape_color_mask] = (np.random.rand(np.sum(shape_color_mask))) * 0.1

color_shape_mask = np.random.rand(N - half_N, half_N) < p_low
W_R[half_N:, :half_N][color_shape_mask] = (np.random.rand(np.sum(color_shape_mask))) * 0.1

color_color_mask = np.random.rand(N - half_N, N - half_N) < p_high
W_R[half_N:, half_N:][color_color_mask] = (np.random.rand(np.sum(color_color_mask)) )* 0.1

np.fill_diagonal(W_R, 0)

# Optional tuning off by default
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
inv_I_minus_WR = np.linalg.inv(np.eye(N) - W_R)
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus
    adjusted_F = W_F @ F
    responses[idx] = inv_I_minus_WR @ adjusted_F

# Fit PCA on the grid responses to define the stimulus PCA space
original_pca = PCA(n_components=2)
original_pca.fit(responses)
responses_pca = original_pca.transform(responses)

# ---------------------------------------------
# Compute Noise-Only Covariance from Trial-to-Trial Variability
# ---------------------------------------------
num_noise_trials = 50  # Number of noisy trials per stimulus
noise_level = 0.1      # noise magnitude
num_stimuli_total = len(stimuli_grid)

noisy_responses = np.zeros((num_stimuli_total, num_noise_trials, N))
I = np.eye(N)

# We'll generate trial-to-trial noise for each stimulus
# This creates a set of noisy responses to each stimulus
for i, (shape_val, color_val) in enumerate(stimuli_grid):
    for t in range(num_noise_trials):
        noisy_shape = shape_val + np.random.randn() * noise_level
        noisy_color = color_val + np.random.randn() * noise_level
        noisy_F = np.array([noisy_shape, noisy_color])
        adjusted_input = W_F @ noisy_F
        noisy_responses[i, t, :] = inv_I_minus_WR @ adjusted_input

# Compute the trial-to-trial deviations
mean_response_per_stim = np.mean(noisy_responses, axis=1, keepdims=True)
noise_only = noisy_responses - mean_response_per_stim

# Flatten noise data
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
subset_size = 500
if subset_size > noise_data.shape[0]:
    subset_size = noise_data.shape[0]
indices = np.random.choice(noise_data.shape[0], subset_size, replace=False)
noisy_pca = original_pca.transform(noise_data[indices])

# Plot both the stimulus-driven grid responses and the noise trials
fig, ax = plt.subplots(figsize=(8,6))
scatter_clean = ax.scatter(responses_pca[:, 0], responses_pca[:, 1],
                           c=stimuli_grid[:, 1], cmap='winter', s=20, alpha=0.7, label='Grid Responses (Mean)')
scatter_noisy = ax.scatter(noisy_pca[:, 0], noisy_pca[:, 1],
                           c='gray', alpha=0.3, s=10, label='Noise Trials')

# Plot the noise-dominant axis as an arrow from the origin
arrow_scale = 3.0
ax.arrow(0, 0, pc_projection[0]*arrow_scale, pc_projection[1]*arrow_scale,
         width=0.005, head_width=0.05, head_length=0.1,
         fc='red', ec='red', label='Noise-Dominant Axis')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
#set y lim
ax.set_ylim(-50, 50)
ax.set_title('Stimulus-Driven PC Space and Noise-Dominant Axis')
ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()
