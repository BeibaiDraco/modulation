import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(15)

# Parameters
N = 300  # Number of neurons
K = 2     # Two features (originally shape and color, now irrelevant as we fix them to 0)
num_stimuli = 10  # Number of stimuli per feature dimension (not used meaningfully anymore)

# Initialize selectivity matrix
S = np.zeros((N, K))

# =======================
# First Half of Neurons
S[:N//2, 0] = np.random.rand(N//2)

# Compute S[:N//2, 1] based on the formula
S[:N//2, 1] = (1/2 - S[:N//2, 0]/2)

# Identify indices where S[:N//2, 1] is negative
negative_indices = S[:N//2, 0] - S[:N//2, 1] < 0
# Replace negative values in S[:N//2, 1] with random values in (0, 0.5)
S[:N//2, 0][negative_indices] = np.random.uniform(0, 0.5, size=np.sum(negative_indices))

# For simplicity, we keep only the first half structure
S[N//2:, 1] = S[:N//2, 0]
S[N//2:, 0] = S[:N//2, 1]

# Initialize and normalize W_F (though it won't matter much if x=0)
W_F = np.zeros((N, K))
for i in range(N):
    if S[i, 0] > 0.5:
       W_F[i, 0] = S[i, 0]
    elif S[i, 1] > 0.5:
       W_F[i, 1] = S[i, 1]
    else:
       W_F[i, 0] = S[i, 0]
       W_F[i, 1] = S[i, 1]

# Normalize W_F rows
row_sums = W_F.sum(axis=1, keepdims=True)
nonzero_mask = row_sums != 0
W_F_normalized = np.zeros_like(W_F)
W_F_normalized[nonzero_mask[:, 0], :] = W_F[nonzero_mask[:, 0], :] / row_sums[nonzero_mask].reshape(-1, 1)
W_F = W_F_normalized

# Initialize W_R with structured connectivity
W_R = np.zeros((N, N))  # Start with all connections as zero

half_N = N // 2

# Define connection probabilities
p_high = 0.25
p_low = 0.25

# Shape-Shape Block (Top-Left)
shape_shape_mask = np.random.rand(half_N, half_N) < p_high
W_R[:half_N, :half_N][shape_shape_mask] = (np.random.rand(np.sum(shape_shape_mask)))  
#now it is all postive, i want it also could be negitive

# Shape-Color Block (Top-Right)
shape_color_mask = np.random.rand(half_N, N - half_N) < p_low
W_R[:half_N, half_N:][shape_color_mask] = (np.random.rand(np.sum(shape_color_mask))) 

# Color-Shape Block (Bottom-Left)
color_shape_mask = np.random.rand(N - half_N, half_N) < p_low
W_R[half_N:, :half_N][color_shape_mask] = np.random.rand(np.sum(color_shape_mask))

# Color-Color Block (Bottom-Right)
color_color_mask = np.random.rand(N - half_N, N - half_N) < p_high
W_R[half_N:, half_N:][color_color_mask] = np.random.rand(np.sum(color_color_mask)) 

# Remove self-connections
np.fill_diagonal(W_R, 0)

# (Optional distance-based tuning skipped)
WR_tuned = False
if WR_tuned:
    threshold = 0.8
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
print(np.linalg.eigvals(W_R)[:10])
#W_R = np.zeros((N, N))
# Since we're fixing x=0, we do not need stimuli as before, but let's keep the grid for indexing
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Without external input (x=0), the deterministic steady-state (without noise) would be zero
# Let's verify that:
responses = np.zeros((len(stimuli_grid), N))  # Will just store zero for reference
# If x=0, adjusted_F = 0, so steady state r = inv(I - W_R)*0 = 0
# responses stays all zeros.



# Introduce noise trials
num_noise_trials = 100   # Number of noisy trials per stimulus
noise_level = 0.05        # Noise magnitude

I = np.eye(N)
inv_I_minus_WR = np.linalg.inv(np.eye(N) - W_R)

# Prepare array for noisy responses
noisy_responses = np.zeros((len(stimuli_grid), num_noise_trials, N))

for i, (shape_val, color_val) in enumerate(stimuli_grid):
    for t in range(num_noise_trials):
        # Instead of W_F * (x + noise) with x variable,
        # We now have no external (deterministic) input: x=0.
        # Just add noise as input directly:
        adjusted_input = np.random.randn(N) * noise_level
        
        # Compute steady-state response with noise-only input
        noisy_responses[i, t, :] = inv_I_minus_WR @ adjusted_input

# Compute mean responses across trials (per stimulus)
mean_response_per_stim = np.mean(noisy_responses, axis=1, keepdims=True)  
noise_only = noisy_responses - mean_response_per_stim




# Reshape noise data: (num_stimuli_total * num_noise_trials, N)
num_stimuli_total = len(stimuli_grid)
noise_data = noise_only.reshape(num_stimuli_total * num_noise_trials, N)

# Fit PCA on the "original responses" (all zero in this scenario, so PCA is trivial)
original_pca = PCA(n_components=4)

#original_pca.fit(responses)
original_pca.fit(noise_data)


# Compute noise covariance
noise_cov = np.cov(noise_data, rowvar=False)

# Eigen-decomposition to find max noise variance axis
eigvals, eigvecs = np.linalg.eig(noise_cov)
idx_max = np.argmax(eigvals)
noise_max_axis = eigvecs[:, idx_max].real

# Project noise-dominant axis onto original PC space
pc_projection = original_pca.components_ @ noise_max_axis

print("Projection of noise-dominant axis onto PC1 and PC2:")
print("PC1 projection:", pc_projection[0])
print("PC2 projection:", pc_projection[1])
print("PC3 projection:", pc_projection[2])
print("PC4 projection:", pc_projection[3])

# --- Visualization ---

# Project original (clean) responses and a subset of noisy responses onto PCA space
responses_pca = original_pca.transform(responses)

# Let's pick a random subset of noisy trials to visualize
subset_size = 500
indices = np.random.choice(noise_data.shape[0], subset_size, replace=False)
noisy_pca = original_pca.transform(noise_data[indices])

fig, ax = plt.subplots(figsize=(8,6))

# Plot the original "grid responses" (all zero)
scatter_clean = ax.scatter(responses_pca[:, 0], responses_pca[:, 1],
                           c=stimuli_grid[:, 1], cmap='winter', s=20, alpha=0.7, label='Clean (Zero) Responses')

# Plot a subset of noisy trials
scatter_noisy = ax.scatter(noisy_pca[:, 0], noisy_pca[:, 1],
                           c='gray', alpha=0.3, s=10, label='Noisy Trials')

# Plot the noise-dominant axis
ax.arrow(0, 0, pc_projection[0]*3, pc_projection[1]*3,
         width=0.005, head_width=0.05, head_length=0.1,
         fc='red', ec='red', label='Noise-Dominant Axis')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('No External Input, Noise-Driven Responses in Original PC Space')
ax.legend()

ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()
