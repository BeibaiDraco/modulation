import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape and color
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

# Feature selectivities
S = np.random.rand(N, K)

# Define matrices and vectors
W_F = np.random.rand(N, K) * 0.1
W_R = np.zeros((N, N))
threshold = 0.1
for i in range(N):
    for j in range(N):
        distance = np.linalg.norm(S[i] - S[j])
        if distance < threshold:
            W_R[i, j] = 1 - distance / threshold

# Normalize W_R
eigenvalues = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues))
W_R = W_R / (scaling_factor + 1)

# Stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Storage for responses
responses = np.zeros((len(stimuli_grid), N))

# Analytical steady states for each stimulus
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

# Compute PCA on original responses
pca_original = PCA(n_components=3)
responses_pca_original = pca_original.fit_transform(responses)

# Rotate the features
theta = -np.deg2rad(45)  # Rotate by -45 degrees to align shape with PC1
R_feat = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])

# Rotate the stimuli grid
stimuli_grid_rotated = stimuli_grid @ R_feat.T

# Storage for rotated responses
responses_rotated = np.zeros((len(stimuli_grid_rotated), N))

# Analytical steady states for each rotated stimulus
for idx, (shape_rot, color_rot) in enumerate(stimuli_grid_rotated):
    F_rot = np.array([shape_rot, color_rot])  # Rotated external stimulus
    adjusted_F_rot = W_F @ F_rot
    responses_rotated[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F_rot

# Compute PCA on rotated responses
# Project rotated responses onto original PCA components for comparison
responses_rotated_projected = pca_original.transform(responses_rotated)

# Plot side by side: Original vs. Rotated Responses in PCA space
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original data
axes[0].scatter(responses_pca_original[:, 0], responses_pca_original[:, 1],
                c=stimuli_grid[:, 0], cmap='Reds')
axes[0].set_title('Original Responses in PCA Space')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True)

# Rotated data
axes[1].scatter(responses_rotated_projected[:, 0], responses_rotated_projected[:, 1],
                c=stimuli_grid[:, 0], cmap='Reds')
axes[1].set_title('Rotated Responses in Original PCA Space')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------
# Additional Plots: Stimuli and Responses Comparison
# -------------------------------------------

# Plot stimuli: Original vs. Rotated
plt.figure(figsize=(8, 6))
plt.scatter(stimuli_grid[:, 0], stimuli_grid[:, 1], color='blue', label='Original Stimuli')
plt.scatter(stimuli_grid_rotated[:, 0], stimuli_grid_rotated[:, 1], color='red', label='Rotated Stimuli')
for i in range(len(stimuli_grid)):
    plt.plot([stimuli_grid[i, 0], stimuli_grid_rotated[i, 0]],
             [stimuli_grid[i, 1], stimuli_grid_rotated[i, 1]], color='gray', linewidth=0.5)
plt.xlabel('Shape Stimulus')
plt.ylabel('Color Stimulus')
plt.title('Original and Rotated Stimuli')
plt.legend()
plt.grid(True)
plt.show()

# Plot responses: Original vs. Rotated in PCA space
plt.figure(figsize=(8, 6))
plt.scatter(responses_pca_original[:, 0], responses_pca_original[:, 1],
            color='blue', label='Original Responses')
plt.scatter(responses_rotated_projected[:, 0], responses_rotated_projected[:, 1],
            color='red', label='Rotated Responses')
for i in range(len(responses_pca_original)):
    plt.plot([responses_pca_original[i, 0], responses_rotated_projected[i, 0]],
             [responses_pca_original[i, 1], responses_rotated_projected[i, 1]],
             color='gray', linewidth=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Original and Rotated Responses in Original PCA Space')
plt.legend()
plt.grid(True)
plt.show()