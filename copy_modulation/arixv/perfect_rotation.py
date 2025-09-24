import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
eigenvalues_W_R = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues_W_R))
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

# Center the responses by subtracting the mean
responses_mean = np.mean(responses, axis=0)
responses_centered = responses - responses_mean

# Perform PCA to get eigenvalues and eigenvectors
pca = PCA(n_components=N)
pca.fit(responses_centered)

eigenvalues = pca.explained_variance_
U = pca.components_.T  # Components are (n_components, n_features), so transpose

# Ensure eigenvalues are non-negative and handle zero or negative values
epsilon = 1e-10
eigenvalues[eigenvalues < epsilon] = epsilon

# Rotation matrix for 30 degrees
alpha_deg_30 = 30
alpha_30 = np.deg2rad(alpha_deg_30)
R_alpha_30 = np.array([[np.cos(alpha_30), -np.sin(alpha_30)],
                       [np.sin(alpha_30),  np.cos(alpha_30)]])

# Rotation matrix for 90 degrees
alpha_deg_90 = 90
alpha_90 = np.deg2rad(alpha_deg_90)
R_alpha_90 = np.array([[np.cos(alpha_90), -np.sin(alpha_90)],
                       [np.sin(alpha_90),  np.cos(alpha_90)]])

# Extend rotation matrices to full size
R_full_30 = np.eye(N)
R_full_30[:2, :2] = R_alpha_30

R_full_90 = np.eye(N)
R_full_90[:2, :2] = R_alpha_90

# Construct the transformation matrices T for 30 and 90 degrees
T_30 = U @ R_full_30 @ U.T
T_90 = U @ R_full_90 @ U.T

# Apply the transformations to the centered responses
responses_transformed_30 = (responses_centered @ T_30) + responses_mean
responses_transformed_90 = (responses_centered @ T_90) + responses_mean

# Perform PCA on original and transformed responses
responses_pca = pca.transform(responses_centered)
responses_pca_transformed_30 = pca.transform(responses_transformed_30)
responses_pca_transformed_90 = pca.transform(responses_transformed_90)

# Extract features based on indices that are multiples of num_stimuli
shape_indices = np.arange(0, len(responses_pca), num_stimuli)

# Before transformation
shape_response = responses_pca[shape_indices]
color_response = responses_pca[:num_stimuli]

# After 30-degree transformation
shape_response_transformed_30 = responses_pca_transformed_30[shape_indices]
color_response_transformed_30 = responses_pca_transformed_30[:num_stimuli]

# After 90-degree transformation
shape_response_transformed_90 = responses_pca_transformed_90[shape_indices]
color_response_transformed_90 = responses_pca_transformed_90[:num_stimuli]

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Original - colored by shape
ax1 = axes[0, 0]
sc1 = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax1.set_title('Original Colored by Shape Stimulus')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_aspect('equal')
plt.colorbar(sc1, ax=ax1, label='Shape Stimulus Value')

# Original - colored by color
ax2 = axes[0, 1]
sc2 = ax2.scatter(responses_pca[:, 0], responses_pca[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
ax2.set_title('Original Colored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_aspect('equal')
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')

# 30-degree transformation - colored by shape
ax3 = axes[1, 0]
sc3 = ax3.scatter(responses_pca_transformed_30[:, 0], responses_pca_transformed_30[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax3.set_title('30-Degree Rotation Colored by Shape Stimulus')
ax3.set_xlabel('PC 1')
ax3.set_ylabel('PC 2')
ax3.set_aspect('equal')
plt.colorbar(sc3, ax=ax3, label='Shape Stimulus Value')

# 30-degree transformation - colored by color
ax4 = axes[1, 1]
sc4 = ax4.scatter(responses_pca_transformed_30[:, 0], responses_pca_transformed_30[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
ax4.set_title('30-Degree Rotation Colored by Color Stimulus')
ax4.set_xlabel('PC 1')
ax4.set_ylabel('PC 2')
ax4.set_aspect('equal')
plt.colorbar(sc4, ax=ax4, label='Color Stimulus Value')

# 90-degree transformation - colored by shape
ax5 = axes[2, 0]
sc5 = ax5.scatter(responses_pca_transformed_90[:, 0], responses_pca_transformed_90[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax5.set_title('90-Degree Rotation Colored by Shape Stimulus')
ax5.set_xlabel('PC 1')
ax5.set_ylabel('PC 2')
ax5.set_aspect('equal')
plt.colorbar(sc5, ax=ax5, label='Shape Stimulus Value')

# 90-degree transformation - colored by color
ax6 = axes[2, 1]
sc6 = ax6.scatter(responses_pca_transformed_90[:, 0], responses_pca_transformed_90[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
ax6.set_title('90-Degree Rotation Colored by Color Stimulus')
ax6.set_xlabel('PC 1')
ax6.set_ylabel('PC 2')
ax6.set_aspect('equal')
plt.colorbar(sc6, ax=ax6, label='Color Stimulus Value')

plt.tight_layout()

# Save the plot as an image
plt.savefig('pca_rotation_comparison.png', dpi=300)

plt.show()


