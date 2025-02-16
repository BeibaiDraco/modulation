import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape and color
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

# Feature selectivities
S = np.random.rand(N, K)

# Define weight matrices for feature and recurrent connectivity
W_F = np.random.rand(N, K) * 0.1
W_R = np.zeros((N, N))
threshold = 0.1
for i in range(N):
    for j in range(N):
        distance = np.linalg.norm(S[i] - S[j])
        if distance < threshold:
            W_R[i, j] = 1 - distance / threshold

# Normalize W_R using its spectral radius
eigenvalues = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues))
W_R = W_R / (scaling_factor + 1)

# Stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Storage for original and rotated responses
responses = np.zeros((len(stimuli_grid), N))
responses_rotated = np.zeros_like(responses)

# Define the rotation matrix in feature space
theta = -np.pi / 4  # Rotate by -45 degrees
R_feat = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])

# Compute responses for each stimulus
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    F_rot = R_feat @ F  # Rotated stimulus
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ (W_F @ F)
    responses_rotated[idx] = np.linalg.inv(np.eye(N) - W_R) @ (W_F @ F_rot)

# PCA transformation
pca = PCA(n_components=2)
responses_pca = pca.fit_transform(responses)
responses_rotated_pca = pca.transform(responses_rotated)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sc1 = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=stimuli_grid[:, 0], cmap='Reds', label='Original Responses')
ax1.set_title('PCA of Original Responses')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.legend()

sc2 = ax2.scatter(responses_rotated_pca[:, 0], responses_rotated_pca[:, 1], c=stimuli_grid[:, 0], cmap='Blues', label='Rotated Responses')
ax2.set_title('PCA of Responses to Rotated Stimuli')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.legend()

plt.colorbar(sc1, ax=ax1)
plt.colorbar(sc2, ax=ax2)
plt.tight_layout()
plt.show()