import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import seaborn as sns

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

# Storage for original responses
responses = np.zeros((len(stimuli_grid), N))

# Analytical steady states for each original stimulus
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

# -----------------------------------------------
# Compute the analytical transformation matrix T
# -----------------------------------------------

# Step 1: Compute A
A = np.linalg.inv(np.eye(N) - W_R) @ W_F  # Shape: (N, K)

# Step 2: Define the rotation matrix in feature space
theta = -np.deg2rad(45)  # Rotate by -45 degrees
R_feat = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])

# Step 3: Compute T analytically
A_inv = np.linalg.pinv(A)  # Pseudo-inverse in case A is not square or invertible
T = A @ R_feat @ A_inv

# -----------------------------------------------
# Apply T to original responses
# -----------------------------------------------

responses_transformed = responses @ T.T  # Transpose because responses are row vectors

# -----------------------------------------------
# Compute responses to rotated stimuli for verification
# -----------------------------------------------

# Rotate the stimuli grid
stimuli_grid_rotated = stimuli_grid @ R_feat.T

# Compute responses to rotated stimuli
responses_rotated = np.zeros((len(stimuli_grid_rotated), N))
for idx, (shape_rot, color_rot) in enumerate(stimuli_grid_rotated):
    F_rot = np.array([shape_rot, color_rot])  # Rotated external stimulus
    adjusted_F_rot = W_F @ F_rot
    responses_rotated[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F_rot

# -----------------------------------------------
# Verify the transformation
# -----------------------------------------------

# Compute the difference between analytically transformed responses and rotated responses
difference = responses_transformed - responses_rotated
transformation_error = np.linalg.norm(difference)
print(f"Transformation error (Frobenius norm): {transformation_error}")

# Compute eigenvalues of T
eigvals_T, eigvecs_T = np.linalg.eig(T)
print("Eigenvalues of the transformation matrix T:")
print(eigvals_T)

# -----------------------------------------------
# Perform PCA and plot results
# -----------------------------------------------

# Perform PCA on original responses
pca = PCA(n_components=3)
responses_3d = pca.fit_transform(responses)

# Project rotated and transformed responses onto original PCA space
responses_rotated_3d = pca.transform(responses_rotated)
responses_transformed_3d = pca.transform(responses_transformed)

# Plotting code remains the same as before
# For example, plot the transformed responses
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121)
ax.scatter(responses_rotated_3d[:, 0], responses_rotated_3d[:, 1],
           c=stimuli_grid[:, 0], cmap='Reds', label='Responses to Rotated Stimuli')
ax.set_title('Responses to Rotated Stimuli (PCA Space)')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.legend()

ax2 = fig.add_subplot(122)
ax2.scatter(responses_transformed_3d[:, 0], responses_transformed_3d[:, 1],
            c=stimuli_grid[:, 0], cmap='Blues', label='Transformed Responses')
ax2.set_title('Transformed Responses (PCA Space)')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------------
# Additional Analysis
# -----------------------------------------------

# Compute the eigenvalues and eigenvectors of T
eigvals_T, eigvecs_T = np.linalg.eig(T)

# Print the eigenvalues
print("Eigenvalues of the transformation matrix T:")
print(eigvals_T)

# Compute the difference between transformed and rotated responses
difference_norm = np.linalg.norm(responses_transformed - responses_rotated)
print(f"Difference between transformed and rotated responses: {difference_norm}")

# Compute the orthogonality error of T
T_T_transpose = T.T @ T
identity_matrix = np.eye(N)
orthogonality_error = np.linalg.norm(T_T_transpose - identity_matrix)
print(f"Orthogonality error of T: {orthogonality_error}")