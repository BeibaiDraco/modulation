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



# Dimensionality reduction with PCA
pca = PCA(n_components=3)
responses_3d = pca.fit_transform(responses)


# Separate response matrices for shape and color variations
shape_responses = np.array([responses[i * num_stimuli] for i in range(num_stimuli)])
color_responses = responses[:num_stimuli]

# Perform PCA on these subsets
pca_shape = PCA(n_components=1)
pca_color = PCA(n_components=1)

pca_shape.fit(shape_responses)
pca_color.fit(color_responses)

# Extract the direction vectors for shape and color
shape_direction = pca_shape.components_[0]
color_direction = pca_color.components_[0]

# Project these directions onto the PCA space
shape_axis = pca.transform(shape_direction.reshape(1, -1))
color_axis = pca.transform(color_direction.reshape(1, -1))


# First plot colored by shape stimulus
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
sc = ax.scatter(responses_3d[:, 0], responses_3d[:, 1], responses_3d[:, 2], c=stimuli_grid[:, 0], cmap='Reds')
plt.colorbar(sc, ax=ax, label='Shape Stimulus Value')
ax.set_title('Responses Colored by Shape Stimulus')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

# Second plot colored by color stimulus
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(responses_3d[:, 0], responses_3d[:, 1], responses_3d[:, 2], c=stimuli_grid[:, 1], cmap='Blues')

plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')
ax2.set_title('Responses Colored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')
plt.tight_layout()
plt.show()

# Extract features based on indices that are multiples of num_stimuli
shape_indices = np.arange(0, len(responses_3d), num_stimuli)

shape_response = responses_3d[shape_indices]
color_response = responses_3d[:num_stimuli]

pc_1_color = color_response[:, 0]
pc_2_color = color_response[:, 1]
pc_1_shape = shape_response[:, 0]
pc_2_shape = shape_response[:, 1]

# Fit linear regression model for color
model_color = LinearRegression()
model_color.fit(pc_1_color.reshape(-1, 1), pc_2_color)
line_x_color = np.linspace(pc_1_color.min(), pc_1_color.max(), 100)
line_y_color = model_color.predict(line_x_color.reshape(-1, 1))

# Fit linear regression model for shape
model_shape = LinearRegression()
model_shape.fit(pc_1_shape.reshape(-1, 1), pc_2_shape)
line_x_shape = np.linspace(pc_1_shape.min(), pc_1_shape.max(), 100)
line_y_shape = model_shape.predict(line_x_shape.reshape(-1, 1))

# First plot colored by shape stimulus
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121)
sc = ax.scatter(responses_3d[:, 0], responses_3d[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax.quiver(0, 0, shape_axis[0, 0], shape_axis[0, 1], scale=5, color='r')
plt.colorbar(sc, ax=ax, label='Shape Stimulus Value')
ax.set_title('Responses Colored by Shape Stimulus')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_aspect('equal')
# Plot linear regression line for shape
ax.plot(line_x_shape, line_y_shape, color='red', linestyle='--', label='Shape Axis')
ax.legend()

# Second plot colored by color stimulus
ax2 = fig.add_subplot(122)
sc2 = ax2.scatter(responses_3d[:, 0], responses_3d[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')
ax2.quiver(0, 0, color_axis[0, 0], color_axis[0, 1], scale=5, color='b')
ax2.set_title('Responses Colored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_aspect('equal')
# Plot linear regression line for color
ax2.plot(line_x_color, line_y_color, color='blue', linestyle='--', label='Color Axis')
ax2.legend()

plt.tight_layout()
plt.show()

# --------------------------------------------
# Additional code for transformation and plotting
# --------------------------------------------

# Angle of rotation in degrees
alpha_deg = 30  # You can adjust this angle as needed
alpha = np.deg2rad(alpha_deg)  # Convert to radians

# Compute the rotation matrix R(alpha)
R_alpha = np.array([[np.cos(alpha), -np.sin(alpha)],
                    [np.sin(alpha),  np.cos(alpha)]])

# Compute the PCA components (already computed earlier)
# We will use the first two principal components
U = pca.components_.T[:, :2]  # Shape (N, 2)

# Compute the projection matrix U U^T
U_proj = U @ U.T  # Shape (N, N)

# Compute the identity matrix
I_N = np.eye(N)

# Compute the transformation matrix T
T = U @ R_alpha @ U.T + (I_N - U_proj)

# Apply the transformation to the responses
responses_transformed = responses @ T.T  # Shape (num_samples, N)

# Project the transformed responses onto the PCA space
responses_3d_transformed = pca.transform(responses_transformed)

# Extract transformed shape and color responses
shape_responses_transformed = np.array([responses_transformed[i * num_stimuli] for i in range(num_stimuli)])
color_responses_transformed = responses_transformed[:num_stimuli]

# Perform PCA on transformed subsets (for axis directions)
pca_shape_transformed = PCA(n_components=1)
pca_color_transformed = PCA(n_components=1)

pca_shape_transformed.fit(shape_responses_transformed)
pca_color_transformed.fit(color_responses_transformed)

# Extract the direction vectors for shape and color after transformation
shape_direction_transformed = pca_shape_transformed.components_[0]
color_direction_transformed = pca_color_transformed.components_[0]

# Project these directions onto the PCA space
shape_axis_transformed = pca.transform(shape_direction_transformed.reshape(1, -1))
color_axis_transformed = pca.transform(color_direction_transformed.reshape(1, -1))

# Extract features based on indices that are multiples of num_stimuli
shape_indices = np.arange(0, len(responses_3d), num_stimuli)

# Before transformation
shape_response = responses_3d[shape_indices]
color_response = responses_3d[:num_stimuli]

# After transformation
shape_response_transformed = responses_3d_transformed[shape_indices]
color_response_transformed = responses_3d_transformed[:num_stimuli]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Before transformation - colored by shape
ax1 = axes[0, 0]
sc1 = ax1.scatter(responses_3d[:, 0], responses_3d[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax1.set_title('Before Transformation\nColored by Shape Stimulus')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.quiver(0, 0, shape_axis[0, 0], shape_axis[0, 1], scale=5, color='r', label='Shape Axis')
ax1.legend()
plt.colorbar(sc1, ax=ax1, label='Shape Stimulus Value')

# Before transformation - colored by color
ax2 = axes[0, 1]
sc2 = ax2.scatter(responses_3d[:, 0], responses_3d[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
ax2.set_title('Before Transformation\nColored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.quiver(0, 0, color_axis[0, 0], color_axis[0, 1], scale=5, color='b', label='Color Axis')
ax2.legend()
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')

# After transformation - colored by shape
ax3 = axes[1, 0]
sc3 = ax3.scatter(responses_3d_transformed[:, 0], responses_3d_transformed[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax3.set_title('After Transformation\nColored by Shape Stimulus')
ax3.set_xlabel('PC 1')
ax3.set_ylabel('PC 2')
ax3.quiver(0, 0, shape_axis_transformed[0, 0], shape_axis_transformed[0, 1], scale=5, color='r', label='Shape Axis')
ax3.legend()
plt.colorbar(sc3, ax=ax3, label='Shape Stimulus Value')

# After transformation - colored by color
ax4 = axes[1, 1]
sc4 = ax4.scatter(responses_3d_transformed[:, 0], responses_3d_transformed[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
ax4.set_title('After Transformation\nColored by Color Stimulus')
ax4.set_xlabel('PC 1')
ax4.set_ylabel('PC 2')
ax4.quiver(0, 0, color_axis_transformed[0, 0], color_axis_transformed[0, 1], scale=5, color='b', label='Color Axis')
ax4.legend()
plt.colorbar(sc4, ax=ax4, label='Color Stimulus Value')

plt.tight_layout()
plt.show()

#------
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

# Perform PCA
pca = PCA(n_components=N)
responses_pca = pca.fit_transform(responses)

# Extract the first two principal components
responses_pca_2d = responses_pca[:, :2]

# Standardize the data in PC1-PC2 space
eigenvalues = pca.explained_variance_[:2]
responses_pca_std = responses_pca_2d / np.sqrt(eigenvalues)

# Angle of rotation in degrees
alpha_deg = 30  # Adjust as needed
alpha = np.deg2rad(alpha_deg)  # Convert to radians

# Rotation matrix
R_alpha = np.array([[np.cos(alpha), -np.sin(alpha)],
                    [np.sin(alpha),  np.cos(alpha)]])

# Apply rotation in standardized space
responses_pca_rotated_std = responses_pca_std @ R_alpha.T

# Transform back to original scaling
responses_pca_rotated = responses_pca_rotated_std * np.sqrt(eigenvalues)

# Replace the first two components with the rotated data
responses_pca_rotated_full = responses_pca.copy()
responses_pca_rotated_full[:, :2] = responses_pca_rotated

# For plotting, we'll focus on the first two principal components
responses_2d = responses_pca[:, :2]
responses_2d_rotated = responses_pca_rotated_full[:, :2]

# Extract features based on indices that are multiples of num_stimuli
shape_indices = np.arange(0, len(responses_2d), num_stimuli)

# Before rotation
shape_response = responses_2d[shape_indices]
color_response = responses_2d[:num_stimuli]

# After rotation
shape_response_rotated = responses_2d_rotated[shape_indices]
color_response_rotated = responses_2d_rotated[:num_stimuli]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Before rotation - colored by shape
ax1 = axes[0, 0]
sc1 = ax1.scatter(responses_2d[:, 0], responses_2d[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax1.set_title('Before Rotation\nColored by Shape Stimulus')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_aspect('equal')
plt.colorbar(sc1, ax=ax1, label='Shape Stimulus Value')

# Before rotation - colored by color
ax2 = axes[0, 1]
sc2 = ax2.scatter(responses_2d[:, 0], responses_2d[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
ax2.set_title('Before Rotation\nColored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_aspect('equal')
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')

# After rotation - colored by shape
ax3 = axes[1, 0]
sc3 = ax3.scatter(responses_2d_rotated[:, 0], responses_2d_rotated[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax3.set_title('After Rotation\nColored by Shape Stimulus')
ax3.set_xlabel('PC 1')
ax3.set_ylabel('PC 2')
ax3.set_aspect('equal')
plt.colorbar(sc3, ax=ax3, label='Shape Stimulus Value')

# After rotation - colored by color
ax4 = axes[1, 1]
sc4 = ax4.scatter(responses_2d_rotated[:, 0], responses_2d_rotated[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
ax4.set_title('After Rotation\nColored by Color Stimulus')
ax4.set_xlabel('PC 1')
ax4.set_ylabel('PC 2')
ax4.set_aspect('equal')
plt.colorbar(sc4, ax=ax4, label='Color Stimulus Value')

plt.tight_layout()
plt.show()







#############
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

# Compute the covariance matrix of the responses
C = np.cov(responses, rowvar=False)

# Perform eigen decomposition
eigenvalues, U = np.linalg.eigh(C)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
U = U[:, sorted_indices]

# Construct the scaling matrix S
Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues))
Lambda_sqrt = np.diag(np.sqrt(eigenvalues))

S = U @ Lambda_inv_sqrt @ U.T

# Rotation angle
alpha_deg = 30  # Rotation angle in degrees
alpha = np.deg2rad(alpha_deg)  # Convert to radians

# Create the rotation matrix R_alpha
R_alpha = np.array([[np.cos(alpha), -np.sin(alpha)],
                    [np.sin(alpha),  np.cos(alpha)]])

# Extend R_alpha to full size
R_full = np.eye(N)
R_full[:2, :2] = R_alpha

# Construct the transformation matrix T
T = U @ Lambda_sqrt @ R_full @ Lambda_inv_sqrt @ U.T

# Apply the transformation to the responses
responses_transformed = responses @ T

# Perform PCA on original and transformed responses
pca = PCA(n_components=2)
responses_pca = pca.fit_transform(responses)
responses_pca_transformed = pca.transform(responses_transformed)

# Extract features based on indices that are multiples of num_stimuli
shape_indices = np.arange(0, len(responses_pca), num_stimuli)

# Before transformation
shape_response = responses_pca[shape_indices]
color_response = responses_pca[:num_stimuli]

# After transformation
shape_response_transformed = responses_pca_transformed[shape_indices]
color_response_transformed = responses_pca_transformed[:num_stimuli]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Before transformation - colored by shape
ax1 = axes[0, 0]
sc1 = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax1.set_title('Before Transformation\nColored by Shape Stimulus')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_aspect('equal')
plt.colorbar(sc1, ax=ax1, label='Shape Stimulus Value')

# Before transformation - colored by color
ax2 = axes[0, 1]
sc2 = ax2.scatter(responses_pca[:, 0], responses_pca[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
ax2.set_title('Before Transformation\nColored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_aspect('equal')
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')

# After transformation - colored by shape
ax3 = axes[1, 0]
sc3 = ax3.scatter(responses_pca_transformed[:, 0], responses_pca_transformed[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
ax3.set_title('After Transformation\nColored by Shape Stimulus')
ax3.set_xlabel('PC 1')
ax3.set_ylabel('PC 2')
ax3.set_aspect('equal')
plt.colorbar(sc3, ax=ax3, label='Shape Stimulus Value')

# After transformation - colored by color
ax4 = axes[1, 1]
sc4 = ax4.scatter(responses_pca_transformed[:, 0], responses_pca_transformed[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
ax4.set_title('After Transformation\nColored by Color Stimulus')
ax4.set_xlabel('PC 1')
ax4.set_ylabel('PC 2')
ax4.set_aspect('equal')
plt.colorbar(sc4, ax=ax4, label='Color Stimulus Value')

plt.tight_layout()
plt.show()