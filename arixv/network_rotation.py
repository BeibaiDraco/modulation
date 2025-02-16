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

# Storage for original responses
responses = np.zeros((len(stimuli_grid), N))

# Analytical steady states for each original stimulus
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

# -----------------------------------------------
# Additional code starts here
# -----------------------------------------------

# 1. Rotate the stimuli grid
theta = -np.deg2rad(45)  # Rotate by -45 degrees
R_feat = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
stimuli_grid_rotated = stimuli_grid @ R_feat.T

# Storage for rotated responses
responses_rotated = np.zeros((len(stimuli_grid_rotated), N))

# Analytical steady states for each rotated stimulus
for idx, (shape_rot, color_rot) in enumerate(stimuli_grid_rotated):
    F_rot = np.array([shape_rot, color_rot])  # Rotated external stimulus
    adjusted_F_rot = W_F @ F_rot
    responses_rotated[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F_rot

# 2. Compute the transformation matrix T
# Solve the equation responses @ T = responses_rotated using least squares
T, residuals, rank, s = np.linalg.lstsq(responses, responses_rotated, rcond=None)

# 3. Compute transformed responses
responses_transformed = responses @ T

# 4. Perform PCA on the original responses
pca = PCA(n_components=3)
responses_3d = pca.fit_transform(responses)

# 5. Project rotated and transformed responses onto the original PCA space
responses_rotated_3d = pca.transform(responses_rotated)
responses_transformed_3d = pca.transform(responses_transformed)

# -------------------------------
# Plot 1: 3D plot of original responses
# -------------------------------

# First plot colored by shape stimulus
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
sc = ax.scatter(responses_3d[:, 0], responses_3d[:, 1], responses_3d[:, 2],
                c=stimuli_grid[:, 0], cmap='Reds')
plt.colorbar(sc, ax=ax, label='Shape Stimulus Value')
ax.set_title('Original Responses Colored by Shape Stimulus')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

# Second plot colored by color stimulus
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(responses_3d[:, 0], responses_3d[:, 1], responses_3d[:, 2],
                  c=stimuli_grid[:, 1], cmap='Blues')
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')
ax2.set_title('Original Responses Colored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_zlabel('PC 3')
plt.tight_layout()
plt.show()

# -------------------------------
# Plot 2: 2D plot of original responses with fitted axes
# -------------------------------

# Extract features based on indices that are multiples of num_stimuli
shape_indices = np.arange(0, len(responses_3d), num_stimuli)
color_indices = np.arange(num_stimuli)

shape_response = responses_3d[shape_indices]
color_response = responses_3d[color_indices]

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
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(121)
sc = ax.scatter(responses_3d[:, 0], responses_3d[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
plt.colorbar(sc, ax=ax, label='Shape Stimulus Value')
ax.set_title('Original Responses Colored by Shape Stimulus')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
# Plot linear regression line for shape
ax.plot(line_x_shape, line_y_shape, color='red', linestyle='--', label='Shape Axis')
ax.legend()

# Second plot colored by color stimulus
ax2 = fig.add_subplot(122)
sc2 = ax2.scatter(responses_3d[:, 0], responses_3d[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')
ax2.set_title('Original Responses Colored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
# Plot linear regression line for color
ax2.plot(line_x_color, line_y_color, color='blue', linestyle='--', label='Color Axis')
ax2.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# Plot 3: 2D plot of rotated responses with new fitted axes
# -------------------------------

# Project rotated responses onto original PCA space
responses_rotated_2d = responses_rotated_3d[:, :2]

# Extract features based on indices
shape_response_rot = responses_rotated_2d[shape_indices]
color_response_rot = responses_rotated_2d[color_indices]

pc_1_color_rot = color_response_rot[:, 0]
pc_2_color_rot = color_response_rot[:, 1]
pc_1_shape_rot = shape_response_rot[:, 0]
pc_2_shape_rot = shape_response_rot[:, 1]

# Fit linear regression model for color (rotated)
model_color_rot = LinearRegression()
model_color_rot.fit(pc_1_color_rot.reshape(-1, 1), pc_2_color_rot)
line_x_color_rot = np.linspace(pc_1_color_rot.min(), pc_1_color_rot.max(), 100)
line_y_color_rot = model_color_rot.predict(line_x_color_rot.reshape(-1, 1))

# Fit linear regression model for shape (rotated)
model_shape_rot = LinearRegression()
model_shape_rot.fit(pc_1_shape_rot.reshape(-1, 1), pc_2_shape_rot)
line_x_shape_rot = np.linspace(pc_1_shape_rot.min(), pc_1_shape_rot.max(), 100)
line_y_shape_rot = model_shape_rot.predict(line_x_shape_rot.reshape(-1, 1))

# Plot rotated responses colored by shape stimulus
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(121)
sc = ax.scatter(responses_rotated_2d[:, 0], responses_rotated_2d[:, 1],
                c=stimuli_grid[:, 0], cmap='Reds')
plt.colorbar(sc, ax=ax, label='Shape Stimulus Value')
ax.set_title('Rotated Responses Colored by Shape Stimulus')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
# Plot linear regression line for shape
ax.plot(line_x_shape_rot, line_y_shape_rot, color='red', linestyle='--', label='Shape Axis')
ax.legend()

# Plot rotated responses colored by color stimulus
ax2 = fig.add_subplot(122)
sc2 = ax2.scatter(responses_rotated_2d[:, 0], responses_rotated_2d[:, 1],
                  c=stimuli_grid[:, 1], cmap='Blues')
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')
ax2.set_title('Rotated Responses Colored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
# Plot linear regression line for color
ax2.plot(line_x_color_rot, line_y_color_rot, color='blue', linestyle='--', label='Color Axis')
ax2.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# Plot 4: 2D plot of transformed responses with new fitted axes
# -------------------------------

# Project transformed responses onto original PCA space
responses_transformed_2d = responses_transformed_3d[:, :2]

# Extract features based on indices
shape_response_trans = responses_transformed_2d[shape_indices]
color_response_trans = responses_transformed_2d[color_indices]

pc_1_color_trans = color_response_trans[:, 0]
pc_2_color_trans = color_response_trans[:, 1]
pc_1_shape_trans = shape_response_trans[:, 0]
pc_2_shape_trans = shape_response_trans[:, 1]

# Fit linear regression model for color (transformed)
model_color_trans = LinearRegression()
model_color_trans.fit(pc_1_color_trans.reshape(-1, 1), pc_2_color_trans)
line_x_color_trans = np.linspace(pc_1_color_trans.min(), pc_1_color_trans.max(), 100)
line_y_color_trans = model_color_trans.predict(line_x_color_trans.reshape(-1, 1))

# Fit linear regression model for shape (transformed)
model_shape_trans = LinearRegression()
model_shape_trans.fit(pc_1_shape_trans.reshape(-1, 1), pc_2_shape_trans)
line_x_shape_trans = np.linspace(pc_1_shape_trans.min(), pc_1_shape_trans.max(), 100)
line_y_shape_trans = model_shape_trans.predict(line_x_shape_trans.reshape(-1, 1))

# Plot transformed responses colored by shape stimulus
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(121)
sc = ax.scatter(responses_transformed_2d[:, 0], responses_transformed_2d[:, 1],
                c=stimuli_grid[:, 0], cmap='Reds')
plt.colorbar(sc, ax=ax, label='Shape Stimulus Value')
ax.set_title('Transformed Responses Colored by Shape Stimulus')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
# Plot linear regression line for shape
ax.plot(line_x_shape_trans, line_y_shape_trans, color='red', linestyle='--', label='Shape Axis')
ax.legend()

# Plot transformed responses colored by color stimulus
ax2 = fig.add_subplot(122)
sc2 = ax2.scatter(responses_transformed_2d[:, 0], responses_transformed_2d[:, 1],
                  c=stimuli_grid[:, 1], cmap='Blues')
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')
ax2.set_title('Transformed Responses Colored by Color Stimulus')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
# Plot linear regression line for color
ax2.plot(line_x_color_trans, line_y_color_trans, color='blue', linestyle='--', label='Color Axis')
ax2.legend()

plt.tight_layout()
plt.show()

# 1. Compare the response matrices (50 x number of stimuli)

# Transpose the response matrices to have neurons on the y-axis and stimuli on the x-axis
responses_T = responses.T  # Shape: (50, 100)
responses_rotated_T = responses_rotated.T  # Shape: (50, 100)
responses_transformed_T = responses_transformed.T  # Shape: (50, 100)

# Plot heatmaps of the response matrices
import seaborn as sns

# Set up the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original responses
sns.heatmap(responses_T, ax=axes[0], cmap='viridis')
axes[0].set_title('Original Responses')
axes[0].set_xlabel('Stimuli')
axes[0].set_ylabel('Neurons')

# Rotated responses
sns.heatmap(responses_rotated_T, ax=axes[1], cmap='viridis')
axes[1].set_title('Rotated Responses')
axes[1].set_xlabel('Stimuli')
axes[1].set_ylabel('Neurons')

# Difference between rotated and original responses
difference_responses = responses_rotated_T - responses_T
sns.heatmap(difference_responses, ax=axes[2], cmap='coolwarm', center=0)
axes[2].set_title('Difference (Rotated - Original)')
axes[2].set_xlabel('Stimuli')
axes[2].set_ylabel('Neurons')

plt.tight_layout()
plt.show()

# 2. Visualize the eigenvalues and properties of the transformation matrix T

# Compute eigenvalues and eigenvectors of T
eigvals_T, eigvecs_T = np.linalg.eig(T)

# Plot the eigenvalues
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, N+1), np.abs(eigvals_T), 'o-')


plt.title('Eigenvalues of the Transformation Matrix T')
plt.xlabel('Index')
plt.ylabel('Absolute Value of Eigenvalues')
plt.grid(True)
plt.show()

# Analyze properties of T

# Check if T is close to orthogonal
T_T_transpose = T.T @ T
identity_matrix = np.eye(N)
orthogonality_error = np.linalg.norm(T_T_transpose - identity_matrix)
print(f"Orthogonality error of T (should be close to zero if T is orthogonal): {orthogonality_error}")

# Visualize the singular values of T
U, s, Vh = np.linalg.svd(T)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, N+1), s, 'o-')
plt.title('Singular Values of the Transformation Matrix T')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.grid(True)
plt.show()

# Visualize the elements of T as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(T, cmap='coolwarm', center=0)
plt.title('Heatmap of the Transformation Matrix T')
plt.xlabel('Original Neurons')
plt.ylabel('Transformed Neurons')
plt.show()


