import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
N = 50  # Number of neurons
num_stimuli = 100  # Number of stimuli for color dimension
constant_shape_value = 0.5  # Keeping shape constant

# Seed for reproducibility
np.random.seed(0)

# Feature selectivities and weight matrices
S = np.random.rand(N, 2)  # Selectivities for shape and color
W_F = np.random.rand(N, 2) * 0.1  # Forward weights
W_R = np.random.rand(N, N)  # Recurrent weights

# Normalize W_R
W_R /= np.max(np.abs(np.linalg.eigvals(W_R))) + 1

# Generate responses for color-only variation
color_stimuli = np.linspace(0, 1, num_stimuli)
responses = np.zeros((num_stimuli, N))
for i, color in enumerate(color_stimuli):
    F = np.array([constant_shape_value, color])  # Fixed shape, variable color
    responses[i] = np.linalg.inv(np.eye(N) - W_R) @ (W_F @ F)

# PCA before neuron shutdown
pca = PCA(n_components=1)
pca.fit(responses)
principal_axis_before = pca.components_[0]

# Shutdown high color-selective neurons
color_selectivity_indices = np.argsort(S[:, 1])[-N//4:]  # Top 25% selective to color
responses[:, color_selectivity_indices] = 0

# PCA after neuron shutdown
pca.fit(responses)
principal_axis_after = pca.components_[0]

# Calculate the rotation angle
cos_theta = np.dot(principal_axis_before, principal_axis_after) / (np.linalg.norm(principal_axis_before) * np.linalg.norm(principal_axis_after))
theta = np.arccos(np.clip(cos_theta, -1, 1))
theta_degrees = np.degrees(theta)

print("Rotation angle in degrees:", theta_degrees)

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(responses[:, 0], responses[:, 1], c=color_stimuli, cmap='Blues')
plt.title("Responses in Neural Space (Color Only)")
plt.xlabel("Neuron 1")
plt.ylabel("Neuron 2")
plt.colorbar(label='Color Stimulus Value')

plt.subplot(122)
plt.quiver(0, 0, principal_axis_before[0], principal_axis_before[1], scale=3, color='r', label='Before Shutdown')
plt.quiver(0, 0, principal_axis_after[0], principal_axis_after[1], scale=3, color='b', label='After Shutdown')
plt.title("Principal Axis Rotation")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()



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
S = np.random.rand(N, K)  # Random selectivities for shape and color

# Define matrices and vectors
W_F = np.random.rand(N, K) * 0.1
W_R = np.zeros((N, N))
threshold = 0.1
for i in range(N):
    for j in range(N):
        distance = np.linalg.norm(S[i] - S[j])
        if distance < threshold:
            W_R[i, j] = 1 - distance / threshold

# Normalize W_R to ensure stability
eigenvalues = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues))
W_R = W_R / (scaling_factor + 1)

# Stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Generate responses
responses = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

# Extract responses for color stimuli (keeping shape constant at median value)
constant_shape_value = np.median(shape_stimuli)
color_only_responses = []
for color in color_stimuli:
    F = np.array([constant_shape_value, color])  # Keeping shape constant
    adjusted_response = np.linalg.inv(np.eye(N) - W_R) @ (W_F @ F)
    color_only_responses.append(adjusted_response)
color_only_responses = np.array(color_only_responses)

# Perform PCA on color responses
pca = PCA(n_components=1)
principal_axis_before = pca.fit_transform(color_only_responses)[:, 0]

# Shut down neurons based on shape selectivity
shape_selectivity_threshold = np.percentile(S[:, 0], 20)
high_shape_selective_neurons = S[:, 0] > shape_selectivity_threshold
responses[:, high_shape_selective_neurons] = 0

# Reanalyze with PCA after neuron shutdown
principal_axis_after = pca.transform(color_only_responses)[:, 0]

# Calculate rotation angle in PCA space
cos_theta = np.dot(principal_axis_before, principal_axis_after) / (np.linalg.norm(principal_axis_before) * np.linalg.norm(principal_axis_after))
theta = np.arccos(np.clip(cos_theta, -1, 1))
theta_degrees = np.degrees(theta)

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(color_stimuli, principal_axis_before, label='Before Shutdown')
plt.title('Principal Component Analysis Before Neuron Shutdown')
plt.xlabel('Color Stimulus')
plt.ylabel('Principal Component Value')

plt.subplot(122)
plt.plot(color_stimuli, principal_axis_after, label='After Shutdown', color='r')
plt.title('Principal Component Analysis After Neuron Shutdown')
plt.xlabel('Color Stimulus')
plt.ylabel('Principal Component Value')

plt.legend()
plt.tight_layout()
plt.show()

print("Rotation angle in degrees:", theta_degrees)