import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
N = 100 # Number of neurons
num_stimuli = 100  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

# Define selectivity for shape and color
shape_selectivity = np.random.rand(N)  # Shape preferences for neurons
color_selectivity = np.random.rand(N)  # Color preferences for neurons

# Generate stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Compute firing rates
responses = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    shape_response = shape_selectivity * shape  # Tuning for shape
    color_response = color_selectivity * color  # Tuning for color
    responses[idx] = shape_response + color_response  # Combined response

# Calculate covariance matrix and eigenvalues/vectors
cov_matrix = np.cov(responses.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Extract the first and second eigenvectors
eigenvector_1 = eigenvectors[:, 0]
eigenvector_2 = eigenvectors[:, 1]

# Sort neurons by color selectivity
#sorted_neuron_indices = np.argsort(color_selectivity)
sorted_neuron_indices = np.argsort(eigenvector_2)

# Plot eigenvector directions and neuron selectivities
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: First eigenvector
im1 = axes[0].imshow(
    np.vstack([eigenvector_1[sorted_neuron_indices], 
               color_selectivity[sorted_neuron_indices], 
               shape_selectivity[sorted_neuron_indices]]),
    aspect='auto', cmap='coolwarm', interpolation='none'
)
axes[0].set_title('First Eigenvector and Selectivities')
axes[0].set_yticks([0, 1, 2])
axes[0].set_yticklabels(['Eigenvector 1', 'Color Selectivity', 'Shape Selectivity'])
axes[0].set_xlabel('Neuron Index (sorted by color selectivity)')
fig.colorbar(im1, ax=axes[0], orientation='vertical')

# Subplot 2: Second eigenvector
im2 = axes[1].imshow(
    np.vstack([eigenvector_2[sorted_neuron_indices], 
               color_selectivity[sorted_neuron_indices], 
               shape_selectivity[sorted_neuron_indices]]),
    aspect='auto', cmap='coolwarm', interpolation='none'
)
axes[1].set_title('Second Eigenvector and Selectivities')
axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(['Eigenvector 2', 'Color Selectivity', 'Shape Selectivity'])
axes[1].set_xlabel('Neuron Index (sorted by color selectivity)')
fig.colorbar(im2, ax=axes[1], orientation='vertical')

plt.tight_layout()
plt.show()

# Print eigenvalues for reference
print(f"First eigenvalue: {eigenvalues[0]:.4f}")
print(f"Second eigenvalue: {eigenvalues[1]:.4f}")