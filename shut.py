import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape (index 0) and color (index 1)
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

# Compute responses to each stimulus
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

# Identify neurons with high selectivity to shape (top quartile of selectivity)
shape_selectivity_threshold = np.percentile(S[:, 0], 20)
high_shape_selectivity_indices = np.where(S[:, 0] > shape_selectivity_threshold)[0]

# Shut down these neurons by setting their responses to zero
responses_modified = np.copy(responses)
responses_modified[:, high_shape_selectivity_indices] = 0

# Perform PCA on the original and modified responses
pca = PCA(n_components=2)
responses_pca = pca.fit_transform(responses)
responses_modified_pca = pca.transform(responses_modified)

# Plotting
plt.figure(figsize=(12, 12))

# Original Responses colored by Shape
ax1 = plt.subplot(2, 2, 1)
sc1 = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
plt.colorbar(sc1, ax=ax1, label='Shape Stimulus Value')
ax1.set_title('Original Responses Colored by Shape')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_aspect('equal')
# Original Responses colored by Color
ax2 = plt.subplot(2, 2, 2)
sc2 = ax2.scatter(responses_pca[:, 0], responses_pca[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
plt.colorbar(sc2, ax=ax2, label='Color Stimulus Value')
ax2.set_title('Original Responses Colored by Color')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_aspect('equal')
# Modified Responses colored by Shape
ax3 = plt.subplot(2, 2, 3)
sc3 = ax3.scatter(responses_modified_pca[:, 0], responses_modified_pca[:, 1], c=stimuli_grid[:, 0], cmap='Reds')
plt.colorbar(sc3, ax=ax3, label='Shape Stimulus Value')
ax3.set_title('Modified Responses Colored by Shape')
ax3.set_xlabel('PC 1')
ax3.set_ylabel('PC 2')
ax3.set_aspect('equal')
# Modified Responses colored by Color
ax4 = plt.subplot(2, 2, 4)
sc4 = ax4.scatter(responses_modified_pca[:, 0], responses_modified_pca[:, 1], c=stimuli_grid[:, 1], cmap='Blues')
plt.colorbar(sc4, ax=ax4, label='Color Stimulus Value')
ax4.set_title('Modified Responses Colored by Color')
ax4.set_xlabel('PC 1')
ax4.set_ylabel('PC 2')
ax4.set_aspect('equal')
plt.tight_layout()
plt.show()

# Add this code at the end of your script

def compute_modified_responses(responses, S, shutdown_percentile):
    """
    Compute modified responses by shutting down neurons with high shape selectivity
    shutdown_percentile: percentile threshold for shape selectivity
    """
    shape_selectivity_threshold = np.percentile(S[:, 0], shutdown_percentile)
    high_shape_selectivity_indices = np.where(S[:, 0] > shape_selectivity_threshold)[0]
    responses_modified = np.copy(responses)
    responses_modified[:, high_shape_selectivity_indices] = 0
    return responses_modified

# Create figure with shared colorbars
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.1])

# Create colorbars once
norm_shape = plt.Normalize(vmin=stimuli_grid[:, 0].min(), vmax=stimuli_grid[:, 0].max())
norm_color = plt.Normalize(vmin=stimuli_grid[:, 1].min(), vmax=stimuli_grid[:, 1].max())
cmap_shape = plt.cm.Reds
cmap_color = plt.cm.Blues

# Fit PCA on original responses
pca = PCA(n_components=2)
responses_pca_original = pca.fit_transform(responses)

# Create modified responses for different conditions
responses_60 = compute_modified_responses(responses, S, 40)  # shutdown top 60%
responses_80 = compute_modified_responses(responses, S, 20)  # shutdown top 80%

# Transform modified responses using the same PCA
responses_pca_60 = pca.transform(responses_60)
responses_pca_80 = pca.transform(responses_80)

# Prepare data for plotting
conditions = [
    ('Original', responses_pca_original),
    ('Shutdown top 60%\nshape-selective', responses_pca_60),
    ('Shutdown top 80%\nshape-selective', responses_pca_80)
]

# Find global limits
all_pca_data = np.vstack([resp for _, resp in conditions])
x_min, x_max = all_pca_data[:, 0].min(), all_pca_data[:, 0].max()
y_min, y_max = all_pca_data[:, 1].min(), all_pca_data[:, 1].max()

# Ensure square aspect ratio
max_range = max(x_max - x_min, y_max - y_min)
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2
x_lim = [x_center - max_range/2, x_center + max_range/2]
y_lim = [y_center - max_range/2, y_center + max_range/2]

# Plot with consistent limits
for row, (condition_name, responses_pca) in enumerate(conditions):
    # Shape plot
    ax1 = fig.add_subplot(gs[row, 0])
    sc1 = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], 
                     c=stimuli_grid[:, 0], cmap=cmap_shape, norm=norm_shape,s=10)
    ax1.set_title(f'{condition_name}\nColored by Shape')
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_aspect('equal')
    
    # Color plot
    ax2 = fig.add_subplot(gs[row, 1])
    sc2 = ax2.scatter(responses_pca[:, 0], responses_pca[:, 1], 
                     c=stimuli_grid[:, 1], cmap=cmap_color, norm=norm_color,s=10)
    ax2.set_title(f'{condition_name}\nColored by Color')
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_aspect('equal')

# Add colorbars on the right
cax_shape = fig.add_subplot(gs[0, 2])
cax_color = fig.add_subplot(gs[2, 2])
plt.colorbar(plt.cm.ScalarMappable(norm=norm_shape, cmap=cmap_shape), 
            cax=cax_shape, label='Shape Stimulus Value')
plt.colorbar(plt.cm.ScalarMappable(norm=norm_color, cmap=cmap_color), 
            cax=cax_color, label='Color Stimulus Value')

plt.tight_layout()
plt.show()