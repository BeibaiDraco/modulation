import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
N = 300  # Number of neurons
num_stimuli = 30  # Number of stimuli per feature dimension
enhancement_therehold_color = 0.2
enhancement_therehold_shape = 0.2
suppression_therehold_color = 0.2
suppression_therehold_shape = 0.4
enhancement_factor = 1.0
suppression_factor = 1.0

# Seed for reproducibility
np.random.seed(0)

# Define selectivity for shape and color
shape_selectivity = np.random.rand(N)  # Shape preferences for neurons
color_selectivity = np.random.rand(N)  # Color preferences for neurons

# Tuning curve function
def tuning_curve(preferred_value, stimulus_value, width=0.2):
    """Gaussian tuning curve."""
    return np.exp(-((stimulus_value - preferred_value)**2) / (2 * width**2))

# Generate stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Compute firing rates
responses = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    shape_response = tuning_curve(shape_selectivity, shape)#shape#tuning_curve(shape_selectivity, shape)
    color_response = tuning_curve(shape_selectivity, shape)#color#tuning_curve(color_selectivity, color)
    responses[idx] = 1*shape_response + 1*color_response  # Combined response


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
sorted_neuron_indices = np.argsort(eigenvector_1)
absolute_eigenvector_weights_1 = np.abs(eigenvector_1)
# Min-Max scaling to [0, 1]
min_weight = np.min(absolute_eigenvector_weights_1)
max_weight = np.max(absolute_eigenvector_weights_1)

scaled_weights_1 = (absolute_eigenvector_weights_1 - min_weight) / (max_weight - min_weight)
# Plot eigenvector directions and neuron selectivities
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: First eigenvector
im1 = axes[0].imshow(
    np.vstack([scaled_weights_1[sorted_neuron_indices], 
               color_selectivity[sorted_neuron_indices], 
               shape_selectivity[sorted_neuron_indices]]),
    aspect='auto', cmap='coolwarm', interpolation='none'
)
axes[0].set_title('First Eigenvector and Selectivities')
axes[0].set_yticks([0, 1, 2])
axes[0].set_yticklabels(['Eigenvector 1', 'Color Selectivity', 'Shape Selectivity'])
axes[0].set_xlabel('Neuron Index (sorted by first eigenvector)')
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
axes[1].set_xlabel('Neuron Index (sorted by first eigenvector)')
fig.colorbar(im2, ax=axes[1], orientation='vertical')

plt.tight_layout()
plt.show()



# Sort neurons by color selectivity
#sorted_neuron_indices = np.argsort(color_selectivity)
sorted_neuron_indices = np.argsort(eigenvector_2)

# Plot eigenvector directions and neuron selectivities
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: First eigenvector
im1 = axes[0].imshow(
    np.vstack([scaled_weights_1[sorted_neuron_indices], 
               color_selectivity[sorted_neuron_indices], 
               shape_selectivity[sorted_neuron_indices]]),
    aspect='auto', cmap='coolwarm', interpolation='none'
)
axes[0].set_title('First Eigenvector and Selectivities')
axes[0].set_yticks([0, 1, 2])
axes[0].set_yticklabels(['Eigenvector 1', 'Color Selectivity', 'Shape Selectivity'])
axes[0].set_xlabel('Neuron Index (sorted by second eigenvector)')
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
axes[1].set_xlabel('Neuron Index (sorted by second eigenvector)')
fig.colorbar(im2, ax=axes[1], orientation='vertical')

plt.tight_layout()
plt.show()

# Print eigenvalues for reference
print(f"First eigenvalue: {eigenvalues[0]:.4f}")
print(f"Second eigenvalue: {eigenvalues[1]:.4f}")


# Thresholds for enhancement and suppression
color_threshold_high = np.percentile(color_selectivity, (1 - enhancement_therehold_color) * 100)
shape_threshold_low = np.percentile(shape_selectivity, enhancement_therehold_shape * 100)
color_threshold_low = np.percentile(color_selectivity, suppression_therehold_color * 100)
shape_threshold_high = np.percentile(shape_selectivity, (1 - suppression_therehold_shape) * 100)

# Identify neurons for enhancement and suppression
neurons_enhance = np.where((color_selectivity > color_threshold_high) & (shape_selectivity < shape_threshold_low))[0]
neurons_suppress = np.where((color_selectivity < color_threshold_low) & (shape_selectivity > shape_threshold_high))[0]

# Initialize gain matrix
gain_matrix = np.eye(N)
gain_matrix[neurons_enhance, neurons_enhance] = enhancement_factor
gain_matrix[neurons_suppress, neurons_suppress] = suppression_factor

# Compute modified responses
#responses_modified = responses @ gain_matrix.T
# New modulation rule
modulation_factors = 1 + 1*(color_selectivity - shape_selectivity)  # New scaling factor

min_modulation = modulation_factors.min()
max_modulation = modulation_factors.max()

#Scale to ensure min=0.7 and max=1.3
modulation_factors_scaled = 0.8 + (modulation_factors - min_modulation) * (1.2 - 0.8) / (max_modulation - min_modulation)

responses_modified = responses * modulation_factors_scaled   # Apply modulation

responses_modified = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    shape_response = shape_selectivity*shape#tuning_curve(shape_selectivity, shape)
    color_response = color_selectivity*color#tuning_curve(color_selectivity, color)
    responses_modified[idx] = 0.8*shape_response + 1.2*color_response  # Combined response


min_modulation = modulation_factors_scaled.min()
max_modulation = modulation_factors_scaled.max()
print(f"Min modulation factor: {min_modulation:.4f}")
print(f"Max modulation factor: {max_modulation:.4f}")

# Compute rank-based scaling factor
#absolute_eigenvector_weights = np.abs(eigenvector_1)
#ranked_indices = np.argsort(absolute_eigenvector_weights)[::-1]  # Rank by absolute value
# Simulate rank-based scaling
#rank = np.arange(1, N + 1)  # Rank of neurons
#eigen_rank_scaler = (N - rank) ** 2  # Quadratic scaling

#rank_based_scaling = np.zeros(N)
#rank_based_scaling[ranked_indices] = eigen_rank_scaler#_sigmoid  # Assign scale factors based on rank
# Compute the min-max normalization
#rank_based_scaling = (rank_based_scaling - rank_based_scaling.min()) / (rank_based_scaling.max() - rank_based_scaling.min())

# Verify the scaling
#print(f"Min value after normalization: {rank_based_scaling.min():.4f}")  # Should be 0
#print(f"Max value after normalization: {rank_based_scaling.max():.4f}")  # Should be 1
#print rank_based_scaling from greatest to smallest, sort it first then print
#print(np.sort(rank_based_scaling)[::-1])    


# Compute new modulation factors
#modulation_factors =  (1+rank_based_scaling) * color_selectivity 
# Get min and max of modulation_factors
#min_modulation = modulation_factors.min()
#max_modulation = modulation_factors.max()

# Scale to ensure min=0.7 and max=1.3
#modulation_factors_scaled = 0.8 + (modulation_factors - min_modulation) * (1.2 - 0.8) / (max_modulation - min_modulation)


# Apply modulation to responses
#responses_modified = responses * modulation_factors_scaled#modulation_factors_scaled

# Compute absolute eigenvector weights
#absolute_eigenvector_weights = np.abs(eigenvector_1)

# Get ranks of eigenvector weights (rank 1 is the highest weight)
#eigen_ranks = np.argsort(-absolute_eigenvector_weights) + 1

# Get ranks of color selectivity (rank 1 is the highest selectivity)
#color_selectivity_ranks = np.argsort(-color_selectivity) + 1

# Compute rank differences
#rank_differences = eigen_ranks - color_selectivity_ranks

# Define acceptable modulation factor limits
#min_modulation_factor = 0.8
#max_modulation_factor = 1.2

# Compute delta for scaling
#delta = (max_modulation_factor - min_modulation_factor) / 2  # In this case, delta = 0.2

# Normalize rank differences to [-1, 1]
#normalized_rank_differences = rank_differences / (N - 1)

# Compute modulation factors
#modulation_factors = 1 - normalized_rank_differences * delta

# Clip modulation factors to acceptable limits
#modulation_factors = np.clip(modulation_factors, min_modulation_factor, max_modulation_factor)

# Apply modulation to responses
#responses_modified = responses * modulation_factors

"""
# Number of neurons (assuming N >= 200)
N = len(eigenvector_1)

# Compute absolute eigenvector weights
absolute_eigenvector_weights = np.abs(eigenvector_1)

# Get sorted indices of eigenvector weights in descending order
sorted_indices = np.argsort(-absolute_eigenvector_weights)

# Initialize modulation factors to 1.0 for all neurons
modulation_factors = np.ones(N)

# Define the number of top and bottom neurons to adjust
top_N = 100  # Adjust if needed, ensure N >= 2 * top_N

# Identify the top 100 neurons
top_indices = sorted_indices[:top_N]

# Adjust modulation factors for the top 100 neurons
for neuron_idx in top_indices:
    if color_selectivity[neuron_idx] < 0.9:
        modulation_factors[neuron_idx] = 1.3
    else:
        modulation_factors[neuron_idx] = 1.0

# Identify the bottom 100 neurons
bottom_indices = sorted_indices[-top_N:]

# Adjust modulation factors for the bottom 100 neurons
for neuron_idx in bottom_indices:
    if color_selectivity[neuron_idx] > 0.2:
        modulation_factors[neuron_idx] = 0.7
    else:
        modulation_factors[neuron_idx] = 1.0
"""
# Apply modulation to responses
#responses_modified = responses * modulation_factors

# PCA Visualization
def create_response_visualization(responses_list, titles, shape_data, color_data):
    fig = plt.figure(figsize=(15, 4 * len(responses_list)))
    pca = PCA(n_components=2)
    pca.fit(responses_list[0])

    all_pca_data = np.vstack([pca.transform(resp) for resp in responses_list])
    x_min, x_max = all_pca_data[:, 0].min(), all_pca_data[:, 0].max()
    y_min, y_max = all_pca_data[:, 1].min(), all_pca_data[:, 1].max()

    for row, (response_data, title) in enumerate(zip(responses_list, titles)):
        responses_pca = pca.transform(response_data)
        ax1 = fig.add_subplot(len(responses_list), 1, row + 1)
        scatter = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=shape_data, cmap='Reds', label='Shape')
        scatter = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=color_data, cmap='Blues', label='Color')
        ax1.set_title(title)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()

# Visualizations
def visualize_neuron_selectivity(shape_selectivity, color_selectivity, neurons_enhance, neurons_suppress):
    plt.figure(figsize=(10, 6))
    plt.scatter(shape_selectivity, color_selectivity, c='gray', alpha=0.6, label='Other Neurons')
    plt.scatter(shape_selectivity[neurons_enhance], color_selectivity[neurons_enhance], 
                c='lime', edgecolors='black', s=100, label='Enhanced Neurons')
    plt.scatter(shape_selectivity[neurons_suppress], color_selectivity[neurons_suppress], 
                c='red', edgecolors='black', s=100, label='Suppressed Neurons')
    plt.axvline(shape_threshold_low, color='r', linestyle='--', label='Shape Threshold (Low)')
    plt.axhline(color_threshold_high, color='b', linestyle='--', label='Color Threshold (High)')
    plt.axvline(shape_threshold_high, color='r', linestyle='--', label='Shape Threshold (High)')
    plt.axhline(color_threshold_low, color='b', linestyle='--', label='Color Threshold (Low)')
    plt.xlabel('Shape Selectivity')
    plt.ylabel('Color Selectivity')
    plt.title('Neuron Selectivity: Enhanced and Suppressed Neurons')
    plt.legend()
    plt.show()



# Visualize responses with PCA
def create_response_visualization(responses_list, titles, shape_data, color_data):
    fig = plt.figure(figsize=(15, 4 * len(responses_list)))
    gs = fig.add_gridspec(len(responses_list), 3, width_ratios=[1, 1, 0.1])

    pca = PCA(n_components=2)
    pca.fit(responses_list[0])

    all_pca_data = np.vstack([pca.transform(resp) for resp in responses_list])
    x_min, x_max = all_pca_data[:, 0].min(), all_pca_data[:, 0].max()
    y_min, y_max = all_pca_data[:, 1].min(), all_pca_data[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_lim = [x_center - max_range/2, x_center + max_range/2]
    y_lim = [y_center - max_range/2, y_center + max_range/2]

    for row, (response_data, title) in enumerate(zip(responses_list, titles)):
        responses_pca = pca.transform(response_data)

        ax1 = fig.add_subplot(gs[row, 0])
        ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=shape_data, cmap='Reds', s=30)
        ax1.set_title(f'{title}\nColored by Shape')
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)

        ax2 = fig.add_subplot(gs[row, 1])
        ax2.scatter(responses_pca[:, 0], responses_pca[:, 1], c=color_data, cmap='Blues', s=30)
        ax2.set_title(f'{title}\nColored by Color')
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)

    plt.tight_layout()
    plt.show()

# Run visualizations
visualize_neuron_selectivity(shape_selectivity, color_selectivity, neurons_enhance, neurons_suppress)
create_response_visualization(
    responses_list=[responses, responses_modified],
    titles=["Original Responses", "Modified Responses"],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)


# Statistics
print(f"Total neurons: {N}")
print(f"Enhanced neurons: {len(neurons_enhance)}, Suppressed neurons: {len(neurons_suppress)}")
print(f"Enhancement factor: {enhancement_factor}, Suppression factor: {suppression_factor}")