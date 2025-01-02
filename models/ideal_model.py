import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Parameters
N = 500  # Number of neurons
num_stimuli = 20  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

# Define selectivity for shape and color
shape_selectivity = np.random.rand(N)  # Shape preferences for neurons
color_selectivity = np.random.rand(N)  # Color preferences for neurons

# Generate stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Compute original firing rates
responses = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    shape_response = shape_selectivity * shape
    color_response = color_selectivity * color
    responses[idx] = shape_response + color_response  # Combined response

# Compute modified firing rates
responses_modified = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    shape_response = shape_selectivity * shape
    color_response = color_selectivity * color
    responses_modified[idx] = 0.8 * shape_response + 1.2 * color_response  # Combined response

# Compute modified firing rates

for idx, (shape, color) in enumerate(stimuli_grid):
    shape_response = shape_selectivity * shape
    color_response = color_selectivity * color
    #responses_modified[idx] = (1+0.2*(color_selectivity-shape_selectivity))* (shape_response + color_response)  # Combined response


# Parameters
N = 500  # Number of neurons
num_stimuli = 20  # Number of stimuli per feature dimension
sigma_shape = 0.4  # Width of Gaussian tuning for shape
sigma_color = 0.4  # Width of Gaussian tuning for color

# Seed for reproducibility
np.random.seed(0)

# Define tuning properties for neurons
preferred_shape = np.random.rand(N)  # Preferred shape for each neuron
preferred_color = np.random.rand(N)  # Preferred color for each neuron

# Compute responses using Gaussian tuning
def gaussian_response(preferred, stimuli, sigma):
    """Compute Gaussian response for a given preferred value, stimuli, and sigma."""
    return np.exp(-0.5 * ((stimuli - preferred) / sigma) ** 2)

responses_gaussian = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    shape_response = gaussian_response(preferred_shape, shape, sigma_shape)
    color_response = gaussian_response(preferred_color, color, sigma_color)
    responses_gaussian[idx] = shape_response + color_response  # Combined response

# Compute modified responses
responses_modified_gaussian = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    shape_response = gaussian_response(preferred_shape, shape, sigma_shape)
    color_response = gaussian_response(preferred_color, color, sigma_color)
    responses_modified_gaussian[idx] = 0.85 * shape_response + 1.15 * color_response  # Weighted response


# Initialize variables to track the maximum gain change
max_gain_change = -np.inf
max_neuron_idx = -1
max_stimulus_idx = -1

# Calculate and find the greatest gain change
for neuron_idx in range(N):
    for stimulus_idx, (original, modified) in enumerate(zip(responses[:, neuron_idx], responses_modified[:, neuron_idx])):
        if original != 0:
            ratio = modified / original
        else:
            ratio = np.nan  # Avoid division by zero
        
        # Update the maximum gain change if the current ratio is greater
        if not np.isnan(ratio) and ratio > max_gain_change:
            max_gain_change = ratio
            max_neuron_idx = neuron_idx
            max_stimulus_idx = stimulus_idx


# Initialize variables to track the maximum gain change
max_gain_change_gaussian = -np.inf
max_neuron_idx = -1
max_stimulus_idx = -1

# Calculate and find the greatest gain change
for neuron_idx in range(N):
    for stimulus_idx, (original, modified) in enumerate(zip(responses_gaussian[:, neuron_idx], responses_modified_gaussian[:, neuron_idx])):
        if original != 0:
            ratio = modified / original
        else:
            ratio = np.nan  # Avoid division by zero
        
        # Update the maximum gain change if the current ratio is greater
        if not np.isnan(ratio) and ratio > max_gain_change_gaussian:
            max_gain_change_gaussian = ratio
            max_neuron_idx = neuron_idx
            max_stimulus_idx = stimulus_idx

# Print the result
print(f"The greatest gain change is {max_gain_change:.3f}, observed for Neuron {max_neuron_idx + 1} and Stimulus {max_stimulus_idx + 1}.")

print(f"The greatest gain change for gaussian is {max_gain_change_gaussian:.3f}, observed for Neuron {max_neuron_idx + 1} and Stimulus {max_stimulus_idx + 1}.")



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

create_response_visualization(
    responses_list=[responses, responses_modified],
    titles=["Original Responses", "Modified Responses"],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)


# Statistics
print(f"Total neurons: {N}")
