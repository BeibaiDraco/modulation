import numpy as np
import matplotlib.pyplot as plt

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


# Compute modulation gain ratio for each neuron
modulation_gain_ratios = np.mean(responses_modified[1:] / responses[1:], axis=0)

# Plot scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(color_selectivity-shape_selectivity, modulation_gain_ratios, alpha=0.7)
plt.xlabel('Feature Selectivity(Color-Shape)')
plt.ylabel('Modulation Gain Ratio')
plt.title('Neuron Color Selectivity vs. Modulation Gain Ratio')
plt.grid(alpha=0.3)
plt.show()