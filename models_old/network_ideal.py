import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape (index=0) and color (index=1)
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(1)

# Generate random selectivities for one feature
S = np.zeros((N, K))


# Shape selectivity: Uniformly distributed between 0.3 and 1.0
#for half of the neurons, assign them a shape selectivity of 0.5+0.5 * np.random.rand(N) 
#S[:, 0] = 0.8 * np.random.rand(N)

# Color selectivity: Independent but constrained to ensure some balance
# Use a complementary relationship with added noise for diversity
#noise = 0.1 * np.random.randn(N)
#S[:, 1] = np.clip(0.8 - 0.7 * S[:, 0] + noise, 0.1, 1.0)  # Ensure values stay within [0.1, 1.0]

########################

# Generate evenly distributed shape selectivities between 0.3 and 1.0
#S[:, 0] = np.linspace(0, 1.0, N)

# Optionally add small jitter to shape selectivities to avoid exact regular spacing
#jitter = 0.1 * (np.random.rand(N) - 0.5)  # Jitter in range [-0.01, 0.01]
#S[:, 0] += jitter
#S[:, 0] = np.clip(S[:, 0], 0.0, 1.0)  # Ensure values remain within [0.3, 1.0]

# Compute color selectivity inversely proportional to shape selectivity with some noise
# This introduces some variability while maintaining an overall inverse relationship
#noise = 0.05 * np.random.randn(N)
#S[:, 1] = 1 - 1 * S[:, 0] + noise
#S[:, 1] = np.clip(S[:, 1], 0.1, 1.0)  # Ensure values stay within [0.1, 1.0]

S[:N//2, 0] = 0.5 + 0.5 * np.random.rand(N//2)
noise = 0.1 * np.random.randn(N//2)
S[:N//2, 1] = np.clip(0.7 - S[:N//2, 0] + noise, 0.0, 1.0)  # Ensure values stay within [0.0, 1.0]

S[N//2:, 1] = 0.5 + 0.5 * np.random.rand(N//2)
noise = 0.1 * np.random.randn(N//2)
S[N//2:, 0] = np.clip(0.7 - S[N//2:, 1] + noise, 0.0, 1.0)  # Ensure values stay within [0.0, 1.0]



# Initialize and normalize W_F
W_F = np.zeros((N, K))
for i in range(N):
    W_F[i, 0] = S[i, 0]
    W_F[i, 1] = S[i, 1]

# Normalize W_F rows
W_F = W_F / W_F.sum(axis=1, keepdims=True)

# Initialize W_R with distance-based modulation
W_R = np.random.rand(N, N) * 0.1

WR_tuned = False
if WR_tuned:
    threshold = 0.2
    for i in range(N):
        for j in range(N):
            distance = np.linalg.norm(S[i] - S[j])
            if distance < threshold:
                W_R[i, j] = W_R[i, j] * (2 - distance / threshold)

np.fill_diagonal(W_R, 0)  # Remove self-connections
eigenvalues = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues))
W_R = W_R / (scaling_factor + 1) *0.5


# Create stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Generate responses
responses = np.zeros((len(stimuli_grid), N))
#W_R = np.zeros((N, N))
# Analytical steady states for each stimulus
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F


# Modify W_R based on neuron preferences
#for i in range(N):
     #W_R[i, :] *= 1+(S[i, 1] - S[i, 0])
    #if (S[i, 1] - S[i, 0]) > 0:  # Color-preferring neuron
    #    W_R[i, :] *= 1.5  # Strengthen outgoing connections from color-preferring neurons
    #elif (S[i, 1] - S[i, 0]) < 0:  # Shape-preferring neuron
    #    W_R[i, :] *= 0.5  # Weaken outgoing connections from shape-preferring neurons

# Modify W_R based on neuron preferences, focusing only on color-to-color and shape-to-shape connections
for i in range(N):
    for j in range(N):
        if (S[i, 1] - S[i, 0]) > 0.0 and (S[j, 1] - S[j, 0]) > 0.0:
            # Both neurons are color-preferring: strengthen connection
            W_R[i, j] *= 1+(S[i, 1] - S[i, 0])*1.2
        elif (S[i, 1] - S[i, 0]) < -0.0 and (S[j, 1] - S[j, 0]) < -0.0:
            # Both neurons are shape-preferring: weaken connection
            W_R[i, j] *= 1+(S[i, 1] - S[i, 0])*2.0
                   
#eigenvalues = np.linalg.eigvals(W_R)
#scaling_factor = np.max(np.abs(eigenvalues))
#W_R /= (scaling_factor + 1)
        
        
# Generate responses
responses_modified = np.zeros((len(stimuli_grid), N))

# Analytical steady states for each stimulus
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    adjusted_F = W_F @ F
    responses_modified[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F


# Compute modulation gain ratio for each neuron
modulation_gain_ratios = np.mean(responses_modified[1:] / responses[1:], axis=0)

# Plot scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(S[:, 1]-S[:, 0], modulation_gain_ratios, alpha=0.7)
plt.xlabel('Feature Selectivity(Color-Shape)')
plt.ylabel('Modulation Gain Ratio')
plt.title('Neuron Color Selectivity vs. Modulation Gain Ratio')
plt.grid(alpha=0.3)
plt.show()


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
