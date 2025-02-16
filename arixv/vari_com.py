import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
N = 50  # Number of neurons
K = 2   # Number of features: shape and color
num_stimuli = 10  # Number of stimuli per feature dimension

np.random.seed(0)

# Neuron selectivity matrix
S = np.random.rand(N, K)

# Identify top color-selective neurons
color_selectivity = S[:, 1]
shape_selectivity = S[:, 0]

# Thresholds for selecting top color-selective neurons
color_threshold = np.percentile(color_selectivity, 70)  # Top 30% in color
shape_threshold = np.percentile(shape_selectivity, 30)  # Bottom 30% in shape

# Indices of top color-selective neurons
color_neurons = np.where((color_selectivity > color_threshold) &
                         (shape_selectivity < shape_threshold))[0]

# Stimuli grid
shape_stimuli = np.linspace(-1, 1, num_stimuli)
color_stimuli = np.linspace(-1, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Function to compute responses
def compute_responses(gain):
    G = np.ones(N)
    G[color_neurons] *= gain
    responses = []
    for stimulus in stimuli_grid:
        r = G * (S @ stimulus)
        responses.append(r)
    return np.array(responses)

# Analyze variances for different gains
gains = np.linspace(1, 5, 5)
shape_variances = []
color_variances = []

for gain in gains:
    responses = compute_responses(gain)
    # Compute variance along shape and color dimensions
    shape_responses = responses @ [1, 0]
    color_responses = responses @ [0, 1]
    shape_variances.append(np.var(shape_responses))
    color_variances.append(np.var(color_responses))

# Plot variances
plt.figure(figsize=(8, 6))
plt.plot(gains, shape_variances, 'o-r', label='Shape Variance')
plt.plot(gains, color_variances, 's-b', label='Color Variance')
plt.xlabel('Gain')
plt.ylabel('Variance')
plt.title('Variance Along Shape and Color Dimensions vs. Gain')
plt.legend()
plt.grid(True)
plt.show()

# PCA Analysis
gain = 5  # Select a gain to visualize
responses = compute_responses(gain)
pca = PCA(n_components=2)
responses_pca = pca.fit_transform(responses)

# Plot PCA results colored by shape and color stimuli
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Color by shape
sc1 = axes[0].scatter(responses_pca[:, 0], responses_pca[:, 1],
                      c=stimuli_grid[:, 0], cmap='viridis', s=50)
axes[0].set_title('Responses Colored by Shape')
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')
plt.colorbar(sc1, ax=axes[0], label='Shape Value')

# Color by color
sc2 = axes[1].scatter(responses_pca[:, 0], responses_pca[:, 1],
                      c=stimuli_grid[:, 1], cmap='plasma', s=50)
axes[1].set_title('Responses Colored by Color')
axes[1].set_xlabel('PC 1')
axes[1].set_ylabel('PC 2')
plt.colorbar(sc2, ax=axes[1], label='Color Value')

plt.tight_layout()
plt.show()