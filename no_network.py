import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape and color
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

# Create selectivity for each neuron independently for shape and color
S = np.random.rand(N, K)  # Random selectivity matrix

# Create stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Calculate selectivity thresholds for enhancement and suppression
color_selectivity = S[:, 1]
shape_selectivity = S[:, 0]

# Thresholds for enhancement
color_threshold_high = np.percentile(color_selectivity, 75)   # Top 25% in color
shape_threshold_low = np.percentile(shape_selectivity, 50)    # Bottom 25% in shape

# Thresholds for suppression
color_threshold_low = np.percentile(color_selectivity, 0)    # Bottom 25% in color
shape_threshold_high = np.percentile(shape_selectivity, 0)   # Top 25% in shape

# Identify neurons to enhance and suppress
neurons_enhance = np.where((color_selectivity > color_threshold_high) & 
                           (shape_selectivity < shape_threshold_low))[0]
neurons_suppress = np.where((color_selectivity < color_threshold_low) & 
                            (shape_selectivity > shape_threshold_high))[0]

# Define gain factors
enhancement_factor = 2.8
suppression_factor = 0.0  # Set to zero to shut down the neurons

# Initialize gain matrix
gain_matrix = np.eye(N)
gain_matrix[neurons_enhance, neurons_enhance] = enhancement_factor
gain_matrix[neurons_suppress, neurons_suppress] = suppression_factor

# Compute responses for each stimulus without gain modification
responses = np.array([S @ stim for stim in stimuli_grid])

# Compute responses with modified gain
responses_modified = np.array([gain_matrix @ S @ stim for stim in stimuli_grid])

def create_response_visualization(responses_list, titles, shape_data, color_data):
    fig = plt.figure(figsize=(15, 4 * len(responses_list)))
    gs = fig.add_gridspec(len(responses_list), 3, width_ratios=[1, 1, 0.1])
    
    # Fit PCA on original responses
    pca = PCA(n_components=2)
    pca.fit(responses_list[0])
    
    # Find global limits for consistent scaling
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
        
        # Shape plot
        ax1 = fig.add_subplot(gs[row, 0])
        sc1 = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1],
                         c=shape_data, cmap=plt.cm.Reds, s=30)
        ax1.set_title(f'{title}\nColored by Shape')
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 2')
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        ax1.set_aspect('equal')
        
        # Color plot
        ax2 = fig.add_subplot(gs[row, 1])
        sc2 = ax2.scatter(responses_pca[:, 0], responses_pca[:, 1],
                         c=color_data, cmap=plt.cm.Blues, s=30)
        ax2.set_title(f'{title}\nColored by Color')
        ax2.set_xlabel('PC 1')
        ax2.set_ylabel('PC 2')
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)
        ax2.set_aspect('equal')
    
    # Add colorbars
    cax_shape = fig.add_subplot(gs[0, 2])
    cax_color = fig.add_subplot(gs[-1, 2])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.Reds),
                cax=cax_shape, label='Shape Value')
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.Blues),
                cax=cax_color, label='Color Value')
    
    plt.tight_layout()
    plt.show()

# Visualize selected neurons' selectivity
plt.figure(figsize=(10, 5))
plt.scatter(S[:, 1], S[:, 0], c='gray', alpha=0.5, label='Other Neurons')
plt.scatter(S[neurons_enhance, 1], S[neurons_enhance, 0], 
           c='green', alpha=0.8, label='Enhanced Neurons')
plt.scatter(S[neurons_suppress, 1], S[neurons_suppress, 0], 
           c='red', alpha=0.8, label='Suppressed Neurons')
plt.axvline(color_threshold_low, color='b', linestyle='--', label='Color Low Threshold')
plt.axvline(color_threshold_high, color='b', linestyle='--', label='Color High Threshold')
plt.axhline(shape_threshold_low, color='r', linestyle='--', label='Shape Low Threshold')
plt.axhline(shape_threshold_high, color='r', linestyle='--', label='Shape High Threshold')
plt.xlabel('Color Selectivity')
plt.ylabel('Shape Selectivity')
plt.title('Neuron Selectivity and Gain Modulation')
plt.legend()
plt.show()

# Create visualization of neuron responses with and without gain modifications
conditions = [
    ('Original Responses', responses),
    ('Modified Gain Responses', responses_modified)
]

create_response_visualization(
    responses_list=[resp for _, resp in conditions],
    titles=[title for title, _ in conditions],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)

# Print analysis
print("\nNeuron Analysis:")
print(f"Total number of neurons: {N}")
print(f"Number of neurons enhanced: {len(neurons_enhance)}")
print(f"Number of neurons suppressed: {len(neurons_suppress)}")

print("\nEnhanced Neurons Statistics:")
print(f"Mean color selectivity: {np.mean(S[neurons_enhance, 1]):.3f}")
print(f"Mean shape selectivity: {np.mean(S[neurons_enhance, 0]):.3f}")
print(f"Enhancement factor: {enhancement_factor}x")

print("\nSuppressed Neurons Statistics:")
print(f"Mean color selectivity: {np.mean(S[neurons_suppress, 1]):.3f}")
print(f"Mean shape selectivity: {np.mean(S[neurons_suppress, 0]):.3f}")
print(f"Suppression factor: {suppression_factor}x")