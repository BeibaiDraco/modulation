import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape (index=0) and color (index=1)
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

# Feature selectivities
S = np.random.rand(N, K)  # S[:, 0] is shape selectivity, S[:, 1] is color selectivity

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

# Create stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Generate responses
responses = np.zeros((len(stimuli_grid), N))

# Compute responses to each stimulus
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

def modify_responses_continuous_gain(responses, S, enhancement_factor=2.0):
    """
    Modify responses based on color selectivity:
    - Neurons with above-mean selectivity get enhanced proportionally
    - Neurons with below-mean selectivity get suppressed proportionally
    
    Args:
        responses: original responses
        S: selectivity matrix
        enhancement_factor: maximum enhancement for most selective neurons
    """
    color_selectivity = S[:, 1]
    mean_selectivity = np.mean(color_selectivity)
    
    # Calculate relative selectivity (centered around mean)
    relative_selectivity = color_selectivity - mean_selectivity
    
    # Normalize relative selectivity to [-1, 1] range
    max_abs_selectivity = np.max(np.abs(relative_selectivity))
    normalized_selectivity = relative_selectivity / max_abs_selectivity
    
    # Convert to gain factors: maps [-1, 1] to [1/enhancement_factor, enhancement_factor]
    gain_factors = enhancement_factor ** normalized_selectivity
    
    # Print diagnostic information
    print("\nGain Analysis:")
    print(f"Mean color selectivity: {mean_selectivity:.3f}")
    print(f"Max gain: {np.max(gain_factors):.3f}")
    print(f"Min gain: {np.min(gain_factors):.3f}")
    print(f"Number of enhanced neurons (gain > 1): {np.sum(gain_factors > 1)}")
    print(f"Number of suppressed neurons (gain < 1): {np.sum(gain_factors < 1)}")
    
    # Apply gains to responses
    responses_modified = responses.copy()
    for i in range(N):
        responses_modified[:, i] *= gain_factors[i]
    
    return responses_modified

# Define consistent colormaps and normalization
SHAPE_CMAP = plt.cm.Reds
COLOR_CMAP = plt.cm.Blues
shape_norm = plt.Normalize(vmin=0, vmax=1)
color_norm = plt.Normalize(vmin=0, vmax=1)

def create_response_visualization(responses_list, titles, shape_data, color_data, output_filename=None):
    """Create a consistent visualization for multiple response conditions"""
    fig = plt.figure(figsize=(15, 4 * len(responses_list)))
    gs = fig.add_gridspec(len(responses_list), 3, width_ratios=[1, 1, 0.1])
    
    # Fit PCA on original responses
    pca = PCA(n_components=2)
    pca.fit(responses_list[0])
    
    # Find global limits for consistent scaling
    all_pca_data = np.vstack([pca.transform(resp) for resp in responses_list])
    x_min, x_max = all_pca_data[:, 0].min(), all_pca_data[:, 0].max()
    y_min, y_max = all_pca_data[:, 1].min(), all_pca_data[:, 1].max()
    
    # Ensure square aspect ratio
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
                         c=shape_data, cmap=SHAPE_CMAP, norm=shape_norm, s=30)
        ax1.set_title(f'{title}\nColored by Shape')
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 2')
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        ax1.set_aspect('equal')
        
        # Color plot
        ax2 = fig.add_subplot(gs[row, 1])
        sc2 = ax2.scatter(responses_pca[:, 0], responses_pca[:, 1],
                         c=color_data, cmap=COLOR_CMAP, norm=color_norm, s=30)
        ax2.set_title(f'{title}\nColored by Color')
        ax2.set_xlabel('PC 1')
        ax2.set_ylabel('PC 2')
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)
        ax2.set_aspect('equal')
    
    # Add colorbars
    cax_shape = fig.add_subplot(gs[0, 2])
    cax_color = fig.add_subplot(gs[-1, 2])
    plt.colorbar(plt.cm.ScalarMappable(norm=shape_norm, cmap=SHAPE_CMAP),
                cax=cax_shape, label='Shape Value')
    plt.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=COLOR_CMAP),
                cax=cax_color, label='Color Value')
    
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

# Generate different response conditions with varying enhancement factors
responses_original = responses
responses_moderate = modify_responses_continuous_gain(responses, S, enhancement_factor=2.0)
responses_strong = modify_responses_continuous_gain(responses, S, enhancement_factor=10.0)

# Create visualization
conditions = [
    ('Original Responses', responses_original),
    ('Moderate Continuous Gain (2x)', responses_moderate),
    ('Strong Continuous Gain (4x)', responses_strong)
]

create_response_visualization(
    responses_list=[resp for _, resp in conditions],
    titles=[title for title, _ in conditions],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)

# Print additional selectivity analysis
print("\nSelectivity Distribution:")
print("Color selectivity percentiles:")
percentiles = [0, 25, 50, 75, 100]
for p in percentiles:
    print(f"{p}th percentile: {np.percentile(S[:, 1], p):.3f}")