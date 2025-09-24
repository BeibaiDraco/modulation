import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape (index=0) and color (index=1)
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

# Feature selectivities - explicitly separate shape and color
# S[:, 0] is shape selectivity, S[:, 1] is color selectivity
S = np.random.rand(N, K)  

# Make some neurons strongly color-selective and others strongly shape-selective
# This ensures a clearer separation between shape and color processing
for i in range(N):
    if i < N/2:  # First half: more color-selective
        S[i, 1] *= 1.5  # Enhance color selectivity
        S[i, 0] *= 0.5  # Reduce shape selectivity
    else:  # Second half: more shape-selective
        S[i, 0] *= 1.5  # Enhance shape selectivity
        S[i, 1] *= 0.5  # Reduce color selectivity

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

# Define consistent colormaps and normalization
SHAPE_CMAP = plt.cm.Reds
COLOR_CMAP = plt.cm.Blues
shape_norm = plt.Normalize(vmin=0, vmax=1)
color_norm = plt.Normalize(vmin=0, vmax=1)

def modify_responses_color_gain(responses, S, percentile, gain_factor):
    """
    Modify responses of color-selective neurons
    S[:, 1] specifically represents color selectivity
    """
    # Use only color selectivity (S[:, 1]) to identify color-selective neurons
    color_selectivity = S[:, 1]
    color_selectivity_threshold = np.percentile(color_selectivity, 100 - percentile)
    high_color_indices = np.where(color_selectivity > color_selectivity_threshold)[0]
    
    # Print diagnostic information
    print(f"Number of color-selective neurons selected: {len(high_color_indices)}")
    print(f"Color selectivity threshold: {color_selectivity_threshold:.3f}")
    
    responses_modified = np.copy(responses)
    responses_modified[:, high_color_indices] *= gain_factor
    return responses_modified

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

# Generate different response conditions
responses_original = responses
responses_moderate = modify_responses_color_gain(responses, S, 20, 10)  # top 40% color-selective neurons, 3x gain
responses_strong = modify_responses_color_gain(responses, S, 10, 10)    # top 5% color-selective neurons, 10x gain

# Generate responses
responses_moderate = np.zeros((len(stimuli_grid), N))

# Compute responses to each stimulus
# Modify W_R based on neuron preferences
for i in range(N):
    if S[i, 1] > S[i, 0]:  # Color-preferring neuron
        W_R[i, :] *= 1.2
    else:  # Shape-preferring neuron
        W_R[i, :] *= 0.8

responses_strong = np.zeros((len(stimuli_grid), N))
     
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = W_F @ F
    responses_moderate[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

responses_strong = np.zeros((len(stimuli_grid), N))

for i in range(N):
    if S[i, 1] > S[i, 0]:  # Color-preferring neuron
        W_R[i, :] *= 1.4
    else:  # Shape-preferring neuron
        W_R[i, :] *= 0.6
        
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = W_F @ F
    responses_strong[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F


# Create visualization
conditions = [
    ('Original Responses', responses_original),
    ('Moderate Color Gain (10x top 20% color-selective)', responses_moderate),
    ('Strong Color Gain (10x top 10% color-selective)', responses_strong)
]

create_response_visualization(
    responses_list=[resp for _, resp in conditions],
    titles=[title for title, _ in conditions],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1],
    output_filename='neural_responses_visualization.png'
)

# Print selectivity analysis
print("\nSelectivity Analysis:")
print("Mean shape selectivity:", np.mean(S[:, 0]))
print("Mean color selectivity:", np.mean(S[:, 1]))
print("Max shape selectivity:", np.max(S[:, 0]))
print("Max color selectivity:", np.max(S[:, 1]))