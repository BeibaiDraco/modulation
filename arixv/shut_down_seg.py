import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape and color
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

def create_segregated_selectivity():
    """Create feature selectivity matrix where neurons respond to only one feature"""
    S = np.zeros((N, K))
    # Randomly select half of neurons for feature 1
    neurons_feature1 = np.random.choice(N, N//2, replace=False)
    neurons_feature2 = np.setdiff1d(np.arange(N), neurons_feature1)
    
    # Set random selectivity values
    S[neurons_feature1, 0] = np.random.rand(len(neurons_feature1))
    S[neurons_feature2, 1] = np.random.rand(len(neurons_feature2))
    
    return S, neurons_feature1, neurons_feature2

def create_selective_network(S, neurons_feature1, neurons_feature2):
    """Create network weights with selective connectivity"""
    # Initialize feedforward weights
    W_F = np.zeros((N, K))
    
    # Set feedforward weights only for preferred features
    W_F[neurons_feature1, 0] = np.random.rand(len(neurons_feature1)) * 0.1  # Shape inputs
    W_F[neurons_feature2, 1] = np.random.rand(len(neurons_feature2)) * 0.1  # Color inputs
    
    # Initialize recurrent weights
    W_R = np.zeros((N, N))
    threshold = 0.1
    
    # Connect neurons that prefer the same feature
    for feature_neurons in [neurons_feature1, neurons_feature2]:
        for i in feature_neurons:
            for j in feature_neurons:
                if i != j:  # Avoid self-connections
                    distance = np.linalg.norm(S[i] - S[j])
                    if distance < threshold:
                        W_R[i, j] = 1 - distance / threshold
    
    # Normalize W_R
    eigenvalues = np.linalg.eigvals(W_R)
    scaling_factor = np.max(np.abs(eigenvalues))
    if scaling_factor > 0:
        W_R = W_R / (scaling_factor + 1)
    
    return W_F, W_R

# Create stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Create segregated network
S, shape_neurons, color_neurons = create_segregated_selectivity()
W_F, W_R = create_selective_network(S, shape_neurons, color_neurons)

# Compute original responses
responses = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

# Shutdown 50% of shape neurons
num_shutdown = len(shape_neurons) // 2
shutdown_indices = np.random.choice(shape_neurons, num_shutdown, replace=False)
responses_shutdown = responses.copy()
responses_shutdown[:, shutdown_indices] = 0

# Print shutdown information
print("\nNetwork Analysis:")
print(f"Total number of shape neurons: {len(shape_neurons)}")
print(f"Number of shape neurons shut down: {num_shutdown}")
print(f"Indices of shut down shape neurons: {shutdown_indices}")
print(f"Number of remaining active shape neurons: {len(shape_neurons) - num_shutdown}")
print(f"Number of color neurons (unchanged): {len(color_neurons)}")

def create_response_visualization(responses_list, titles, shape_data, color_data):
    fig = plt.figure(figsize=(15, 4 * len(responses_list)))
    gs = fig.add_gridspec(len(responses_list), 3, width_ratios=[1, 1, 0.1])
    
    # Fit PCA on original responses
    pca = PCA(n_components=2)
    pca.fit(responses_list[0])
    
    # Find global limits
    all_pca_data = np.vstack([pca.transform(resp) for resp in responses_list])
    x_min, x_max = all_pca_data[:, 0].min(), all_pca_data[:, 0].max()
    y_min, y_max = all_pca_data[:, 1].min(), all_pca_data[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_lim = [x_center - max_range/2-0.3, x_center + max_range/2+0.3]
    y_lim = [y_center - max_range/2-0.3, y_center + max_range/2+0.3]
    
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

# Create conditions for visualization
conditions = [
    ('Segregated Network - Original', responses),
    ('Segregated Network - 50% Shape Neurons Shutdown', responses_shutdown)
]

# Visualize results
create_response_visualization(
    responses_list=[resp for _, resp in conditions],
    titles=[title for title, _ in conditions],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)

# Additional analysis of representation
print("\nRepresentation Analysis:")
pca = PCA(n_components=2)
pca_original = pca.fit(responses)
print(f"Original variance explained: {pca.explained_variance_ratio_}")

pca_shutdown = pca.fit(responses_shutdown)
print(f"Post-shutdown variance explained: {pca.explained_variance_ratio_}")