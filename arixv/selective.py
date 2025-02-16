import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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
    """Create network weights with selective connectivity and feedforward"""
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
    eigenvalues_W_R = np.linalg.eigvals(W_R)
    scaling_factor = np.max(np.abs(eigenvalues_W_R))
    if scaling_factor > 0:  # Avoid division by zero
        W_R = W_R / (scaling_factor + 1)
    
    return W_F, W_R

def compute_responses(W_F, W_R, stimuli):
    """Compute neural responses for given stimuli"""
    responses = np.zeros((len(stimuli), N))
    
    for idx, stimulus in enumerate(stimuli):
        adjusted_F = W_F @ stimulus
        responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F
    
    return responses

def plot_responses(responses, stimuli, feature_idx, subplot_3d, subplot_2d, title):
    """Plot 3D and 2D PCA representations"""
    responses_centered = responses - np.mean(responses, axis=0)
    
    pca = PCA(n_components=3)
    responses_pca = pca.fit_transform(responses_centered)
    
    # Choose colormap based on feature
    cmap = 'Reds' if feature_idx == 0 else 'Blues'
    feature_name = 'Shape' if feature_idx == 0 else 'Color'
    
    # 3D plot
    scatter = subplot_3d.scatter(responses_pca[:, 0], responses_pca[:, 1], responses_pca[:, 2],
                               c=stimuli[:, feature_idx], cmap=cmap)
    subplot_3d.set_title(f'{title}\nColored by {feature_name}')
    subplot_3d.set_xlabel('PC1')
    subplot_3d.set_ylabel('PC2')
    subplot_3d.set_zlabel('PC3')
    
    # 2D plot
    scatter = subplot_2d.scatter(responses_pca[:, 0], responses_pca[:, 1],
                               c=stimuli[:, feature_idx], cmap=cmap)
    subplot_2d.set_title(f'{title}\nColored by {feature_name}')
    subplot_2d.set_xlabel('PC1')
    subplot_2d.set_ylabel('PC2')
    
    # Set equal aspect ratio and scales for 2D plot
    subplot_2d.set_aspect('equal')
    max_range = max(np.ptp(responses_pca[:, 0]), np.ptp(responses_pca[:, 1]))
    mean_x = np.mean(responses_pca[:, 0])
    mean_y = np.mean(responses_pca[:, 1])
    subplot_2d.set_xlim(mean_x - max_range/2-0.5, mean_x + max_range/2+0.5)
    subplot_2d.set_ylim(mean_y - max_range/2-0.5, mean_y + max_range/2+0.5)
    
    plt.colorbar(scatter, ax=subplot_2d, label=f'{feature_name} Value')

# Create stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Create selectivity matrix and get neuron groups
S, neurons_feature1, neurons_feature2 = create_segregated_selectivity()

# Create network with selective connectivity and feedforward
W_F, W_R = create_selective_network(S, neurons_feature1, neurons_feature2)

# Compute responses
responses = compute_responses(W_F, W_R, stimuli_grid)

# Create subplots for both features
fig = plt.figure(figsize=(15, 10))

# Shape feature plots
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
plot_responses(responses, stimuli_grid, 0, ax1, ax2, 'Selective Connectivity\nand Feedforward')

# Color feature plots
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224)
plot_responses(responses, stimuli_grid, 1, ax3, ax4, 'Selective Connectivity\nand Feedforward')

plt.suptitle('Neural Responses with Segregated Features,\nSelective Connectivity and Feedforward', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# Visualize both matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Visualize feedforward weights
im1 = ax1.imshow(W_F, cmap='viridis')
ax1.set_title('Feedforward Weights (W_F)')
ax1.set_xlabel('Feature Index\n(0: Shape, 1: Color)')
ax1.set_ylabel('Neuron Index')
plt.colorbar(im1, ax=ax1, label='Weight')

# Visualize connectivity matrix
im2 = ax2.imshow(W_R, cmap='viridis')
ax2.set_title('Connectivity Matrix (W_R)')
ax2.set_xlabel('Neuron Index')
ax2.set_ylabel('Neuron Index')
plt.colorbar(im2, ax=ax2, label='Connection Strength')

plt.tight_layout()
plt.show()