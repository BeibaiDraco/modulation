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
    neurons_feature1 = np.random.choice(N, N//2, replace=False)
    neurons_feature2 = np.setdiff1d(np.arange(N), neurons_feature1)
    
    S[neurons_feature1, 0] = np.random.rand(len(neurons_feature1))
    S[neurons_feature2, 1] = np.random.rand(len(neurons_feature2))
    return S

def create_overlapping_selectivity():
    """Create feature selectivity matrix where neurons can respond to both features"""
    return np.random.rand(N, K)

def create_network(S):
    """Create network weights based on feature selectivity"""
    W_F = np.random.rand(N, K) * 0.1
    W_R = np.zeros((N, N))
    threshold = 0.1
    
    for i in range(N):
        for j in range(N):
            distance = np.linalg.norm(S[i] - S[j])
            if distance < threshold:
                W_R[i, j] = 1 - distance / threshold

    eigenvalues_W_R = np.linalg.eigvals(W_R)
    scaling_factor = np.max(np.abs(eigenvalues_W_R))
    W_R = W_R / (scaling_factor + 1)
    
    return W_F, W_R

def compute_responses(W_F, W_R, stimuli):
    """Compute neural responses for given stimuli"""
    responses = np.zeros((len(stimuli), N))
    
    for idx, stimulus in enumerate(stimuli):
        adjusted_F = W_F @ stimulus
        responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F
    
    return responses

def plot_responses(responses, stimuli, ax_3d, ax_2d, title_prefix, color_feature_idx, equal_axes=True):
    """Plot 3D and 2D PCA representations"""
    responses_centered = responses - np.mean(responses, axis=0)
    
    pca = PCA(n_components=3)
    responses_pca = pca.fit_transform(responses_centered)
    
    # Choose colormap based on feature
    cmap = 'Reds' if color_feature_idx == 0 else 'Blues'
    feature_name = 'Shape' if color_feature_idx == 0 else 'Color'
    
    # 3D plot
    scatter = ax_3d.scatter(responses_pca[:, 0], responses_pca[:, 1], responses_pca[:, 2],
                           c=stimuli[:, color_feature_idx], cmap=cmap)
    ax_3d.set_title(f'{title_prefix}\nColored by {feature_name}')
    ax_3d.set_xlabel('PC1')
    ax_3d.set_ylabel('PC2')
    ax_3d.set_zlabel('PC3')
    
    # 2D plot
    scatter = ax_2d.scatter(responses_pca[:, 0], responses_pca[:, 1],
                           c=stimuli[:, color_feature_idx], cmap=cmap)
    ax_2d.set_title(f'{title_prefix}\nColored by {feature_name}')
    ax_2d.set_xlabel('PC1')
    ax_2d.set_ylabel('PC2')
    
    if equal_axes:
        ax_2d.set_aspect('equal')
        
    # Find max extent for equal scaling
    max_range = max(
        np.ptp(responses_pca[:, 0]),
        np.ptp(responses_pca[:, 1])
    )
    mean_x = np.mean(responses_pca[:, 0])
    mean_y = np.mean(responses_pca[:, 1])
    ax_2d.set_xlim(mean_x - max_range/2, mean_x + max_range/2)
    ax_2d.set_ylim(mean_y - max_range/2, mean_y + max_range/2)
    
    plt.colorbar(scatter, ax=ax_2d, label=f'{feature_name} Value')

# Create stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Create networks
S_segregated = create_segregated_selectivity()
W_F_segregated, W_R_segregated = create_network(S_segregated)
responses_segregated = compute_responses(W_F_segregated, W_R_segregated, stimuli_grid)

S_overlapping = create_overlapping_selectivity()
W_F_overlapping, W_R_overlapping = create_network(S_overlapping)
responses_overlapping = compute_responses(W_F_overlapping, W_R_overlapping, stimuli_grid)

# Create two figures for shape and color features
for feature_idx in range(2):
    fig = plt.figure(figsize=(15, 10))
    feature_name = 'Shape' if feature_idx == 0 else 'Color'
    
    # Segregated selectivity plots
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    plot_responses(responses_segregated, stimuli_grid, ax1, ax2, 
                  'Segregated Features', feature_idx)
    
    # Overlapping selectivity plots
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)
    plot_responses(responses_overlapping, stimuli_grid, ax3, ax4, 
                  'Overlapping Features', feature_idx)
    
    plt.suptitle(f'Neural Responses Colored by {feature_name}', y=1.02, fontsize=14)
    plt.tight_layout()

plt.show()