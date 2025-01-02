import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape and color
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

# Create mixed selectivity network
S = np.random.rand(N, K)

# Define matrices and vectors
W_F = np.random.rand(N, K) * 0.1
W_R = np.zeros((N, N))
threshold = 0.10  # Adjustable threshold for connectivity

# Set up recurrent connections
for i in range(N):
    for j in range(N):
        distance = np.linalg.norm(S[i] - S[j])
        if distance < threshold:
            W_R[i, j] = 1 - distance / threshold

# Normalize W_R
eigenvalues = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues))
W_R = W_R / (scaling_factor + 1)

# Create stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Calculate selectivity thresholds
color_selectivity = S[:, 1]
shape_selectivity = S[:, 0]

# Define thresholds for enhancement and suppression
color_threshold_high = np.percentile(color_selectivity, 75)   # Top 25% in color
shape_threshold_low = np.percentile(shape_selectivity, 25)    # Bottom 25% in shape
color_threshold_low = np.percentile(color_selectivity, 25)    # Bottom 25% in color
shape_threshold_high = np.percentile(shape_selectivity, 75)   # Top 25% in shape

# Neurons to enhance
neurons_enhance = np.where(
    (color_selectivity > color_threshold_high) & 
    (shape_selectivity < shape_threshold_low)
)[0]

# Neurons to suppress
neurons_suppress = np.where(
    (color_selectivity < color_threshold_low) & 
    (shape_selectivity > shape_threshold_high)
)[0]

# Define gain factors
enhancement_factor = 2.8
suppression_factor = 0.0  # Set to zero to shut down the neurons

# Initialize gain matrix
gain_matrix = np.eye(N)
gain_matrix[neurons_enhance, neurons_enhance] = enhancement_factor
gain_matrix[neurons_suppress, neurons_suppress] = suppression_factor

# Compute responses with original gain
responses = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

# Compute responses with modified gain
responses_modified = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = gain_matrix @ W_F @ F  # Apply gain to feedforward input
    responses_modified[idx] = np.linalg.inv(np.eye(N) - gain_matrix @ W_R) @ adjusted_F

# Visualize PCA of responses
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
    x_lim = [(x_max + x_min) / 2 - max_range / 2, (x_max + x_min) / 2 + max_range / 2]
    y_lim = [(y_max + y_min) / 2 - max_range / 2, (y_max + y_min) / 2 + max_range / 2]
    
    for row, (response_data, title) in enumerate(zip(responses_list, titles)):
        responses_pca = pca.transform(response_data)
        
        # Plot for shape
        ax1 = fig.add_subplot(gs[row, 0])
        sc1 = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=shape_data, cmap=plt.cm.Reds, s=30)
        ax1.set_title(f'{title}\nColored by Shape')
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        
        # Plot for color
        ax2 = fig.add_subplot(gs[row, 1])
        sc2 = ax2.scatter(responses_pca[:, 0], responses_pca[:, 1], c=color_data, cmap=plt.cm.Blues, s=30)
        ax2.set_title(f'{title}\nColored by Color')
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)
    
    # Colorbars
    cax_shape = fig.add_subplot(gs[0, 2])
    cax_color = fig.add_subplot(gs[-1, 2])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.Reds), cax=cax_shape, label='Shape Value')
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.Blues), cax=cax_color, label='Color Value')
    
    plt.tight_layout()
    plt.show()

# Visualize selected neurons' selectivity
plt.figure(figsize=(10, 5))
plt.scatter(S[:, 1], S[:, 0], c='gray', alpha=0.5, label='Other Neurons')
plt.scatter(S[neurons_enhance, 1], S[neurons_enhance, 0], c='lime', label='Enhanced Neurons')
plt.scatter(S[neurons_suppress, 1], S[neurons_suppress, 0], c='red', label='Suppressed Neurons')
plt.axvline(color_threshold_low, color='blue', linestyle='--')
plt.axvline(color_threshold_high, color='blue', linestyle='--')
plt.axhline(shape_threshold_low, color='red', linestyle='--')
plt.axhline(shape_threshold_high, color='red', linestyle='--')
plt.xlabel('Color Selectivity')
plt.ylabel('Shape Selectivity')
plt.legend()
plt.title('Neuron Selectivity and Gain Modulation')
plt.show()

# Connectivity matrix and graph visualization
def visualize_connectivity(W_R, enhanced_neurons, suppressed_neurons):
    plt.figure(figsize=(10, 8))
    plt.imshow(W_R, cmap='viridis', interpolation='none')
    plt.colorbar(label='Connection Strength')
    plt.scatter(enhanced_neurons, enhanced_neurons, color='lime', s=100, edgecolors='black')
    plt.scatter(suppressed_neurons, suppressed_neurons, color='red', s=100, edgecolors='black')
    plt.title('Connectivity Matrix with Enhanced and Suppressed Neurons')
    plt.show()

def draw_network_graph(W_R, enhanced_neurons, suppressed_neurons, threshold=0.01):
    G = nx.DiGraph()
    for i in range(len(W_R)):
        for j in range(len(W_R[i])):
            if W_R[i][j] > threshold:
                G.add_edge(i, j, weight=W_R[i][j])
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, nodelist=range(N), node_color='gray', alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=enhanced_neurons, node_color='lime', label='Enhanced Neurons')
    nx.draw_networkx_nodes(G, pos, nodelist=suppressed_neurons, node_color='red', label='Suppressed Neurons')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=10)
    nx.draw_networkx_labels(G, pos)
    plt.legend()
    plt.title('Network Graph of Neurons')
    plt.show()

# Create visualizations
conditions = [('Original Responses', responses), ('Modified Gain Responses', responses_modified)]
create_response_visualization([resp for _, resp in conditions], [title for title, _ in conditions], stimuli_grid[:, 0], stimuli_grid[:, 1])
visualize_connectivity(W_R, neurons_enhance, neurons_suppress)
draw_network_graph(W_R, neurons_enhance, neurons_suppress)