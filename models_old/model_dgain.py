import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx

# Parameters
N = 100  # Number of neurons
K = 2   # Two features: shape and color
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(0)

# Create mixed selectivity network
S = np.random.rand(N, K)  # Selectivity matrix

# Define matrices and vectors
W_F = np.random.rand(N, K) * 0.1  # Feedforward weights
W_R = np.zeros((N, N))  # Recurrent weight matrix
threshold = 0.15  # Distance threshold for connectivity

# Set up recurrent connections
for i in range(N):
    for j in range(N):
        distance = np.linalg.norm(S[i] - S[j])
        if distance < threshold:
            W_R[i, j] = 1 - distance / threshold

# Remove self-connections in W_R
np.fill_diagonal(W_R, 0)

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

enhancement_therehold_color = 0.3 #top xxx in color
enhancement_therehold_shape = 0.25 #bottom xxx in shape

suppression_therehold_color = 0.0 #bottom xxx in color
suppression_therehold_shape = 0.0 #top xxx in shape


# Thresholds for enhancement and suppression
color_threshold_high = np.percentile(color_selectivity, (1-enhancement_therehold_color)*100)   # Top 25% in color
shape_threshold_low = np.percentile(shape_selectivity, enhancement_therehold_shape*100)    # Bottom 25% in shape
color_threshold_low = np.percentile(color_selectivity, suppression_therehold_color*100)    # Bottom 25% in color
shape_threshold_high = np.percentile(shape_selectivity, (1-suppression_therehold_shape)*100)   # Top 25% in shape

# Identify neurons for enhancement and suppression
neurons_enhance = np.where((color_selectivity > color_threshold_high) &
                           (shape_selectivity < shape_threshold_low))[0]
neurons_suppress = np.where((color_selectivity < color_threshold_low) &
                            (shape_selectivity > shape_threshold_high))[0]

# Define gain factors
enhancement_factor = 1.6
suppression_factor = 0.0  # Shut down suppressed neurons

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
    adjusted_F = gain_matrix @ W_F @ F
    responses_modified[idx] = np.linalg.inv(np.eye(N) - gain_matrix @ W_R) @ adjusted_F

# Visualize connectivity matrix
def visualize_connectivity(W_R, enhanced_neurons, suppressed_neurons):
    plt.figure(figsize=(10, 8))
    plt.imshow(W_R, cmap='viridis', interpolation='none')
    plt.colorbar(label='Connection Strength')
    plt.scatter(enhanced_neurons, enhanced_neurons, color='lime', label='Enhanced Neurons', s=100, edgecolors='black')
    plt.scatter(suppressed_neurons, suppressed_neurons, color='red', label='Suppressed Neurons', s=100, edgecolors='black')
    plt.title('Connectivity Matrix')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    plt.legend()
    plt.show()

# Draw network graph
def draw_network_graph(W_R, enhanced_neurons, suppressed_neurons, threshold=0.01):
    import networkx as nx
    
    # Initialize a directed graph
    G = nx.DiGraph()
    
    # Add all nodes to ensure positions for isolated nodes
    G.add_nodes_from(range(len(W_R)))
    
    # Add edges with weights above the threshold
    for i in range(len(W_R)):
        for j in range(len(W_R[i])):
            if W_R[i, j] > threshold:
                G.add_edge(i, j, weight=W_R[i, j])
    
    # Compute layout
    pos = nx.spring_layout(G)  # Assign positions to all nodes
    
    plt.figure(figsize=(12, 10))
    
    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, nodelist=range(len(W_R)), node_color='gray', node_size=500, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=enhanced_neurons, node_color='lime', node_size=700, label='Enhanced Neurons')
    nx.draw_networkx_nodes(G, pos, nodelist=suppressed_neurons, node_color='red', node_size=700, label='Suppressed Neurons')
    
    # Draw edges with colors based on weight
    edges = G.edges(data=True)
    weights = [G[u][v]['weight'] for u, v, d in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues, width=2, alpha=0.5, arrowsize=10)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='whitesmoke')
    
    plt.title('Network Graph of Neurons')
    plt.legend()
    plt.axis('off')
    plt.show()
    
# Visualize neuron selectivity and selected neurons
def visualize_neuron_selectivity(S, neurons_enhance, neurons_suppress):
    plt.figure(figsize=(10, 6))
    plt.scatter(S[:, 0], S[:, 1], c='gray', alpha=0.6, label='Other Neurons')  # Non-selected neurons
    plt.scatter(S[neurons_enhance, 0], S[neurons_enhance, 1], 
                c='lime', edgecolors='black', s=100, label='Enhanced Neurons')
    plt.scatter(S[neurons_suppress, 0], S[neurons_suppress, 1], 
                c='red', edgecolors='black', s=100, label='Suppressed Neurons')
    plt.axvline(np.percentile(S[:, 0], enhancement_therehold_shape*100), color='r', linestyle='--', label='Shape Threshold (Low)')
    plt.axhline(np.percentile(S[:, 1], (1-enhancement_therehold_color)*100), color='b', linestyle='--', label='Color Threshold (High)')
    plt.axvline(np.percentile(S[:, 0], (1-suppression_therehold_shape)*100), color='r', linestyle='--', label='Shape Threshold (High)')
    plt.axhline(np.percentile(S[:, 1], suppression_therehold_color*100), color='b', linestyle='--', label='Color Threshold (Low)')
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

# Run visualizations

visualize_neuron_selectivity(S, neurons_enhance, neurons_suppress)
visualize_connectivity(W_R, neurons_enhance, neurons_suppress)
draw_network_graph(W_R, neurons_enhance, neurons_suppress)

create_response_visualization(
    responses_list=[responses, responses_modified],
    titles=["Original Responses", "Modified Gain Responses"],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)

# Print statistics
print(f"Total neurons: {N}")
print(f"Enhanced neurons: {len(neurons_enhance)}, Suppressed neurons: {len(neurons_suppress)}")
print(f"Enhancement factor: {enhancement_factor}, Suppression factor: {suppression_factor}")