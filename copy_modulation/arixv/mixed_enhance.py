import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
threshold = 0.1

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

# Select neurons with high color selectivity AND low shape selectivity
color_selectivity = S[:, 1]
shape_selectivity = S[:, 0]

color_threshold = np.percentile(color_selectivity, 30)  # top 30% in color
shape_threshold = np.percentile(shape_selectivity, 70)  # bottom 30% in shape

pure_color_neurons = np.where((color_selectivity > color_threshold) & 
                            (shape_selectivity < shape_threshold))[0]

# Create gain matrix for selected neurons
enhancement_factor = 2.5
gain_matrix = np.eye(N)  # Start with identity matrix
gain_matrix[pure_color_neurons, pure_color_neurons] = enhancement_factor

# Compute responses with original gain
responses = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

# Compute responses with modified gain
# Modified equation: r = inv(I - G*W_R) @ (G*W_F*x)
# where G is the gain matrix
responses_enhanced = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = gain_matrix @ W_F @ F  # Apply gain to feedforward input
    # Modified network equation with gain
    responses_enhanced[idx] = np.linalg.inv(np.eye(N) - gain_matrix @ W_R) @ adjusted_F

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
plt.scatter(S[:, 1], S[:, 0], c='gray', alpha=0.5, label='Non-selected')
plt.scatter(S[pure_color_neurons, 1], S[pure_color_neurons, 0], 
           c='red', alpha=0.8, label='Selected')
plt.axvline(color_threshold, color='b', linestyle='--', label='Color threshold')
plt.axhline(shape_threshold, color='r', linestyle='--', label='Shape threshold')
plt.xlabel('Color Selectivity')
plt.ylabel('Shape Selectivity')
plt.title('Selected Neurons: High Color, Low Shape Selectivity')
plt.legend()
plt.show()

# Create visualization of network responses
conditions = [
    ('Mixed Network - Original', responses),
    ('Mixed Network - Modified Gain', responses_enhanced)
]

create_response_visualization(
    responses_list=[resp for _, resp in conditions],
    titles=[title for title, _ in conditions],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)

# Print analysis
print("\nNetwork Analysis:")
print(f"Total number of neurons: {N}")
print(f"Number of pure color-selective neurons: {len(pure_color_neurons)}")
print("\nSelected neurons statistics:")
print(f"Mean color selectivity: {np.mean(S[pure_color_neurons, 1]):.3f}")
print(f"Mean shape selectivity: {np.mean(S[pure_color_neurons, 0]):.3f}")
print(f"Enhancement factor: {enhancement_factor}x")

# Analyze network stability
print("\nNetwork Stability Analysis:")
eig_orig = np.max(np.abs(np.linalg.eigvals(W_R)))
eig_mod = np.max(np.abs(np.linalg.eigvals(gain_matrix @ W_R)))
print(f"Max eigenvalue magnitude (original): {eig_orig:.3f}")
print(f"Max eigenvalue magnitude (modified): {eig_mod:.3f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
threshold = 0.25

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

# Thresholds for enhancement
color_threshold_high = np.percentile(color_selectivity, 75)   # Top 30% in color
shape_threshold_low = np.percentile(shape_selectivity, 25)    # Bottom 30% in shape

# Thresholds for suppression
color_threshold_low = np.percentile(color_selectivity, 100)    # Bottom 30% in color
shape_threshold_high = np.percentile(shape_selectivity, 50)   # Top 30% in shape

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

# Create visualization of network responses
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
print("\nNetwork Analysis:")
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

# Analyze network stability
eig_orig = np.max(np.abs(np.linalg.eigvals(W_R)))
eig_mod = np.max(np.abs(np.linalg.eigvals(gain_matrix @ W_R)))
print("\nNetwork Stability Analysis:")
print(f"Max eigenvalue magnitude (original): {eig_orig:.3f}")
print(f"Max eigenvalue magnitude (modified): {eig_mod:.3f}")


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
threshold = 0.15

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

# Thresholds for enhancement
color_threshold_high = np.percentile(color_selectivity, 75)   # Top 25% in color
shape_threshold_low = np.percentile(shape_selectivity, 25)    # Bottom 25% in shape

# Thresholds for suppression
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

def visualize_connectivity(W_R, enhanced_neurons, suppressed_neurons, title='Connectivity Matrix'):
    plt.figure(figsize=(10, 8))
    plt.imshow(W_R, cmap='viridis', interpolation='none')
    plt.colorbar(label='Connection Strength')
    plt.scatter(enhanced_neurons, enhanced_neurons, color='lime', label='Enhanced Neurons', s=100, edgecolors='black')
    plt.scatter(suppressed_neurons, suppressed_neurons, color='red', label='Suppressed Neurons', s=100, edgecolors='black')
    plt.title(title)
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    plt.legend()
    plt.grid(False)
    plt.show()

def draw_network_graph(W_R, enhanced_neurons, suppressed_neurons, threshold=0.01):
    G = nx.DiGraph()
    for i in range(len(W_R)):
        for j in range(len(W_R[i])):
            if W_R[i][j] > threshold:  # Only add significant connections to avoid clutter
                G.add_edge(i, j, weight=W_R[i][j])
    
    pos = nx.spring_layout(G)  # Positions for all nodes
    
    plt.figure(figsize=(12, 10))
    
    # Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=range(len(W_R)), node_color='gray', node_size=500, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=enhanced_neurons, node_color='lime', node_size=700, label='Enhanced Neurons')
    nx.draw_networkx_nodes(G, pos, nodelist=suppressed_neurons, node_color='red', node_size=700, label='Suppressed Neurons')
    
    # Edges
    edges = G.edges(data=True)
    weights = [G[u][v]['weight'] for u, v, d in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues, width=2, alpha=0.5, arrowstyle='-|>', arrowsize=10)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='whitesmoke')
    
    plt.title('Network Graph of Neurons')
    plt.legend()
    plt.axis('off')
    plt.show()

# Visualization of connectivity matrix and network graph
visualize_connectivity(W_R, neurons_enhance, neurons_suppress)
draw_network_graph(W_R, neurons_enhance, neurons_suppress)