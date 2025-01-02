import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np
np.random.seed(15)

# Parameters
N = 300  # Number of neurons
K = 2     # Two features: shape (index=0) and color (index=1)
num_stimuli = 10  # Number of stimuli per feature dimension


# Initialize selectivity matrix
S = np.zeros((N, K))

# =======================
# First Half of Neurons
S[:N//2, 0] = np.random.rand(N//2)

# Compute S[:N//2, 1] based on the formula
S[:N//2, 1] = (1/2 - S[:N//2, 0]/2)

# Identify indices where S[:N//2, 1] is negative
negative_indices = S[:N//2, 0] - S[:N//2, 1] < 0
# Replace negative values in S[:N//2, 1] with random values in (0, 0.1)
S[:N//2, 0][negative_indices] = np.random.uniform(0, 0.5, size=np.sum(negative_indices))

# Ensure S[i, 0] >= S[i, 1] for all i in the first half
#flip_indices = S[:N//2, 1] > S[:N//2, 0]  # Identify indices where S[i, 1] < S[i, 0]

# Flip S[i, 0] and S[i, 1] for those indices
#S[:N//2][flip_indices] = S[:N//2][flip_indices][:, ::-1]  # Flip the columns for those rows

# =======================
# Second Half of Neurons (S[N//2:, 1] is Primary)
# =======================
# Assign random values to S[N//2:, 1]
#S[N//2:, 1] = np.random.rand(N//2)

# Compute S[N//2:, 0] based on the formula
#S[N//2:, 0] = (1/3 - S[N//2:, 1]/3)

# Identify indices where S[N//2:, 0] is negative
#negative_indices = S[N//2:, 0] < 0

# Replace negative values in S[N//2:, 0] with random values in (0, 0.1)
#S[N//2:, 0][negative_indices] = np.random.uniform(0, 0.1, size=np.sum(negative_indices))

#Ensure S[i, 0] <= S[i, 1] for all i in the second half
#flip_indices = S[N//2:, 1] < S[N//2:, 0]  # Identify indices where S[i, 1] < S[i, 0]

# Flip S[i, 0] and S[i, 1] for those indices
#S[N//2:][flip_indices] = S[N//2:][flip_indices][:, ::-1]  # Flip the columns for those rows

S[N//2:, 1] = S[:N//2, 0]
S[N//2:, 0] = S[:N//2, 1]

# Initialize and normalize W_F
W_F = np.zeros((N, K))
for i in range(N):
    if S[i, 0]>0.5:
       W_F[i, 0] = S[i, 0]
    elif S[i, 1]>0.5:
       W_F[i, 1] = S[i, 1]
    else:
         W_F[i, 0] = S[i, 0]
         W_F[i, 1] = S[i, 1]

# Normalize W_F rows, handling rows that sum to zero
row_sums = W_F.sum(axis=1, keepdims=True)  # Shape: (1000, 1)

# Create a boolean mask where row sums are not zero
nonzero_mask = row_sums != 0  # Shape: (1000, 1)

# Initialize W_F_normalized with zeros, same shape as W_F
W_F_normalized = np.zeros_like(W_F)  # Shape: (1000, 2)

# Normalize only the rows where the sum is not zero
# Reshape row_sums[nonzero_mask] to (568, 1) for correct broadcasting
W_F_normalized[nonzero_mask[:, 0], :] = W_F[nonzero_mask[:, 0], :] / row_sums[nonzero_mask].reshape(-1, 1)

# Replace W_F with the normalized matrix
W_F = W_F_normalized

# Initialize W_R with structured connectivity
W_R = np.zeros((N, N))  # Start with all connections as zero

# Define block sizes
half_N = N // 2

# Define connection probabilities
p_high = 0.25  # Probability for shape-shape and color-color connections
p_low = 0.25  # Probability for shape-color and color-shape connections

# Shape-Shape Block (Top-Left)
shape_shape_mask = np.random.rand(half_N, half_N) < p_high
W_R[:half_N, :half_N][shape_shape_mask] = np.random.rand(np.sum(shape_shape_mask))  # Initialize with small weights

# Shape-Color Block (Top-Right)
shape_color_mask = np.random.rand(half_N, N - half_N) < p_low
W_R[:half_N, half_N:][shape_color_mask] = np.random.rand(np.sum(shape_color_mask))

# Color-Shape Block (Bottom-Left)
color_shape_mask = np.random.rand(N - half_N, half_N) < p_low
W_R[half_N:, :half_N][color_shape_mask] = np.random.rand(np.sum(color_shape_mask))

# Color-Color Block (Bottom-Right)
color_color_mask = np.random.rand(N - half_N, N - half_N) < p_high
W_R[half_N:, half_N:][color_color_mask] = np.random.rand(np.sum(color_color_mask))


# Remove self-connections
np.fill_diagonal(W_R, 0)

# Optional: Tune W_R based on distance (if desired)
WR_tuned = False  # Set to True if distance-based modulation is still needed
if WR_tuned:
    threshold = 0.2
    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # Skip self-connections
            distance = np.linalg.norm(S[i] - S[j])
            if distance < threshold:
                W_R[i, j] *= (2 - distance / threshold)



# Scale W_R to ensure stability (spectral radius < 1)
eigenvalues = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues))
desired_radius = 0.9  # You can adjust this value as needed
W_R = W_R * (desired_radius / scaling_factor)

# Continue with the rest of your code...

#W_R = np.zeros((N, N))

# Create stimuli grid
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

# Generate responses
responses = np.zeros((len(stimuli_grid), N))
#W_R = np.zeros((N, N))
# Analytical steady states for each stimulus
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    adjusted_F = W_F @ F
    responses[idx] = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F



g_vector = 1.0 + 0.1 * (S[:, 1] - S[:, 0]) 

# Define the sigmoid function
def sigmoid(x, k=10):
    """
    Sigmoid function to introduce nonlinearity.
    
    Parameters:
    - x (np.ndarray): Input array.
    - k (float, optional): Steepness parameter. Higher values make the function steeper. Default is 10.
    
    Returns:
    - np.ndarray: Output array after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-k * x))-0.5

# ... [Previous code sections remain unchanged] ...

# **New Section: Compute `responses_modified` with Dynamic Per-Neuron Gain Factors**

# 1. Compute the difference between color and shape selectivity
selectivity_difference = S[:, 1] - S[:, 0]

# 2. Apply the nonlinear function f (sigmoid) to the selectivity difference
f_values = selectivity_difference #sigmoid(selectivity_difference, k=2)  # 'k' can be adjusted based on desired nonlinearity

#noise from -0.1 to 0.1
noise = np.random.uniform(-0.1, 0.1, N)


###noise
# 3. Assign gain factors based on the nonlinear mapping
#g_vector = 1.0 +  0.10* f_values + np.random.uniform(-0.008, 0.008, N)
g_vector = 1.0 +  0.15* f_values
G = np.diag(g_vector)  # Diagonal matrix of gains

# Precompute (I - G W_R) and its inverse
I = np.eye(N)
I_minus_GWR = I - G @ W_R

# Check if (I - GWR) is invertible
if np.linalg.cond(I_minus_GWR) > 1 / np.finfo(I_minus_GWR.dtype).eps:
    raise ValueError("Matrix (I - G W_R) is poorly conditioned and may not be invertible.")

# Compute the inverse once since G and W_R are constant
inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)

# Compute G W_F once
G_WF = G @ W_F

# Generate modified responses
responses_modified = np.zeros((len(stimuli_grid), N))

for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])  # External stimulus for shape and color
    # Compute adjusted external input with gains
    adjusted_F = G_WF @ F
    # Compute steady-state response
    responses_modified[idx] = inv_I_minus_GWR @ adjusted_F

# Compute modulation gain ratio for each neuron
modulation_gain_ratios = np.mean(responses_modified[1:] / responses[1:], axis=0)

# Plot scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(S[:, 1]-S[:, 0], modulation_gain_ratios, alpha=0.7)
plt.xlabel('Feature Selectivity(Color-Shape)')
plt.ylabel('Modulation Gain Ratio')
plt.title('Neuron Color Selectivity vs. Modulation Gain Ratio')
plt.grid(alpha=0.3)
plt.show()


# Initialize variables to track the maximum gain change
max_gain_change = -np.inf
max_neuron_idx = -1
max_stimulus_idx = -1

# Calculate and find the greatest gain change
for neuron_idx in range(N):
    for stimulus_idx, (original, modified) in enumerate(zip(responses[:, neuron_idx], responses_modified[:, neuron_idx])):
        if original != 0:
            ratio = modified / original
        else:
            ratio = np.nan  # Avoid division by zero
        
        # Update the maximum gain change if the current ratio is greater
        if not np.isnan(ratio) and ratio > max_gain_change:
            max_gain_change = ratio
            max_neuron_idx = neuron_idx
            max_stimulus_idx = stimulus_idx




# PCA Visualization
def create_response_visualization(responses_list, titles, shape_data, color_data):
    fig = plt.figure(figsize=(15, 4 * len(responses_list)))
    pca = PCA(n_components=2)
    pca.fit(responses_list[0])

    all_pca_data = np.vstack([pca.transform(resp) for resp in responses_list])
    x_min, x_max = all_pca_data[:, 0].min(), all_pca_data[:, 0].max()
    y_min, y_max = all_pca_data[:, 1].min(), all_pca_data[:, 1].max()

    for row, (response_data, title) in enumerate(zip(responses_list, titles)):
        responses_pca = pca.transform(response_data)
        ax1 = fig.add_subplot(len(responses_list), 1, row + 1)
        scatter = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=shape_data, cmap='autumn', label='Shape')
        scatter = ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=color_data, cmap='winter', label='Color')
        ax1.set_title(title)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()

# Visualizations
def visualize_neuron_selectivity(shape_selectivity, color_selectivity, neurons_enhance, neurons_suppress):
    plt.figure(figsize=(10, 6))
    plt.scatter(shape_selectivity, color_selectivity, c='gray', alpha=0.6, label='Other Neurons')
    plt.scatter(shape_selectivity[neurons_enhance], color_selectivity[neurons_enhance], 
                c='lime', edgecolors='black', s=100, label='Enhanced Neurons')
    plt.scatter(shape_selectivity[neurons_suppress], color_selectivity[neurons_suppress], 
                c='red', edgecolors='black', s=100, label='Suppressed Neurons')
    plt.axvline(shape_threshold_low, color='r', linestyle='--', label='Shape Threshold (Low)')
    plt.axhline(color_threshold_high, color='b', linestyle='--', label='Color Threshold (High)')
    plt.axvline(shape_threshold_high, color='r', linestyle='--', label='Shape Threshold (High)')
    plt.axhline(color_threshold_low, color='b', linestyle='--', label='Color Threshold (Low)')
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
    x_max_range = (x_max - x_min)*1.2
    y_max_range = (y_max - y_min)*2
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_lim = [x_center - x_max_range/2, x_center + x_max_range/2]
    y_lim = [y_center - y_max_range/2, y_center + y_max_range/2]

    for row, (response_data, title) in enumerate(zip(responses_list, titles)):
        responses_pca = pca.transform(response_data)

        ax1 = fig.add_subplot(gs[row, 0])
        ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=shape_data, cmap='autumn', s=30)
        ax1.set_title(f'{title}\nColored by Shape')
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        #same scale for x and y
        ax1.set_aspect('equal', adjustable='box')

        ax2 = fig.add_subplot(gs[row, 1])
        ax2.scatter(responses_pca[:, 0], responses_pca[:, 1], c=color_data, cmap='winter', s=30)
        ax2.set_title(f'{title}\nColored by Color')
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)
        ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

create_response_visualization(
    responses_list=[responses, responses_modified],
    titles=["Original Responses", "Modified Responses"],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)




# Statistics
print(f"Total neurons: {N}")


def plot_color_selectivity_distribution(
    S,
    bins=10,
    title='Distribution of Color Selectivity (S[:, 1])',
    xlabel='Color Selectivity (S[:, 1])',
    ylabel='Number of Neurons',
    show_plot=True,
    save_path=None
):
    """
    Plots the distribution of color selectivity (S[:, 1]) among neurons.

    Parameters:
    - S (np.ndarray): Selectivity matrix of shape (N, K), where K >= 2.
    - bins (int, optional): Number of bins for the histogram. Default is 10.
    - title (str, optional): Title of the plot. Default is 'Distribution of Color Selectivity (S[:, 1])'.
    - xlabel (str, optional): Label for the x-axis. Default is 'Color Selectivity (S[:, 1])'.
    - ylabel (str, optional): Label for the y-axis. Default is 'Number of Neurons'.
    - show_plot (bool, optional): Whether to display the plot. Default is True.
    - save_path (str, optional): File path to save the plot image. If None, the plot is not saved. Default is None.
    """

    # Extract color selectivity
    color_selectivity = S
    
    # Create the histogram
    plt.figure(figsize=(8, 6))
    counts, bins_edges, patches = plt.hist(color_selectivity, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add titles and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.75)
    
    # Optional: Annotate bar counts
    for count, edge in zip(counts, bins_edges):
        if count > 0:
            plt.text(edge + (bins_edges[1] - bins_edges[0])/2, count, int(count), ha='center', va='bottom', fontsize=10)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    # Show the plot if required
    if show_plot:
        plt.show()
    else:
        plt.close()
plot_color_selectivity_distribution(
    S[:, 1] - S[:, 0],
    bins=30,
    title='Neuron Color Selectivity Distribution',
    xlabel='Color Selectivity (S[:, 1]-S[:, 0])',
    ylabel='Number of Neurons',
    show_plot=True,
    save_path='color_selectivity_distribution.png'  # Optional: Set to None if you don't want to save
)














num_noise_trials = 50  # Number of noisy trials per stimulus
noise_level = 0.1      # Adjust this value to control the amount of noise

# We'll add noise only to the color dimension of the stimulus.
# stimuli_grid is of shape (num_stimuli^2, 2): [shape, color]
num_stimuli_total = len(stimuli_grid)
N = responses.shape[1]  # number of neurons
original_pca = PCA(n_components=2)
original_pca.fit(responses)  # Fit PCA on the original responses (no noise)

# Precompute inverse for modified condition if needed
# If you only want to show noise effects on baseline responses, you can skip or adapt accordingly.

# Let's do noise trials on the "original" scenario first:
# We have (I - W_R) and W_F from the original code
I = np.eye(N)
inv_I_minus_WR = np.linalg.inv(I - W_R)  # from original scenario (no gain changes)
W_F_eff = W_F  # If you want to analyze original scenario, no gain G involved.

# Prepare a 3D array for noisy responses
# Dimensions: (num_stimuli_total, num_noise_trials, N)
noisy_responses = np.zeros((num_stimuli_total, num_noise_trials, N))

for i, (shape_val, color_val) in enumerate(stimuli_grid):
    # mean stimulus input
    F_mean = np.array([shape_val, color_val])
    for t in range(num_noise_trials):
        # Add Gaussian or uniform noise only to the color dimension
        # This noise simulates trial-to-trial variability of the input
        noisy_F = np.array([shape_val+ np.random.randn() * noise_level, color_val + np.random.randn() * noise_level])
        # Compute response
        adjusted_input = W_F_eff @ noisy_F
        # Solve steady-state
        noisy_responses[i, t, :] = inv_I_minus_WR @ adjusted_input

# Compute the mean response across trials to isolate noise variability
mean_response_per_stim = np.mean(noisy_responses, axis=1, keepdims=True) # (num_stimuli_total, 1, N)
noise_only = noisy_responses - mean_response_per_stim  # deviations from mean

# Reshape noise_only to (num_stimuli_total * num_noise_trials, N)
noise_data = noise_only.reshape(num_stimuli_total * num_noise_trials, N)

# Compute noise covariance
noise_cov = np.cov(noise_data, rowvar=False)  # shape (N, N)

# Find eigenvectors of the noise covariance
eigvals, eigvecs = np.linalg.eig(noise_cov)
idx_max = np.argmax(eigvals)
noise_max_axis = eigvecs[:, idx_max].real  # the axis of maximum noise variance

# Project this noise-dominant axis onto the original PCA space
# original_pca.components_ shape: (2, N) -> each row is a PC, each column a neuron weight
# Project the noise axis onto PC space: projection = U * noise_max_axis, where U = original_pca.components_
# Actually, let's just compute how it aligns with PC1 and PC2.
pc_projection = original_pca.components_ @ noise_max_axis

# pc_projection[0] is the component along PC1
# pc_projection[1] is the component along PC2

print("Projection of noise-dominant axis onto PC1 and PC2:")
print("PC1 projection:", pc_projection[0])
print("PC2 projection:", pc_projection[1])

# If pc_projection[0] is large relative to pc_projection[1], the noise axis aligns more with PC1.
# Otherwise, it may align more with PC2 or an intermediate direction.

# For visualization, you can plot the noise axis in the space of PC1-PC2:
fig, ax = plt.subplots()
ax.plot([0, pc_projection[0]], [0, pc_projection[1]], 'k-', linewidth=3, label='Noise Max Axis')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Noise-Dominant Axis in Original PC Space')
ax.legend()
plt.grid(True)
plt.show()