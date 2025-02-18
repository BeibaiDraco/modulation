import numpy as np

def initialize_selectivity_matrix(N, K):
    """
    Create an N x K selectivity matrix S, where each neuron i has selectivity 
    values for K features. For simplicity, assign random selectivity preferences.
    For example, each neuron can have a preferred shape and color value in [0,1].
    """
    # Random preferred values for each of the K features (uniform [0,1])
    S = np.random.rand(N, K)
    return S

def initialize_W_F(S):
    """
    Initialize feedforward weight matrix W_F (dimensions N x K) using the selectivity matrix S.
    Each neuron's feedforward weights to each feature can be proportional to its selectivity.
    """
    # We can simply use S as the weight matrix (assuming input features normalized [0,1]).
    W_F = S.copy()
    # Optionally, scale or normalize W_F rows if needed (e.g., to equalize input drive).
    # Here we ensure each neuron has unit total feedforward weight for stability.
    row_norms = np.linalg.norm(W_F, axis=1, keepdims=True)
    W_F = W_F / row_norms
    return W_F

def initialize_W_R(N, p_high, p_low, S, WR_tuned=True, desired_radius=0.9):
    """
    Initialize recurrent weight matrix W_R of size N x N.
    If WR_tuned is True, use feature selectivity S to assign higher weights between neurons 
    with similar selectivity (probability p_high) and lower weights (p_low) for dissimilar neurons.
    Otherwise, initialize randomly with a fixed connection probability.
    Then scale W_R to have spectral radius equal to desired_radius.
    """
    # Initialize an empty weight matrix
    W_R = np.zeros((N, N))
    # Determine pairwise similarity between neurons' selectivities (e.g., by correlation or distance in S)
    # Here, we'll use a simple criterion: if neurons have closest feature index or high correlation in S.
    # For simplicity, assume K=2 (shape, color) and classify neurons by dominant selectivity:
    if S.shape[1] >= 2:
        # Classify neurons as "shape-tuned" or "color-tuned" based on which feature has higher selectivity value
        feature_pref = np.argmax(S, axis=1)  # index of max selectivity for each neuron
    else:
        feature_pref = np.zeros(N)  # if only one feature, treat all same
    
    # Fill W_R with random weights based on probabilities
    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # no self-connection for simplicity
            if WR_tuned:
                # If neuron i and j prefer the same feature, use p_high; else p_low
                if feature_pref[i] == feature_pref[j]:
                    if np.random.rand() < p_high:
                        # Random weight (can be positive or negative small value)
                        W_R[i, j] = np.random.normal(0, 0.1)
                else:
                    if np.random.rand() < p_low:
                        W_R[i, j] = np.random.normal(0, 0.1)
            else:
                # Untuned: all connections with equal probability (e.g., average of p_high and p_low)
                if np.random.rand() < (0.5 * (p_high + p_low)):
                    W_R[i, j] = np.random.normal(0, 0.1)
    # Scale W_R to set spectral radius (max eigenvalue magnitude) to desired_radius
    eigvals = np.linalg.eigvals(W_R)
    max_eig = np.max(np.abs(eigvals))
    if max_eig > 0:
        W_R *= (desired_radius / max_eig)
    return W_R

# Example usage:
N = 500   # number of neurons
K = 2     # number of stimulus features (shape, color)
S = initialize_selectivity_matrix(N, K)
W_F = initialize_W_F(S)
W_R = initialize_W_R(N, p_high=0.2, p_low=0.2, S=S, WR_tuned=False, desired_radius=0.9)
print("Selectivity matrix S shape:", S.shape)
print("Feedforward W_F shape:", W_F.shape, "; Recurrent W_R shape:", W_R.shape)


def compute_responses(W_F, W_R, shape_stimuli, color_stimuli):
    """
    Compute neural responses for each combination of shape and color stimulus.
    shape_stimuli and color_stimuli should be lists or 1D arrays of equal length, 
    representing a set of stimuli (each stimulus has a shape value and a color value).
    Returns an array of shape (num_stimuli, N) with neuron responses.
    """
    num_stimuli = len(shape_stimuli)
    responses = np.zeros((num_stimuli, W_F.shape[0]))  # W_F.shape[0] is N (neurons)
    for idx, (s_val, c_val) in enumerate(zip(shape_stimuli, color_stimuli)):
        # Construct input vector u for this stimulus. 
        # Assuming input has two features: [shape, color].
        u = np.array([s_val, c_val])
        # Iteratively find RNN response (fixed point) using tanh nonlinearity
        x = np.tanh(W_F.dot(u))  # start with feedforward response
        for t in range(100):  # iterate a few steps to include recurrence
            x = np.tanh(W_F.dot(u) + W_R.dot(x))
        responses[idx] = x
    return responses

def generate_noisy_responses(W_R, noise_level, stimuli_grid, num_noise_trials=10):
    """
    For each stimulus in stimuli_grid (a list of (shape, color) pairs), generate multiple 
    noisy response trials. We add Gaussian noise with given noise_level to the recurrent 
    activity on each iteration.
    Returns an array of shape (num_noise_trials, num_stimuli, N).
    """
    num_stimuli = len(stimuli_grid)
    noisy_trials = np.zeros((num_noise_trials, num_stimuli, W_R.shape[0]))
    for trial in range(num_noise_trials):
        trial_responses = []
        for i, (s_val, c_val) in enumerate(stimuli_grid):
            u = np.array([s_val, c_val])
            # Simulate one step with noise in recurrent activity
            x = np.tanh(W_F.dot(u))
            # Add noise to initial response
            x += np.random.normal(0, noise_level, size=x.shape)
            # Iterate recurrently with noise added at each step
            for t in range(100):
                x = np.tanh(W_F.dot(u) + W_R.dot(x) + np.random.normal(0, noise_level, size=x.shape))
            trial_responses.append(x)
        noisy_trials[trial] = np.array(trial_responses)
    return noisy_trials

# Example usage:
# Define a small set of stimuli for demonstration
shape_vals = np.linspace(0, 1, 5)   # e.g., 5 distinct shapes (could be shape identity or a continuous feature)
color_vals = np.linspace(0, 1, 5)   # 5 distinct colors
stimuli_grid = [(s, c) for s in shape_vals for c in color_vals]  # all combinations
# Separate lists for compute_responses
shape_stimuli_list = [s for s, c in stimuli_grid]
color_stimuli_list = [c for s, c in stimuli_grid]

responses = compute_responses(W_F, W_R, shape_stimuli_list, color_stimuli_list)
print("Responses shape (num_stimuli x N):", responses.shape)
# Generate noisy responses (e.g., 5 trials)
noisy_responses = generate_noisy_responses(W_R, noise_level=0.05, stimuli_grid=stimuli_grid, num_noise_trials=5)
print("Noisy responses shape (trials x stimuli x N):", noisy_responses.shape)

import numpy.linalg as LA

def measure_color_axis_angle(responses, shape_vals, color_vals):
    """
    Compute the color axis (difference between mean responses to max vs min color, averaging over shapes),
    perform PCA on responses, and return the angle (in degrees) between color axis and first PC.
    """
    # Determine which stimuli are min color and max color
    min_color = np.min(color_vals); max_color = np.max(color_vals)
    # Average responses across all stimuli with min color and max color
    # (Here, stimuli are aligned in order with shape_vals and color_vals lists)
    stimuli_arr = np.column_stack((shape_vals, color_vals))
    min_color_mask = (stimuli_arr[:,1] == min_color)
    max_color_mask = (stimuli_arr[:,1] == max_color)
    mean_resp_minC = responses[min_color_mask].mean(axis=0)
    mean_resp_maxC = responses[max_color_mask].mean(axis=0)
    color_axis = mean_resp_maxC - mean_resp_minC  # vector in neural space
    
    # PCA on responses (subtract mean for PCA)
    X = responses - responses.mean(axis=0)
    U, s, Vt = LA.svd(X, full_matrices=False)  # SVD for PCA
    PC1 = Vt[0]  # first principal component (as a vector of length N)
    # Angle between color_axis and PC1
    cos_angle = np.dot(color_axis, PC1) / (LA.norm(color_axis) * LA.norm(PC1) + 1e-9)
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Measure initial angle without gain modulation
initial_angle = measure_color_axis_angle(responses, np.array(shape_stimuli_list), np.array(color_stimuli_list))
print(f"Initial color axis vs PC1 angle: {initial_angle:.2f} degrees")

# Gain modulation ranges to test
gain_ranges = [
    (0.5, 1.5),
    (0.6, 1.4),
    (0.7, 1.3),
    (0.8, 1.2),
    (0.9, 1.1),
]
# Identify neuron groups by selectivity (using feature_pref from earlier classification)
feature_pref = np.argmax(S, axis=1)

# Store results
best_angles = []
best_gain_pairs = []  # store (gain_shape, gain_color) that achieved best alignment

for (g_min, g_max) in gain_ranges:
    best_angle = 180.0  # start with max (worst) angle
    best_pair = (1.0, 1.0)
    # Search gains for shape-preferring and color-preferring groups
    scan_values = np.linspace(g_min, g_max, 11)  # sample 11 points within range
    print(scan_values)
    for g_shape in scan_values:
        for g_color in scan_values:
            # Apply gains: create a diagonal gain matrix for neurons
            gains = np.ones(N)
            # shape-pref neurons (feature_pref==0) get g_shape, color-pref (==1) get g_color
            gains[feature_pref == 0] = g_shape
            gains[feature_pref == 1] = g_color
            # Modulate responses by these gains
            mod_responses = responses * gains  # broadcasts along neurons dimension
            # Measure new angle
            angle = measure_color_axis_angle(mod_responses, np.array(shape_stimuli_list), np.array(color_stimuli_list))
            if angle < best_angle:
                best_angle = angle
                best_pair = (g_shape, g_color)
    best_angles.append(best_angle)
    best_gain_pairs.append(best_pair)
    print(f"Range [{g_min},{g_max}]: best angle = {best_angle:.2f} deg at gains (shape={best_pair[0]:.2f}, color={best_pair[1]:.2f})")

# Calculate improvement in alignment (angle reduction) relative to initial
angle_improvements = [initial_angle - a for a in best_angles]


import matplotlib.pyplot as plt

# Plot improvement vs max gain of range
max_gains = [rg[1] for rg in gain_ranges]
plt.figure(figsize=(5,4))
plt.plot(max_gains, angle_improvements, marker='o')
plt.xlabel('Max gain in range')
plt.ylabel('Alignment Improvement (deg reduction in color-PC1 angle)')
plt.title('Gain Range vs Alignment Improvement')
plt.grid(True)
plt.xticks(max_gains)
for x, y in zip(max_gains, angle_improvements):
    plt.text(x, y+0.5, f"{y:.1f}°", ha='center')  # annotate improvement
plt.show()

# Scatter plots for before and after modulation
# Perform PCA on original and modulated responses for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# Original responses PCA
X = responses - responses.mean(axis=0)
pcs_orig = pca.fit_transform(X)  # shape (num_stimuli, 2)
# Modulated responses (for [0.5,1.5] optimal gains)
opt_g_shape, opt_g_color = best_gain_pairs[0]  # assuming [0.5,1.5] was first in list
gains = np.ones(N)
gains[feature_pref == 0] = opt_g_shape
gains[feature_pref == 1] = opt_g_color
mod_responses = responses * gains
pcs_mod = pca.fit_transform(mod_responses - mod_responses.mean(axis=0))

# Prepare color mapping for points (based on color stimulus value)
colors = np.array(color_stimuli_list)
shapes = np.array(shape_stimuli_list)
# Normalize color values for a colormap
cmap = plt.cm.get_cmap('coolwarm')
norm_colors = (colors - colors.min()) / (colors.max() - colors.min())

plt.figure(figsize=(10,4))
for i, (pcs, title) in enumerate([(pcs_orig, "Baseline"), (pcs_mod, "After Gain Modulation")]):
    plt.subplot(1,2,i+1)
    scatter = plt.scatter(pcs[:,0], pcs[:,1], c=cmap(norm_colors), edgecolors='k')
    plt.title(title + " PC1-PC2 Projection")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # Plot arrows for color axis
    # Compute mean PC coordinates for min and max color stimuli
    minC = colors.min(); maxC = colors.max()
    mean_minC = pcs[colors == minC].mean(axis=0)
    mean_maxC = pcs[colors == maxC].mean(axis=0)
    plt.arrow(mean_minC[0], mean_minC[1],
              mean_maxC[0]-mean_minC[0], mean_maxC[1]-mean_minC[1],
              color='black', width=0.02, head_width=0.1, length_includes_head=True)
    plt.text(mean_maxC[0], mean_maxC[1], 'Color axis', color='k', fontweight='bold')
    # Optionally, distinguish different shapes by marker (not explicitly requested, but could)
    # Here, just annotate one shape vs another if needed:
    # for shape_val in np.unique(shapes):
    #     plt.annotate(f"Shape{shape_val:.1f}", 
    #                  (pcs[shapes==shape_val,0].mean(), pcs[shapes==shape_val,1].mean()))
    plt.grid(True)
plt.tight_layout()
plt.show()

