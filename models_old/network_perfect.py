import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize

np.random.seed(15)

# Parameters
N = 100
K = 2
num_stimuli = 10

# ====== Same setup as before ======
S = np.zeros((N, K))
S[:N//2, 0] = np.random.rand(N//2)
S[:N//2, 1] = (1/2 - S[:N//2, 0]/2)
negative_indices = S[:N//2, 0] - S[:N//2, 1] < 0
S[:N//2, 0][negative_indices] = np.random.uniform(0, 0.5, size=np.sum(negative_indices))
S[N//2:, 1] = S[:N//2, 0]
S[N//2:, 0] = S[:N//2, 1]

W_F = np.zeros((N, K))
for i in range(N):
    if S[i, 0] > 0.5:
        W_F[i, 0] = S[i, 0]
    elif S[i, 1] > 0.5:
        W_F[i, 1] = S[i, 1]
    else:
        W_F[i, 0] = S[i, 0]
        W_F[i, 1] = S[i, 1]

row_sums = W_F.sum(axis=1, keepdims=True)
nonzero_mask = (row_sums != 0)
W_F_normalized = np.zeros_like(W_F)
W_F_normalized[nonzero_mask[:, 0], :] = W_F[nonzero_mask[:, 0], :] / row_sums[nonzero_mask].reshape(-1, 1)
W_F = W_F_normalized

W_R = np.zeros((N, N))
half_N = N // 2
p_high = 0.25
p_low = 0.25

shape_shape_mask = np.random.rand(half_N, half_N) < p_high
W_R[:half_N, :half_N][shape_shape_mask] = np.random.rand(np.sum(shape_shape_mask))
shape_color_mask = np.random.rand(half_N, N - half_N) < p_low
W_R[:half_N, half_N:][shape_color_mask] = np.random.rand(np.sum(shape_color_mask))
color_shape_mask = np.random.rand(N - half_N, half_N) < p_low
W_R[half_N:, :half_N][color_shape_mask] = np.random.rand(np.sum(color_shape_mask))
color_color_mask = np.random.rand(N - half_N, N - half_N) < p_high
W_R[half_N:, half_N:][color_color_mask] = np.random.rand(np.sum(color_color_mask))

np.fill_diagonal(W_R, 0)

eigenvalues = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues))
desired_radius = 0.9
W_R = W_R * (desired_radius / scaling_factor)

shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)

I = np.eye(N)
inv_I_minus_WR = np.linalg.inv(I - W_R)

responses = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = W_F @ F
    responses[idx] = inv_I_minus_WR @ adjusted_F

responses_mean = responses.mean(axis=0)
responses_centered = responses - responses_mean

pca = PCA(n_components=N)
pca.fit(responses_centered)
U = pca.components_.T  # N x N

alpha_deg_30 = 30
alpha_30 = np.deg2rad(alpha_deg_30)
R_alpha_30 = np.array([[np.cos(alpha_30), -np.sin(alpha_30)],
                       [np.sin(alpha_30),  np.cos(alpha_30)]])

R_full_30 = np.eye(N)
R_full_30[:2, :2] = R_alpha_30
T_30 = U @ R_full_30 @ U.T

# Compute target rotated responses
responses_rotated = (responses_centered @ T_30) + responses_mean

# We'll optimize G so that responses_G(F) ~ responses_rotated(F)
# Let's vectorize: we have a set of stimuli and their responses.
# For a given G, responses_G = inv(I - G W_R) G W_F [F_1, F_2, ...] computed together.

F_matrix = stimuli_grid  # shape: (num_stimuli^2, 2)
# We'll handle (I - G W_R) inverse inside the objective.

def compute_responses(g):
    G = np.diag(g)
    I_minus_GWR = I - G @ W_R
    if np.linalg.cond(I_minus_GWR) > 1e8:
        return None
    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)
    # Compute all responses at once
    # adjusted_input = G W_F F for all stimuli
    adjusted_input = (G @ W_F) @ F_matrix.T  # shape (N, num_stimuli^2)
    # responses_g = inv(I - G W_R) adjusted_input
    responses_g = inv_I_minus_GWR @ adjusted_input  # shape (N, num_stimuli^2)
    return responses_g.T  # shape (num_stimuli^2, N)

def objective(g):
    resp_g = compute_responses(g)
    if resp_g is None:
        return 1e10
    diff = resp_g - responses_rotated
    return np.sum(diff**2)

g_initial = np.ones(N)
res = minimize(objective, g_initial, method='BFGS', options={'maxiter':1000, 'disp':True})
g_optimized = res.x
print("Optimization done. Final loss:", res.fun)

G_opt = np.diag(g_optimized)
I_minus_GWR_opt = I - G_opt @ W_R
inv_I_minus_GWR_opt = np.linalg.inv(I_minus_GWR_opt)
G_opt_WF = G_opt @ W_F

responses_optimized = np.zeros((len(stimuli_grid), N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = G_opt_WF @ F
    responses_optimized[idx] = inv_I_minus_GWR_opt @ adjusted_F

def create_response_visualization(responses_list, titles, shape_data, color_data):
    fig = plt.figure(figsize=(15, 4 * len(responses_list)))
    gs = fig.add_gridspec(len(responses_list), 3, width_ratios=[1, 1, 0.1])

    pca_vis = PCA(n_components=2)
    pca_vis.fit(responses_list[0])

    all_pca_data = np.vstack([pca_vis.transform(resp) for resp in responses_list])
    x_min, x_max = all_pca_data[:, 0].min(), all_pca_data[:, 0].max()
    y_min, y_max = all_pca_data[:, 1].min(), all_pca_data[:, 1].max()
    x_max_range = (x_max - x_min)*1.2
    y_max_range = (y_max - y_min)*2
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_lim = [x_center - x_max_range/2, x_center + x_max_range/2]
    y_lim = [y_center - y_max_range/2, y_center + y_max_range/2]

    for row, (response_data, title) in enumerate(zip(responses_list, titles)):
        responses_pca = pca_vis.transform(response_data)

        ax1 = fig.add_subplot(gs[row, 0])
        ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=shape_data, cmap='autumn', s=30)
        ax1.set_title(f'{title}\nColored by Shape')
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
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
    responses_list=[responses, responses_rotated, responses_optimized],
    titles=["Original Responses", "Rotated (Target) Responses", "Responses with Optimized G"],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize

np.random.seed(42)

# Parameters
N = 20  # Reduced from 100 for demonstration
K = 2
num_stimuli = 10

# Generate feature selectivities and W_F similar to your approach
S = np.zeros((N, K))
S[:N//2, 0] = np.random.rand(N//2)
S[:N//2, 1] = (1/2 - S[:N//2, 0]/2)
negative_indices = S[:N//2, 0] - S[:N//2, 1] < 0
S[:N//2, 0][negative_indices] = np.random.uniform(0, 0.5, size=np.sum(negative_indices))
S[N//2:, 1] = S[:N//2, 0]
S[N//2:, 0] = S[:N//2, 1]

W_F = np.zeros((N, K))
for i in range(N):
    if S[i, 0] > 0.5:
        W_F[i, 0] = S[i, 0]
    elif S[i, 1] > 0.5:
        W_F[i, 1] = S[i, 1]
    else:
        W_F[i, 0] = S[i, 0]
        W_F[i, 1] = S[i, 1]

row_sums = W_F.sum(axis=1, keepdims=True)
nonzero_mask = (row_sums != 0)
W_F_normalized = np.zeros_like(W_F)
W_F_normalized[nonzero_mask[:, 0], :] = W_F[nonzero_mask[:, 0], :] / row_sums[nonzero_mask].reshape(-1, 1)
W_F = W_F_normalized

# Initialize a random W_R
W_R_init = np.random.randn(N, N)*0.01
np.fill_diagonal(W_R_init, 0)

# Stimuli
shape_stimuli = np.linspace(0, 1, num_stimuli)
color_stimuli = np.linspace(0, 1, num_stimuli)
stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)
num_stimuli_total = len(stimuli_grid)

# Compute original responses with initial W_R (just to have a baseline)
I = np.eye(N)
def stable_inv(I_minus_W):  # safe inverse
    if np.linalg.cond(I_minus_W) > 1e8:
        return None
    return np.linalg.inv(I_minus_W)

inv_I_minus_WR_init = stable_inv(I - W_R_init)
if inv_I_minus_WR_init is None:
    inv_I_minus_WR_init = np.linalg.inv(I - W_R_init + 1e-3*np.eye(N))

responses_init = np.zeros((num_stimuli_total, N))
for idx, (shape, color) in enumerate(stimuli_grid):
    F = np.array([shape, color])
    adjusted_F = W_F @ F
    responses_init[idx] = inv_I_minus_WR_init @ adjusted_F

responses_mean = responses_init.mean(axis=0)
responses_centered = responses_init - responses_mean

pca = PCA(n_components=N)
pca.fit(responses_centered)
U = pca.components_.T  # N x N

alpha_deg_30 = 30
alpha_30 = np.deg2rad(alpha_deg_30)
R_alpha_30 = np.array([
    [np.cos(alpha_30), -np.sin(alpha_30)],
    [np.sin(alpha_30),  np.cos(alpha_30)]
])

R_full_30 = np.eye(N)
R_full_30[:2, :2] = R_alpha_30
T_30 = U @ R_full_30 @ U.T

responses_rotated = (responses_centered @ T_30) + responses_mean

# We'll optimize G and W_R together.
# Parameter vector x will consist of:
# first N entries = g (gains)
# next N*N entries = W_R (flattened)
x0 = np.concatenate([np.ones(N), W_R_init.flatten()])

def unpack_parameters(x):
    g = x[:N]
    W_R = x[N:].reshape(N, N)
    return g, W_R

def compute_responses(W_R, g):
    G = np.diag(g)
    I_minus_GWR = I - G @ W_R
    inv_mat = stable_inv(I_minus_GWR)
    if inv_mat is None:
        return None
    G_WF = G @ W_F
    # Compute all responses for all stimuli
    # responses = inv(I - G W_R) G W_F F
    F_matrix = stimuli_grid.T  # shape (2, num_stimuli_total)
    adjusted_input = G_WF @ F_matrix  # shape (N, num_stimuli_total)
    resp = inv_mat @ adjusted_input   # shape (N, num_stimuli_total)
    return resp.T  # shape (num_stimuli_total, N)

def spectral_radius(M):
    vals = np.linalg.eigvals(M)
    return np.max(np.abs(vals))

def objective(x):
    g, W_R = unpack_parameters(x)
    resp_g = compute_responses(W_R, g)
    if resp_g is None:
        return 1e10
    # Match rotated responses
    diff = resp_g - responses_rotated
    loss = np.sum(diff**2)

    # Regularizations:
    # 1. Keep W_R stable (penalize if spectral radius > 0.9)
    sr = spectral_radius(W_R)
    if sr > 0.9:
        loss += (sr - 0.9)**2 * 1e5

    # 2. Keep W_R close to initial (optional)
    diff_wr = W_R - W_R_init
    loss += np.sum(diff_wr**2)*1e2

    # 3. Keep G close to 1
    loss += np.sum((g - 1.0)**2)*1e1

    return loss

# Run optimization
res = minimize(objective, x0, method='BFGS', options={'maxiter':1000, 'disp':True})
opt_x = res.x
opt_g, opt_WR = unpack_parameters(opt_x)

print("Optimization finished.")
print("Final loss:", res.fun)
print("Final spectral radius:", spectral_radius(opt_WR))

# Compute optimized responses
resp_optimized = compute_responses(opt_WR, opt_g)

# Visualization
def create_response_visualization(responses_list, titles, shape_data, color_data):
    fig = plt.figure(figsize=(15, 4 * len(responses_list)))
    gs = fig.add_gridspec(len(responses_list), 3, width_ratios=[1, 1, 0.1])

    pca_vis = PCA(n_components=2)
    pca_vis.fit(responses_list[0])

    all_pca_data = np.vstack([pca_vis.transform(resp) for resp in responses_list])
    x_min, x_max = all_pca_data[:, 0].min(), all_pca_data[:, 0].max()
    y_min, y_max = all_pca_data[:, 1].min(), all_pca_data[:, 1].max()
    x_max_range = (x_max - x_min)*1.2
    y_max_range = (y_max - y_min)*2
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_lim = [x_center - x_max_range/2, x_center + x_max_range/2]
    y_lim = [y_center - y_max_range/2, y_center + y_max_range/2]

    for row, (response_data, title) in enumerate(zip(responses_list, titles)):
        responses_pca = pca_vis.transform(response_data)

        ax1 = fig.add_subplot(gs[row, 0])
        ax1.scatter(responses_pca[:, 0], responses_pca[:, 1], c=shape_data, cmap='autumn', s=30)
        ax1.set_title(f'{title}\nColored by Shape')
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
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
    responses_list=[responses_init, responses_rotated, resp_optimized],
    titles=["Initial Responses", "Rotated (Target) Responses", "Optimized (W_R & G) Responses"],
    shape_data=stimuli_grid[:, 0],
    color_data=stimuli_grid[:, 1]
)

print(f"Optimized gains (first 10): {opt_g[:10]}")
