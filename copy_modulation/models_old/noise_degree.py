import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# Parameters
# -------------------------
np.random.seed(15)
N = 300           
K = 2             # shape=0, color=1
num_stimuli = 10  
num_noise_trials = 50  
noise_level = 0.05
desired_radius = 0.9    
p_high = 0.25
p_low = 0.05

# -------------------------------------------------
# 1) Initialization
# -------------------------------------------------
def initialize_selectivity_matrix(N, K):
    S = np.zeros((N, K))
    S[:N//2, 0] = np.random.rand(N//2)
    S[:N//2, 1] = 0.5 - S[:N//2, 0] / 2
    negative_indices = (S[:N//2, 0] - S[:N//2, 1]) < 0
    S[:N//2, 0][negative_indices] = np.random.uniform(0, 0.5, size=np.sum(negative_indices))
    S[N//2:, 1] = S[:N//2, 0]
    S[N//2:, 0] = S[:N//2, 1]
    return S

def initialize_W_F(S):
    W_F = np.zeros_like(S)
    for i in range(S.shape[0]):
        row_sum = np.sum(S[i])
        if row_sum > 0:
            W_F[i] = S[i] / row_sum
        else:
            W_F[i] = S[i]
    return W_F

def initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=0.9):
    W_R = np.zeros((N, N))
    half_N = N // 2
    # shape-shape
    shape_shape_mask = np.random.rand(half_N, half_N) < p_high
    W_R[:half_N, :half_N][shape_shape_mask] = np.random.rand(np.sum(shape_shape_mask)) * 0.1
    # shape-color
    shape_color_mask = np.random.rand(half_N, N - half_N) < p_low
    W_R[:half_N, half_N:][shape_color_mask] = np.random.rand(np.sum(shape_color_mask)) * 0.1
    # color-shape
    color_shape_mask = np.random.rand(N - half_N, half_N) < p_low
    W_R[half_N:, :half_N][color_shape_mask] = np.random.rand(np.sum(color_shape_mask)) * 0.1
    # color-color
    color_color_mask = np.random.rand(N - half_N, N - half_N) < p_high
    W_R[half_N:, half_N:][color_color_mask] = np.random.rand(np.sum(color_color_mask)) * 0.1
    # No self-connections
    np.fill_diagonal(W_R, 0)
    # Optional tuning
    if WR_tuned:
        threshold = 0.2
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(S[i] - S[j])
                    if dist < threshold:
                        W_R[i, j] *= (2 - dist/threshold)
    # Scale
    eigvals = np.linalg.eigvals(W_R)
    max_eval = np.max(np.abs(eigvals))
    if max_eval > 0:
        W_R *= (desired_radius / max_eval)
    return W_R

# -------------------------------------------------
# 2) Basic Response Computations
# -------------------------------------------------
def compute_responses(W_F, W_R, shape_stimuli, color_stimuli):
    stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)
    inv_mat = np.linalg.inv(np.eye(N) - W_R)
    responses = []
    for (sh, co) in stimuli_grid:
        F = np.array([sh, co])
        responses.append(inv_mat @ (W_F @ F))
    return np.array(responses), stimuli_grid

def generate_noisy_responses(W_R, noise_level, stimuli_grid, num_noise_trials):
    inv_mat = np.linalg.inv(np.eye(N) - W_R)
    noisy_list = []
    for (sh, co) in stimuli_grid:
        for _ in range(num_noise_trials):
            inp = np.random.randn(N) * noise_level
            out = inv_mat @ inp
            noisy_list.append(out)
    return np.array(noisy_list)

# -------------------------------------------------
# 3) Original Modulated (for comparison)
# -------------------------------------------------
def compute_modulated_responses(W_R, W_F, S, stimuli_grid):
    selectivity_diff = S[:,1] - S[:,0]
    g_vector = 1.0 + 0.2*selectivity_diff
    G = np.diag(g_vector)
    inv_mat = np.linalg.inv(np.eye(N) - G@W_R)
    G_WF = G@W_F
    out_list = []
    for (sh,co) in stimuli_grid:
        F = np.array([sh,co])
        out_list.append(inv_mat @ (G_WF @ F))
    return np.array(out_list)

# -------------------------------------------------
# 4) Revised Modulation for PC1
# -------------------------------------------------
def compute_modulated_responses_pc1_alignment(W_R, W_F, S, stimuli_grid, pc1_vector, alpha=0.5):
    """
    Gains: g[i] = 1 + alpha*(S[i,1] - S[i,0]) * pc1_vector[i].
    Then clipped to [0.1, 5.0].
    Explanation:
      - (S[i,1] - S[i,0]) means "color minus shape" preference.
      - pc1_vector[i] is how neuron i loads on PC1 (positive or negative).
      - alpha is strength.
    """
    selectivity_diff = S[:,1] - S[:,0]
    # direct push-pull
    g_vector = 1.0 + alpha * (selectivity_diff) * pc1_vector
    g_vector = np.clip(g_vector, 0.1, 5.0)

    G = np.diag(g_vector)
    inv_mat = np.linalg.inv(np.eye(N) - G@W_R)
    G_WF = G@W_F
    out_list = []
    for (sh,co) in stimuli_grid:
        F = np.array([sh,co])
        out_list.append(inv_mat @ (G_WF @ F))
    return np.array(out_list), g_vector

# -------------------------------------------------
# 5) "Color Axis" as shape=0, color=0..1
# -------------------------------------------------
def compute_color_line(W_R, W_F, g_vector, color_vals):
    """
    For shape=0, color in color_vals, returns the
    line of responses in raw (N-dim).
    if g_vector is None -> unmod
    else -> modded
    """
    I = np.eye(N)
    if g_vector is None:
        inv_mat = np.linalg.inv(I - W_R)
        W_F_eff = W_F
    else:
        G = np.diag(g_vector)
        inv_mat = np.linalg.inv(I - G@W_R)
        W_F_eff = G@W_F

    out_list = []
    for c in color_vals:
        F = np.array([0.0, c])
        out_list.append(inv_mat @ (W_F_eff @ F))
    return np.array(out_list)

def measure_axis_in_pca(pca_model, line_data):
    """
    line_data shape (num_points, N).
    project to 2D, define axis as [last - first].
    measure angle to x-axis (PC1).
    returns angle_deg, axis_2d, line_2d
    """
    line_2d = pca_model.transform(line_data)
    axis_vec = line_2d[-1] - line_2d[0]
    angle_rad = np.arctan2(np.abs(axis_vec[1]), np.abs(axis_vec[0]))
    angle_deg = np.degrees(angle_rad)
    return angle_deg, axis_vec, line_2d

# -------------------------------------------------
# 6) Subplot Visualization
# -------------------------------------------------
def visualize_four_subplots(
    responses_grid_unmod, 
    responses_noise_unmod,
    responses_grid_mod,
    responses_noise_mod,
    stimuli_grid,
    color_line_unmod,
    color_line_mod,
    title_main
):
    """
    4 subplots as before, using PCA(2).
    We'll overlay the color line in top-left (unmod) and bottom-left (mod).
    We'll measure & print angles in top-left & bottom-left.
    """
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    fig.suptitle(title_main, fontsize=16)

    # PCA from unmod GRID
    pca_grid = PCA(n_components=2)
    pca_grid.fit(responses_grid_unmod)

    grid_unmod_2d = pca_grid.transform(responses_grid_unmod)
    noise_unmod_2d = pca_grid.transform(responses_noise_unmod)
    grid_mod_2d = pca_grid.transform(responses_grid_mod)
    noise_mod_2d = pca_grid.transform(responses_noise_mod)

    # 1) top-left: unmod -> PCA from grid
    ax1 = axes[0,0]
    ax1.scatter(grid_unmod_2d[:,0], grid_unmod_2d[:,1],
                c=stimuli_grid[:,1], cmap='winter',
                alpha=0.8, s=30, label='Unmod Grid')
    ax1.scatter(noise_unmod_2d[:,0], noise_unmod_2d[:,1],
                c='gray', alpha=0.3, s=10, label='Unmod Noise')
    ax1.set_title("Unmod – PCA from Grid")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()

    # Overlay unmod color line
    angle_unmod, axis_unmod_2d, line_unmod_2d = measure_axis_in_pca(pca_grid, color_line_unmod)
    # plot line
    ax1.scatter(line_unmod_2d[:,0], line_unmod_2d[:,1],
                c=np.linspace(0,1,len(line_unmod_2d)), cmap='cool',
                s=40, alpha=0.8, label='Unmod ColorLine')
    ax1.arrow(line_unmod_2d[0,0], line_unmod_2d[0,1],
              axis_unmod_2d[0], axis_unmod_2d[1],
              head_width=0.05, color='blue')
    # We'll store the angle to print after the figure
    txt_unmod = f"Unmod color axis angle: {angle_unmod:.2f}°"

    # 2) top-right: unmod – PCA from noise
    pca_noise = PCA(n_components=2)
    pca_noise.fit(responses_noise_unmod)
    grid_unmod_2d_innoise = pca_noise.transform(responses_grid_unmod)
    noise_unmod_2d_innoise = pca_noise.transform(responses_noise_unmod)
    ax2 = axes[0,1]
    ax2.scatter(noise_unmod_2d_innoise[:,0], noise_unmod_2d_innoise[:,1],
                c='gray', alpha=0.3, s=10, label='Unmod Noise')
    ax2.scatter(grid_unmod_2d_innoise[:,0], grid_unmod_2d_innoise[:,1],
                c=stimuli_grid[:,1], cmap='winter',
                alpha=0.8, s=30, label='Unmod Grid')
    ax2.set_title("Unmod – PCA from Noise")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.grid(True)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()

    # 3) bottom-left: mod – PCA from grid
    ax3 = axes[1,0]
    mod_grid_2d = pca_grid.transform(responses_grid_mod)
    mod_noise_2d = pca_grid.transform(responses_noise_mod)
    ax3.scatter(mod_grid_2d[:,0], mod_grid_2d[:,1],
                c=stimuli_grid[:,1], cmap='spring',
                alpha=0.8, s=30, label='Mod Grid')
    ax3.scatter(mod_noise_2d[:,0], mod_noise_2d[:,1],
                c='gray', alpha=0.3, s=10, label='Mod Noise')
    ax3.set_title("Mod – PCA from Grid")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.grid(True)
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend()

    # Overlay mod color line
    angle_mod, axis_mod_2d, line_mod_2d = measure_axis_in_pca(pca_grid, color_line_mod)
    ax3.scatter(line_mod_2d[:,0], line_mod_2d[:,1],
                c=np.linspace(0,1,len(line_mod_2d)), cmap='cool',
                s=40, alpha=0.8, label='Mod ColorLine')
    ax3.arrow(line_mod_2d[0,0], line_mod_2d[0,1],
              axis_mod_2d[0], axis_mod_2d[1],
              head_width=0.05, color='red')
    txt_mod = f"Mod color axis angle: {angle_mod:.2f}°"

    # 4) bottom-right: mod – PCA from noise
    mod_grid_2d_innoise = pca_noise.transform(responses_grid_mod)
    mod_noise_2d_innoise = pca_noise.transform(responses_noise_mod)
    ax4 = axes[1,1]
    ax4.scatter(mod_noise_2d_innoise[:,0], mod_noise_2d_innoise[:,1],
                c='gray', alpha=0.3, s=10, label='Mod Noise')
    ax4.scatter(mod_grid_2d_innoise[:,0], mod_grid_2d_innoise[:,1],
                c=stimuli_grid[:,1], cmap='spring',
                alpha=0.8, s=30, label='Mod Grid')
    ax4.set_title("Mod – PCA from Noise")
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.grid(True)
    ax4.set_aspect('equal', adjustable='box')
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # Print angles and difference
    angle_diff = angle_unmod - angle_mod
    print(txt_unmod)
    print(txt_mod)
    print(f"Delta angle = {angle_diff:.2f}° (positive => mod axis is more aligned with PC1)")

# -------------------------------------------------
# 7) MAIN
# -------------------------------------------------
if __name__ == "__main__":
    # 1) Build
    S = initialize_selectivity_matrix(N, K)
    W_F = initialize_W_F(S)
    W_R_untuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)

    # 2) Stimuli & unmod
    shape_stimuli = np.linspace(0,1,num_stimuli)
    color_stimuli = np.linspace(0,1,num_stimuli)
    responses_grid_unmod, stimuli_grid = compute_responses(W_F, W_R_untuned, shape_stimuli, color_stimuli)
    responses_noise_unmod = generate_noisy_responses(W_R_untuned, noise_level, stimuli_grid, num_noise_trials)

    # 3) A basic "original" mod for reference
    responses_grid_mod_original = compute_modulated_responses(W_R_untuned, W_F, S, stimuli_grid)
    # We'll just re-use unmod noise for it, or do a quick replicate:
    responses_noise_mod_original = responses_noise_unmod.copy()

    # 4) Our new PC1-based mod
    # Fit 2D PCA on unmod grid, extract pc1_vector
    pca_2 = PCA(n_components=2)
    pca_2.fit(responses_grid_unmod)
    pc1_vector = pca_2.components_[0]  # shape (N,)

    # Build the "pc1-alignment" mod
    responses_grid_mod_pc1, g_vector_pc1 = compute_modulated_responses_pc1_alignment(
        W_R_untuned, W_F, S, stimuli_grid, pc1_vector, alpha=0.7
    )
    # For noise mod, we can skip or replicate
    responses_noise_mod_pc1 = responses_noise_unmod.copy()

    # 5) Build color lines (shape=0, color=0..1)
    color_vals = np.linspace(0,1,10)
    color_line_unmod = compute_color_line(W_R_untuned, W_F, g_vector=None, color_vals=color_vals)
    color_line_mod   = compute_color_line(W_R_untuned, W_F, g_vector_pc1, color_vals=color_vals)

    # 6) Visualize subplots, overlay color axis (unmod vs mod)
    visualize_four_subplots(
        responses_grid_unmod,
        responses_noise_unmod,
        responses_grid_mod_pc1,
        responses_noise_mod_pc1,
        stimuli_grid,
        color_line_unmod,
        color_line_mod,
        title_main="2D PCA with PC1-Aligned Modulation"
    )

    print("All done!")
