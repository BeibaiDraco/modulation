import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# Parameters
# -------------------------
np.random.seed(15)
N = 400
K = 2  # shape=0, color=1
num_stimuli = 20
num_noise_trials = 50
noise_level = 0.05
desired_radius = 0.9
p_high = 0.2
p_low = 0.03

# -------------------------------------------------
# 1) Initialization
# -------------------------------------------------
def initialize_selectivity_matrix(N, K):
    """
    Half are shape-based, half are color-based, with a random distribution.
    """
    S = np.zeros((N, K))
    S[:N//2, 0] = np.random.rand(N//2)
    S[:N//2, 1] = 0.5 - S[:N//2, 0] / 2
    neg_idx = (S[:N//2, 0] - S[:N//2, 1]) < 0
    S[:N//2, 0][neg_idx] = np.random.uniform(0, 0.5, size=np.sum(neg_idx))
    S[N//2:, 1] = S[:N//2, 0]
    S[N//2:, 0] = S[:N//2, 1]
    return S

def initialize_W_F(S):
    """
    W_F divides each neuron's (shape,color) by its sum.
    """
    W_F = np.zeros_like(S)
    for i in range(S.shape[0]):
        r = np.sum(S[i])
        if r > 0:
            W_F[i] = S[i] / r
        else:
            W_F[i] = S[i]
    return W_F

def initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=0.9):
    """
    Build a recurrent matrix with four blocks and scale it to have max eigenvalue near desired_radius.
    """
    W_R = np.zeros((N, N))
    halfN = N // 2
    
    ss_mask = np.random.rand(halfN, halfN) < p_high
    W_R[:halfN, :halfN][ss_mask] = np.random.rand(np.sum(ss_mask)) * 0.1
    
    sc_mask = np.random.rand(halfN, N - halfN) < p_low
    W_R[:halfN, halfN:][sc_mask] = np.random.rand(np.sum(sc_mask)) * 0.1
    
    cs_mask = np.random.rand(N - halfN, halfN) < p_low
    W_R[halfN:, :halfN][cs_mask] = np.random.rand(np.sum(cs_mask)) * 0.1
    
    cc_mask = np.random.rand(N - halfN, N - halfN) < p_high
    W_R[halfN:, halfN:][cc_mask] = np.random.rand(np.sum(cc_mask)) * 0.1
    
    np.fill_diagonal(W_R, 0)
    
    if WR_tuned:
        thresh = 0.2
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(S[i] - S[j])
                    if d < thresh:
                        W_R[i, j] *= (2 - d / thresh)
                        
    eivals = np.linalg.eigvals(W_R)
    max_ev = np.max(np.abs(eivals))
    if max_ev > 0:
        W_R *= (desired_radius / max_ev)
    return W_R

# -------------------------------------------------
# 2) Basic Response Computation
# -------------------------------------------------
def compute_responses(W_F, W_R, shape_stimuli, color_stimuli):
    shape_vals = np.array(shape_stimuli)
    color_vals = np.array(color_stimuli)
    stimuli_grid = np.array(np.meshgrid(shape_vals, color_vals)).T.reshape(-1, 2)
    inv_mat = np.linalg.inv(np.eye(N) - W_R)
    out = []
    for (sh, co) in stimuli_grid:
        F = np.array([sh, co])
        out.append(inv_mat @ (W_F @ F))
    return np.array(out), stimuli_grid

def generate_noisy_responses(W_R, noise_level, stimuli_grid, num_noise_trials):
    inv_mat = np.linalg.inv(np.eye(N) - W_R)
    noise = []
    for (sh, co) in stimuli_grid:
        for _ in range(num_noise_trials):
            inp = np.random.randn(N) * noise_level
            noise.append(inv_mat @ inp)
    return np.array(noise)

# -------------------------------------------------
# 3) Helper: Build Color/Shape Lines
# -------------------------------------------------
def compute_color_line(W_R, W_F, shape_val, color_vals, g_vector=None):
    N = W_R.shape[0]
    I = np.eye(N)
    if g_vector is None:
        inv_mat = np.linalg.inv(I - W_R)
        WF_eff = W_F
    else:
        G = np.diag(g_vector)
        inv_mat = np.linalg.inv(I - G @ W_R)
        WF_eff = G @ W_F
    out = []
    for c in color_vals:
        F = np.array([shape_val, c])
        out.append(inv_mat @ (WF_eff @ F))
    return np.array(out)

def compute_shape_line(W_R, W_F, shape_vals, color_val, g_vector=None):
    N = W_R.shape[0]
    I = np.eye(N)
    if g_vector is None:
        inv_mat = np.linalg.inv(I - W_R)
        WF_eff = W_F
    else:
        G = np.diag(g_vector)
        inv_mat = np.linalg.inv(I - G @ W_R)
        WF_eff = G @ W_F
    out = []
    for s in shape_vals:
        F = np.array([s, color_val])
        out.append(inv_mat @ (WF_eff @ F))
    return np.array(out)

def measure_axis_in_pca(pca_model, line_data):
    """
    Return the angle of a line in PCA(PC1,PC2) space,
    plus the axis vector and the PCA-projected line.
    """
    line_2d = pca_model.transform(line_data)
    axis_vec = line_2d[-1] - line_2d[0]
    angle = np.arctan2(np.abs(axis_vec[1]), np.abs(axis_vec[0]))
    return np.degrees(angle), axis_vec, line_2d

# -------------------------------------------------
# 4) Visualization with Axis Overlays
# -------------------------------------------------
def visualize_four_subplots_with_axis(
    responses_grid_unmod,
    responses_noise_unmod,
    responses_grid_mod,
    responses_noise_mod,
    stimuli_grid,
    color_line_unmod,   # (#pts, N)
    color_line_mod,
    shape_line_unmod,
    shape_line_mod,
    title_main
):
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    fig.suptitle(title_main, fontsize=16)

    # PCA on unmodulated grid
    pca_grid = PCA(n_components=2)
    pca_grid.fit(responses_grid_unmod)
    grid_unmod_2d = pca_grid.transform(responses_grid_unmod)
    noise_unmod_2d = pca_grid.transform(responses_noise_unmod)
    grid_mod_2d = pca_grid.transform(responses_grid_mod)
    noise_mod_2d = pca_grid.transform(responses_noise_mod)

    # Subplot 1: Unmod Grid
    ax1 = axes[0,0]
    ax1.scatter(grid_unmod_2d[:,0], grid_unmod_2d[:,1],
                c=stimuli_grid[:,1], cmap='winter', s=30, alpha=0.8, label='Unmod Grid')
    ax1.scatter(noise_unmod_2d[:,0], noise_unmod_2d[:,1],
                c='gray', alpha=0.3, s=10, label='Unmod Noise')
    ax1.set_title("Unmod – PCA from Grid")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True)
    ax1.legend()

    angle_col_un, axis_col_un, line_col_un = measure_axis_in_pca(pca_grid, color_line_unmod)
    ax1.scatter(line_col_un[:,0], line_col_un[:,1],
                c=np.linspace(0,1,len(line_col_un)), cmap='cool', s=40, alpha=0.8, label='Unmod ColorLine')
    ax1.arrow(line_col_un[0,0], line_col_un[0,1],
              axis_col_un[0], axis_col_un[1], head_width=0.05, color='blue')
    txt_col_un = f"Unmod color axis angle: {angle_col_un:.2f}°"

    angle_shp_un, axis_shp_un, line_shp_un = measure_axis_in_pca(pca_grid, shape_line_unmod)
    ax1.scatter(line_shp_un[:,0], line_shp_un[:,1],
                c=np.linspace(0,1,len(line_shp_un)), cmap='autumn', s=40, alpha=0.8, label='Unmod ShapeLine')
    ax1.arrow(line_shp_un[0,0], line_shp_un[0,1],
              axis_shp_un[0], axis_shp_un[1], head_width=0.05, color='red')
    ax1.set_aspect('equal', adjustable='box')
    txt_shp_un = f"Unmod shape axis angle: {angle_shp_un:.2f}°"

    # Subplot 2: Unmod Noise PCA
    pca_noise = PCA(n_components=2)
    pca_noise.fit(responses_noise_unmod)
    grid_unmod_2d_innoise = pca_noise.transform(responses_grid_unmod)
    noise_unmod_2d_innoise = pca_noise.transform(responses_noise_unmod)
    ax2 = axes[0,1]
    ax2.scatter(noise_unmod_2d_innoise[:,0], noise_unmod_2d_innoise[:,1],
                c='gray', alpha=0.3, s=10, label='Unmod Noise')
    ax2.scatter(grid_unmod_2d_innoise[:,0], grid_unmod_2d_innoise[:,1],
                c=stimuli_grid[:,1], cmap='winter', s=30, alpha=0.8, label='Unmod Grid')
    ax2.set_title("Unmod – PCA from Noise")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.grid(True)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()

    # Subplot 3: Modulated Grid PCA
    ax3 = axes[1,0]
    ax3.scatter(grid_mod_2d[:,0], grid_mod_2d[:,1],
                c=stimuli_grid[:,1], cmap='spring', s=30, alpha=0.8, label='Mod Grid')
    ax3.scatter(noise_mod_2d[:,0], noise_mod_2d[:,1],
                c='gray', alpha=0.3, s=10, label='Mod Noise')
    ax3.set_title("Mod – PCA from Grid")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.grid(True)
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend()

    angle_col_mod, axis_col_mod, line_col_mod = measure_axis_in_pca(pca_grid, color_line_mod)
    ax3.scatter(line_col_mod[:,0], line_col_mod[:,1],
                c=np.linspace(0,1,len(line_col_mod)), cmap='cool', s=40, alpha=0.8, label='Mod ColorLine')
    ax3.arrow(line_col_mod[0,0], line_col_mod[0,1],
              axis_col_mod[0], axis_col_mod[1], head_width=0.05, color='blue')
    txt_col_mod = f"Mod color axis angle: {angle_col_mod:.2f}°"

    angle_shp_mod, axis_shp_mod, line_shp_mod = measure_axis_in_pca(pca_grid, shape_line_mod)
    ax3.scatter(line_shp_mod[:,0], line_shp_mod[:,1],
                c=np.linspace(0,1,len(line_shp_mod)), cmap='autumn', s=40, alpha=0.8, label='Mod ShapeLine')
    ax3.arrow(line_shp_mod[0,0], line_shp_mod[0,1],
              axis_shp_mod[0], axis_shp_mod[1], head_width=0.05, color='red')
    txt_shp_mod = f"Mod shape axis angle: {angle_shp_mod:.2f}°"
    ax3.legend()

    # Subplot 4: Mod Noise PCA
    grid_mod_2d_innoise  = pca_noise.transform(responses_grid_mod)
    noise_mod_2d_innoise = pca_noise.transform(responses_noise_mod)
    ax4 = axes[1,1]
    ax4.scatter(noise_mod_2d_innoise[:,0], noise_mod_2d_innoise[:,1],
                c='gray', alpha=0.3, s=10, label='Mod Noise')
    ax4.scatter(grid_mod_2d_innoise[:,0], grid_mod_2d_innoise[:,1],
                c=stimuli_grid[:,1], cmap='spring', s=30, alpha=0.8, label='Mod Grid')
    ax4.set_title("Mod – PCA from Noise")
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.grid(True)
    ax4.set_aspect('equal', adjustable='box')
    ax4.legend()

    plt.tight_layout()
    plt.show()

    print(txt_col_un)
    print(txt_shp_un)
    print(txt_col_mod)
    print(txt_shp_mod)
    dcol = angle_col_un - angle_col_mod
    dshp = angle_shp_un - angle_shp_mod
    print(f"Color Axis Angle Diff: {dcol:.2f}° (positive => mod color axis more aligned w/ PC1)")
    print(f"Shape Axis Angle Diff: {dshp:.2f}° (positive => mod shape axis more aligned w/ PC1)")


# -------------------------------------------------
# 5) The Strict PC1-Alignment Strategy
# -------------------------------------------------
def solve_delta_g_for_color_axis(dcolor, eigvecs):
    """
    Solve for delta_g such that (I + diag(delta_g)) dcolor
    is parallel to eigvecs[:, 0] (which is v1).
    Equivalently, for j>=1, v_j^T((I + diag(delta_g)) dcolor) = 0.
    We skip j=0 (the top PC1) because we *allow* overlap with v1
    but remove overlap from v2..vN.

    The matrix 'M' arises from the condition:
       for j>=1:   v_j^T dcolor + v_j^T diag(delta_g) dcolor = 0
                   a_j + sum_i delta_g[i] * v_j[i] * dcolor[i] = 0
    => M[j-1, i] = v_j[i] * dcolor[i],  and  RHS[j-1] = -a_j

    Return the minimal-norm delta_g via least-squares solution.
    """
    N = len(dcolor)
    # Coeffs a_j = v_j^T dcolor
    a = eigvecs.T @ dcolor  # shape (N,)
    # Build M from j=1..(N-1)
    rows = []
    rhs = []
    for j in range(1, N):  # j=1..N-1
        row_j = eigvecs[:, j] * dcolor  # elementwise product => length N
        rows.append(row_j)
        rhs.append(-a[j])
    M = np.vstack(rows)        # shape ((N-1), N)
    rhs = np.array(rhs)        # shape (N-1,)

    # solve minimal ||delta_g||^2 s.t. M delta_g = rhs
    delta_g, residuals, rank, svals = np.linalg.lstsq(M, rhs, rcond=None)
    return delta_g

def compute_modulated_responses_strict_pc1(
    W_R, W_F, S, stimuli_grid,
    responses_grid_unmod,  # to define color axis
    pca_full,              # full PCA object with all components
    alpha=1.0, clamp_range=(0.8,1.2)
):
    """
    1) Define color axis dcolor as difference in average response 
       between color=1.0 and color=0.0 (averaged over shape).
    2) Solve for delta_g that kills components in v2..vN.
    3) Scale by alpha, clamp, and build G. 
    4) Compute modded responses.
    """
    # 1) Build a "color axis" from data. E.g. difference of average responses: color=1 minus color=0
    #    We'll average over shape in [0..1].
    shape_vals = np.linspace(0,1, num_stimuli)
    # responses_grid_unmod has length num_stimuli^2
    # we want to find those stimuli with color ~0 and color ~1.
    # We'll pick color=0 index and color=1 index exactly:
    stimuli_grid_reshaped = stimuli_grid.reshape(num_stimuli, num_stimuli, 2)
    # gather all responses where color=0 => col idx=0
    color0_indices = np.where(np.isclose(stimuli_grid[:,1], 0.0))[0]
    color1_indices = np.where(np.isclose(stimuli_grid[:,1], 1.0))[0]
    # average
    resp_color0 = np.mean(responses_grid_unmod[color0_indices], axis=0)  # shape (N,)
    resp_color1 = np.mean(responses_grid_unmod[color1_indices], axis=0)  # shape (N,)
    dcolor = resp_color1 - resp_color0   # shape (N,)
    # optional normalization
    dcolor /= (np.linalg.norm(dcolor) + 1e-9)

    # 2) Extract all PCA eigenvectors from pca_full
    #    pca_full.components_ shape => (N, N) if we fit n_components=N
    #    each row is an eigenvector in sklearn => we want columns => we'll transpose
    eigvecs = pca_full.components_.T  # shape (N,N), columns = v_j
    # We'll solve for delta_g s.t. G*dcolor is parallel to v1 => v2..vN comp=0
    delta_g_raw = solve_delta_g_for_color_axis(dcolor, eigvecs)

    # 3) Apply a scale alpha, then clamp in [0.8,1.2] around 1.0
    #    Gains = 1.0 + alpha * delta_g_raw
    delta_scaled = alpha * delta_g_raw
    gains = 1.0 + delta_scaled
    gains_clamped = np.clip(gains, clamp_range[0], clamp_range[1])

    # 4) Build G, compute final responses
    G = np.diag(gains_clamped)
    I_minus = np.eye(N) - G @ W_R
    condVal = np.linalg.cond(I_minus)
    if condVal > 1/np.finfo(I_minus.dtype).eps:
        raise ValueError("(I - G*W_R) nearly singular. Reduce alpha or check network.")
    inv_mat = np.linalg.inv(I_minus)
    GW_F = G @ W_F

    out = []
    for (sh, co) in stimuli_grid:
        F = np.array([sh, co])
        out.append(inv_mat @ (GW_F @ F))
    out = np.array(out)

    return out, gains_clamped, dcolor, delta_g_raw

# -------------------------------------------------
# 6) Main
# -------------------------------------------------
if __name__=="__main__":
    # 1) Setup
    S = initialize_selectivity_matrix(N, K)
    W_F = initialize_W_F(S)
    W_R_untuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)

    # 2) Full grid responses and noise (unmodulated)
    shape_vals = np.linspace(0, 1, num_stimuli)
    color_vals = np.linspace(0, 1, num_stimuli)
    responses_grid_unmod, stimuli_grid = compute_responses(W_F, W_R_untuned, shape_vals, color_vals)
    responses_noise_unmod = generate_noisy_responses(W_R_untuned, noise_level, stimuli_grid, num_noise_trials)

    # 3) PCA for visualization is only 2D, but we need full PCA for all eigenvectors.
    #    We'll do a separate fit with n_components = N (or as large as feasible).
    #    This can be big for N=1000, but is doable offline.
    pca_full = PCA(n_components=min(N, responses_grid_unmod.shape[0]))  # can be up to N
    pca_full.fit(responses_grid_unmod)

    # 4) Strict PC1-based modulation with small alpha
    responses_grid_mod_strict, g_vector_strict, dcolor_strict, delta_g_raw = compute_modulated_responses_strict_pc1(
        W_R_untuned, W_F, S, stimuli_grid,
        responses_grid_unmod, pca_full,
        alpha=1.0,      # you can try smaller alpha if clamping is triggered
        clamp_range=(0.8, 1.2)
    )
    # we won't modify the noise: we'll assume the same input noise
    responses_noise_mod_strict = responses_noise_unmod.copy()

    print("[Main] Gains stats:")
    print(f"  min(g)={g_vector_strict.min():.3f}, max(g)={g_vector_strict.max():.3f}, mean(g)={g_vector_strict.mean():.3f}")

    # 5) Axis lines for unmod vs mod
    color_line_vals = np.linspace(0, 1, 10)
    color_line_unmod = compute_color_line(W_R_untuned, W_F, 0.0, color_line_vals, g_vector=None)
    color_line_mod = compute_color_line(W_R_untuned, W_F, 0.0, color_line_vals, g_vector_strict)
    shape_line_vals = np.linspace(0, 1, 10)
    shape_line_unmod = compute_shape_line(W_R_untuned, W_F, shape_line_vals, 0.0, g_vector=None)
    shape_line_mod = compute_shape_line(W_R_untuned, W_F, shape_line_vals, 0.0, g_vector_strict)

    # 6) Visualize
    visualize_four_subplots_with_axis(
        responses_grid_unmod,
        responses_noise_unmod,
        responses_grid_mod_strict,
        responses_noise_mod_strict,
        stimuli_grid,
        color_line_unmod,
        color_line_mod,
        shape_line_unmod,
        shape_line_mod,
        title_main="Strict PC1 Alignment – Small Diagonal Gain"
    )

    print("Done!")
