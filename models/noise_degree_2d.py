import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# Parameters
# -------------------------
np.random.seed(15)
N = 1000
K = 2  # shape=0, color=1
num_stimuli = 10
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
    Build a recurrent matrix with four blocks and scale it.
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
# 2) Response Computations
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
# 3) Axis Lines
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
    line_2d = pca_model.transform(line_data)
    axis_vec = line_2d[-1] - line_2d[0]
    angle = np.arctan2(np.abs(axis_vec[1]), np.abs(axis_vec[0]))
    return np.degrees(angle), axis_vec, line_2d

# -------------------------------------------------
# 4) Visualization
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
    #ax3.set_aspect('equal', adjustable='box')
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
    #ax4.set_aspect('equal', adjustable='box')
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


# ============== New "Small-Gain" Modulation with Automatic Testing ====================
def compute_modulated_responses_pc1_small_gain_auto(
    W_R, W_F, S, stimuli_grid, pc1,
    pca_unmod_grid,   # PCA fitted on unmod grid
    color_line_unmod, # unmod color line data
    angle_tolerance=0.5,
    max_iter=15,
    alpha_init=0.05,
    alpha_step=0.05,
    gamma_init=0.1,
    gamma_step=0.05
):
    """
    We attempt a small-gain push/pull that rotates color axis to PC1.
    We'll iteratively adjust alpha,gamma in small steps until
    color axis angle is improved by at least 'angle_tolerance' degrees.
    or we do max_iter steps.

    Gains in [0.8..1.2].

    Return final (responses_mod, g_vector, angle_col_mod).
    """

    def small_gain_function(W_R, W_F, S, stimuli_grid, pc1, alpha, gamma):
        """
        Gains:
          color_ratio = S[i,1]/(S[i,0]+S[i,1]+eps)
          shape_ratio = S[i,0]/(S[i,0]+S[i,1]+eps)
          overlap_pc1 = abs(pc1_unit[i]) => [0..1]
          g[i] = 1 + alpha*(color_ratio*overlap) - gamma*(shape_ratio*overlap)
          then clip to [0.8..1.2]
        """
        eps = 1e-9
        N = W_R.shape[0]
        pc1_unit = pc1/(np.linalg.norm(pc1)+eps)

        color_part = np.maximum(S[:,1],0.0)
        shape_part = np.maximum(S[:,0],0.0)
        denom = color_part+shape_part+eps
        color_ratio = color_part/denom
        shape_ratio = shape_part/denom

        overlap = np.abs(pc1_unit)
        overlap/= (overlap.max()+eps)

        raw_g = 1.0 - alpha*(color_ratio*overlap) + gamma*(shape_ratio*overlap)
        # Suppose pc1_unit has real sign for each neuron i
        # Then:
        #raw_g = 1.0 + alpha_color * color_ratio * pc1_unit + alpha_shape * shape_ratio * pc1_unit

        g_clipped = np.clip(raw_g,0.8,1.2)

        G = np.diag(g_clipped)
        A = np.eye(N) - G@W_R
        condA = np.linalg.cond(A)
        if condA>1/np.finfo(A.dtype).eps:
            raise ValueError("Nearly singular. Lower alpha/gamma or adjust clipping.")
        invA = np.linalg.inv(A)
        G_WF = G@W_F

        mod_resp = []
        for (sh,co) in stimuli_grid:
            F = np.array([sh,co])
            mod_resp.append(invA@(G_WF@F))
        return np.array(mod_resp), g_clipped
    
    
    def small_gain_function(W_R, W_F, S, stimuli_grid, pc1, alpha_color, alpha_shape):
        """
    Gains:
      color_ratio = S[i,1] / (S[i,0] + S[i,1] + eps)
      shape_ratio = S[i,0] / (S[i,0] + S[i,1] + eps)
      For each neuron i:
        pc1_unit[i] = pc1[i] / ||pc1||
        raw_g[i] = 1.0 + alpha_color * color_ratio[i] * pc1_unit[i] 
                         + alpha_shape * shape_ratio[i] * pc1_unit[i]
      and clip to [0.8, 1.2].
    """
        eps = 1e-9
        N = W_R.shape[0]
        pc1_unit = pc1 / (np.linalg.norm(pc1) + eps)

        color_part = np.maximum(S[:, 1], 0.0)
        shape_part = np.maximum(S[:, 0], 0.0)
        denom = color_part + shape_part + eps
        color_ratio = color_part / denom
        shape_ratio = shape_part / denom

        raw_g = 1.0 + alpha_color * color_ratio * pc1_unit + alpha_shape * shape_ratio * pc1_unit
        g_clipped = np.clip(raw_g, 0.8, 1.2)

        G = np.diag(g_clipped)
        A = np.eye(N) - G @ W_R
        condA = np.linalg.cond(A)
        if condA > 1 / np.finfo(A.dtype).eps:
            raise ValueError("Nearly singular. Lower alpha_color/alpha_shape or adjust clipping.")
        invA = np.linalg.inv(A)
        G_WF = G @ W_F

        mod_resp = []
        for (sh, co) in stimuli_grid:
            F = np.array([sh, co])
            mod_resp.append(invA @ (G_WF @ F))
        mod_resp = np.array(mod_resp)
        return mod_resp, g_clipped


    # measure unmod color angle
    angle_col_un,_,_ = measure_axis_in_pca(pca_unmod_grid, color_line_unmod)

    alpha = alpha_init
    gamma = gamma_init
    best_resps = None
    best_g = None
    best_angle_col_mod = 999.0
    iteration=0

    while iteration<max_iter:
        iteration+=1
        try:
            resp_try, g_try = small_gain_function(W_R, W_F, S, stimuli_grid, pc1, alpha, gamma)
        except ValueError:
            # if singular, we skip
            print(f"Singular at alpha={alpha}, gamma={gamma}, skipping..")
            alpha+=alpha_step
            continue

        # measure color line angle
        color_line_mod_try = compute_color_line(W_R, W_F, 0.0, np.linspace(0,1,10), g_try)
        angle_col_mod_try,_,_ = measure_axis_in_pca(pca_unmod_grid, color_line_mod_try)

        improvement = angle_col_un - angle_col_mod_try
        print(f"Iter={iteration}, alpha={alpha:.3f}, gamma={gamma:.3f}, colorAngle={angle_col_mod_try:.2f}, improvement={improvement:.2f}")

        if improvement>angle_tolerance:
            # success
            best_resps=resp_try
            best_g=g_try
            best_angle_col_mod=angle_col_mod_try
            print("Requirement met! color axis improved by more than angle_tolerance.")
            break
        else:
            # increment alpha,gamma slightly
            alpha+=alpha_step
            gamma+=gamma_step

    if best_resps is None:
        # final attempt
        print("Could not find suitable small gain that improves color axis angle enough.")
        # we at least return the last attempt
        best_resps=resp_try
        best_g=g_try
        best_angle_col_mod=angle_col_mod_try

    return best_resps,best_g,best_angle_col_mod

# -------------------------------------------------
# 8) MAIN
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

    # 3) Fit 2D PCA on unmod, get pc1; flip if needed to ensure dot>0 w/ pure color
    pca_2 = PCA(n_components=2)
    pca_2.fit(responses_grid_unmod)
    pc1 = pca_2.components_[0].copy()

    x_color_pure = np.linalg.inv(np.eye(N)-W_R_untuned) @ (W_F @ np.array([0.0,1.0]))
    if np.dot(pc1, x_color_pure)<0:
        pc1=-pc1

    # 4) Build unmod color/shape lines & measure angles
    color_line_vals = np.linspace(0,1,10)
    color_line_unmod= compute_color_line(W_R_untuned, W_F, 0.0, color_line_vals, g_vector=None)
    shape_line_vals = np.linspace(0,1,10)
    shape_line_unmod= compute_shape_line(W_R_untuned, W_F, shape_line_vals, 0.0, g_vector=None)

    # 5) Attempt a small-gain push/pull that meets a certain angle improvement
    angle_tolerance=2.0  # want color axis angle improved by at least 0.5 deg
    responses_grid_mod_auto,g_vector_auto,angle_col_mod_auto = compute_modulated_responses_pc1_small_gain_auto(
        W_R_untuned, W_F, S, stimuli_grid, pc1,
        pca_unmod_grid=pca_2, 
        color_line_unmod=color_line_unmod,
        angle_tolerance=angle_tolerance,
        max_iter=1000,
        alpha_init=3.650,
        alpha_step=0.05,
        gamma_init=2.260,
        gamma_step=0.03
    )

    responses_noise_mod_auto= responses_noise_unmod.copy()

    # lines w/ final g_vector
    color_line_mod_auto= compute_color_line(W_R_untuned, W_F, 0.0, color_line_vals, g_vector_auto)
    shape_line_mod_auto= compute_shape_line(W_R_untuned, W_F, shape_line_vals, 0.0, g_vector_auto)

    # 6) visualize
    visualize_four_subplots_with_axis(
        responses_grid_unmod,
        responses_noise_unmod,
        responses_grid_mod_auto,
        responses_noise_mod_auto,
        stimuli_grid,
        color_line_unmod,
        color_line_mod_auto,
        shape_line_unmod,
        shape_line_mod_auto,
        title_main="Small-Gain PC1 Color Axis w/ Automatic Parameter Tuning"
    )

    print("All done!")
