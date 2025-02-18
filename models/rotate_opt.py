import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize

np.random.seed(15)
N = 300
K = 2  # shape=0, color=1
desired_radius = 0.9
p_high = 0.2
p_low = 0.2

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

    # Rescale W_R so spectral radius = desired_radius
    eivals = np.linalg.eigvals(W_R)
    max_ev = np.max(np.abs(eivals))
    if max_ev > 0:
        W_R *= (desired_radius / max_ev)
    return W_R

# -------------------------------------------------
# 2) Response Computations
# -------------------------------------------------
def compute_response(W_R, W_F, shape_val, color_val, g_vector=None):
    """
    Returns steady-state response for a single (shape_val, color_val).
    """
    I = np.eye(W_R.shape[0])
    if g_vector is None:
        inv_mat = np.linalg.inv(I - W_R)
        WF_eff = W_F
    else:
        G = np.diag(g_vector)
        inv_mat = np.linalg.inv(I - G @ W_R)
        WF_eff = G @ W_F

    F = np.array([shape_val, color_val])
    return inv_mat @ (WF_eff @ F)

def compute_grid_responses(W_R, W_F, shape_vals, color_vals, g_vector=None):
    """
    Returns a [num_shape * num_color, N] array of responses
    by scanning over shape_vals x color_vals.
    """
    responses = []
    for s in shape_vals:
        for c in color_vals:
            resp = compute_response(W_R, W_F, s, c, g_vector)
            responses.append(resp)
    return np.array(responses)

# define a "color line" function if needed
def compute_color_line(W_R, W_F, shape_val, color_vals, g_vector=None):
    return np.array([
        compute_response(W_R, W_F, shape_val, c, g_vector=g_vector)
        for c in color_vals
    ])

# ================================================================
# MAIN SCRIPT / EXAMPLE
# ================================================================

if __name__ == "__main__":
    # 1) Create the network
    S = initialize_selectivity_matrix(N, K)
    W_F = initialize_W_F(S)
    W_R = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)

    # 2) Build a 2D grid of shape & color, e.g., 11 x 11
    grid_points = 11
    shape_vals = np.linspace(0, 1, grid_points)
    color_vals = np.linspace(0, 1, grid_points)

    # 2.1) Compute unmodulated responses for the entire grid
    responses_grid_unmod = compute_grid_responses(W_R, W_F, shape_vals, color_vals, g_vector=None)
    # shape => [11*11, N]

    # 2.2) Do PCA on these full-grid responses
    # We can keep 3 PCs, for instance
    pca_grid = PCA(n_components=3)
    pca_grid.fit(responses_grid_unmod)   # shape = [121, N]
    v1 = pca_grid.components_[0]        # first principal component, shape (N,)
    v1 /= np.linalg.norm(v1)            # normalize

    # 3) Define how we measure "color-axis" (for a specific shape slice)
    #    We'll pick shape_val=0.3 for demonstration. Then vary color from 0 to 1.
    shape_for_color_line = 0.3
    color_line_vals = np.linspace(0, 1, 10)

    def color_axis_direction(g):
        """
        For a given g-vector, define color-axis as the difference between
        response at color=1.0 and color=0.0 for shape=0.3,
        or you could sum more points if you wish.
        """
        resp_c0 = compute_response(W_R, W_F, shape_for_color_line, 0.0, g)
        resp_c1 = compute_response(W_R, W_F, shape_for_color_line, 1.0, g)
        return resp_c1 - resp_c0  # shape (N,)

    # 4) Define the objective to maximize alignment with v1
    #    => We'll minimize negative cos^2
    def alignment_objective(g):
        d_col = color_axis_direction(g)
        dot_val = np.dot(v1, d_col)
        denom = (np.linalg.norm(d_col) * np.linalg.norm(v1))
        if denom < 1e-15:
            return 0.0  # or some fallback if color-axis is degenerate
        cos_val = dot_val / denom
        # clamp to [-1, 1] to avoid numeric overshoot
        cos_val = max(min(cos_val, 1.0), -1.0)
        cos_sq = cos_val**2
        return -cos_sq  # negative => maximizing cos^2

    # 5) Box-constrained optimization in [L, U]^N
    L, U = 0.9, 1.1
    init_g = np.ones(N)  # start from unmodulated
    bounds = [(L, U)] * N

    print("RUNNING L-BFGS-B...")
    res = minimize(fun=alignment_objective,
                   x0=init_g,
                   method='L-BFGS-B',
                   bounds=bounds,
                   options={'maxiter': 200, 'disp': True})

    g_opt = res.x
    final_loss = res.fun
    print(f"Optimization complete. Final loss = {final_loss:.6f}")

    # 5.1) Compare angles pre vs. post
    def angle_with_v1(d_vec):
        # angle in degrees between v1 and d_vec
        dot_v = np.dot(v1, d_vec)
        denom = np.linalg.norm(v1)*np.linalg.norm(d_vec)
        if denom < 1e-15:
            return np.nan
        val = dot_v/denom
        #val = np.clip(val, -1, 1)  # ensure in [-1, 1]
        return np.degrees(np.arccos(val))

    d_unmod = color_axis_direction(init_g)  # g=1 => unmodulated
    d_mod = color_axis_direction(g_opt)

    angle_pre = angle_with_v1(d_unmod)
    angle_post = angle_with_v1(d_mod)
    print(f"Unmodulated angle = {angle_pre:.3f} degrees")
    print(f"Post-optimization angle = {angle_post:.3f} degrees")

    # 6) Visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # (A) Plot angle pre vs. post
    axes[0].bar([0,1], [angle_pre, angle_post], width=0.4, color=['gray','orange'])
    axes[0].set_xticks([0,1])
    axes[0].set_xticklabels(['Pre (g=1)', 'Post (g_opt)'])
    axes[0].set_ylabel("Angle with PC1 (degrees)")
    axes[0].set_title("Pre vs. Post Modulation Angle")

    # (B) Distribution of learned g
    axes[1].hist(g_opt, bins=30, color='green', alpha=0.7)
    axes[1].axvline(L, color='r', linestyle='--', label='Lower bound')
    axes[1].axvline(U, color='r', linestyle='--', label='Upper bound')
    axes[1].set_title("Distribution of g_opt")
    axes[1].set_xlabel("g value")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
