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

def compute_color_line(W_R, W_F, shape_val, color_vals, g_vector=None):
    """
    Optional helper for a 1D sweep in color, if needed.
    """
    return np.array([
        compute_response(W_R, W_F, shape_val, c, g_vector=g_vector)
        for c in color_vals
    ])

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    # 1) Create the network
    S = initialize_selectivity_matrix(N, K)
    W_F = initialize_W_F(S)
    W_R = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)

    # 2) Build a 2D grid of shape & color, e.g., 11 x 11
    grid_points = 11
    shape_vals = np.linspace(0, 1, grid_points)
    color_vals = np.linspace(0, 1, grid_points)

    # Compute unmodulated responses over the entire grid
    responses_grid_unmod = compute_grid_responses(W_R, W_F, shape_vals, color_vals, g_vector=None)
    # shape => [11*11, N] = [121, N]

    # PCA on full grid
    pca_grid = PCA(n_components=3)
    pca_grid.fit(responses_grid_unmod)
    v1 = pca_grid.components_[0]  # shape (N,)
    v1 /= np.linalg.norm(v1)      # normalize

    # We'll define the color-axis for shape=0.3 from color=0->1
    shape_for_color_line = 0.3

    def color_axis_direction(g):
        """
        For a given g-vector, define color-axis as the difference between
        the response at color=1.0 and color=0.0 for shape=0.3.
        """
        resp_c0 = compute_response(W_R, W_F, shape_for_color_line, 0.0, g)
        resp_c1 = compute_response(W_R, W_F, shape_for_color_line, 1.0, g)
        return resp_c1 - resp_c0

    def alignment_objective(g):
        """
        We want to MAXIMIZE cos^2(angle) => MINIMIZE -cos^2(angle).
        """
        d_col = color_axis_direction(g)
        dot_val = np.dot(v1, d_col)
        denom = np.linalg.norm(d_col) * np.linalg.norm(v1)
        if denom < 1e-15:
            return 0.0
        cos_val = dot_val / denom
        # clamp to avoid numerical overshoot
        cos_val = max(min(cos_val, 1.0), -1.0)
        cos_sq = cos_val**2
        return -cos_sq

    def angle_with_v1(d_vec):
        """ Return the smaller angle [0..90] w.r.t. v1 in degrees. """
        dot_v = np.dot(v1, d_vec)
        denom = np.linalg.norm(v1)*np.linalg.norm(d_vec)
        if denom < 1e-15:
            return np.nan
        val = dot_v / denom
        val = np.clip(val, -1, 1)
        angle = np.degrees(np.arccos(val))
        # We can treat angle & 180-angle as equivalent if sign doesn't matter
        return min(angle, 180 - angle)

    # ---------------------------------------------------------
    # Sweep over different box intervals [L, U], step = 0.025
    # from [0.8, 1.2] down to [0.975, 1.025].
    # We'll do 8 steps: i=0..7
    #   L_i = 0.8 + i*0.025
    #   U_i = 1.2 - i*0.025
    # ---------------------------------------------------------
    num_steps = 20
    all_L = []
    all_U = []
    angle_improvements = []
    interval_widths = []

    # Pre-compute unmodulated angle
    d_unmod = color_axis_direction(np.ones(N))
    angle_pre = angle_with_v1(d_unmod)

    for i in range(num_steps):
        L = 0.8 + i*0.01
        U = 1.2 - i*0.01

        init_g = np.ones(N)  # always start from the unmodulated g=1
        bounds = [(L, U)] * N

        res = minimize(fun=alignment_objective,
                       x0=init_g,
                       method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 200, 'disp': False})

        g_opt = res.x
        d_mod = color_axis_direction(g_opt)
        angle_post = angle_with_v1(d_mod)

        angle_improvement = angle_pre - angle_post  # how many degrees improved
        width = U - L

        all_L.append(L)
        all_U.append(U)
        interval_widths.append(width)
        angle_improvements.append(angle_improvement)

        print(f"Bounds [{L:.3f}, {U:.3f}], angle pre={angle_pre:.2f}, post={angle_post:.2f}, "
              f"improvement={angle_improvement:.3f}")

    # ---------------------------------------------------------
    # Plot the improvement vs. L,U interval
    # ---------------------------------------------------------
    plt.figure(figsize=(8,5))

    # Option A: Plot vs. interval width (U-L)
    # Option B: Plot each bound pair on x-axis
    # Here let's do the width on x-axis:
    #for each elements in interval_widths, divide by 2
    interval_widths = np.array(interval_widths) /2
    
    plt.plot(interval_widths, angle_improvements, marker='o', color='b')
    for i, (w, imp) in enumerate(zip(interval_widths, angle_improvements)):
        plt.text(w, imp, f"[{all_L[i]:.2f}, {all_U[i]:.2f}]", fontsize=9,
                 ha='center', va='bottom')

    plt.xlabel("Interval width (U - L)/2")
    plt.ylabel("Angle improvement (deg)")
    plt.title("Gain Range vs. Improvement in Alignment")
    plt.grid(True)
    plt.show()
