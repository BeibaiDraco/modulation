import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# 1) Parameters
# -------------------------
np.random.seed(15)
N = 800
K = 2  # shape=0, color=1
num_stimuli = 29
noise_level = 0.05
desired_radius = 0.9
p_high = 0.2
p_low = 0.2

# We'll sweep these [L,U] pairs
GAIN_RANGES = [
    (0.95, 1.05),
    (0.90, 1.10),
    (0.80, 1.20),
    (0.70, 1.30),
    (0.50, 1.50),
]

# -------------------------------------------------
# 2) Initialization
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
    # mirror to second half
    S[N//2:, 1] = S[:N//2, 0]
    S[N//2:, 0] = S[:N//2, 1]
    return S

def initialize_W_F(S):
    """ W_F divides each neuron's (shape,color) by sum """
    W_F = np.zeros_like(S)
    for i in range(S.shape[0]):
        total = np.sum(S[i])
        if total > 0:
            W_F[i] = S[i] / total
        else:
            W_F[i] = S[i]
    return W_F

def initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=0.9):
    """ Recurrent matrix, rescale to have max eigenvalue ~ desired_radius """
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
        # optional short-range tuning
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
# 3) Response Computation
# -------------------------------------------------
def compute_responses(W_F, W_R, shape_vals, color_vals):
    """
    Return responses for all shape-color combos in [shape_vals x color_vals].
    """
    stimuli_grid = np.array(np.meshgrid(shape_vals, color_vals)).T.reshape(-1, 2)
    inv_mat = np.linalg.inv(np.eye(N) - W_R)
    out = []
    for (sh, co) in stimuli_grid:
        F = np.array([sh, co])
        out.append(inv_mat @ (W_F @ F))
    return np.array(out), stimuli_grid

def compute_average_response(W_F, W_R, shape_vals, color_val):
    """
    Average responses across all shape in shape_vals, for a fixed color_val.
    """
    all_resp = []
    inv_mat = np.linalg.inv(np.eye(N) - W_R)
    for s in shape_vals:
        F = np.array([s, color_val])
        all_resp.append(inv_mat @ (W_F @ F))
    all_resp = np.array(all_resp)
    return np.mean(all_resp, axis=0)

# -------------------------------------------------
# 4) Color Axis (Unmod + New) and Angles
# -------------------------------------------------
def define_color_axis_unmod(responses_grid_unmod, stimuli_grid, N):
    """
    dcolor = average(resp(color=1)) - average(resp(color=0)), 
             also normalized in R^N.
    """
    # Indices for color=0 and color=1
    color0_idx = np.where(np.isclose(stimuli_grid[:,1], 0.0))[0]
    color1_idx = np.where(np.isclose(stimuli_grid[:,1], 1.0))[0]
    
    mean_c0 = np.mean(responses_grid_unmod[color0_idx], axis=0)
    mean_c1 = np.mean(responses_grid_unmod[color1_idx], axis=0)
    dc = mean_c1 - mean_c0
    norm_val = np.linalg.norm(dc)
    if norm_val < 1e-9:
        return dc
    return dc / norm_val

def compute_color_axis_after_gains(W_F, W_R, shape_vals):
    """
    Return new color axis for color=1 vs color=0 after the 
    diagonal gains have been applied inside W_R, W_F.
    (We assume W_R, W_F are already the 'modified' versions.)
    """
    # average response at color=0
    resp_c0 = compute_average_response(W_F, W_R, shape_vals, color_val=0.0)
    # average response at color=1
    resp_c1 = compute_average_response(W_F, W_R, shape_vals, color_val=1.0)
    dnew = resp_c1 - resp_c0
    norm_val = np.linalg.norm(dnew)
    if norm_val < 1e-9:
        return dnew
    return dnew / norm_val

def angle_degrees(v1, x):
    """
    Angle in degrees between two vectors in R^N.
    We take absolute dot, ignoring sign.
    """
    dot_val = np.abs(np.dot(v1, x))
    cos_val = dot_val / (np.linalg.norm(v1)*np.linalg.norm(x) + 1e-12)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))

# -------------------------------------------------
# 5) Extreme Gains Approach
# -------------------------------------------------
def build_extreme_gain_vector(v1, dcolor, L, U, sign_threshold=0.0):
    """
    If (v1_i * dcolor_i) >= sign_threshold => g_i=U, else g_i=L.
    sign_threshold=0 => strictly negative => L
    """
    g = np.empty_like(v1)
    for i in range(len(v1)):
        val = v1[i]*dcolor[i]
        # set a small threshold if you want "near zero => L" or ">=0 => U"
        if val >= sign_threshold:
            g[i] = U
        else:
            g[i] = L
    return g

def apply_gains_and_get_newWF_WR(W_F_unmod, W_R_unmod, g):
    """
    Build G=diag(g). Then define:
      W_F_new = G @ W_F_unmod
      W_R_new = ??? 
    The diagonal gain can be applied either to feedforward or 
    to entire recurrent input => G*(x).
    Typically, we do: new response = (I - G*W_R)^(-1)(G*W_F * [sh,co]).
    So we keep a wrapped version of W_R => W_R_new = G*W_R for the effective system.
    We'll do that explicitly.
    """
    G = np.diag(g)
    W_F_mod = G @ W_F_unmod
    W_R_mod = G @ W_R_unmod
    return W_F_mod, W_R_mod

# -------------------------------------------------
# 6) Main
# -------------------------------------------------
def main():
    # A) Setup
    S = initialize_selectivity_matrix(N, K)
    W_F_base = initialize_W_F(S)
    W_R_base = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)
    shape_vals = np.linspace(0,1, num_stimuli)
    color_vals = np.linspace(0,1, num_stimuli)
    
    # B) Unmodulated responses
    responses_grid_unmod, stimuli_grid = compute_responses(W_F_base, W_R_base, shape_vals, color_vals)
    
    # C) PCA to get v1
    pca_full = PCA(n_components=min(N, responses_grid_unmod.shape[0]))
    pca_full.fit(responses_grid_unmod)
    v1 = pca_full.components_[0,:]  # shape (N,)
    
    # D) Original color axis
    dcolor_unmod = define_color_axis_unmod(responses_grid_unmod, stimuli_grid, N)
    angle_init = angle_degrees(v1, dcolor_unmod)
    print(f"Initial angle between color-axis and PC1: {angle_init:.2f} deg")
    
    # E) Sweep multiple [L,U] => build extreme gains => measure new angle
    results = []
    for (Lval, Uval) in GAIN_RANGES:
        # 1) Build gain vector
        g_vec = build_extreme_gain_vector(v1, dcolor_unmod, Lval, Uval, sign_threshold=0.0)
        
        # 2) Build new W_F and W_R
        W_F_mod, W_R_mod = apply_gains_and_get_newWF_WR(W_F_base, W_R_base, g_vec)
        
        # 3) Compute new color axis => angle with old v1
        dcolor_new = compute_color_axis_after_gains(W_F_mod, W_R_mod, shape_vals)
        angle_new = angle_degrees(v1, dcolor_new)
        
        # store
        results.append((Lval, Uval, angle_new, g_vec))
        print(f"[L={Lval}, U={Uval}] => Final angle: {angle_new:.2f} deg, #U={np.sum(g_vec==Uval)} #L={np.sum(g_vec==Lval)}")
    
    # F) Compare with Theoretical Bound
    #   x_new_min = arctan( (L/U)*tan(x_init) )
    x_init_rad = np.radians(angle_init)
    angles_measured = [r[2] for r in results]
    angles_theory = []
    for (Lval,Uval,_,_) in results:
        ratio = (Lval/Uval)*np.tan(x_init_rad)
        x_min_rad = np.arctan(ratio)
        angles_theory.append(np.degrees(x_min_rad))
    
    # G) Plot angle vs. gain-range index
    xvals = np.arange(len(results))
    plt.figure(figsize=(8,4))
    plt.plot(xvals, angles_measured, 'o-', label='Measured final angle')
    plt.plot(xvals, angles_theory, 's--', label='Theory bound')
    plt.axhline(angle_init, color='gray', alpha=0.5, label='Initial angle')
    plt.xticks(xvals, [f"[{r[0]:.2f},{r[1]:.2f}]" for r in results], rotation=20)
    plt.ylabel("Angle (deg)")
    plt.title("Angle vs. Gain Ranges (Extreme Gains)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # H) Example: Plot Gains for the last range
    #    We'll do a histogram or so to see distribution
    last_g = results[-1][3]
    plt.figure(figsize=(6,4))
    plt.hist(last_g, bins=20, alpha=0.7)
    plt.title(f"Gains distribution for last range {GAIN_RANGES[-1]}")
    plt.xlabel("Gain")
    plt.ylabel("Count of neurons")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # I) Optional: Visualize unmod vs. mod responses for one range
    #    e.g. pick the last one in GAIN_RANGES
    (L_val, U_val, _, g_pick) = results[-1]
    W_F_pick, W_R_pick = apply_gains_and_get_newWF_WR(W_F_base, W_R_base, g_pick)
    responses_grid_mod, _ = compute_responses(W_F_pick, W_R_pick, shape_vals, color_vals)
    
    # We'll do a 2D PCA (trained on unmod data for consistent axes)
    pca_2d = PCA(n_components=2)
    pca_2d.fit(responses_grid_unmod)  # unmod as reference
    unmod_2d = pca_2d.transform(responses_grid_unmod)
    mod_2d = pca_2d.transform(responses_grid_mod)
    
    plt.figure(figsize=(7,5))
    plt.scatter(unmod_2d[:,0], unmod_2d[:,1],
                c=stimuli_grid[:,1], cmap='winter', alpha=0.5, label='Unmod')
    plt.scatter(mod_2d[:,0], mod_2d[:,1],
                c=stimuli_grid[:,1], cmap='autumn', alpha=0.5, label='Mod')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Scatter: Range [{L_val},{U_val}], Extreme Gains")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
