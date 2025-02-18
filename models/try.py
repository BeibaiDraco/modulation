import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# 1) RNN Initialization
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

def initialize_selectivity_matrix(N, K):
    """
    EXACT, as you provided.
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
    EXACT, as you provided.
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
    EXACT, as you provided.
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

# -------------------------
# 2) Response Computation
# -------------------------
def compute_responses(W_F, W_R, shape_stimuli, color_stimuli):
    """
    EXACT, as you provided: linear system solution x = (I - W_R)^(-1) W_F * [shape, color].
    We'll do a grid of shape, color.
    """
    shape_vals = np.array(shape_stimuli)
    color_vals = np.array(color_stimuli)
    stimuli_grid = np.array(np.meshgrid(shape_vals, color_vals)).T.reshape(-1, 2)
    inv_mat = np.linalg.inv(np.eye(N) - W_R)
    out = []
    for (sh, co) in stimuli_grid:
        F = np.array([sh, co])
        out.append(inv_mat @ (W_F @ F))
    return np.array(out), stimuli_grid

# -------------------------------------------------
# Step A) Build RNN, compute unmodulated responses
# -------------------------------------------------
S = initialize_selectivity_matrix(N, K)
W_F = initialize_W_F(S)
W_R = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)

shape_vals = np.linspace(0,1,num_stimuli)
color_vals = np.linspace(0,1,num_stimuli)
unmod_responses, stimuli_grid = compute_responses(W_F, W_R, shape_vals, color_vals)

# Flatten (num_stimuli^2, N)
unmod_responses = unmod_responses.reshape(-1, N)

# Perform PCA on unmodulated responses
X_centered = unmod_responses - unmod_responses.mean(axis=0)
pca = PCA(n_components=2)
pca.fit(X_centered)
v1_unmod = pca.components_[0]  # shape (N,)

# -------------------------------------------------
# Step B) Define the color axis under modulation
# -------------------------------------------------
def color_axis_modulated(g_vector, W_F, W_R, shape_vals, color0=0.0, color1=1.0):
    """
    1) Build G=diag(g_vector).
    2) For each shape in shape_vals, compute response at color=0.0 and color=1.0:
       x_c0 = (I - G W_R)^(-1) G W_F [shape, 0]
       x_c1 = (I - G W_R)^(-1) G W_F [shape, 1]
    3) Average across shape, then subtract => color axis in modulated space.
    """
    N = len(g_vector)
    G = np.diag(g_vector)
    inv_mat = np.linalg.inv(np.eye(N) - G @ W_R)
    c0_accum = np.zeros(N)
    c1_accum = np.zeros(N)
    for sh in shape_vals:
        F0 = np.array([sh, color0])  # shape, color=0
        F1 = np.array([sh, color1])  # shape, color=1
        x_c0 = inv_mat @ (G @ (W_F @ F0))
        x_c1 = inv_mat @ (G @ (W_F @ F1))
        c0_accum += x_c0
        c1_accum += x_c1
    c0_mean = c0_accum / len(shape_vals)
    c1_mean = c1_accum / len(shape_vals)
    c_axis = c1_mean - c0_mean
    return c_axis

def angle_unmod_v1_vs_modulated_coloraxis(g_vector, W_F, W_R, shape_vals, v1_unmod):
    """
    Return the angle (in degrees) between unmod_v1 and the color axis 
    in the modulated system.
    """
    c_axis_mod = color_axis_modulated(g_vector, W_F, W_R, shape_vals)
    dot_val = np.dot(v1_unmod, c_axis_mod)
    denom = (np.linalg.norm(v1_unmod)*np.linalg.norm(c_axis_mod)+1e-12)
    cos_val = dot_val/denom
    cos_val = np.clip(cos_val, -1, 1)
    angle_rad = np.arccos(cos_val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# measure the unmodded color axis vs v1_unmod
# (this is just to see the original angle, though user wants modded color axis vs unmodded v1)
def color_axis_unmod(W_F, W_R, shape_vals):
    inv_mat = np.linalg.inv(np.eye(N)-W_R)
    c0_accum = np.zeros(N)
    c1_accum = np.zeros(N)
    for sh in shape_vals:
        x0 = inv_mat @ (W_F @ [sh, 0.0])
        x1 = inv_mat @ (W_F @ [sh, 1.0])
        c0_accum += x0
        c1_accum += x1
    c0_mean = c0_accum/len(shape_vals)
    c1_mean = c1_accum/len(shape_vals)
    return c1_mean - c0_mean

unmod_color_axis = color_axis_unmod(W_F, W_R, shape_vals)
orig_angle = angle_unmod_v1_vs_modulated_coloraxis(np.ones(N), W_F, W_R, shape_vals, v1_unmod)
print(f"Original angle between unmodded color axis and unmodded v1: {orig_angle:.3f} deg")

# -------------------------------------------------
# Step C) Projected Gradient Approach to Minimize Angle
# -------------------------------------------------
def projected_gradient_search(
    W_F, W_R, shape_vals, v1_unmod, 
    L=0.8, U=1.2, 
    max_iter=30, step_size=0.01
):
    """
    We'll do a naive gradient-based approach to 
      minimize angle( g_vector ), i.e. angle_unmod_v1_vs_modulated_coloraxis(g).
    We'll define alignment measure = cos(angle), and ascend it.
    
    Because N=1000, we can't do naive finite-difference on all neurons each iteration 
    (that would be 1000 evaluations per iteration). 
    We'll do a small random subset each time to approximate gradient.
    
    This is just a demonstration. 
    """
    g = np.ones(N)  # start from no gain change
    def objective(g_):
        # we'll maximize cos(angle) => equivalently minimize angle
        # so let cost = cos_val = dot / (norms)
        c_axis = color_axis_modulated(g_, W_F, W_R, shape_vals)
        dot_val = np.dot(v1_unmod, c_axis)
        denom = (np.linalg.norm(v1_unmod)*np.linalg.norm(c_axis)+1e-12)
        return dot_val/denom  # bigger is better => ascend
    
    # Project in [L,U]
    def project(g_):
        return np.clip(g_, L, U)
    
    current_obj = objective(g)
    for it in range(max_iter):
        # pick random subset of neurons to approximate gradient
        subset_size = 500  # e.g., pick 50 indices
        idx_subset = np.random.choice(N, size=subset_size, replace=False)
        grad_approx = np.zeros(N)
        base_obj = current_obj
        
        eps = 1e-3
        for i in idx_subset:
            # finite difference on i-th dimension
            old_gi = g[i]
            g[i] = old_gi + eps
            obj_plus = objective(g)
            grad_approx[i] = (obj_plus - base_obj)/eps
            g[i] = old_gi  # revert
        
        # gradient ascent step
        g = g + step_size*grad_approx
        # project
        g = project(g)
        new_obj = objective(g)
        if np.abs(new_obj - current_obj) < 1e-6:
            break
        current_obj = new_obj
        # can print debug
        #print(f"Iter={it}, objective={current_obj:.4f}, avg_g={g.mean():.4f}")
    # convert final cos_val to angle
    final_angle = angle_unmod_v1_vs_modulated_coloraxis(g, W_F, W_R, shape_vals, v1_unmod)
    return g, final_angle, current_obj

# We'll do a quick test with range [0.5,1.5], 20 iterations
g_optimized, angle_final, obj_final = projected_gradient_search(
    W_F, W_R, shape_vals, v1_unmod, L=0.5, U=1.5, max_iter=100, step_size=0.02
)
print(f"After gradient search => final angle with old v1: {angle_final:.3f} deg, mean_gain={g_optimized.mean():.3f}")

# We can also do multiple ranges:
ranges_list = [(0.5,1.5), (0.6,1.4), (0.7,1.3), (0.8,1.2), (0.9,1.1)]
angle_results = []
for (lowg, highg) in ranges_list:
    g_opt, ang_fin, _ = projected_gradient_search(
        W_F, W_R, shape_vals, v1_unmod, L=lowg, U=highg, max_iter=20, step_size=0.02
    )
    angle_results.append(ang_fin)
    print(f"Range [{lowg},{highg}] => final angle={ang_fin:.3f} deg")

improvements = [orig_angle - a for a in angle_results]

plt.figure(figsize=(6,4))
xvals = np.arange(len(ranges_list))
plt.bar(xvals, improvements, color='cadetblue')
plt.xticks(xvals, [f"[{a},{b}]" for (a,b) in ranges_list])
plt.ylabel("Angle Improvement (deg) vs. original")
plt.title("Gain Range vs. Color-axis alignment with old PC1")
for i, val in enumerate(improvements):
    plt.text(i, val+0.1, f"{val:.2f}", ha='center')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

