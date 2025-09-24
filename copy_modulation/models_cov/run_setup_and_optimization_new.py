#!/usr/bin/env python3
"""
Linear 1‑D network on a line (not a ring)
=========================================
• Feed‑forward drive and recurrent weight share the same slope vector,
  so stimulus axis ≈ noise PC‑1
• Gain optimisation keeps both total norm and PC‑plane norm unchanged
  while rotating the stimulus line inside the plane
"""
import numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from sklearn.decomposition import PCA

# ───────── 1. basic parameters ─────────
np.random.seed(42)
N               = 120
q               = np.linspace(0, 1, N)              # neuron positions on a line
target_radius   = 0.90
noise_std       = 0.20
noise_trials    = 600
gain_bounds     = (0.5, 1.5)
max_iter        = 150
feature_range   = (0.0, 1.0)
num_stim_steps  = 21
out_file        = "linear_line_network.npz"

# ───────── 2. feed‑forward: strictly linear drive ─────────
b  = (q - q.mean()).reshape(-1, 1)           # slope vector  (N,1)
WF = lambda f: b * f                         # FF input   ff_i(f)=b_i·f

# ───────── 3. recurrent weight: rank‑1 outer product (+jitter) ─────────
def build_W_R(J=1.0, jitter_amp=0.02):
    W = J * (b @ b.T)                        # rank‑1, eigen‑vector = b
    W *= 1 + jitter_amp*(np.random.rand(N,N)-0.5)  # break perfect alignment
    # rescale to desired spectral radius
    W *= target_radius / np.max(np.abs(np.linalg.eigvals(W)))
    return W
W_R = build_W_R()

# ───────── 4. response & noise PCA ─────────
def response(f, g=None, xi=None):
    g = np.ones(N) if g is None else g
    G = np.diag(g)
    ff = G @ WF(f)
    xi = np.zeros((N,1)) if xi is None else xi.reshape(N,1)
    return np.linalg.solve(np.eye(N) - G@W_R, ff + xi).flatten()

# noise samples
X = np.zeros((noise_trials, N))
for k in range(noise_trials):
    X[k] = response(0.0, xi=np.random.normal(0, noise_std, N))
X -= X.mean(0)
pc1, pc2 = PCA(2).fit(X).components_         # noise PC space
P = np.vstack((pc1, pc2)).T                  # projector N×2

# ───────── 5. stimulus axis helpers ─────────
def stim_axis(g):
    f0,f1 = feature_range
    return response(f1, g) - response(f0, g)

g0      = np.ones(N)
s0      = stim_axis(g0)
L_total = np.linalg.norm(s0)**2
L_plane = np.linalg.norm(P @ (P.T @ s0))**2  # projected length squared

def c_total(g): return np.linalg.norm(stim_axis(g))**2 - L_total
def c_plane(g): return np.linalg.norm(P@(P.T@stim_axis(g)))**2 - L_plane
constraints = [{'type':'eq','fun':c_total},
               {'type':'eq','fun':c_plane}]
bounds = Bounds([gain_bounds[0]]*N, [gain_bounds[1]]*N)

# objective for most‑aligned
obj_align = lambda g: - (np.dot(stim_axis(g), pc1) /
                         (np.linalg.norm(stim_axis(g))*np.linalg.norm(pc1)))**2

# objective for least‑aligned (rotate 60° inside plane)
target = np.cos(np.deg2rad(60))*pc1 + np.cos(np.deg2rad(30))*pc2
target /= np.linalg.norm(target)
obj_misal= lambda g: - (np.dot(stim_axis(g), target) /
                        (np.linalg.norm(stim_axis(g))*np.linalg.norm(target)))**2

# optimise
resA = minimize(obj_align, g0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter':max_iter,'disp':False})
g_align = resA.x if resA.success else g0

resM = minimize(obj_misal, g0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter':max_iter,'disp':False})
g_mis  = resM.x if resM.success else g0

# check norms
for name,g in [("base",g0),("aligned",g_align),("mis-aligned",g_mis)]:
    s = stim_axis(g)
    print(f"{name:10s}  full ‖s‖={np.linalg.norm(s):.3f}   "
          f"plane ‖proj‖={np.linalg.norm(P@(P.T@s)):.3f}")

# ───────── 6. visualise straight trajectories ─────────
f_vals = np.linspace(*feature_range, num_stim_steps)
def proj_curve(g):
    R = np.array([response(f,g) for f in f_vals]) - X.mean(0)
    return R @ np.vstack((pc1, pc2)).T
traj_base = proj_curve(g0)
traj_align= proj_curve(g_align)
traj_mis  = proj_curve(g_mis)
noise_proj= X @ np.vstack((pc1, pc2)).T

plt.style.use('seaborn-v0_8-whitegrid')
fig,ax=plt.subplots(figsize=(8,6))
ax.scatter(noise_proj[:,0],noise_proj[:,1],s=10,alpha=0.2,color='gray')
ax.plot(*traj_base.T,'.-',c='k',   label='baseline')
ax.plot(*traj_align.T,'.-',c='blue',label='aligned')
ax.plot(*traj_mis.T,  '.-',c='red', label='mis‑aligned')
ax.set_xlabel("PC‑1"); ax.set_ylabel("PC‑2"); ax.set_aspect('equal')
ax.set_title("Straight stimulus lines with equal norms") ; ax.legend(frameon=False)
plt.tight_layout(); plt.show()

# ───────── 7. save everything ─────────
np.savez(out_file,
         W_F=W_F, W_R=W_R, q=q, pc1=pc1, pc2=pc2,
         g_aligned=g_align, g_misaligned=g_mis)
print("saved →", out_file)
