#!/usr/bin/env python3
# -------------------------------------------------------------
#  viz_unique_EI_ring.py
#  All‑in‑one: builds E–I ring with UNIQUE leading eigenvalue,
#  measures alignment between v_slow and noise PC1, visualises.
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ========== 1. global settings ==========
np.random.seed(42)
N                = 120                 # neurons
frac_E           = 0.8
N_E, N_I         = int(N*frac_E), N-int(N*frac_E)
feature_range    = (0.0, 1.0)
sigma_tune_R     = 0.15
target_radius    = 0.90
jitter_amp       = 0.02                # ±2 % weight jitter
noise_std        = 0.2
noise_trials     = 600
eigen_gap_thresh = 1e-3

# ========== 2. helper utilities ==========
def prefs_and_types(N, N_E):
    prefs = np.linspace(*feature_range, N)
    np.random.shuffle(prefs)
    typ   = np.array(['E']*N_E + ['I']*(N-N_E))
    np.random.shuffle(typ)
    return prefs, (typ == 'E')

def build_W_R(prefs, is_E, gamma_inh):
    """Gaussian E/I ring with jitter and inhibition scale γ."""
    J = {('E','E'):  0.8/N_E,
         ('E','I'):  0.6/N_E,
         ('I','E'): -gamma_inh*1.0/N_I,
         ('I','I'): -gamma_inh*0.8/N_I}
    N = len(prefs); W = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            Δ = prefs[i] - prefs[j]
            tuning = np.exp(-0.5*(Δ/sigma_tune_R)**2)
            jitter = 1.0 + jitter_amp*(np.random.rand()-0.5)
            pre = 'E' if is_E[j] else 'I'
            post= 'E' if is_E[i] else 'I'
            W[i,j] = J[(pre,post)] * tuning * jitter
    # rescale spectral radius
    eigs = np.linalg.eigvals(W)
    W   *= target_radius / np.max(np.abs(eigs))
    return W

def unique_leader_matrix():
    prefs, is_E = prefs_and_types(N, N_E)
    for γ in np.linspace(1.0, 0.05, 20):            # weaken inhibition gradually
        for _ in range(10):                         # ten jittered tries per γ
            W = build_W_R(prefs, is_E, γ)
            eigs = np.linalg.eigvals(W)
            idx  = np.argsort(-np.abs(eigs))
            gap  = abs(abs(eigs[idx[0]]) - abs(eigs[idx[1]]))
            if gap > eigen_gap_thresh and np.isreal(eigs[idx[0]]):
                print(f"Success: γ={γ:.3f}, "
                      f"|λ₁|={abs(eigs[idx[0]]):.3f}, "
                      f"|λ₂|={abs(eigs[idx[1]]):.3f}, gap={gap:.3f}")
                return W, prefs, is_E
    raise RuntimeError("Could not obtain unique leading eigenvalue; "
                       "try looser inhibition or larger jitter.")

def noise_PCs(W, σ, trials):
    """Generate noise responses analytically: r = (I-W)^-1 ξ."""
    N = W.shape[0]
    L = np.linalg.inv(np.eye(N) - W)      # linear operator
    ξ = np.random.normal(0, σ, size=(trials, N))
    X = ξ @ L.T                           # responses
    X -= X.mean(axis=0, keepdims=True)
    return PCA(n_components=3).fit(X).components_

def decay_curve(A, vec, steps=60):
    norms, x = [], vec.copy()
    for _ in range(steps):
        norms.append(np.linalg.norm(x))
        x = A @ x
    return np.array(norms)

# ========== 3. build network with unique leader ==========
W_R, prefs, is_E = unique_leader_matrix()
eigvals, eigvecs = np.linalg.eig(W_R)
idx_lead         = np.argmax(np.abs(eigvals))
v_slow           = np.real(eigvecs[:, idx_lead])
v_slow          /= np.linalg.norm(v_slow)

pcs  = noise_PCs(W_R, noise_std, noise_trials)
pc1  = pcs[0]
cos_sim = np.abs(v_slow @ pc1)
print(f"cosine similarity (v_slow, PC1) = {cos_sim:.3f}")

# ========== 4. visualisation ==========
fig = plt.figure(figsize=(11, 8))
gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

# (A) eigen‑spectrum
axA = fig.add_subplot(gs[0, 0])
axA.plot(np.sort(np.abs(eigvals))[::-1], '.-')
axA.set_xlabel("eigen‑index (sorted)"); axA.set_ylabel(r"$|\lambda_i|$")
axA.set_title("(A) eigen‑spectrum"); axA.axhline(1, ls='--', c='k', lw=0.6)

# highlight gap
λ_sorted = np.sort(np.abs(eigvals))[::-1]
axA.annotate(f"gap={λ_sorted[0]-λ_sorted[1]:.3f}",
             xy=(1, λ_sorted[1]),
             xytext=(4, λ_sorted[1]*1.05),
             arrowprops=dict(arrowstyle="->"))

# (B) profiles around the ring
θ = np.linspace(0, 2*np.pi, N, endpoint=False)
axB = fig.add_subplot(gs[0, 1])
axB.plot(θ, v_slow, label=r"$v_{\rm slow}$")
axB.plot(θ, pc1, ls='--', label="PC‑1")
axB.set_xlabel("preferred feature angle"); axB.set_ylabel("component")
axB.set_title("(B) mode profiles"); axB.legend(frameon=False)

# (C) projection in PC1‑PC2 plane
proj_slow = pcs[:2] @ v_slow
axC = fig.add_subplot(gs[1, 0])
axC.quiver(0, 0, proj_slow[0], proj_slow[1],
           angles='xy', scale_units='xy', scale=1,
           color='tab:blue', label=r"$v_{\rm slow}$")
axC.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1,
           color='tab:orange', label="PC‑1")
axC.set_aspect('equal'); axC.set_xlabel("PC‑1"); axC.set_ylabel("PC‑2")
axC.set_title("(C) projections"); axC.legend(frameon=False)

# (D) decay curves
axD = fig.add_subplot(gs[1, 1])
steps = 60; ts = np.arange(steps)
axD.semilogy(ts, decay_curve(W_R, v_slow)/np.linalg.norm(v_slow),
             label=r"$v_{\rm slow}$")
axD.semilogy(ts, decay_curve(W_R, pcs[2])/np.linalg.norm(pcs[2]),
             ls='--', label="PC‑3 (faster)")
axD.set_xlabel("time‑step"); axD.set_ylabel("norm (log)")
axD.set_title("(D) decay comparison"); axD.legend(frameon=False)

fig.suptitle(f"Unique‑leader E–I ring  |  cos(v_slow, PC1) = {cos_sim:.3f}",
             fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
