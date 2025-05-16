#!/usr/bin/env python3
# -------------------------------------------------------------
#  viz_alignment_E_only.py
#  Same analysis as before, but with an all‑excitatory network
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------------------------------------
# 1.  Configuration
# -------------------------------------------------------------
np.random.seed(42)

N               = 100          # neurons (all excitatory)
feature_range   = (0.0, 1.0)   # 1‑D stimulus
sigma_tuning_R  = 0.15         # width of Gaussian tuning for W_R
desired_radius  = 0.90         # spectral radius after rescaling
num_noise_tr    = 500          # noise trials for PCA
noise_std       = 0.20         # std of private noise

# -------------------------------------------------------------
# 2.  Helper functions
# -------------------------------------------------------------
def initialise_prefs(N, f_range):
    prefs = np.linspace(*f_range, N)
    np.random.shuffle(prefs)
    return prefs

def build_W_F(N):
    """Simple feed‑forward weights (all ones)."""
    return np.ones((N, 1))

def build_W_R(prefs, base_J, sigma, target_radius):
    N = len(prefs)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            Δ = prefs[i] - prefs[j]
            W[i, j] = base_J * np.exp(-0.5 * (Δ / sigma) ** 2)  # positive weight
    # rescale so largest |λ| is target_radius
    eigs = np.linalg.eigvals(W)
    W *= target_radius / np.max(np.abs(eigs))
    return W

def steady_response(W_R, W_F, feat_val, g=None, ξ=None):
    N = W_R.shape[0]
    I = np.eye(N)
    g = np.ones(N) if g is None else g
    G = np.diag(g)
    ξ = np.zeros((N, 1)) if ξ is None else ξ.reshape(N, 1)
    ff = (G @ W_F) * feat_val
    r  = np.linalg.solve(I - G @ W_R, ff + ξ)
    return r.flatten()

def noise_PCA(W_R, W_F, trials, σ_noise, g=None):
    N = W_R.shape[0]
    g = np.ones(N) if g is None else g
    X = np.zeros((trials, N))
    for k in range(trials):
        ξ = np.random.normal(0, σ_noise, size=N)
        X[k] = steady_response(W_R, W_F, 0.0, g, ξ)
    X -= X.mean(axis=0)
    pca = PCA(n_components=3).fit(X)
    return X, pca.components_

# -------------------------------------------------------------
# 3.  Build network and obtain noise PCs
# -------------------------------------------------------------
prefs           = initialise_prefs(N, feature_range)
W_F             = build_W_F(N)
W_R             = build_W_R(prefs, base_J=0.8 / N,  # uniform positive strength
                            sigma=sigma_tuning_R,
                            target_radius=desired_radius)
g_vec           = np.ones(N)
A               = np.diag(g_vec) @ W_R

_, pcs          = noise_PCA(W_R, W_F, num_noise_tr, noise_std, g_vec)
pc1             = pcs[0]

# -------------------------------------------------------------
# 4.  Slowest eigen‑mode
# -------------------------------------------------------------
eigvals, eigvecs = np.linalg.eig(A)
idx_slow         = np.argmax(np.abs(eigvals))
v_slow           = np.real(eigvecs[:, idx_slow])
v_slow          /= np.linalg.norm(v_slow)

cos_sim          = np.abs(v_slow @ pc1)
print(f"cosine similarity = {cos_sim:.3f}")

# -------------------------------------------------------------
# 5.  Visualisation
# -------------------------------------------------------------
fig = plt.figure(figsize=(11, 8))
gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

# (A) spectrum
axA = fig.add_subplot(gs[0, 0])
axA.plot(np.sort(np.abs(eigvals))[::-1], '.-')
axA.set_xlabel("eigen‑index (sorted)")
axA.set_ylabel(r"$|\lambda_i|$")
axA.set_title("(A) spectrum of $A$")
axA.axhline(1, ls='--', c='k', lw=0.7)

# (B) profiles
axB = fig.add_subplot(gs[0, 1])
θ = np.linspace(0, 2*np.pi, N, endpoint=False)
axB.plot(θ, v_slow, label=r"$v_{\rm slow}$")
axB.plot(θ, pc1,   ls='--', label="PC‑1")
axB.set_xlabel("preferred feature angle")
axB.set_ylabel("component value")
axB.set_title("(B) mode profiles")
axB.legend(frameon=False)

# (C) PC‑1/PC‑2 plane
proj_slow = pcs[:2] @ v_slow
axC = fig.add_subplot(gs[1, 0])
axC.quiver([0], [0], [proj_slow[0]], [proj_slow[1]],
           angles='xy', scale_units='xy', scale=1,
           color='tab:blue', label=r"$v_{\rm slow}$")
axC.quiver([0], [0], [1], [0],
           angles='xy', scale_units='xy', scale=1,
           color='tab:orange', label="PC‑1")
axC.set_aspect('equal')
axC.set_xlabel("PC‑1")
axC.set_ylabel("PC‑2")
axC.set_title("(C) projections in PC space")
axC.legend(frameon=False)

# (D) decay curves
def decay(vec, steps=60):
    x = vec.copy(); norms = [np.linalg.norm(x)]
    for _ in range(1, steps):
        x = A @ x
        norms.append(np.linalg.norm(x))
    return np.array(norms)

axD = fig.add_subplot(gs[1, 1])
ts  = np.arange(60)
axD.semilogy(ts, decay(v_slow)/decay(v_slow)[0], label=r"$v_{\rm slow}$")
axD.semilogy(ts, decay(pcs[2])/decay(pcs[2])[0], ls='--', label="PC‑3 (faster)")
axD.set_xlabel("time‑step")
axD.set_ylabel("norm (log)")
axD.set_title("(D) decay comparison")
axD.legend(frameon=False)

fig.suptitle(f"All‑E network: slowest mode vs noise PC‑1 (cos = {cos_sim:.3f})",
             fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
