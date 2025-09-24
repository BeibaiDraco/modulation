#!/usr/bin/env python3
# -------------------------------------------------------------
#  viz_linear_line.py
#  Build a *line* (non‑periodic) network whose feed‑forward slope
#  and recurrent slow mode are the same, confirm unique leading
#  eigen‑value, compare it with noise PC‑1, visualise.
# -------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ========== 1. global settings ==========
np.random.seed(45)
N                = 120                     # neurons
q                = np.linspace(0, 1, N)    # positions on a line
target_radius    = 0.90
jitter_amp       = 0.02                    # ±2 % jitter
noise_std        = 0.20
noise_trials     = 600
eigen_gap_thresh = 1e-3

# ========== 2. build W^R (rank‑1 + jitter, rescaled) ==========
b = (q - q.mean()).reshape(-1, 1)          # slope / slow eigen‑vector

def build_W_R(rank_J=1.0):
    W = rank_J * (b @ b.T)                 # rank‑1 outer product
    W *= 1 + jitter_amp*(np.random.rand(N,N)-0.5)   # break exact rank‑1
    W *= target_radius / np.max(np.abs(np.linalg.eigvals(W)))
    return W

def unique_leader_matrix():
    for _ in range(30):                    # a few jittered tries
        W = build_W_R()
        eigs = np.linalg.eigvals(W)
        idx  = np.argsort(-np.abs(eigs))
        gap  = abs(abs(eigs[idx[0]])-abs(eigs[idx[1]]))
        if gap > eigen_gap_thresh and np.isreal(eigs[idx[0]]):
            print(f"Success: |λ₁|={abs(eigs[idx[0]]):.3f}, "
                  f"|λ₂|={abs(eigs[idx[1]]):.3f}, gap={gap:.3f}")
            return W
    raise RuntimeError("unique leading eigen‑value not obtained")

W_R = unique_leader_matrix()
eigvals, eigvecs = np.linalg.eig(W_R)
idx_lead  = np.argmax(np.abs(eigvals))
v_slow    = np.real(eigvecs[:, idx_lead])
v_slow   /= np.linalg.norm(v_slow)

# ========== 3. noise PCs ==========
def noise_PCs(W, σ, trials):
    L = np.linalg.inv(np.eye(N)-W)
    X = np.random.normal(0,σ,(trials,N)) @ L.T
    X -= X.mean(0,keepdims=True)
    return PCA(3).fit(X).components_

pcs = noise_PCs(W_R, noise_std, noise_trials)
pc1, pc2 = pcs[:2]
cos_sim = abs(v_slow @ pc1)
print(f"cosine similarity (v_slow, PC1) = {cos_sim:.3f}")

# ========== 4. decay curve helper ==========
def decay_curve(A, vec, steps=60):
    norms, x = [], vec.copy()
    for _ in range(steps):
        norms.append(np.linalg.norm(x))
        x = A @ x
    return np.array(norms)

# ========== 5. visualisation ==========
fig = plt.figure(figsize=(11,8))
gs  = fig.add_gridspec(2,2,hspace=0.35,wspace=0.30)

# (A) eigen‑spectrum
axA = fig.add_subplot(gs[0,0])
axA.plot(np.sort(np.abs(eigvals))[::-1],'.-')
axA.set_xlabel("eigen‑index (sorted)"); axA.set_ylabel(r"$|\lambda_i|$")
axA.set_title("(A) eigen‑spectrum"); axA.axhline(1,ls='--',c='k',lw=0.6)
λ_sorted=np.sort(np.abs(eigvals))[::-1]
axA.annotate(f"gap={λ_sorted[0]-λ_sorted[1]:.3f}",xy=(1,λ_sorted[1]),
             xytext=(4,λ_sorted[1]*1.05),arrowprops=dict(arrowstyle="->"))

# (B) profiles along the line
axB = fig.add_subplot(gs[0,1])
axB.plot(q, v_slow, label=r"$v_{\rm slow}$")
axB.plot(q, pc1,  ls='--', label="PC‑1")
axB.set_xlabel("neuron position q"); axB.set_ylabel("component")
axB.set_title("(B) mode profiles"); axB.legend(frameon=False)

# (C) projection in PC1‑PC2 plane
proj_slow = pcs[:2] @ v_slow
axC = fig.add_subplot(gs[1,0])
axC.quiver(0,0,proj_slow[0],proj_slow[1],angles='xy',scale_units='xy',
           scale=1,color='tab:blue',label=r"$v_{\rm slow}$")
axC.quiver(0,0,1,0,angles='xy',scale_units='xy',scale=1,
           color='tab:orange',label="PC‑1")
axC.set_aspect('equal'); axC.set_xlabel("PC‑1"); axC.set_ylabel("PC‑2")
axC.set_title("(C) projections"); axC.legend(frameon=False)

# (D) decay curves
axD = fig.add_subplot(gs[1,1])
ts = np.arange(60)
axD.semilogy(ts,decay_curve(W_R,v_slow)/np.linalg.norm(v_slow),
             label=r"$v_{\rm slow}$")
axD.semilogy(ts,decay_curve(W_R,pcs[2])/np.linalg.norm(pcs[2]),
             ls='--',label="PC‑3 (faster)")
axD.set_xlabel("time‑step"); axD.set_ylabel("norm (log)")
axD.set_title("(D) decay comparison"); axD.legend(frameon=False)

fig.suptitle(f"Linear line network  |  cos(v_slow, PC1) = {cos_sim:.3f}",
             fontsize=14,y=0.98)
plt.tight_layout(rect=[0,0,1,0.96]); plt.show()
