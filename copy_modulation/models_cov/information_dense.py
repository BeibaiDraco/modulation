#!/usr/bin/env python3
"""
Information vs. decoder angle in a smoother, dense network
----------------------------------------------------------
• Feed-forward w  = Gaussian bump across feature space
• Recurrent W_R   = ρ w wᵀ  +  ε noise   (ρ = 0.4, ε = 0.05ρ)
"""

import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import norm, eig, solve

# ─── 1. parameters ────────────────────────────────────────────────────
np.random.seed(42)
N           = 120
rho         = 0.4             # desired spectral radius (slow mode)
jitter_frac = 0.05            # 5 % random jitter
sigma_eta   = 1.0
trials_sim  = 50_000

# ─── 2. feed-forward vector w  (Gaussian tuning) ──────────────────────
x     = np.linspace(-1, 1, N)                           # feature axis
kappa = 4.0                                             # bump width
w     = np.exp(-kappa * x**2).reshape(-1, 1)
w    /= norm(w)                                         # unit length

# ─── 3. recurrent matrix  W_R  (rank-1 + jitter) ──────────────────────
W_dom   = rho * (w @ w.T)
jitter  = jitter_frac * rho * np.random.randn(N, N)
jitter  = 0.5 * (jitter + jitter.T)                     # make symmetric
W_R     = W_dom + jitter

# rescale so the largest eigen-value = ρ exactly
evals, evecs = eig(W_R)
idx_max      = np.argmax(evals.real)
W_R         *= rho / evals.real[idx_max]

u_rec = evecs[:, idx_max].real
u_rec /= norm(u_rec)

# ─── 4. handy inverses and stimulus difference Δμ ─────────────────────
I        = np.eye(N)
inv_I_W  = solve(I - W_R, I)
Delta_mu = inv_I_W @ w

# ─── 5. noise covariance Σ ────────────────────────────────────────────
Sigma = sigma_eta**2 * inv_I_W @ inv_I_W.T

# ─── 6. build decoding axes  v(θ) = cosθ u_rec + sinθ u_perp ──────────
# pick a deterministic orthogonal vector
v_perp = np.random.randn(N)
v_perp -= v_perp @ u_rec * u_rec
v_perp /= norm(v_perp)

def axis(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.cos(t)*u_rec + np.sin(t)*v_perp

thetas  = np.linspace(-90, 90, 181)
S_th, N_th, J_th = [], [], []

for th in thetas:
    v = axis(th)
    S_th.append( (v @ Delta_mu)**2 )
    N_th.append( v @ Sigma @ v )
    J_th.append( S_th[-1] / N_th[-1] )

S_th, N_th, J_th = map(np.array, (S_th, N_th, J_th))

# ─── 7. Monte-Carlo noise check (optional) ────────────────────────────
eta  = sigma_eta * np.random.randn(trials_sim, N)
x_n  = eta @ inv_I_W.T
proj = x_n @ np.vstack([axis(th) for th in thetas]).T
N_emp = proj.var(0)

# ─── 8. plotting  (signal, noise, Fisher) ─────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

axs[0].plot(thetas, S_th/S_th.max(),  lw=2, color='green', label='Signal power')
axs[0].plot(thetas, N_th/N_th.max(),  lw=2, color='grey',  label='Noise variance')
axs[0].set_ylabel("Response Variance (normalised)")
axs[0].set_title("Signal and Noise vs. decoding-axis angle")
axs[0].legend(frameon=False);  axs[0].grid(False)

axs[1].plot(thetas, J_th/J_th.max(), lw=2, color='black')
axs[1].set_ylabel("Fisher ratio  J(θ)  (normalised)")
axs[1].set_xlabel("angle θ (degrees)")
axs[1].set_title("Discriminability vs. angle");  axs[1].grid(False)

for ax in axs:
    ax.axvline(0, ls="--", c="gray", lw=0.8)
    ax.set_xlim(-90, 90)

plt.tight_layout()
plt.savefig("information_plot_dense.png", dpi=300)
plt.show()

# ─── 9. visualise W_R ─────────────────────────────────────────────────
print(W_R)