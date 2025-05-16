#!/usr/bin/env python3
"""
Cheap 1-D decoder in a perfectly aligned network
-----------------------------------------------
• feed-forward vector w   ∥ recurrent Perron vector  u
• study how S(θ), N(θ), and J(θ)=S/N vary when we decode
  along an axis that is rotated by θ relative to u
"""

import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import eig, norm, solve

# ───────── 1. network construction ─────────
np.random.seed(0)
N           = 120
rho         = 0.5         # spectral radius of W_R
sigma_eta   = 1.0          # private-noise SD
trials_sim  = 50000        # Monte-Carlo samples for noise check

# feed-forward vector w = first canonical basis vector
w = np.zeros((N, 1))
w[0, 0] = 1.0
w /= norm(w)                # normalise (only the direction matters)

# recurrent matrix: rank-1 outer product along the SAME direction
W_R = rho * (w @ w.T)       # (outer-product gives Perron vector = w)

# handy matrices
I = np.eye(N)
inv_I_W  = solve(I - W_R, I)              # (I - W_R)^{-1}
inv_I_WT = solve(I - W_R.T, I)            # (I - W_R^T)^{-1}

# stimulus difference (Δμ) for s=1 minus s=0
beta        = 1.0                         # stimulus amplitude difference
Delta_mu    = beta * inv_I_W @ w          # shape (N,1)
u_stim      = Delta_mu.flatten() / norm(Delta_mu)

# noise covariance Σ = σ² (I-W)^{-1}(I-Wᵀ)^{-1}
Sigma = sigma_eta**2 * inv_I_W @ inv_I_WT

# ───────── 2. pick an orthonormal complement vector ─────────
# choose a random vector orthogonal to u_stim
v_rand = np.random.randn(N)
v_rand -= v_rand.dot(u_stim) * u_stim       # Gram-Schmidt
u_perp  = v_rand / norm(v_rand)

# function that returns axis for an angle θ
def axis(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.cos(theta) * u_stim + np.sin(theta) * u_perp

# ───────── 3. analytic curves ─────────
thetas = np.linspace(-90, 90, 181)
S_th   = []        # signal power
N_th   = []        # noise power
J_th   = []        # Fisher ratio

for th in thetas:
    v  = axis(th)
    s  = (v @ Delta_mu)**2
    n  = v @ Sigma @ v
    S_th.append(s)
    N_th.append(n)
    J_th.append(s / n)

S_th = np.array(S_th)
N_th = np.array(N_th)
J_th = np.array(J_th)

# ───────── 4. Monte-Carlo noise check (optional) ─────────
# make many noise samples and project them
eta   = np.random.randn(trials_sim, N) * sigma_eta
x_n   = eta @ inv_I_W.T                   # shape (trials,N)
proj  = x_n @ np.vstack([axis(th) for th in thetas]).T  # (trials, θ)

N_emp = proj.var(0)                       # empirical noise variance

# ───────── 5. plots ─────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

axs[0].plot(thetas, S_th, lw=2, label="theory")
axs[0].scatter(thetas, S_th, s=10, c="k", alpha=0.3)
axs[0].set_ylabel("signal power $\\beta^2\\cos^2\\theta$")
axs[0].set_title("Signal vs. decoding-axis angle")

axs[1].plot(thetas, N_th, lw=2, label="theory")
axs[1].scatter(thetas, N_emp, s=10, c="r", alpha=0.3, label="simulation")
axs[1].set_ylabel("noise variance $v^\\top\\Sigma v$")
axs[1].set_title("Noise vs. decoding-axis angle")
axs[1].legend(frameon=False, fontsize=9)

axs[2].plot(thetas, J_th, lw=2)
axs[2].set_ylabel("Fisher ratio $J(\\theta)$")
axs[2].set_xlabel("angle $\\theta$ (degrees)")
axs[2].set_title("Discriminability vs. angle")

for ax in axs:
    ax.axvline(0, ls="--", c="gray", lw=0.8)
    ax.set_xlim(-90, 90)

plt.tight_layout()
plt.show()
