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
rho         = 0.4        # spectral radius of W_R
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
# Set figure size parameters
fig_width = 6  # width in inches
fig_height = 7  # height in inches
fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)

# First subplot with both signal and noise curves
axs[0].plot(thetas, S_th, lw=2, color='green', label="Signal power")
axs[0].plot(thetas, N_th, lw=2, color='grey', label="Noise variance")
axs[0].set_ylabel("Response Variance (Normalized)")
axs[0].set_title("Signal and Noise vs. decoding-axis angle")
axs[0].legend(frameon=False, fontsize=9)
axs[0].grid(False)

# Second subplot for Fisher ratio
axs[1].plot(thetas, J_th, lw=2, color='black')
axs[1].set_ylabel("Fisher ratio $J(\\theta)$")
axs[1].set_xlabel("angle $\\theta$ (degrees)")
axs[1].set_title("Discriminability vs. angle")
axs[1].grid(False)

for ax in axs:
    ax.axvline(0, ls="--", c="gray", lw=0.8)
    ax.set_xlim(-90, 90)

plt.tight_layout()

# ───────── 6. save plots ─────────
# Save the figure in both PNG and SVG formats
plt.savefig('information_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('information_plot.svg', bbox_inches='tight')
plt.show()
print("Plots saved as 'information_plot.png' and 'information_plot.svg'")

# ───────── 7. save data ─────────
# Save the plotting data to a file
data = np.column_stack((thetas, S_th, N_th, J_th))
header = "theta_degrees,signal_power,noise_variance,fisher_ratio"
np.savetxt('information_data.csv', data, delimiter=',', header=header)
print("Data saved as 'information_data.csv'")

