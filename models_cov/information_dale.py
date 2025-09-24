#!/usr/bin/env python3
"""
Cheap 1-D decoder in an EI network obeying Dale’s rule
=====================================================
• 80 % excitatory rows → positive weights only
• 20 % inhibitory rows → negative weights only
• Leading eigen-vector  z  (mixed sign) defines
    – slow/noise axis
    – stimulus mean difference Δμ
and is aligned with the decoder’s “best” axis (θ = 0°).
"""

import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import norm, solve, eig

# ─── 1. parameters ────────────────────────────────────────────────────
np.random.seed(42)
N          = 120
frac_E     = 0.8           # 80 % excitatory
rho        = 0.4          # desired Perron eigen-value
sigma_eta  = 1.0
trials_sim = 50_000

# ─── 2. build signed magnitude vector  z  (Dale) ──────────────────────
idx_E = np.arange(int(frac_E * N))
idx_I = np.arange(int(frac_E * N), N)

x_pos = np.linspace(-1, 1, N)             # “feature” axis
kappa  = 4.0
v_mag  = np.exp(-kappa * x_pos**2)
v_mag /= norm(v_mag)

sign_vec       = np.ones(N)
sign_vec[idx_I]= -1                       # I rows negative

z = (sign_vec * v_mag)[:, None]           # column vector shape (N,1)
w = z / norm(z)                           # feed-forward drive

# ─── 3. Dale-compliant W_R  = ρ z v_magᵀ + rescale ───────────────────
W_R = rho * (z @ v_mag[None, :])          # rank-1 with correct signs

# rescale exactly so that  λ_max = rho
evals, evecs = eig(W_R)
idx_max      = np.argmax(evals.real)
W_R         *= rho / evals.real[idx_max]
# recompute eigen-vectors after scaling
evals, evecs = eig(W_R)
u_rec        = evecs[:, np.argmax(evals.real)].real
u_rec       /= norm(u_rec)

# quick sanity check
assert np.isclose(evals.real.max(), rho, atol=1e-9)

# ─── 4. inverse, Δμ, noise covariance Σ ───────────────────────────────
I         = np.eye(N)
inv_I_W   = solve(I - W_R, I)
Delta_mu  = inv_I_W @ w                    # stimulus difference
Sigma     = sigma_eta**2 * inv_I_W @ inv_I_W.T

# ─── 5. define decoding axis family  v(θ) ─────────────────────────────
v_rand = np.random.randn(N)
v_rand -= (v_rand @ u_rec) * u_rec         # orthogonalise
u_perp  = v_rand / norm(v_rand)

def axis(theta_deg):
    θ = np.deg2rad(theta_deg)
    return np.cos(θ)*u_rec + np.sin(θ)*u_perp

thetas = np.linspace(-90, 90, 181)
S_th, N_th, J_th = [], [], []
for th in thetas:
    v = axis(th)
    S_th.append( (v @ Delta_mu)**2 )
    N_th.append( v @ Sigma @ v )
    J_th.append( S_th[-1] / N_th[-1] )
S_th, N_th, J_th = map(np.array, (S_th, N_th, J_th))

# ─── 6. plots ─────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(2, 1, figsize=(6,7), sharex=True)

axs[0].plot(thetas, S_th/S_th.max(), lw=2, c='green', label="Signal power")
axs[0].plot(thetas, N_th/N_th.max(), lw=2, c='grey',  label="Noise variance")
axs[0].set_ylabel("Variance (normalised)")
axs[0].set_title("Signal & Noise vs. decoding-axis angle")
axs[0].legend(frameon=False); axs[0].grid(False)

axs[1].plot(thetas, J_th/J_th.max(), lw=2, c='black')
axs[1].set_ylabel("Fisher ratio  J(θ)  (normalised)")
axs[1].set_xlabel("angle θ (degrees)")
axs[1].set_title("Discriminability vs. angle"); axs[1].grid(False)

for ax in axs:
    ax.axvline(0, ls="--", c="gray", lw=0.8)
    ax.set_xlim(-90, 90)

plt.tight_layout()

# ───────── Save plots ─────────
plt.savefig('information_dale_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('information_dale_plot.svg', bbox_inches='tight')
plt.show()
print("Plots saved as 'information_dale_plot.png' and 'information_dale_plot.svg'")

# ───────── Save data ─────────
data = np.column_stack((thetas, S_th, N_th, J_th))
header = "theta_degrees,signal_power,noise_variance,fisher_ratio"
np.savetxt('information_dale_data.csv', data, delimiter=',', header=header)
print("Data saved as 'information_dale_data.csv'")

# ─── 7. visualise W_R with Dale signs ────────────────────────────────
plt.figure(figsize=(4.5,4))
plt.imshow(W_R, cmap='coolwarm',
           vmin=-abs(W_R).max(), vmax=abs(W_R).max())
plt.colorbar(label='connection strength')
plt.title('Recurrent Matrix (Dale)')
plt.tight_layout()
plt.savefig('recurrent_connectivity_dale.png', dpi=300, bbox_inches='tight')
plt.savefig('recurrent_connectivity_dale.svg', bbox_inches='tight')
plt.show()
print("Connectivity matrix plots saved as 'recurrent_connectivity_dale.png' and 'recurrent_connectivity_dale.svg'")

# ─── 8. Check the rank of W_R ────────────────────────────────────────
# Calculate the singular values of W_R
singular_values = np.linalg.svd(W_R, compute_uv=False)

# Print the rank and singular values
print("\n─── Rank analysis of W_R ───")
print(f"Rank of W_R: {np.sum(singular_values > 1e-10)}")
print(f"Largest singular value: {singular_values[0]:.6f}")

# Plot the singular values
plt.figure(figsize=(6, 4))
plt.semilogy(singular_values, 'o-', markersize=6)
plt.grid(True, alpha=0.3)
plt.xlabel('Index')
plt.ylabel('Singular value (log scale)')
plt.title('Singular Value Spectrum of W_R')
plt.tight_layout()
plt.savefig('singular_values_dale.png', dpi=300, bbox_inches='tight')
plt.savefig('singular_values_dale.svg', bbox_inches='tight')
plt.show()
print("Singular value spectrum saved as 'singular_values_dale.png' and 'singular_values_dale.svg'")

theta_star = thetas[np.argmax(J_th)]
print(f"Peak at θ = {theta_star:.2f}°")