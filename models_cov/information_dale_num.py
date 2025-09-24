#!/usr/bin/env python3
"""
Cheap 1-D decoder in an EI network obeying Dale’s rule
======================================================
• 80 % excitatory rows → positive weights only
• 20 % inhibitory rows → negative weights only
• Right Perron vector   z  (mixed sign) defines
    – slow/noise axis
    – stimulus mean difference Δμ
and is aligned with the decoder’s “best” axis (θ = 0°).
"""

import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import norm, solve, eig

# ─── 1. parameters ────────────────────────────────────────────────────
np.random.seed(42)
N          = 120
frac_E     = 0.8            # 80 % excitatory
rho        = 0.4            # desired Perron eigen-value
sigma_eta  = 1.0
trials_sim = 50_000         # Monte-Carlo trials

# ─── 2. build signed magnitude vector  z  (Dale) ──────────────────────
idx_E = np.arange(int(frac_E * N))
idx_I = np.arange(int(frac_E * N), N)

x_pos = np.linspace(-1, 1, N)           # “feature” axis
kappa  = 4.0
v_mag  = np.exp(-kappa * x_pos**2)
v_mag /= norm(v_mag)                    # ‖v‖ = 1

sign_vec        = np.ones(N)
sign_vec[idx_I] = -1                    # I rows negative

z = (sign_vec * v_mag)[:, None]         # column vector  (N,1)
w = z / norm(z)                         # feed-forward drive  (unit)

d = float(v_mag @ z)                    # overlap needed later   # NEW -----
print(f"Overlap d = {d:.6f}")           # NEW -----

# ─── 3. Dale-compliant  W_R  = ρ z v_magᵀ  (rescaled) ─────────────────
W_R = rho * (z @ v_mag[None, :])        # rank-1 with correct signs

# rescale so that  λ_max = ρ  exactly
evals, evecs = eig(W_R)
W_R *= rho / evals.real.max()

# right Perron vector (unit length)
u_rec = evecs[:, np.argmax(evals.real)].real
u_rec /= norm(u_rec)

# sanity check
assert np.isclose(eig(W_R)[0].real.max(), rho, atol=1e-9)

# ─── 4. inverse, Δμ, covariance Σ ─────────────────────────────────────
I         = np.eye(N)
inv_I_W   = solve(I - W_R, I)
Delta_mu  = inv_I_W @ w                  # stimulus difference vector
Sigma     = sigma_eta**2 * inv_I_W @ inv_I_W.T

# ─── 5. decoder family  v(θ)  (unit vectors) ─────────────────────────
v_rand = np.random.randn(N)
v_rand -= (v_rand @ u_rec) * u_rec       # orthogonalise
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

# ─── 6. Monte-Carlo simulation (empirical curves) ──────────────────── # NEW
# draw independent noise for s=0 and s=1
eta0 = np.random.randn(trials_sim, N) * sigma_eta
eta1 = np.random.randn(trials_sim, N) * sigma_eta

# steady-state responses
r0 = eta0 @ inv_I_W.T                                # s = 0
r1 = (w.ravel() + eta1) @ inv_I_W.T                  # s = 1

Δμ_emp = r1.mean(axis=0) - r0.mean(axis=0)           # empirical Δμ

S_emp, N_emp, J_emp = [], [], []
for th in thetas:
    v = axis(th)
    proj0 = r0 @ v
    S_emp.append( (v @  Δμ_emp)**2 )
    N_emp.append( proj0.var() )                      # noise from s = 0
    J_emp.append( S_emp[-1] / N_emp[-1] )
S_emp, N_emp, J_emp = map(np.array, (S_emp, N_emp, J_emp))

# ─── 7. plot analytic vs. empirical (共有1 figure) ────────────────────
plt.figure(figsize=(6, 7))

# top panel: signal & noise
plt.subplot(2, 1, 1)
plt.plot(thetas, S_th, lw=2,  c='green',  label='Signal (analytic)')
plt.plot(thetas, N_th, lw=2,  c='grey',   label='Noise  (analytic)')
plt.scatter(thetas, S_emp, 20, c='none', edgecolors='darkgreen', label='Signal (MC)')
plt.scatter(thetas, N_emp, 20, c='none', edgecolors='dimgray',   label='Noise  (MC)')
plt.ylabel('Variance')
plt.title('Signal & Noise vs. decoding-axis angle')
plt.legend(frameon=False, fontsize=8)
plt.axvline(0, ls='--', c='gray', lw=0.8)
plt.xlim(-90, 90)

# bottom panel: Fisher ratio
plt.subplot(2, 1, 2)
plt.plot(thetas, J_th, lw=2, c='black', label='Fisher ratio (analytic)')
plt.scatter(thetas, J_emp, 20, c='none', edgecolors='black', label='Fisher ratio (MC)')
plt.ylabel('Fisher ratio  $J(\\theta)$')
plt.xlabel('angle $\\theta$ (degrees)')
plt.title('Discriminability vs. angle')
plt.legend(frameon=False, fontsize=8)
plt.axvline(0, ls='--', c='gray', lw=0.8)
plt.xlim(-90, 90)

plt.tight_layout()
plt.savefig('information_dale_compare.png', dpi=300, bbox_inches='tight')
plt.savefig('information_dale_compare.svg',  bbox_inches='tight')
plt.show()
print("Comparison plots saved as 'information_dale_compare.(png/svg)'")

# ─── 8. save raw data ────────────────────────────────────────────────
data = np.column_stack((thetas, S_th, N_th, J_th, S_emp, N_emp, J_emp))
header = ("theta_degrees,signal_th,noise_th,fisher_th,"
          "signal_emp,noise_emp,fisher_emp")
np.savetxt('information_dale_compare.csv', data, delimiter=',', header=header)
print("Data saved as 'information_dale_compare.csv'")

# report optimum
theta_star = thetas[np.argmax(J_th)]
print(f"Analytic peak at θ = {theta_star:+.2f}°")
