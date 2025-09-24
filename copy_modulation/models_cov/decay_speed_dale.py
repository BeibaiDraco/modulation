#!/usr/bin/env python3
"""
Decay of three directions in an EI network obeying Dale’s rule
--------------------------------------------------------------
(1)  u_rec  : Perron eigen-vector  (λ = ρ)        → slow
(2)  e_fast : any orthogonal vector (λ = 0)       → fast
(3)  mix    : (u_rec + e_fast)/√2                → intermediate

Network is the same Dale-compliant matrix used in information_dale.py
"""

import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import norm, eig, solve
from scipy.linalg import expm

# ─── 1. network parameters ──────────────────────────────────────────
np.random.seed(42)
N       = 120
rho     = 0.8            # set a slow mode (τ = 1/(1-ρ) = 5)
frac_E  = 0.8            # 80 % excitatory
T_max   = 10
t_vals  = np.arange(T_max + 1)

# ─── 2. build Dale-compliant rank-1 matrix  W_R  ─────────────────────
idx_E = np.arange(int(frac_E * N))
idx_I = np.arange(int(frac_E * N), N)

x_pos = np.linspace(-1, 1, N)
kappa  = 4.0
v_mag  = np.exp(-kappa * x_pos**2)
v_mag /= norm(v_mag)

signs        = np.ones(N)
signs[idx_I] = -1
z = (signs * v_mag)[:, None]              # column  (N,1)

# initial rank-1 matrix
W_R = rho * (z @ v_mag[None, :])

# rescale exactly so that λ_max = ρ (numerical hygiene)
λs, vecs = eig(W_R)
W_R *= rho / λs.real.max()

# recompute Perron eigen-vector (real part) after rescaling
λs, vecs = eig(W_R)
u_rec = vecs[:, λs.real.argmax()].real
u_rec /= norm(u_rec)

# ─── 3. choose fast orthogonal vector and mixture ───────────────────
v_rand = np.random.randn(N)
v_rand -= (v_rand @ u_rec) * u_rec        # orthogonalise
e_fast = v_rand / norm(v_rand)            # eigen-value ≈ 0
mix    = (u_rec + e_fast) / np.sqrt(2)

# ─── 4. propagator for continuous-time decay  ṙ = −(I−W_R) r ────────
A    = -(np.eye(N) - W_R)
props = [expm(A*t) for t in t_vals]

def decay_curve(v):
    """return |vᵀ r(t)|² for r(0)=v, normalised to 1 at t=0"""
    return np.array([(v @ P @ v) for P in props]) / (v @ v)

dec_slow  = decay_curve(u_rec)
dec_fast  = decay_curve(e_fast)
dec_mix   = decay_curve(mix)

# ─── 5. plot on semi-log scale ───────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(6, 4.5))

ax.semilogy(t_vals, dec_slow, lw=2, c='tab:grey',
            label=r"$u_{\mathrm{rec}}\;(\lambda=\rho)$")
ax.semilogy(t_vals, dec_fast, lw=2, c='tab:orange',
            label=r"orthogonal $(\lambda=0)$")
ax.semilogy(t_vals, dec_mix,  lw=2, c='tab:green',
            label="mixture")

ax.set_xlabel("time  $t$")
ax.set_ylabel(r"normalised power  $|v^\top r(t)|^2$")
ax.set_title("Decay rates in a Dale-compliant EI network")
ax.legend(frameon=False)
ax.grid(False)
plt.tight_layout()
plt.savefig("decay_dale.png", dpi=300)
plt.savefig("decay_dale.svg")
plt.show()

print("Figure saved → decay_dale.png  /  decay_dale.svg")

# Save the data for reproducibility
decay_data = {
    't_vals': t_vals,
    'decay_slow': dec_slow,
    'decay_fast': dec_fast,
    'decay_mix': dec_mix,
    'parameters': {
        'N': N,
        'rho': rho,
        'T_max': T_max
    }
}

# Save to numpy compressed format
np.savez('decay_dale_data.npz', **decay_data)

print("Data saved → decay_dale_data.npz")
