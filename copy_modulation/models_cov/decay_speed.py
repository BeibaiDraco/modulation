#!/usr/bin/env python3
"""
Decay along three directions in an *aligned* rank-1 network
----------------------------------------------------------
(1) u_rec           : eigen-value ρ  → slow
(2) e_fast          : eigen-value 0  → fast
(3) mix = (u_rec+e_fast)/√2 → intermediate (sum of two exponentials)
"""

import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import norm, solve
from scipy.linalg import expm

# ─── 1. network parameters ─────────────────────────────────────────────
np.random.seed(0)
N       = 120
rho     = 0.8            # leading eigen-value of W_R
T_max   = 10
t_vals  = np.arange(T_max+1)

# ─── 2. analytical eigen-basis of W_R  (rank-1) ────────────────────────
u_rec = np.zeros((N,1));  u_rec[0,0] = 1.0
u_rec /= norm(u_rec)                      # eigen-vector with λ=ρ

# pick one explicit vector orthogonal to u_rec  → eigen-value 0
e_fast = np.zeros((N,1));  e_fast[1,0] = 1.0
e_fast -= (u_rec.T @ e_fast) * u_rec
e_fast /= norm(e_fast)

# mixture to give a decay that is neither purely slow nor purely fast
mix = (u_rec + 2*e_fast) / np.sqrt(2)

# ─── 3. build matrices & propagators ───────────────────────────────────
W_R = rho * (u_rec @ u_rec.T)
A   = -(np.eye(N) - W_R)                  # generator  ẋ = A x
prop = [expm(A*t) for t in t_vals]        # exp(A t) list

def decay(v):
    """|vᵀ r(t)|² with initial r(0)=v (unit-impulse)"""
    return np.array([(v.T @ P @ v)[0,0]**2 for P in prop])

decay_slow  = decay(u_rec)
decay_fast  = decay(e_fast)
decay_mix   = decay(mix)

# ─── 4. plotting ───────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(6,4.5))

ax.semilogy(t_vals, decay_slow/decay_slow[0],
            lw=2, label="u_rec  (λ = ρ)", color="tab:grey")
ax.semilogy(t_vals, decay_fast/decay_fast[0],
            lw=2, label="orthogonal (λ = 0)", color="tab:orange")
ax.semilogy(t_vals, decay_mix/decay_mix[0],
            lw=2, label="mixture", color="tab:green")

ax.set_xlabel("time  $t$")
ax.set_ylabel(r"normalised power  $|v^\top r(t)|^2$")
ax.set_title("Decay rates: slow eigen-mode vs. fast mode vs. mixture")
ax.legend(frameon=False)
ax.grid(False)
plt.tight_layout()
plt.savefig("decay_eigen_vs_mix.png", dpi=300)
plt.savefig("decay_eigen_vs_mix.svg")
plt.show()
# Save the data for reproducibility
decay_data = {
    't_vals': t_vals,
    'decay_slow': decay_slow/decay_slow[0],
    'decay_fast': decay_fast/decay_fast[0],
    'decay_mix': decay_mix/decay_mix[0],
    'parameters': {
        'N': N,
        'rho': rho,
        'T_max': T_max
    }
}

# Save to numpy compressed format
np.savez('decay_eigen_vs_mix_data.npz', **decay_data)

print("Figure saved → decay_eigen_vs_mix.png")
