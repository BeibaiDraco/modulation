#!/usr/bin/env python3
"""
Visualise 40° mis-alignment between stimulus axis and noise axis
================================================================
Everything is shown in the 2-D plane spanned by
   • u_rec  (noise / Perron eigen-vector)
   • Δμ     (stimulus mean difference)
so the angle is exact, not a projection artefact.
"""

import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import norm, solve

# ─── 1. parameters ─────────────────────────────────────────────────────
np.random.seed(1)
N          = 120          # neurons
rho        = 0.8          # spectral radius of W_R
phi_deg    = 40           # desired angle between Δμ and u_rec
sigma_eta  = 1.0          # private noise SD
trials     = 5000        # noise samples

# ─── 2. define noise axis  u_rec  (PC-1) ───────────────────────────────
u_rec = np.zeros((N, 1));  u_rec[0, 0] = 1.0
u_rec /= norm(u_rec)

# pick an orthogonal companion  u_perp
v_rand = np.random.randn(N, 1)
v_rand -= (u_rec.T @ v_rand) * u_rec
u_perp  = v_rand / norm(v_rand)

# ─── 3. choose feed-forward vector w so that  Δμ  is φ° away ───────────
phi  = np.deg2rad(phi_deg)
c, s = np.cos(phi), np.sin(phi)

g    = rho / (1 - rho)                  # gain of (I − W)⁻¹ along u_rec
a    = c / (1 + g)                      # coefficient on u_rec  before inverse
b    = s                                # coefficient on u_perp before inverse
w    = a * u_rec + b * u_perp           # **do NOT renormalise**

# ─── 4. build matrices, compute Δμ and Σ ───────────────────────────────
W_R  = rho * (u_rec @ u_rec.T)
inv  = solve(np.eye(N) - W_R, np.eye(N))   # (I − W)⁻¹

beta      = 1.0
Delta_mu  = beta * (inv @ w)               # (N,1)
angle_real = np.rad2deg(
    np.arccos((Delta_mu.T @ u_rec) /
              (norm(Delta_mu) * norm(u_rec)))).item()
print(f"target angle = {phi_deg}°, realised = {angle_real:.2f}°")

Sigma = sigma_eta**2 * inv @ inv.T         # noise covariance

# ─── 5. noise samples and projection onto the 2-D plane ────────────────
eta   = sigma_eta * np.random.randn(trials, N)
x_no  = eta @ inv.T

# orthonormal basis of the plane:  e1 = u_rec,  e2 = Δμ⊥
e1 = u_rec.flatten()
dmu_vec = Delta_mu.flatten()
e2 = dmu_vec - (dmu_vec @ e1) * e1         # make orthogonal to e1
e2 /= norm(e2)

proj_mat = np.vstack([e1, e2]).T           # N×2

noise_xy = x_no @ proj_mat                 # (trials,2)
u_rec_xy = e1 @ proj_mat                   # → (1,2) = (1,0)
dmu_xy   = dmu_vec @ proj_mat

# ─── 6. plot ───────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(noise_xy[:, 0], noise_xy[:, 1],
           s=5, alpha=0.15, color='grey', label='noise trials')

def draw_arrow(vec, **kw):
    ax.arrow(0, 0, vec[0], vec[1],
             head_width=0.08, length_includes_head=True, **kw)

draw_arrow(10 * u_rec_xy / norm(u_rec_xy), color='k',   lw=2, label='noise PC-1')
draw_arrow(10 * dmu_xy   / norm(dmu_xy),   color='red', lw=2, label='stimulus axis')

ax.set_xlabel("axis 1  (u_rec)")
ax.set_ylabel("axis 2  (Δμ)")
ax.set_aspect('equal')
ax.set_title(f"Stimulus axis {phi_deg}° from noise axis\n(in their own plane)")
ax.legend(frameon=False)
plt.tight_layout();  plt.show()
