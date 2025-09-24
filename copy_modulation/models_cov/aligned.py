#!/usr/bin/env python3
"""
Aligned case: stimulus axis == noise axis
========================================
Everything plotted in the plane spanned by
   • u_rec  (noise / Perron eigen-vector)
   • Δμ     (stimulus mean difference = same axis here)
"""

import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import norm, solve

# ─── 1. parameters ─────────────────────────────────────────────────────
np.random.seed(2)          # different seed just for variety
N          = 120
rho        = 0.8           # same recurrent “slowness”
phi_deg    = 0             # ***aligned!***
sigma_eta  = 1.0
trials     = 5000

# ─── 2. noise axis  u_rec  ─────────────────────────────────────────────
u_rec = np.zeros((N, 1));  u_rec[0, 0] = 1.0
u_rec /= norm(u_rec)

# ─── 3. feed-forward vector  w  perfectly parallel to u_rec  ───────────
w = u_rec.copy()           # already unit norm, keep raw amplitude

# ─── 4. build matrices, compute Δμ, angle check ────────────────────────
W_R = rho * (u_rec @ u_rec.T)
inv  = solve(np.eye(N) - W_R, np.eye(N))   # (I − W)⁻¹
beta = 1.0
Delta_mu = beta * (inv @ w)

angle_real = np.rad2deg(
    np.arccos((Delta_mu.T @ u_rec) /
              (norm(Delta_mu) * norm(u_rec)))).item()
print(f"target angle = {phi_deg}°, realised = {angle_real:.2f}°")

Sigma = sigma_eta**2 * inv @ inv.T

# ─── 5. noise samples and projection onto plane (u_rec, Δμ⊥) ───────────
eta  = sigma_eta * np.random.randn(trials, N)
x_no = eta @ inv.T

# plane basis: e1 = u_rec, e2 arbitrary orthogonal unit vector
v_rand = np.random.randn(N, 1)
v_rand -= (u_rec.T @ v_rand) * u_rec
e2 = v_rand.flatten() / norm(v_rand)
proj_mat = np.vstack([u_rec.flatten(), e2]).T

noise_xy = x_no @ proj_mat
u_rec_xy = u_rec.flatten() @ proj_mat
dmu_xy   = Delta_mu.flatten() @ proj_mat      # should be ± same as u_rec_xy

# ─── 6. plot ───────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(noise_xy[:, 0], noise_xy[:, 1],
           s=5, alpha=0.15, color='grey', label='noise trials')

def draw_arrow(vec, **kw):
    ax.arrow(0, 0, vec[0], vec[1],
             head_width=0.08, length_includes_head=True, **kw)

draw_arrow(10 * u_rec_xy / norm(u_rec_xy), color='k',   lw=2, label='noise / stimulus axis')
draw_arrow(10 * dmu_xy   / norm(dmu_xy),   color='red', lw=2, label='same axis (Δμ)')

ax.set_xlabel("axis 1  (u_rec = stimulus)")
ax.set_ylabel("axis 2  (orthogonal)")
ax.set_aspect('equal')
ax.set_title("Perfect alignment: stimulus axis = noise axis")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
