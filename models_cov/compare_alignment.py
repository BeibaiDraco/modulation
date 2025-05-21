#!/usr/bin/env python3
"""
Compare aligned vs misaligned cases
==================================
Shows two subplots:
1. 40° misalignment between stimulus and noise axes
2. Perfect alignment (0°) between stimulus and noise axes
"""

import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import norm, solve

def setup_network(N, rho, phi_deg, sigma_eta, trials, seed):
    """Setup network parameters and compute key vectors"""
    np.random.seed(seed)
    
    # Define noise axis u_rec (PC-1)
    u_rec = np.zeros((N, 1))
    u_rec[0, 0] = 1.0
    u_rec /= norm(u_rec)
    
    if phi_deg == 0:  # Aligned case
        w = u_rec.copy()  # already unit norm, keep raw amplitude
    else:  # Misaligned case
        # Pick orthogonal companion u_perp
        v_rand = np.random.randn(N, 1)
        v_rand -= (u_rec.T @ v_rand) * u_rec
        u_perp = v_rand / norm(v_rand)
        
        # Choose feed-forward vector w
        phi = np.deg2rad(phi_deg)
        c, s = np.cos(phi), np.sin(phi)
        
        g = rho / (1 - rho)
        a = c / (1 + g)
        b = s
        w = a * u_rec + b * u_perp
    
    # Build matrices and compute Δμ
    W_R = rho * (u_rec @ u_rec.T)
    inv = solve(np.eye(N) - W_R, np.eye(N))
    beta = 1.0
    Delta_mu = beta * (inv @ w)
    
    # Compute angle
    angle_real = np.rad2deg(
        np.arccos((Delta_mu.T @ u_rec) /
                  (norm(Delta_mu) * norm(u_rec)))).item()
    
    # Generate noise samples
    eta = sigma_eta * np.random.randn(trials, N)
    x_no = eta @ inv.T
    
    # Project onto 2D plane
    if phi_deg == 0:  # Aligned case
        # plane basis: e1 = u_rec, e2 arbitrary orthogonal unit vector
        v_rand = np.random.randn(N, 1)
        v_rand -= (u_rec.T @ v_rand) * u_rec
        e2 = v_rand.flatten() / norm(v_rand)
        proj_mat = np.vstack([u_rec.flatten(), e2]).T
    else:  # Misaligned case
        e1 = u_rec.flatten()
        dmu_vec = Delta_mu.flatten()
        e2 = dmu_vec - (dmu_vec @ e1) * e1
        e2 /= norm(e2)
        proj_mat = np.vstack([e1, e2]).T
    
    noise_xy = x_no @ proj_mat
    u_rec_xy = u_rec.flatten() @ proj_mat
    dmu_xy = Delta_mu.flatten() @ proj_mat
    
    return noise_xy, u_rec_xy, dmu_xy, angle_real

def draw_arrow(ax, vec, **kw):
    """Helper function to draw arrows"""
    ax.arrow(0, 0, vec[0], vec[1],
             head_width=0.08, length_includes_head=True, **kw)

# Parameters
N = 120
rho = 0.8
sigma_eta = 1.0
trials = 5000

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

# ─── Misaligned case (40°) ─────────────────────────────────────────────
noise_xy, u_rec_xy, dmu_xy, angle_real = setup_network(
    N, rho, 40, sigma_eta, trials, seed=1)

ax1.grid(False)
ax1.scatter(noise_xy[:, 0], noise_xy[:, 1],
           s=5, alpha=0.15, color='grey', label='noise trials')
#draw_arrow(ax1, 10 * u_rec_xy / norm(u_rec_xy), color='k', lw=2, label='noise PC-1')
draw_arrow(ax1, 10 * dmu_xy / norm(dmu_xy), color='red', lw=2, label='Stimulus Axis')

ax1.set_xlabel("Noise Axis")
ax1.set_ylabel("Stimulus Basis")
ax1.set_aspect('equal')
ax1.set_title(f"Stimulus Axis {angle_real:.1f}° from Noise Axis")

# ─── Aligned case (0°) ────────────────────────────────────────────────
noise_xy, u_rec_xy, dmu_xy, angle_real = setup_network(
    N, rho, 0, sigma_eta, trials, seed=2)

ax2.grid(False)
ax2.scatter(noise_xy[:, 0], noise_xy[:, 1],
           s=5, alpha=0.15, color='grey', label='noise trials')
#draw_arrow(ax2, 10 * u_rec_xy / norm(u_rec_xy), color='k', lw=2, label='noise / stimulus axis')
draw_arrow(ax2, 10 * dmu_xy / norm(dmu_xy), color='red', lw=2, label='Stimulus Axis')

ax2.set_xlabel("Noise Axis")
ax2.set_ylabel("Orthogonal Axis")
ax2.set_aspect('equal')
ax2.set_title("Stimulus Axis = Noise Axis")

# Calculate the overall limits for both plots
x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

# Make x and y ranges independently symmetric
x_range = max(abs(x_min), abs(x_max))
y_range = max(abs(y_min), abs(y_max))
x_min = -x_range
x_max = x_range
y_min = -y_range
y_max = y_range

# Apply the same limits to both plots
for ax in [ax1, ax2]:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# Add legends to each subplot
for ax in [ax1, ax2]:
    legend = ax.legend(loc='upper right', fontsize=8)
    # Make the lines in the legend thinner
    for line in legend.get_lines():
        line.set_linewidth(1.0)

plt.tight_layout()
plt.savefig('compare_alignment.png', dpi=300)
# Save the figure in SVG format
plt.savefig('compare_alignment.svg', bbox_inches='tight')

# Save the data for plotting
data_misaligned = {
    'noise_xy': noise_xy,
    'u_rec_xy': u_rec_xy,
    'dmu_xy': dmu_xy,
    'angle_real': angle_real
}

data_aligned = {
    'noise_xy': noise_xy,  # This is from the aligned case (second setup_network call)
    'u_rec_xy': u_rec_xy,
    'dmu_xy': dmu_xy,
    'angle_real': angle_real
}

# Save the data to a numpy file
np.savez('compare_alignment_data.npz', 
         misaligned=data_misaligned,
         aligned=data_aligned)

print("Figure saved as 'compare_alignment.png' and 'compare_alignment.svg'")
print("Data saved as 'compare_alignment_data.npz'")
# Save the data to CSV files for easier analysis
# Misaligned data
np.savetxt('misaligned_data.csv', 
           np.column_stack((
               noise_xy[:, 0], noise_xy[:, 1],  # Noise projections
               np.full(noise_xy.shape[0], u_rec_xy[0]),  # Repeat u_rec_xy for each row
               np.full(noise_xy.shape[0], u_rec_xy[1]),
               np.full(noise_xy.shape[0], dmu_xy[0]),    # Repeat dmu_xy for each row
               np.full(noise_xy.shape[0], dmu_xy[1]),
               np.full(noise_xy.shape[0], angle_real)    # Repeat angle for each row
           )),
           delimiter=',',
           header='noise_x,noise_y,u_rec_x,u_rec_y,dmu_x,dmu_y,angle_real',
           comments='')

# Aligned data (using the same variable names since they're from the aligned case)
np.savetxt('aligned_data.csv', 
           np.column_stack((
               noise_xy[:, 0], noise_xy[:, 1],  # Noise projections
               np.full(noise_xy.shape[0], u_rec_xy[0]),  # Repeat u_rec_xy for each row
               np.full(noise_xy.shape[0], u_rec_xy[1]),
               np.full(noise_xy.shape[0], dmu_xy[0]),    # Repeat dmu_xy for each row
               np.full(noise_xy.shape[0], dmu_xy[1]),
               np.full(noise_xy.shape[0], angle_real)    # Repeat angle for each row
           )),
           delimiter=',',
           header='noise_x,noise_y,u_rec_x,u_rec_y,dmu_x,dmu_y,angle_real',
           comments='')

print("CSV data saved as 'misaligned_data.csv' and 'aligned_data.csv'")


plt.show() 