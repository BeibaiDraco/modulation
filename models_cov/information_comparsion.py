import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve, eig

# ====================================================================
# 1. Common parameters and helper functions
# ====================================================================
np.random.seed(42)  # Consistent seed for both models
N = 120
rho = 0.4           # Spectral radius of recurrent matrix
sigma_eta = 1.0     # Private-noise standard deviation
beta = 1.0          # Stimulus amplitude difference
frac_E = 0.8        # Fraction of excitatory neurons (Dale only)
thetas = np.linspace(-90, 90, 181)  # Decoding angles

def compute_fisher_ratio(Delta_mu, Sigma):
    """Compute Fisher ratio for different decoding angles"""
    u_stim = Delta_mu.flatten() / norm(Delta_mu)
    v_rand = np.random.randn(N)
    v_rand -= v_rand.dot(u_stim) * u_stim
    u_perp = v_rand / norm(v_rand)
    
    S_th, N_th, J_th = [], [], []
    for th in thetas:
        theta_rad = np.deg2rad(th)
        v_decoder = np.cos(theta_rad) * u_stim + np.sin(theta_rad) * u_perp
        S_th.append((v_decoder @ Delta_mu)**2)
        N_th.append(v_decoder @ Sigma @ v_decoder)
        J_th.append(S_th[-1] / N_th[-1])
    
    return np.array(S_th), np.array(N_th), np.array(J_th)

# ====================================================================
# 2. Symmetric network model (information.py)
# ====================================================================
def run_symmetric_network():
    # Feed-forward vector
    w = np.zeros((N, 1))
    w[0] = 1.0
    w /= norm(w)
    
    # Recurrent matrix (rank-1 symmetric)
    W_R = rho * (w @ w.T)
    
    # Dynamics matrices
    I = np.eye(N)
    inv_I_W = solve(I - W_R, I)
    inv_I_WT = solve(I - W_R.T, I)
    
    # Statistics
    Delta_mu = beta * inv_I_W @ w
    Sigma = sigma_eta**2 * inv_I_W @ inv_I_WT
    
    return compute_fisher_ratio(Delta_mu, Sigma)

# ====================================================================
# 3. Dale-compliant network model (information_dale.py)
# ====================================================================
def run_dale_network():
    # Create magnitude profile
    x_pos = np.linspace(-1, 1, N)
    v_mag = np.exp(-4.0 * x_pos**2)
    v_mag /= norm(v_mag)
    
    # Apply Dale's law signs
    sign_vec = np.ones(N)
    sign_vec[int(frac_E * N):] = -1  # Inhibitory neurons
    z = (sign_vec * v_mag)[:, None]
    w_ff = z / norm(z)  # Feed-forward drive
    
    # Recurrent matrix (Dale-compliant)
    W_R = rho * (z @ v_mag[None, :])
    
    # Ensure spectral radius = rho
    evals = eig(W_R)[0]
    W_R *= rho / np.max(evals.real)
    
    # Dynamics matrices
    I = np.eye(N)
    inv_I_W = solve(I - W_R, I)
    
    # Statistics
    Delta_mu = inv_I_W @ w_ff
    Sigma = sigma_eta**2 * inv_I_W @ inv_I_W.T
    
    return compute_fisher_ratio(Delta_mu, Sigma)

# ====================================================================
# 4. Run both models
# ====================================================================
print("Running symmetric network model...")
S_sym, N_sym, J_sym = run_symmetric_network()

print("Running Dale-compliant network model...")
S_dale, N_dale, J_dale = run_dale_network()

# ====================================================================
# 5. Plotting
# ====================================================================
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Normalized Fisher ratio
plt.figure(figsize=(10, 5))
plt.plot(thetas, J_sym/J_sym.max(), 'b-', lw=2.5, 
         label='Symmetric Network')
plt.plot(thetas, J_dale/J_dale.max(), 'r-', lw=2.5, 
         label='Dale-Compliant Network')
plt.axvline(0, ls='--', c='gray', lw=1)
plt.title('Normalized Fisher Ratio Comparison', fontsize=14)
plt.xlabel('Decoding Angle (degrees)', fontsize=12)
plt.ylabel('Normalized $J(θ)$', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('normalized_fisher_comparison.png', dpi=300)
plt.savefig('normalized_fisher_comparison.svg')

# Plot 2: Unnormalized Fisher ratio
plt.figure(figsize=(10, 5))
plt.plot(thetas, J_sym, 'b-', lw=2.5, label='Symmetric Network')
plt.plot(thetas, J_dale, 'r-', lw=2.5, label='Dale-Compliant Network')
plt.axvline(0, ls='--', c='gray', lw=1)
plt.title('Unnormalized Fisher Ratio Comparison', fontsize=14)
plt.xlabel('Decoding Angle (degrees)', fontsize=12)
plt.ylabel('Fisher Ratio $J(θ)$', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('unnormalized_fisher_comparison.png', dpi=300)
plt.savefig('unnormalized_fisher_comparison.svg')

# Plot 3: Signal and Noise components
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Symmetric network
axs[0].plot(thetas, S_sym, 'g-', lw=2, label='Signal Power')
axs[0].plot(thetas, N_sym, 'm-', lw=2, label='Noise Variance')
axs[0].set_title('Symmetric Network', fontsize=13)
axs[0].set_xlabel('Angle (degrees)', fontsize=11)
axs[0].set_ylabel('Variance', fontsize=11)
axs[0].legend(fontsize=10)
axs[0].grid(True, alpha=0.2)

# Dale network
axs[1].plot(thetas, S_dale, 'g-', lw=2, label='Signal Power')
axs[1].plot(thetas, N_dale, 'm-', lw=2, label='Noise Variance')
axs[1].set_title('Dale-Compliant Network', fontsize=13)
axs[1].set_xlabel('Angle (degrees)', fontsize=11)
axs[1].legend(fontsize=10)
axs[1].grid(True, alpha=0.2)

plt.suptitle('Signal and Noise Components', fontsize=15)
plt.tight_layout()
plt.savefig('signal_noise_components.png', dpi=300)

plt.show()
print("All plots saved successfully!")