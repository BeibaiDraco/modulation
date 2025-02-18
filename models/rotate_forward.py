import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh, norm

# =============================================================================
# Helper functions
# =============================================================================

def generate_covariance_matrix(N, eigenvalues=None):
    """
    Generate an N x N covariance matrix with prescribed eigenvalues.
    We first sample a random orthonormal matrix Q (via QR) and then set:
        C = Q * D * Q.T
    where D is diagonal with the eigenvalues.
    """
    Q, _ = np.linalg.qr(np.random.randn(N, N))
    if eigenvalues is None:
        # Choose eigenvalues with a clear gap: e.g., largest 10, second 5, rest 1.
        eigenvalues = np.concatenate(([10, 5], np.ones(N-2)))
    D = np.diag(eigenvalues)
    C = Q @ D @ Q.T
    return C, eigenvalues

def principal_eigen(C):
    """
    Compute the largest eigenvalue and its corresponding eigenvector of symmetric C.
    (We use np.linalg.eigh, which returns sorted eigenvalues.)
    """
    evals, evecs = eigh(C)
    return evals[-1], evecs[:, -1]

# =============================================================================
# Experiment 1: Gain modulation in full-dimensional space
# =============================================================================

def simulate_gain_modulation(N=20, gain_strength=1.1):
    """
    (a) Create a random covariance matrix C for an N-neuron population.
    (b) Compute its principal eigenvector v1.
    (c) Define a "color axis" d_color that is nearly aligned with v1 but with a small
        component along v2.
    (d) Apply a diagonal gain G that (for example) boosts neurons with positive d_color.
    (e) Compute the new covariance C' = G C G and its top eigenvector.
    (f) Also compute the modulated color axis w = G*d_color.
    (g) Plot the vectors projected into the v1–v2 plane.
    """
    # (a) Generate a covariance matrix.
    C, evals = generate_covariance_matrix(N)
    # (b) Compute the top eigenvector of C.
    lam1, v1 = principal_eigen(C)
    
    # (c) Define d_color as a combination of v1 and v2.
    evals_sorted, evecs_sorted = eigh(C)
    v1 = evecs_sorted[:, -1]   # largest eigenvector (PC1)
    v2 = evecs_sorted[:, -2]   # second-largest eigenvector
    alpha1 = 1.0
    alpha2 = 0.2  # small admixture so that d_color is nearly along v1.
    d_color = alpha1 * v1 + alpha2 * v2
    d_color = d_color / norm(d_color)
    
    # Compute the initial angle between d_color and v1.
    cos_initial = np.abs(np.dot(v1, d_color))
    angle_initial = np.arccos(cos_initial) * 180/np.pi
    print("Initial angle between d_color and v1: {:.2f}°".format(angle_initial))
    
    # (d) Define a gain modulation.
    # For example, if d_color[i] is positive, set g_i = gain_strength, else 1/gain_strength.
    g = np.ones(N)
    for i in range(N):
        if d_color[i] > 0:
            g[i] = gain_strength
        else:
            g[i] = 1.0 / gain_strength
    G = np.diag(g)
    
    # (e) Compute the modulated covariance and its principal eigenvector.
    C_mod = G @ C @ G
    lam1_mod, v1_mod = principal_eigen(C_mod)
    
    # (f) The modulated color axis is w = G*d_color.
    w = g * d_color
    w = w / norm(w)
    
    # (g) Report and compare the angles.
    angle_mod = np.arccos(np.abs(np.dot(v1, w))) * 180/np.pi
    angle_v1 = np.arccos(np.abs(np.dot(v1, v1_mod))) * 180/np.pi
    print("After gain modulation:")
    print("  Angle between modulated color axis (w) and original v1: {:.2f}°".format(angle_mod))
    print("  Angle between new PC1 (from C') and original v1: {:.2f}°".format(angle_v1))
    
    # For visualization we project our vectors onto the 2D plane spanned by v1 and v2.
    def proj_2d(vec):
        return np.array([np.dot(vec, v1), np.dot(vec, v2)])
    v1_2d      = proj_2d(v1)
    d_color_2d = proj_2d(d_color)
    w_2d       = proj_2d(w)
    v1_mod_2d  = proj_2d(v1_mod)
    
    plt.figure(figsize=(6,6))
    plt.quiver(0, 0, v1_2d[0], v1_2d[1], angles='xy', scale_units='xy', scale=1,
               color='k', label='v1 (original PC1)')
    plt.quiver(0, 0, d_color_2d[0], d_color_2d[1], angles='xy', scale_units='xy', scale=1,
               color='b', label='d_color (original)')
    plt.quiver(0, 0, w_2d[0], w_2d[1], angles='xy', scale_units='xy', scale=1,
               color='g', label='w = G*d_color')
    plt.quiver(0, 0, v1_mod_2d[0], v1_mod_2d[1], angles='xy', scale_units='xy', scale=1,
               color='r', label='v1 (modulated)')
    
    plt.xlabel("Component along v1")
    plt.ylabel("Component along v2")
    plt.title("Vector alignments in the v1-v2 plane")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# =============================================================================
# Experiment 2: 2D Optimal Alignment under Gain Constraints
# =============================================================================

def test_optimal_alignment_2d(x_deg, L, U):
    """
    In a 2D setting, let the original color axis be
        d_color = [cos(x), sin(x)]
    so that its angle from the x-axis (PC1) is x.
    
    With independent gains, the modulated vector is:
         w = [g1*cos(x), g2*sin(x)].
    
    To maximize alignment with PC1, we choose g1 = U (boosting PC1 component)
    and g2 = L (shrinking the PC2 component). Then the new angle is:
         theta_new = arctan((L/U)*tan(x)).
    
    This function plots the original and modulated d_color.
    """
    x = np.deg2rad(x_deg)
    d = np.array([np.cos(x), np.sin(x)])  # original vector (PC1 = [1,0])
    w = np.array([U * np.cos(x), L * np.sin(x)])  # modulated vector
    theta_new = np.arctan2(L * np.sin(x), U * np.cos(x))
    theta_new_deg = np.rad2deg(theta_new)
    
    print("2D Example:")
    print("  Original angle: {}°  -->  Optimal new angle: {:.2f}°".format(x_deg, theta_new_deg))
    
    plt.figure(figsize=(5,5))
    origin = np.array([0, 0])
    plt.quiver(*origin, d[0], d[1], angles='xy', scale_units='xy', scale=1,
               color='b', label='d_color (original)')
    plt.quiver(*origin, w[0], w[1], angles='xy', scale_units='xy', scale=1,
               color='r', label='w = diag(g)*d_color')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Optimal Alignment in 2D (x = {}°)".format(x_deg))
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# =============================================================================
# Experiment 3: Plot Gain Range vs. Maximum Alignment Improvement
# =============================================================================

def plot_gain_range_vs_improvement():
    """
    For a fixed original angle x (or multiple choices), suppose that the allowed gains
    are in the range [1-d, 1+d]. The theory predicts that the best attainable new angle is:
         theta_new = arctan((1-d)/(1+d) * tan(x)).
    Hence, the improvement in alignment (in degrees) is:
         Delta theta = x - theta_new.
    
    This function plots Delta theta vs. d for several values of x.
    """
    d_values = np.linspace(0, 0.5, 100)  # d from 0 (no gain change) to 0.5 (±50% change)
    angles_deg = [10, 30, 60, 80]  # original angles in degrees
    plt.figure(figsize=(8,6))
    for x_deg in angles_deg:
        improvements = []
        for d in d_values:
            L = 1 - d
            U = 1 + d
            theta_new = np.arctan((L/U)*np.tan(np.deg2rad(x_deg)))
            improvement = x_deg - np.rad2deg(theta_new)
            improvements.append(improvement)
        improvements = np.array(improvements)
        plt.plot(d_values, improvements, label=f'Initial x = {x_deg}°')
    plt.xlabel('Gain deviation d (with L = 1-d, U = 1+d)')
    plt.ylabel('Improvement in alignment (°)')
    plt.title('Maximum Improvement vs. Gain Range')
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# Main: run experiments and plots
# =============================================================================

def main():
    # --- Experiment 1: Full-dimensional simulation ---
    print("=== Full-dimensional gain modulation simulation ===")
    simulate_gain_modulation(N=20, gain_strength=1.1)
    
    # --- Experiment 2: 2D optimal alignment ---
    print("\n=== 2D Optimal Alignment Examples ===")
    angles_deg = [10, 30, 60, 80]
    L = 0.9  # lower bound for gain
    U = 1.1  # upper bound for gain
    for x_deg in angles_deg:
        test_optimal_alignment_2d(x_deg, L, U)
    
    # --- Experiment 3: Plot gain range vs. maximum alignment improvement ---
    print("\n=== Gain Range vs. Maximum Alignment Improvement ===")
    plot_gain_range_vs_improvement()

if __name__ == "__main__":
    main()
