import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 2D Optimal Alignment: Theoretical and Grid Search Verification
# =============================================================================

def theoretical_new_angle(x_rad, d):
    """
    For a 2D color axis d_color = [cos x, sin x] and gain bounds
    g1 in [1-d, 1+d] and g2 in [1-d, 1+d], the optimal gain to maximize
    alignment with PC1 = [1, 0] is to take g1 = 1+d (boost the x component)
    and g2 = 1-d (shrink the y component). Then the new angle is:
      theta_new = arctan( ((1-d)/(1+d))*tan(x) )
    """
    return np.arctan(((1-d)/(1+d)) * np.tan(x_rad))

def optimize_gain_2d(x_rad, L, U, steps=200):
    """
    Grid search over g1 and g2 in [L, U] (for a 2D vector)
    to find the gain vector that minimizes the angle between
    w = [g1*cos(x), g2*sin(x)] and PC1 = [1, 0].
    
    Returns:
      best_angle: the minimal new angle (in radians)
      best_g: (g1, g2) that achieved that angle.
    """
    grid = np.linspace(L, U, steps)
    best_angle = np.pi
    best_g = (None, None)
    for g1 in grid:
        for g2 in grid:
            # New vector after gain modulation:
            w = np.array([g1 * np.cos(x_rad), g2 * np.sin(x_rad)])
            # The angle relative to [1,0] is:
            angle = np.arctan2(np.abs(w[1]), w[0])  # w[0]>0 if cos(x) > 0 and g1>0
            if angle < best_angle:
                best_angle = angle
                best_g = (g1, g2)
    return best_angle, best_g

def test_and_plot_2d(x_deg, d, steps=200):
    """
    For a given initial angle x (in degrees) and gain deviation d,
    (1) compute the theoretical new angle,
    (2) run the grid search to find the optimal gains,
    (3) print the results,
    (4) plot the original vector and the optimally rotated one.
    """
    x_rad = np.deg2rad(x_deg)
    L = 1 - d
    U = 1 + d
    
    # Theoretical new angle (in radians)
    theta_new_theo = theoretical_new_angle(x_rad, d)
    theta_new_theo_deg = np.rad2deg(theta_new_theo)
    
    # Grid search to find optimal gains
    best_angle, best_g = optimize_gain_2d(x_rad, L, U, steps=steps)
    best_angle_deg = np.rad2deg(best_angle)
    
    print(f"--- 2D Optimization for x = {x_deg}° and d = {d:.2f} ---")
    print(f"Theoretical new angle: {theta_new_theo_deg:.2f}°")
    print(f"Optimal (grid search) new angle: {best_angle_deg:.2f}°")
    print(f"Optimal gains: g1 = {best_g[0]:.3f}, g2 = {best_g[1]:.3f}")
    
    # Plot the original and modulated vectors
    d_color = np.array([np.cos(x_rad), np.sin(x_rad)])
    # Use the optimal gain vector from grid search:
    w = np.array([best_g[0] * np.cos(x_rad), best_g[1] * np.sin(x_rad)])
    
    # Normalize for plotting (only direction matters)
    d_color_unit = d_color / np.linalg.norm(d_color)
    w_unit = w / np.linalg.norm(w)
    
    plt.figure(figsize=(5,5))
    origin = np.array([0, 0])
    plt.quiver(*origin, d_color_unit[0], d_color_unit[1],
               angles='xy', scale_units='xy', scale=1,
               color='b', label='d_color (original)')
    plt.quiver(*origin, w_unit[0], w_unit[1],
               angles='xy', scale_units='xy', scale=1,
               color='r', label='w = diag(g)*d_color (optimal)')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"2D Optimal Alignment: x = {x_deg}°, d = {d:.2f}")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_optimal_improvement_vs_d(x_deg, d_values, steps=200):
    """
    For a fixed initial angle x (in degrees), and for a set of d-values,
    run the grid search optimization and compare the optimal new angle (in degrees)
    with the theoretical prediction. Plot the improvement (x - theta_new).
    """
    improvements_theo = []
    improvements_opt = []
    
    x_rad = np.deg2rad(x_deg)
    for d in d_values:
        L = 1 - d
        U = 1 + d
        theta_new_theo = theoretical_new_angle(x_rad, d)
        improvements_theo.append(x_deg - np.rad2deg(theta_new_theo))
        
        best_angle, _ = optimize_gain_2d(x_rad, L, U, steps=steps)
        improvements_opt.append(x_deg - np.rad2deg(best_angle))
    
    improvements_theo = np.array(improvements_theo)
    improvements_opt = np.array(improvements_opt)
    
    plt.figure(figsize=(8,6))
    plt.plot(d_values, improvements_theo, 'r--', label='Theoretical improvement')
    plt.plot(d_values, improvements_opt, 'bo', label='Optimal (grid search)')
    plt.xlabel("Gain deviation d (with L=1-d, U=1+d)")
    plt.ylabel("Improvement in alignment (°)")
    plt.title(f"Optimal Alignment Improvement vs. Gain Range (x = {x_deg}°)")
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# Main: Run tests and plots for the 2D case
# =============================================================================

def main_2d():
    # Test a single example: choose an initial angle x and a gain deviation d.
    x_deg = 30   # initial angle in degrees
    d = 0.1      # allowed gain deviation: gains in [0.9, 1.1]
    test_and_plot_2d(x_deg, d, steps=200)
    
    # Now plot the improvement vs. gain range for x = 30°.
    d_values = np.linspace(0, 0.5, 50)
    plot_optimal_improvement_vs_d(x_deg, d_values, steps=100)

if __name__ == "__main__":
    main_2d()
