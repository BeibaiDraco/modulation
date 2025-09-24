import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting in some environments

# -------------------------
# Parameters
# -------------------------
np.random.seed(15)
N = 300           # Number of neurons
K = 2             # Two features: shape (0) and color (1)
num_stimuli = 10  # Number of stimuli per feature dimension
num_noise_trials = 50  # Number of noisy trials per stimulus
noise_level = 1       # Noise magnitude
desired_radius = 0.9    # Desired spectral radius for stability scaling
p_high = 0.25
p_low = 0.05

# -------------------------------------------------
# 1) Initialization Functions
# -------------------------------------------------
def initialize_selectivity_matrix(N, K):
    """
    Initialize the selectivity matrix S with constraints.
    First half of neurons selective mainly to shape,
    the second half mirrors this for color.
    """
    S = np.zeros((N, K))
    # First half selectivity
    S[:N//2, 0] = np.random.rand(N//2)
    S[:N//2, 1] = 0.5 - S[:N//2, 0] / 2

    # Adjust negative indices
    negative_indices = (S[:N//2, 0] - S[:N//2, 1]) < 0
    S[:N//2, 0][negative_indices] = np.random.uniform(0, 0.5, size=np.sum(negative_indices))

    # Mirror for second half
    S[N//2:, 1] = S[:N//2, 0]
    S[N//2:, 0] = S[:N//2, 1]
    return S

def initialize_W_F(S):
    """
    Initialize W_F based on the selectivity matrix S.
    """
    W_F = np.zeros_like(S)
    for i in range(S.shape[0]):
        row_sum = np.sum(S[i])
        if row_sum > 0:
            W_F[i] = S[i] / row_sum
        else:
            W_F[i] = S[i]
    return W_F

def initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=0.9):
    """
    Initialize the recurrent connectivity matrix W_R.
    """
    W_R = np.zeros((N, N))
    half_N = N // 2

    # Shape-shape block
    shape_shape_mask = np.random.rand(half_N, half_N) < p_high
    W_R[:half_N, :half_N][shape_shape_mask] = np.random.rand(np.sum(shape_shape_mask)) * 0.1

    # Shape-color block
    shape_color_mask = np.random.rand(half_N, N - half_N) < p_low
    W_R[:half_N, half_N:][shape_color_mask] = np.random.rand(np.sum(shape_color_mask)) * 0.1

    # Color-shape block
    color_shape_mask = np.random.rand(N - half_N, half_N) < p_low
    W_R[half_N:, :half_N][color_shape_mask] = np.random.rand(np.sum(color_shape_mask)) * 0.1

    # Color-color block
    color_color_mask = np.random.rand(N - half_N, N - half_N) < p_high
    W_R[half_N:, half_N:][color_color_mask] = np.random.rand(np.sum(color_color_mask)) * 0.1

    # No self-connections
    np.fill_diagonal(W_R, 0)

    # Optional tuning
    if WR_tuned:
        threshold = 0.2
        for i in range(N):
            for j in range(N):
                if i != j:
                    distance = np.linalg.norm(S[i] - S[j])
                    if distance < threshold:
                        W_R[i, j] *= (2 - distance / threshold)

    # Scale W_R to ensure stability (spectral radius <= desired_radius)
    eigenvalues = np.linalg.eigvals(W_R)
    max_eval = np.max(np.abs(eigenvalues))
    if max_eval > 0:
        W_R *= (desired_radius / max_eval)

    return W_R

def initialize_W_R_cycle(N, desired_radius=0.9):
    """
    Cycle W_R: each neuron connects to the next in a ring with weight = desired_radius.
    """
    W_R = np.zeros((N, N))
    for i in range(N):
        W_R[i, (i + 1) % N] = desired_radius
    return W_R

# -------------------------------------------------
# 2) Response Computations (Unmodulated)
# -------------------------------------------------
def compute_responses(W_F, W_R, shape_stimuli, color_stimuli):
    """
    Compute the network responses for a grid of stimuli.
    Returns:
      responses: (num_stimuli * num_stimuli, N)
      stimuli_grid: (num_stimuli * num_stimuli, 2)
    """
    stimuli_grid = np.array(np.meshgrid(shape_stimuli, color_stimuli)).T.reshape(-1, 2)
    inv_I_minus_WR = np.linalg.inv(np.eye(N) - W_R)
    responses = np.zeros((len(stimuli_grid), N))

    for idx, (shape_val, color_val) in enumerate(stimuli_grid):
        F = np.array([shape_val, color_val])
        adjusted_F = W_F @ F
        responses[idx] = inv_I_minus_WR @ adjusted_F

    return responses, stimuli_grid

def generate_noisy_responses(W_R, noise_level, stimuli_grid, num_noise_trials):
    """
    Generate noise-only responses in unmodulated form.
    Returns shape = (num_stimuli_grid * num_noise_trials, N)
    """
    inv_I_minus_WR = np.linalg.inv(np.eye(N) - W_R)
    noisy_responses = []

    for (shape_val, color_val) in stimuli_grid:
        for _ in range(num_noise_trials):
            noise_input = np.random.randn(N) * noise_level
            noisy_responses.append(inv_I_minus_WR @ noise_input)

    return np.array(noisy_responses)

# -------------------------------------------------
# 3) Modulated Responses
# -------------------------------------------------
def compute_modulated_responses(W_R, W_F, S, stimuli_grid):
    """
    Compute modulated grid responses using G = diag(1 + 0.15*(color-shape)).
    """
    selectivity_diff = S[:, 1] - S[:, 0]
    g_vector = 1.0 + 0.2 * selectivity_diff  # can adjust factor

    G = np.diag(g_vector)

    I = np.eye(N)
    I_minus_GWR = I - G @ W_R

    # Check invertibility
    cond_number = np.linalg.cond(I_minus_GWR)
    if cond_number > 1 / np.finfo(I_minus_GWR.dtype).eps:
        raise ValueError("(I - G W_R) is nearly singular.")

    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)
    G_WF = G @ W_F

    mod_responses = np.zeros((len(stimuli_grid), N))
    for idx, (shape_val, color_val) in enumerate(stimuli_grid):
        F = np.array([shape_val, color_val])
        mod_responses[idx] = inv_I_minus_GWR @ (G_WF @ F)
        
    return np.array(mod_responses)


def compute_modulated_responses_pc3(
    W_R,
    W_F,
    S,
    stimuli_grid,
    pc3,          # principal component 3 (length N)
    alpha=0.3     # how strongly to weight color × pc3 overlap
):
    """
    Compute modulated grid responses, but now color is preferentially 
    amplified in the direction of PC3 (rather than PC1).

    Steps:
      1) pc3_unit = pc3 / ||pc3||
      2) Let color_pref = S[:,1]
      3) For each neuron i, define gain_i = 1 + alpha * color_pref[i] * |pc3_unit[i]|
         (or possibly keep the sign of pc3_unit[i] if you prefer)
      4) G = diag(gain_i)
      5) Final response = (I - G W_R)^(-1) * G * W_F * F
    """
    N = W_R.shape[0]
    I = np.eye(N)

    # (1) Normalize pc3
    pc3_unit = pc3 / np.linalg.norm(pc3)

    # (2) Extract color preference
    color_pref = S[:, 1]  # how much neuron i prefers "color"

    # (3) Build gain vector:
    #     Option A: use absolute value of pc3 so that any neuron with large |pc3[i]| gets boosted
    #     Option B: keep the sign of pc3[i] if you want “push–pull” along pc3
    overlap_pc3 = np.abs(pc3_unit)  # or just pc3_unit if you want sign

    # Gains: 1 + alpha * (color) * (overlap)
    g_vector = 1.0 + alpha * color_pref * overlap_pc3

    # Build G
    G = np.diag(g_vector)

    # (4) Solve (I - G W_R) and check invertibility
    I_minus_GWR = I - G @ W_R
    cond_number = np.linalg.cond(I_minus_GWR)
    if cond_number > 1 / np.finfo(I_minus_GWR.dtype).eps:
        raise ValueError("(I - G W_R) is nearly singular.")

    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)
    G_WF = G @ W_F

    # (5) Compute final modulated responses for each (shape_val, color_val)
    mod_responses = np.zeros((len(stimuli_grid), N))
    for idx, (shape_val, color_val) in enumerate(stimuli_grid):
        F = np.array([shape_val, color_val])
        mod_responses[idx] = inv_I_minus_GWR @ (G_WF @ F)

    return mod_responses


def compute_modulated_noise(W_R, W_F, S, stimuli_grid, noise_level, num_noise_trials):
    """
    Similar to generate_noisy_responses, but with gain modulation G.
    For each stimulus, we create noise input and pass it through (I - G W_R)^(-1) * G.
    """
    selectivity_diff = S[:, 1] - S[:, 0]
    g_vector = 1.0 + 0.15 * selectivity_diff  # can adjust factor
    G = np.diag(g_vector)

    I = np.eye(N)
    I_minus_GWR = I - G @ W_R
    inv_I_minus_GWR = np.linalg.inv(I_minus_GWR)

    noisy_responses = []
    for (shape_val, color_val) in stimuli_grid:
        for _ in range(num_noise_trials):
            noise_input = np.random.randn(N) * noise_level
            modded_noise = G @ noise_input
            noisy_responses.append(inv_I_minus_GWR @ modded_noise)

    return np.array(noisy_responses)



def set_axes_equal(ax):
    """
    Set 3D plot axes to have equal scale so that spheres appear as spheres,
    or cubes as cubes.  This is one of the simplest methods to accomplish
    this that works for all three axes.
    """
    import numpy as np
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # Set plot ranges so they are all the same
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

# -------------------------------------------------
# 4) Visualization in 3D: 4 Subplots
# -------------------------------------------------
def visualize_four_subplots_3d(
    responses_grid_unmod,    # shape (grid_size, N)
    responses_noise_unmod,   # shape (grid_size * num_noise_trials, N)
    responses_grid_mod,      # shape (grid_size, N)
    responses_noise_mod,     # shape (grid_size * num_noise_trials, N)
    stimuli_grid,
    title_main
):
    """
    4 subplots (2x2) in 3D, each with BOTH grid and noise:
      1) PCA from unmod GRID; plot unmod GRID (colored) + unmod NOISE
      2) PCA from unmod NOISE; plot unmod NOISE (gray) + unmod GRID
      3) Re-use PCA from unmod GRID, but plot mod GRID + mod NOISE
      4) Re-use PCA from unmod NOISE, but plot mod GRID + mod NOISE
    """

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title_main, fontsize=16)

    # ========== 1) PCA from Unmod GRID (3D) ==========
    pca_grid = PCA(n_components=3)
    pca_grid.fit(responses_grid_unmod)  # fit on unmod grid only

    # Transform both unmod grid & noise
    grid_unmod_3d = pca_grid.transform(responses_grid_unmod)
    noise_unmod_3d_in_grid = pca_grid.transform(responses_noise_unmod)

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(
        grid_unmod_3d[:, 0],
        grid_unmod_3d[:, 1],
        grid_unmod_3d[:, 2],
        c=stimuli_grid[:, 1],  # color dimension from the second feature
        cmap='winter',
        s=30,
        alpha=0.8,
        label='Unmod Grid'
    )
    ax1.scatter(
        noise_unmod_3d_in_grid[:, 0],
        noise_unmod_3d_in_grid[:, 1],
        noise_unmod_3d_in_grid[:, 2],
        c='gray',
        alpha=0.3,
        s=10,
        label='Unmod Noise'
    )
    ax1.set_title("Unmod – PCA from Grid")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.legend()
    set_axes_equal(ax1)

    # ========== 2) PCA from Unmod NOISE (3D) ==========
    pca_noise = PCA(n_components=3)
    pca_noise.fit(responses_noise_unmod)  # fit on unmod noise

    # Transform both unmod noise & grid
    noise_unmod_3d = pca_noise.transform(responses_noise_unmod)
    grid_unmod_3d_in_noise = pca_noise.transform(responses_grid_unmod)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.scatter(
        noise_unmod_3d[:, 0],
        noise_unmod_3d[:, 1],
        noise_unmod_3d[:, 2],
        c='gray',
        alpha=0.3,
        s=10,
        label='Unmod Noise'
    )
    ax2.scatter(
        grid_unmod_3d_in_noise[:, 0],
        grid_unmod_3d_in_noise[:, 1],
        grid_unmod_3d_in_noise[:, 2],
        c=stimuli_grid[:, 1],
        cmap='winter',
        s=30,
        alpha=0.8,
        label='Unmod Grid'
    )
    ax2.set_title("Unmod – PCA from Noise")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.legend()
    set_axes_equal(ax2)

    # ========== 3) Modulated in Grid PCA-Space (3D) ==========
    grid_mod_3d = pca_grid.transform(responses_grid_mod)
    noise_mod_3d_in_grid = pca_grid.transform(responses_noise_mod)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(
        grid_mod_3d[:, 0],
        grid_mod_3d[:, 1],
        grid_mod_3d[:, 2],
        c=stimuli_grid[:, 1],
        cmap='spring',
        s=30,
        alpha=0.8,
        label='Mod Grid'
    )
    ax3.scatter(
        noise_mod_3d_in_grid[:, 0],
        noise_mod_3d_in_grid[:, 1],
        noise_mod_3d_in_grid[:, 2],
        c='gray',
        alpha=0.3,
        s=10,
        label='Mod Noise'
    )
    ax3.set_title("Mod – PCA from Grid")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.legend()
    set_axes_equal(ax3)

    # ========== 4) Modulated in Noise PCA-Space (3D) ==========
    grid_mod_3d_in_noise = pca_noise.transform(responses_grid_mod)
    noise_mod_3d_in_noise = pca_noise.transform(responses_noise_mod)

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(
        noise_mod_3d_in_noise[:, 0],
        noise_mod_3d_in_noise[:, 1],
        noise_mod_3d_in_noise[:, 2],
        c='gray',
        alpha=0.3,
        s=10,
        label='Mod Noise'
    )
    ax4.scatter(
        grid_mod_3d_in_noise[:, 0],
        grid_mod_3d_in_noise[:, 1],
        grid_mod_3d_in_noise[:, 2],
        c=stimuli_grid[:, 1],
        cmap='spring',
        s=30,
        alpha=0.8,
        label='Mod Grid'
    )
    ax4.set_title("Mod – PCA from Noise")
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_zlabel("PC3")
    ax4.legend()
    set_axes_equal(ax4)
    
    
    

    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# 5) External Noise in 3rd PC
# -------------------------------------------------
import numpy as np

def generate_external_noise_in_third_pc(
    W_R,
    pc3,               # The PCA-derived 3rd principal component, shape (N,)
    stimuli_grid,
    noise_level,
    num_noise_trials
):
    """
    Generate 'external' noise so that the *final network response* is truly
    in the direction of PC3 (rather than just the input being in PC3).

    Steps:
      1) We choose a random amplitude alpha for each trial: alpha ~ N(0, noise_level)
      2) The desired final state is y = alpha * pc3.
      3) The required input is x = (I - W_R) y.
      4) We verify the final network response is (I - W_R)^(-1} x = y. 
    """

    N = W_R.shape[0]
    I = np.eye(N)
    M = I - W_R              # so (I - W_R)^(-1) is M^{-1}

    # Normalize pc3 (just in case)
    pc3_unit = pc3 / np.linalg.norm(pc3)

    inv_M = np.linalg.inv(M)  # reuse so we don't invert repeatedly
    external_noise_responses = []

    for (shape_val, color_val) in stimuli_grid:
        for _ in range(num_noise_trials):
            # Draw a random amplitude
            alpha = np.random.randn() * noise_level
            
            # Desired final output = alpha * pc3_unit
            final_desired = alpha * pc3_unit
            
            # Required input to produce 'final_desired'
            noise_input = M @ final_desired  # shape (N,)

            # Pass through network (should yield final_desired exactly)
            response = inv_M @ noise_input

            external_noise_responses.append(response)

    return np.array(external_noise_responses)

def plot_3d_pca_grid_and_external_noise(responses_grid, external_noise_responses, stimuli_grid, title_main="3D PCA"):
    """
    Plot the grid responses plus the external-noise responses in 3D PCA space.
    We'll fit PCA(n_components=3) on the grid responses to define the axes.
    """
    pca_3 = PCA(n_components=3)
    pca_3.fit(responses_grid)

    # Project both sets into 3D PCA space
    grid_3d = pca_3.transform(responses_grid)             # shape: (grid_size, 3)
    noise_3d = pca_3.transform(external_noise_responses)  # shape: (grid_size * num_noise_trials, 3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc1 = ax.scatter(
        grid_3d[:, 0],
        grid_3d[:, 1],
        grid_3d[:, 2],
        c=stimuli_grid[:, 1],
        cmap='coolwarm',
        s=40,
        alpha=0.8,
        label='Grid Responses'
    )

    sc2 = ax.scatter(
        noise_3d[:, 0],
        noise_3d[:, 1],
        noise_3d[:, 2],
        c='gray',
        s=10,
        alpha=0.3,
        label='External Noise (3rd PC)'
    )

    ax.set_title(title_main, fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    set_axes_equal(ax)

    # Optional colorbar for the grid
    cbar = fig.colorbar(sc1, ax=ax, label="Color Stimulus Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# 6) Main Code
# -------------------------------------------------
if __name__ == "__main__":

    # 1) Selectivity and Feedforward
    S = initialize_selectivity_matrix(N, K)
    W_F = initialize_W_F(S)

    # 2) Stimuli
    shape_stimuli = np.linspace(0, 1, num_stimuli)
    color_stimuli = np.linspace(0, 1, num_stimuli)

    # -------------------------------------------------
    # Example: UNTUNED W_R
    # -------------------------------------------------
    W_R_untuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)
    responses_grid_unmod, stimuli_grid = compute_responses(W_F, W_R_untuned, shape_stimuli, color_stimuli)
    responses_noise_unmod = generate_noisy_responses(W_R_untuned, noise_level, stimuli_grid, num_noise_trials)
    responses_grid_mod = compute_modulated_responses(W_R_untuned, W_F, S, stimuli_grid)
    responses_noise_mod = compute_modulated_noise(W_R_untuned, W_F, S, stimuli_grid, noise_level, num_noise_trials)
    
    
    # 3D PCA visualization of unmod vs. mod + noise
    visualize_four_subplots_3d(
        responses_grid_unmod,
        responses_noise_unmod,
        responses_grid_mod,
        responses_noise_mod,
        stimuli_grid,
        "UNTUNED W_R (3D PCA)"
    )

    # -------------------------------------------------
    # Example: TUNED W_R
    # -------------------------------------------------
    W_R_tuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=True, desired_radius=desired_radius)
    responses_grid_unmod_tuned, stimuli_grid_tuned = compute_responses(W_F, W_R_tuned, shape_stimuli, color_stimuli)
    responses_noise_unmod_tuned = generate_noisy_responses(W_R_tuned, noise_level, stimuli_grid_tuned, num_noise_trials)
    responses_grid_mod_tuned = compute_modulated_responses(W_R_tuned, W_F, S, stimuli_grid_tuned)
    responses_noise_mod_tuned = compute_modulated_noise(W_R_tuned, W_F, S, stimuli_grid_tuned, noise_level, num_noise_trials)

    visualize_four_subplots_3d(
        responses_grid_unmod_tuned,
        responses_noise_unmod_tuned,
        responses_grid_mod_tuned,
        responses_noise_mod_tuned,
        stimuli_grid_tuned,
        "TUNED W_R (3D PCA)"
    )

    # -------------------------------------------------
    # Example: CYCLE W_R
    # -------------------------------------------------
    W_R_cycle = initialize_W_R_cycle(N, desired_radius=desired_radius)
    responses_grid_unmod_cycle, stimuli_grid_cycle = compute_responses(W_F, W_R_cycle, shape_stimuli, color_stimuli)
    responses_noise_unmod_cycle = generate_noisy_responses(W_R_cycle, noise_level, stimuli_grid_cycle, num_noise_trials)
    responses_grid_mod_cycle = compute_modulated_responses(W_R_cycle, W_F, S, stimuli_grid_cycle)
    responses_noise_mod_cycle = compute_modulated_noise(W_R_cycle, W_F, S, stimuli_grid_cycle, noise_level, num_noise_trials)

    visualize_four_subplots_3d(
        responses_grid_unmod_cycle,
        responses_noise_unmod_cycle,
        responses_grid_mod_cycle,
        responses_noise_mod_cycle,
        stimuli_grid_cycle,
        "CYCLE W_R (3D PCA)"
    )

    # -------------------------------------------------
    # NEW: EXTERNAL NOISE in the 3rd PC for the UNTUNED case
    # -------------------------------------------------
    # 1) Fit a 3-component PCA to the (unmodulated) grid responses:
# 1) Fit a 3-component PCA to the (unmodulated) grid responses
    pca_3 = PCA(n_components=3)
    pca_3.fit(responses_grid_unmod)

    # Extract just PC3
    pc3_untuned = pca_3.components_[2]  # shape (N,)
    
    external_noise_responses_3rd_pc = generate_external_noise_in_third_pc(
    W_R_untuned,
    pc3_untuned,          # only PC3 is needed now
    stimuli_grid,
    noise_level=10.0,
    num_noise_trials=50
)



    # 3) Plot the original grid responses + the new external noise responses in 3D PCA space
    plot_3d_pca_grid_and_external_noise(
        responses_grid_unmod,
        external_noise_responses_3rd_pc,
        stimuli_grid,
        title_main="3D PCA – External Noise in 3rd PC (Untuned W_R)"
    )

    print("All done!")
