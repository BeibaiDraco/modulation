# main_script.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import all the functions from my_network.py:
from my_network import (
    initialize_selectivity_matrix,
    initialize_W_F,
    initialize_W_R,
    initialize_W_R_cycle,
    compute_responses,
    generate_noisy_responses,
    compute_modulated_responses,
    compute_modulated_responses_pc3,
    compute_modulated_noise,
    visualize_four_subplots_3d,
    generate_external_noise_in_third_pc,
    plot_3d_pca_grid_and_external_noise,
)

def main():
    # 1) Hyperparams
    np.random.seed(15)
    N = 300           
    K = 2             
    num_stimuli = 10  
    num_noise_trials = 50  
    noise_level = 1.0       
    desired_radius = 0.9    
    p_high = 0.25
    p_low = 0.00

    # 2) Build S, W_F
    S = initialize_selectivity_matrix(N, K)
    W_F = initialize_W_F(S)

    # Stimuli
    shape_stimuli = np.linspace(0, 1, num_stimuli)
    color_stimuli = np.linspace(0, 1, num_stimuli)

    # -------------------------------------------------
    # 2a) UNTUNED W_R Example
    # -------------------------------------------------
    W_R_untuned = initialize_W_R(N, p_high, p_low, S, WR_tuned=False, desired_radius=desired_radius)
    responses_grid_unmod, stimuli_grid = compute_responses(W_F, W_R_untuned, shape_stimuli, color_stimuli)
    responses_noise_unmod = generate_noisy_responses(W_R_untuned, noise_level, stimuli_grid, num_noise_trials)
    responses_grid_mod = compute_modulated_responses(W_R_untuned, W_F, S, stimuli_grid)
    responses_noise_mod = compute_modulated_noise(W_R_untuned, W_F, S, stimuli_grid, noise_level, num_noise_trials)


    # -------------------------------------------------
    # 3) EXTERNAL NOISE in the 3rd PC (UNTUNED case)
    # -------------------------------------------------
    # Fit a 3-component PCA on unmodulated grid responses
    pca_3 = PCA(n_components=3)
    pca_3.fit(responses_grid_unmod)
    pc3_untuned = pca_3.components_[2]  # shape (N,)

    external_noise_responses_3rd_pc = generate_external_noise_in_third_pc(
        W_R_untuned,
        pc3_untuned,
        stimuli_grid,
        noise_level=5.0,
        num_noise_trials=50
    )

    plot_3d_pca_grid_and_external_noise(
        responses_grid_unmod,
        external_noise_responses_3rd_pc,
        stimuli_grid,
        title_main="3D PCA – External Noise in 3rd PC (Untuned W_R)"
    )

    # -------------------------------------------------
    # 4) PC3-based Modulation
    # -------------------------------------------------
    # Suppose we want color dimension to be pushed more into PC3
    # We'll do a small example with the UNTUNED matrix
    responses_grid_mod_pc3 = compute_modulated_responses_pc3(
        W_R_untuned,
        W_F,
        S,
        stimuli_grid,
        pc3_untuned
    )

    # We can visualize that with or without noise...
    # Let's just show it in the "Unmod Grid PCA" space
    # so we see how it changed from the standard unmod
    pca_grid_again = PCA(n_components=3)
    pca_grid_again.fit(responses_grid_unmod)

    grid_mod_3d = pca_grid_again.transform(responses_grid_mod_pc3)
    grid_unmod_3d = pca_grid_again.transform(responses_grid_unmod)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        grid_unmod_3d[:, 0], grid_unmod_3d[:, 1], grid_unmod_3d[:, 2],
        c=stimuli_grid[:, 0],alpha=0.3, s=40, label='Unmod Grid'
    )
    ax.scatter(
        grid_mod_3d[:, 0], grid_mod_3d[:, 1], grid_mod_3d[:, 2],
        c=stimuli_grid[:, 0],alpha=0.6, s=40, label='Mod PC3 Grid'
    )
    ax.set_title("PC3-based Modulation vs. Unmodulated (Untuned W_R)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    from my_network import set_axes_equal
    #set_axes_equal(ax)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------
    # 5) Plot Modulation vs. Selectivity (Per Neuron)
    # ---------------------------------------------
    # We'll define "average unmodded response" and "average modded response" per neuron
    avg_unmod_per_neuron = np.mean(responses_grid_unmod, axis=0)        # shape (N,)
    avg_mod_pc3_per_neuron = np.mean(responses_grid_mod_pc3, axis=0)    # shape (N,)

    # Avoid divide-by-zero: small epsilon
    epsilon = 1e-9
    ratio_per_neuron = avg_mod_pc3_per_neuron / (avg_unmod_per_neuron + epsilon)  # shape (N,)

    # Now the x-axis: each neuron's "color selectivity" = S[:,1] - S[:,0]
    selectivity_diff = S[:, 1] - S[:, 0]  # shape (N,)

    # Create a simple 2D scatter
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(
        selectivity_diff,
        ratio_per_neuron,
        c=selectivity_diff,          # Color the dots by the same selectivity, or anything else
        cmap='coolwarm',
        alpha=0.8,
        s=40
    )
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.7)  # reference line at ratio = 1
    ax2.set_xlabel("Selectivity Difference (S[:,1] - S[:,0])")
    ax2.set_ylabel("Modulated / Unmodulated (avg response ratio)")
    ax2.set_title("Per-Neuron Modulation vs. Selectivity")
    plt.colorbar(sc, ax=ax2, label="S[:,1] - S[:,0]")
    plt.tight_layout()
    plt.show()


    print("All done!")

if __name__ == "__main__":
    main()
