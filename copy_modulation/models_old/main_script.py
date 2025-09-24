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
    identify_dominant_neurons_for_rotation,
    plot_dominant_neurons,
    analyze_axis_rotation_in_pca
)

def main():
    # 1) Hyperparams
    np.random.seed(15)
    N = 100           
    K = 2             
    num_stimuli = 10  
    num_noise_trials = 50  
    noise_level = 1.0       
    desired_radius = 0.9    
    p_high = 0.25
    p_low = 0.25

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
    responses_grid_mod_pc3,g_vector = compute_modulated_responses_pc3(
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
        # Save the scatter plot objects
    sc1 = ax.scatter(
        grid_unmod_3d[:, 0], grid_unmod_3d[:, 1], grid_unmod_3d[:, 2],
        c=stimuli_grid[:, 0], alpha=0.3, s=40, label='Unmod Grid', cmap='winter'
    )
    sc2 = ax.scatter(
        grid_mod_3d[:, 0], grid_mod_3d[:, 1], grid_mod_3d[:, 2],
        c=stimuli_grid[:, 0], alpha=1, s=40, label='Mod PC3 Grid', cmap='winter'
    )


    ax.set_title("PC3-based Modulation vs. Unmodulated (Untuned W_R)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    
    cbar = fig.colorbar(sc2, ax=ax, shrink=0.7, aspect=20)
    cbar.set_label('Stimulus value')
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
    

    G = np.diag(g_vector)

    # 2) Identify rotation changes
    shape_diff, color_diff = identify_dominant_neurons_for_rotation(W_R_untuned, W_F, G)

    # 3) Plot or analyze the top-changed neurons
    plot_dominant_neurons(shape_diff, color_diff, top_k=20)
    
        # Suppose we pick a single stimulus (or an average over a set of stimuli).
    # For demonstration, let's pick shape=0.0, color=1.0 as an example "pure color" stimulus:

    # 1) Identify the index for that stimulus:
    stim_index_color1 = np.where((stimuli_grid[:,0]==0.0) & (stimuli_grid[:,1]==1.0))[0]
    # (Assumes you have shape_stimuli and color_stimuli that contain exactly 0.0 and 1.0)

    # 2) Extract the 300D firing for that stimulus, unmod vs mod
    x_unmod = responses_grid_unmod[stim_index_color1[0]]  # shape (N,)
    x_mod   = responses_grid_mod_pc3[stim_index_color1[0]]  # shape (N,)

    # 3) Compute the difference in raw space
    delta_x = x_mod - x_unmod  # shape (N,)

    # 4) Project both unmod and mod into PCA space
    z_unmod = pca_grid_again.transform(x_unmod.reshape(1,-1))  # shape (1,3)
    z_mod   = pca_grid_again.transform(x_mod.reshape(1,-1))    # shape (1,3)
    delta_z = (z_mod - z_unmod)[0]   # shape (3,)

    # 5) The PCA components matrix is pca_grid_again.components_, shape (3, N).
    #    But we often handle the transposed version to get U[i, :] for each neuron i.

    U = pca_grid_again.components_.T  # shape (N,3)

    # 6) Now let's define "per-neuron contribution in PCA space" as:
    #    delta_z_i = delta_x[i] * U[i,:]
    #    We can store these in an (N,3) array:

    delta_z_per_neuron = np.zeros((x_unmod.shape[0], 3))
    for i in range(x_unmod.shape[0]):
        delta_z_per_neuron[i,:] = delta_x[i] * U[i,:]

    # 7) If you want a single scalar measure of magnitude:
    contrib_magnitude = np.linalg.norm(delta_z_per_neuron, axis=1)  # shape (N,)

    # 8) Plot or analyze correlation with g_vector[i]
    # Suppose g_vector is shape (N,). We can scatter:
    plt.figure(figsize=(8,6))
    plt.scatter(g_vector, contrib_magnitude, alpha=0.5)
    plt.xlabel("g_vector[i]")
    plt.ylabel("Norm of neuron i's contribution in PCA space")
    plt.title("Per-Neuron Contribution in PCA Space vs. Gain")
    plt.grid(True)
    plt.show()

    # 9) You can also check shape or color selectivity S[:,1]-S[:,0]:
    selectivity_diff = S[:,1] - S[:,0]
    plt.figure(figsize=(8,6))
    plt.scatter(selectivity_diff, contrib_magnitude, alpha=0.5)
    plt.xlabel("selectivity_diff (color - shape)")
    plt.ylabel("Norm of i's PCA contribution")
    plt.title("Per-Neuron PCA Contribution vs. Selectivity")
    plt.grid(True)
    plt.show()
    
    # 1) Suppose you have pca_grid_again fitted on responses_grid_unmod
    # 2) Suppose you define your G = np.diag(g_vector) already

    results = analyze_axis_rotation_in_pca(W_R_untuned, W_F, G, pca_grid_again)

    shape_contrib_mag = results["shape_contrib_mag"]  # shape (N,)
    color_contrib_mag = results["color_contrib_mag"]  # shape (N,)

    # e.g., correlation with g_vector
    plt.figure(figsize=(8,6))
    plt.scatter(g_vector, shape_contrib_mag, alpha=0.5, color='blue', label='Shape Contribution')
    plt.scatter(g_vector, color_contrib_mag, alpha=0.5, color='red',  label='Color Contribution')
    plt.xlabel("g_vector[i]")
    plt.ylabel("Contribution magnitude in PCA space")
    plt.title("Neuron-by-Neuron Contribution vs. Gain Factor")
    plt.legend()
    plt.grid(True)
    plt.show()





    print("All done!")

if __name__ == "__main__":
    main()
