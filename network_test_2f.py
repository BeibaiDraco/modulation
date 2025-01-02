import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 50  # Number of neurons
K = 2   # Two features: shape and color

# Seed for reproducibility
np.random.seed(0)

# Feature selectivities
S = np.random.rand(N, K)  # Random selectivities for each feature

# Define matrices and vectors
W_F = np.random.rand(N, K) * 0.1  # Feedforward weights
F = np.random.rand(K)  # External stimuli for each feature

# Recurrent connectivity based on similarity in feature selectivities
W_R = np.zeros((N, N))
threshold = 0.1  # Similarity threshold
for i in range(N):
    for j in range(N):
        distance = np.linalg.norm(S[i] - S[j])
        if distance < threshold:
            W_R[i, j] = 1 - distance / threshold  # Stronger connections for more similar neurons

# Normalize W_R
eigenvalues = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues))
W_R = W_R / (scaling_factor + 1)  # Ensure stability

# Test each stimulus independently
results = {}
stimuli = {'Shape Only': np.array([1, 0]), 'Color Only': np.array([0, 1])}

for test, stimulus in stimuli.items():
    # Adjust external stimulus
    adjusted_F = W_F @ stimulus

    # Analytical solution
    analytical_steady_state = np.linalg.inv(np.eye(N) - W_R) @ adjusted_F

    # Simulation parameters
    dt = 0.001
    time_steps = 30000
    r = np.zeros(N)
    noise_std = 0.1

    # Simulation
    for t in range(time_steps):
        noise = np.random.normal(0, noise_std, N) * np.sqrt(dt)
        r += dt * (-r + np.tanh(adjusted_F + W_R @ r + noise))
    
    results[test] = r

    # Plotting the result for this stimulus
    plt.figure(figsize=(10, 5))
    plt.title(f'Neuron Firing Rates Over Time - {test}')
    plt.plot(r, label=f'Simulated Steady State - {test}')
    plt.plot(analytical_steady_state, 'r--', label=f'Analytical Steady State - {test}')
    plt.xlabel('Neuron Index')
    plt.ylabel('Firing Rate')
    plt.legend()
    plt.show()

# Compare responses
plt.figure(figsize=(10, 5))
for test, response in results.items():
    plt.plot(response, label=f'{test}')
plt.title('Comparison of Responses to Different Stimuli')
plt.xlabel('Neuron Index')
plt.ylabel('Firing Rate')
plt.legend()
plt.show()

