import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 50  # Number of neurons
K = 1    # One input feature

# Seed for reproducibility
np.random.seed(0)

# Define matrices and vectors
W_F = np.random.rand(N, K) * 0.1  # Feedforward weights
W_R = np.random.rand(N, N) * 0.1  # Recurrent weights
F = np.random.rand(K)  # External stimuli

# Normalize W_R to ensure all eigenvalues have absolute values less than 1
eigenvalues = np.linalg.eigvals(W_R)
scaling_factor = np.max(np.abs(eigenvalues))
W_R = W_R / (scaling_factor + 0.1)  # Scale down to ensure stability

# Identity matrix
I = np.eye(N)

# Analytical solution using the inverse matrix
inv_matrix = np.linalg.inv(I - W_R)
analytical_steady_state = inv_matrix @ W_F @ F

# Simulation parameters
dt = 0.01  # Time step
time_steps = 500  # Number of time steps for simulation
r = np.random.rand(N)  # Initial firing rates
noise_std = 0.001  # Standard deviation of noise

# Store firing rate history for plotting
r_history = np.zeros((N, time_steps))

# Numerical simulation using Euler's method
for t in range(time_steps):
    noise = np.random.normal(0, noise_std, N)  # Gaussian noise
    r += dt * (-r + (W_F @ F + W_R @ r + noise/np.sqrt(dt)))  # Update rule
    r_history[:, t] = r

# Plotting the results
plt.figure(figsize=(12, 8))
time = np.linspace(0, time_steps * dt, time_steps)
for neuron_index in range(min(N, 10)):  # Plotting only first 10 neurons for clarity
    plt.plot(time, r_history[neuron_index, :], label=f'Neuron {neuron_index + 1}')
    # Plot analytical steady state as a horizontal line for comparison
    # plt.hlines(analytical_steady_state[neuron_index], 0, time_steps * dt, colors='k', linestyles='dashed', label='Analytical Steady State')


plt.title('Evolution of Neuron Firing Rates Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Firing Rate')
plt.xlim(0, time_steps * dt)
plt.ylim(0, 1.5)  # Assuming firing rates are between 0 and 1.5 due to the tanh nonlinearity
plt.legend()
plt.show()

# Scatter plot of numerical vs analytical steady states
plt.figure(figsize=(8, 8))
plt.scatter(r, analytical_steady_state, color='blue', label='Neuron Comparison')
plt.plot([0, max(r)], [0, max(r)], 'r--', label='x=y Line')
plt.xlabel('Numerical Steady State Firing Rates')
plt.ylabel('Analytical Steady State Firing Rates')
plt.title('Comparison of Steady States')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Display analytical and simulated steady-state values
print("Analytical Steady State Firing Rates:\n", analytical_steady_state)
print("Simulated Steady State Firing Rates at final timestep:\n", r)
print("Difference (norm):\n", np.linalg.norm(analytical_steady_state - r))