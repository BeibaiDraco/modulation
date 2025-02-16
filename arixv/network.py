import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 200  # Total number of neurons
K = 2    # Number of features

# Feature selectivities
S = np.random.rand(N, K)

# Feedforward weights
W = np.random.normal(0, 1, (N, K))

# Recurrent connectivity based on similarity in feature selectivities
R = np.zeros((N, N))
threshold = 0.2
for i in range(N):
    for j in range(N):
        distance = np.linalg.norm(S[i] - S[j])
        if distance < threshold:
            R[i, j] = 1 - distance / threshold  # Stronger connections for more similar neurons

# Initial firing rates
r = np.zeros(N)

# Initial stimulus
F = np.array([0.5, 0.5])

# Simulation settings
time_steps = 100

# Simulation loop for initial stimulus
for t in range(time_steps):
    input_drive = np.dot(W, F)
    recurrent_drive = np.dot(R, r)
    r = np.tanh(input_drive + recurrent_drive)

firing_rates_initial = r.copy()

# Change the stimulus
F = np.array([1.0, 0.0])
r = np.zeros(N)  # Reset firing rates for new stimulus

# Simulation loop for changed stimulus
for t in range(time_steps):
    input_drive = np.dot(W, F)
    recurrent_drive = np.dot(R, r)
    r = np.tanh(input_drive + recurrent_drive)

firing_rates_changed = r.copy()

# Plotting both firing rates on the same graph
plt.figure(figsize=(10, 5))
plt.plot(firing_rates_initial, label='Initial Firing Rates', linestyle='-', marker='o')
plt.plot(firing_rates_changed, label='Changed Firing Rates', linestyle='-', marker='x')
plt.title('Comparison of Neuron Firing Rates')
plt.xlabel('Neuron Index')
plt.ylabel('Firing Rate')
plt.legend()
plt.show()