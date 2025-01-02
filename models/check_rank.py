import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100  # Number of neurons
k = 0.1  # Adjustment penalty parameter

# Simulate eigenvector and color selectivity
np.random.seed(0)
eigenvector_1 = np.random.randn(N)
color_selectivity = np.random.rand(N)

# Compute ranks
rank_eigenvector = np.argsort(np.abs(eigenvector_1))[::-1]  # Rank by |eigenvector_1|
rank_color = np.argsort(color_selectivity)[::-1]  # Rank by color selectivity

# Compute rank disparity
rank_disparity = np.zeros(N)
for i in range(N):
    rank_disparity[i] = np.where(rank_eigenvector == i)[0][0] - np.where(rank_color == i)[0][0]

# Compute adjustment factor
adjustment_factor = np.exp(-k * np.abs(rank_disparity))

# Compute rank-based scaling (e.g., quadratic scaling)
rank_based_scaling = np.zeros(N)
eigen_rank_scaler = (N - np.arange(1, N + 1)) ** 2  # Quadratic scaling
rank_based_scaling[rank_eigenvector] = eigen_rank_scaler
rank_based_scaling = (rank_based_scaling - rank_based_scaling.min()) / (rank_based_scaling.max() - rank_based_scaling.min())

# Compute modulation factors
modulation_factors = (1 + rank_based_scaling) * adjustment_factor * color_selectivity

# Scale to ensure min=0.8 and max=1.2
min_modulation = modulation_factors.min()
max_modulation = modulation_factors.max()
modulation_factors_scaled = modulation_factors#0.8 + (modulation_factors - min_modulation) * (1.2 - 0.8) / (max_modulation - min_modulation)

# Visualize modulation factors
plt.figure(figsize=(10, 6))
plt.bar(range(N), modulation_factors_scaled, label='Modulation Factors (Scaled)', color='blue', alpha=0.7)
plt.xlabel('Neuron Index')
plt.ylabel('Modulation Factor')
plt.title('Neuron Modulation Factors (Adjusted to Match Ranks)')
plt.legend()
plt.grid()
plt.show()

# Print rank matching statistics
print(f"Rank Disparity (mean): {rank_disparity.mean():.4f}")
print(f"Adjustment Factor (mean): {adjustment_factor.mean():.4f}")
print(f"Modulation Factors (scaled min): {modulation_factors_scaled.min():.4f}")
print(f"Modulation Factors (scaled max): {modulation_factors_scaled.max():.4f}")