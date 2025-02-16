import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import shapiro, normaltest
import seaborn as sns

# ============================
# 1. Parameters and Initialization
# ============================

# Parameters
N = 1000  # Number of neurons
K = 2     # Two features: shape (index=0) and color (index=1)
num_stimuli = 10  # Number of stimuli per feature dimension

# Seed for reproducibility
np.random.seed(1)

# Initialize the selectivity matrix
S = np.zeros((N, K))

# ============================
# 2. Assigning Selectivity with Normal Distribution Differences
# ============================

# Define standard deviation for the noise
sigma = 0.1  # Adjust as needed

# Define the range for S[:, 0] to ensure S[:, 1] stays within [0, 1] after adding noise
shape_min = 0.2
shape_max = 0.8

# Assign shape selectivity for all neurons
S[:, 0] = np.random.uniform(shape_min, shape_max, size=N)

# Generate Gaussian noise for the difference S[:,1] - S[:,0]
noise = np.random.normal(loc=0.0, scale=sigma, size=N)

# Assign color selectivity
S[:, 1] = S[:, 0] + noise

# Clip S[:,1] to ensure values are within [0, 1]
S[:, 1] = np.clip(S[:, 1], 0.0, 1.0)

# ============================
# 3. Reordering Neurons: Shape-Preferring and Color-Preferring
# ============================

# Compute the differences
differences = S[:, 1] - S[:, 0]

# Sort the indices based on differences
sorted_indices = np.argsort(differences)

# Reorder S so that first half are shape-preferring and second half are color-preferring
S = S[sorted_indices]

# Verify the Reordering
half_N = N // 2
differences_sorted = differences[sorted_indices]

print(f"First half (Shape-Preferring) differences: min = {differences_sorted[:half_N].min():.4f}, max = {differences_sorted[:half_N].max():.4f}")
print(f"Second half (Color-Preferring) differences: min = {differences_sorted[half_N:].min():.4f}, max = {differences_sorted[half_N:].max():.4f}")

# Optional: Plotting to Verify Distribution and Reordering
plt.figure(figsize=(14, 6))

# Histogram of differences before reordering
plt.subplot(1, 2, 1)
sns.histplot(differences, bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('Original Distribution of Differences (S[:,1] - S[:,0])')
plt.xlabel('Difference (Color - Shape)')
plt.ylabel('Number of Neurons')

# Histogram of differences after reordering
plt.subplot(1, 2, 2)
sns.histplot(differences_sorted[:half_N], bins=30, kde=True, color='salmon', label='Shape-Preferring', edgecolor='black')
sns.histplot(differences_sorted[half_N:], bins=30, kde=True, color='skyblue', label='Color-Preferring', edgecolor='black')
plt.title('Reordered Distribution of Differences')
plt.xlabel('Difference (Color - Shape)')
plt.ylabel('Number of Neurons')
plt.legend()

plt.tight_layout()
plt.show()
