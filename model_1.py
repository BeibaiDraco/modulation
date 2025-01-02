import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_neurons = 100
n_trials = 1000
n_features = 2  # curvature and color

# Create neural population
def create_population(n_neurons, n_features):
    # Random preferred stimuli for each neuron and feature
    preferred_stimuli = np.random.rand(n_neurons, n_features)
    # Random tuning widths
    tuning_widths = np.random.uniform(0.1, 0.5, (n_neurons, n_features))
    return preferred_stimuli, tuning_widths

# Neural response function
def neural_response(stimulus, preferred_stimuli, tuning_widths):
    diff = stimulus - preferred_stimuli
    return np.exp(-0.5 * np.sum((diff / tuning_widths)**2, axis=1))

# Generate correlated variability
def generate_correlated_variability(n_neurons, n_trials, correlation_strength=0.2):
    cov_matrix = np.eye(n_neurons) * (1 - correlation_strength) + np.ones((n_neurons, n_neurons)) * correlation_strength
    return np.random.multivariate_normal(np.zeros(n_neurons), cov_matrix, n_trials).T

# Simulate neural activity
def simulate_activity(stimulus, preferred_stimuli, tuning_widths, variability):
    mean_response = neural_response(stimulus, preferred_stimuli, tuning_widths)
    return mean_response[:, np.newaxis] + variability

# Create population and generate correlated variability
preferred_stimuli, tuning_widths = create_population(n_neurons, n_features)
correlated_variability = generate_correlated_variability(n_neurons, n_trials)

# Simulate different task conditions
def simulate_task(attended_feature, stimulus):
    # Implement a simple attention mechanism by scaling the response to the attended feature
    attention_gain = np.ones(n_features)
    attention_gain[attended_feature] = 1.5
    
    responses = simulate_activity(stimulus * attention_gain, preferred_stimuli, tuning_widths, correlated_variability)
    return responses

# Readout mechanism
def linear_readout(responses, weights):
    return np.dot(weights, responses)

# Analyze alignment
def analyze_alignment(responses, feature_axis, variability_axis):
    feature_projection = np.abs(np.dot(responses.T, feature_axis))
    variability_projection = np.abs(np.dot(responses.T, variability_axis))
    alignment = np.corrcoef(feature_projection, variability_projection)[0, 1]
    return alignment

# Run simulation
stimulus = np.random.rand(n_features)
curvature_responses = simulate_task(0, stimulus)
color_responses = simulate_task(1, stimulus)

# Calculate correlated variability axis (simplified as the first principal component)
_, variability_axis = np.linalg.eig(np.cov(correlated_variability))
variability_axis = variability_axis[:, 0]

# Calculate feature axes (simplified as the average gradient of responses)
curvature_axis = np.mean(np.gradient(curvature_responses, axis=1), axis=1)
color_axis = np.mean(np.gradient(color_responses, axis=1), axis=1)

# Analyze alignment
curvature_alignment = analyze_alignment(curvature_responses, curvature_axis, variability_axis)
color_alignment = analyze_alignment(color_responses, color_axis, variability_axis)

print(f"Curvature task alignment: {curvature_alignment:.3f}")
print(f"Color task alignment: {color_alignment:.3f}")

# Visualize results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(curvature_responses, aspect='auto', cmap='viridis')
plt.title("Curvature Task Responses")
plt.subplot(122)
plt.imshow(color_responses, aspect='auto', cmap='viridis')
plt.title("Color Task Responses")
plt.tight_layout()
plt.show()