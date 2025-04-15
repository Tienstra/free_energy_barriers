import jax.numpy as jnp
from jax import random, jit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib.colors as mcolors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import your modules
from models.regression import LogisticRegression
from kernels import MALAKernel
from sampler import MCMC


# Set up a simple logistic regression problem
def generate_synthetic_data(n_samples=100, n_features=2, seed=42):
    """Generate synthetic data for logistic regression"""
    np.random.seed(seed)

    # Generate random features
    X = np.random.randn(n_samples, n_features)

    # True coefficients
    # true_beta = np.array([1.5, -2.0])
    true_beta = np.zeros(n_features)

    # Generate probabilities and labels
    logits = X @ true_beta
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)

    return X, y, true_beta


# Generate synthetic data
n_samples = 100
n_features = 100
X, y, true_beta = generate_synthetic_data(n_samples, n_features)

# Convert to JAX arrays
X_jax = jnp.array(X)
y_jax = jnp.array(y)

# Initialize the logistic regression model
model = LogisticRegression(N=n_samples, p=n_features)


# Define the log posterior function for MCMC
def create_log_posterior(model, X, y, n_features):
    """
    Creates a log posterior function that doesn't modify any state

    Args:
        model: The model class (not an instance)
        X: Input data
        y: Target data
        sigma_prior: Prior standard deviation

    Returns:
        A pure function that computes log posterior
    """
    sigma_prior = 1 / jnp.sqrt(n_features)

    @jit
    def log_posterior(beta):
        # Create model predictions without storing state
        model_pred = 1 / (1 + jnp.exp(-X @ beta))

        # Calculate log likelihood (needs to match your model's implementation)
        # For logistic regression:
        log_likelihood = jnp.sum(
            y * jnp.log(model_pred + 1e-10) + (1 - y) * jnp.log(1 - model_pred + 1e-10)
        )

        # Add log prior (Gaussian)
        log_prior = -0.5 * jnp.sum(beta**2) / sigma_prior**2

        return log_likelihood + log_prior

    return log_posterior


log_posterior_fn = create_log_posterior(LogisticRegression, X_jax, y_jax, n_features)
# Initialize the MALA kernel with our log posterior
mala_kernel = MALAKernel(log_posterior_fn, epsilon=0.01)


# Create an initializer function for the chains
def initialize_chains(n_chains, D, scale=0.1):
    """Initialize chains with small random values"""
    key = random.PRNGKey(0)
    return scale * random.normal(key, shape=(n_chains, D))


# Set up and run the MCMC sampler
mcmc = MCMC(
    kernel=mala_kernel,
    D=n_features,
    n_steps=50000,
    n_chains=5,
    initializer="sample_annuli",
    init_args=[0, 0.33],
    seed=45,
)

# Run sampling
samples = mcmc.sample(thin=10)

# Print acceptance ratio
print(f"Acceptance ratio: {mcmc.acceptance_ratio:.4f}")

if n_features < 2:
    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot trace plots for each parameter
    for i in range(n_features):
        plt.subplot(2, 2, i + 1)
        for chain in range(samples.shape[0]):
            plt.plot(samples[chain, :, i], alpha=0.7)
        plt.axhline(true_beta[i], color="r", linestyle="--")
        plt.title(f"Trace plot for β{i}")

    # Plot posterior distributions
    for i in range(n_features):
        plt.subplot(2, 2, i + 3)
        for chain in range(samples.shape[0]):
            # Skip some initial samples as burn-in
            burn_in = int(samples.shape[1] * 0.2)
            sns.kdeplot(samples[chain, burn_in:, i], fill=True, alpha=0.3)
        plt.axvline(true_beta[i], color="r", linestyle="--")
        plt.title(f"Posterior distribution for β{i}")

    plt.tight_layout()
    plt.show()

# Calculate posterior means
# burn_in = int(samples.shape[1] * 0.2)
# posterior_means = samples[:, burn_in:, :].mean(axis=(0, 1))
posterior_means = samples.mean(axis=(0, 1))
# print("True coefficients:", true_beta)
# print("Posterior means:", posterior_means)
print("Error :", np.linalg.norm(posterior_means - true_beta))
# Use the fitted model to make predictions
model.beta = posterior_means
predicted_probs = model.evaluate(X_jax @ model.beta)
predicted_classes = (predicted_probs > 0.5).astype(int)

# Calculate accuracy
accuracy = (predicted_classes == y_jax).mean()
print(f"Prediction accuracy: {accuracy:.4f}")
print("shape of samples", samples.shape)


teal = np.array(mcolors.to_rgb("#009688"))  # Teal
lavender = np.array(mcolors.to_rgb("#B57EDC"))  # Lavender


for chain_idx in range(samples.shape[0]):
    norms = np.linalg.norm(samples[chain_idx, ::100], axis=1)
    color = teal + (lavender - teal) * (chain_idx / (samples.shape[0] - 1))
    plt.plot(norms, color=color, alpha=0.3)
plt.axhline(0.33, color="red", linestyle="--")
plt.axhline(0.66, color="blue", linestyle="--")
plt.xlabel("t")
plt.ylabel("Norm of X_t")
plt.grid(True)
plt.title(r"$\mathcal{N}(0,1/D)$")
print(samples[chain_idx].shape)
print(np.linalg.norm(true_beta))


plt.tight_layout()
plt.show()
