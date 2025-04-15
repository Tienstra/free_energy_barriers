import numpy as np
import jax.numpy as jnp
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import os
from jax import random, jit
from functools import partial


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.spherical_coords import sample_spherical_coords, spherical_to_cartesian


def sample_annuli(D, n_samples, args):
    """
    Samples from the annuli with specified radius

    args tuple:
        - n_dims : dimension of samples
        - n_samples : number of samples to be drawn
        - r_low : lower bound for radius (inner radius)
        - r_high: upper bound for radius (outer radius)
    """
    if args == []:
        print(
            """
            You must pass in args. The format of the list is:
                - r_low : lower bound for radius (inner radius)
                - r_high: upper bound for radius (outer radius)
              """
        )
    else:
        r_low = args[0]  # Lower bound for radial coordinate
        r_high = args[1]  # Upper bound for radial coordinate

    # Set random seed for reproducibility
    key = random.PRNGKey(42)

    # Generate samples
    r, phis = sample_spherical_coords(key, D, n_samples, r_low, r_high)

    # Convert to Cartesian coordinates
    samples = spherical_to_cartesian(r, phis)

    return samples


def sample_stdnorm_prior(D, n_samples, args):
    """
    Samples from the standard norm of dim D

    args tuple:
        - D : dimension of samples
        - n_samples : number of samples to be drawn
        - args : key for reproducibility
    """
    if args == []:
        print(
            """
            You must pass in args. The format of the list is:
                - key 
              """
        )
    else:
        key = args[0]
    mean = jnp.zeros(D)
    covariance = jnp.eye(D)

    return random.multivariate_normal(key, mean, covariance, shape=(n_samples,))


def generate_bounds(start=0, stop=1, length=0.33):
    arr = np.arange(start, stop, length, dtype=np.float16)
    pairs = [[arr[i], arr[i + 1]] for i in range(len(arr) - 1)]
    return pairs

@jit
def gaussian_log_prior(theta, sigma_prior):
    return -0.5 * jnp.sum(theta**2) / sigma_prior**2

def create_log_posterior(regression_model, y, sigma_prior, sigma_noise=1.0, log_prior_fn=None ):
    # Use provided prior or default to Gaussian
    if log_prior_fn is None:
        print(sigma_prior)
        log_prior_fn = partial(gaussian_log_prior, sigma_prior=sigma_prior)


    def log_posterior_fn(theta):
        log_like = regression_model.log_likelihood(theta, y)
        log_prior = log_prior_fn(theta)
        return log_like + log_prior

    return log_posterior_fn


# Generate samples from logistic regression problem
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


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    samples = sample_annuli(D=3, n_samples=1000, args=[0.9, 1])
    # Convert JAX arrays to NumPy for plotting
    samples_np = jnp.asarray(samples).copy()
    print(np.linalg.norm(samples, axis=1))
    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211, projection="3d")
    ax1.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], alpha=0.1)
    ax2 = fig.add_subplot(212)
    ax2.hist(np.linalg.norm(samples, axis=1), bins=100)
    ax2.set_xlabel("norm")
    ax2.set_ylabel("Samples")
    plt.title("Samples from Annuli")
    plt.tight_layout()
    plt.show()
