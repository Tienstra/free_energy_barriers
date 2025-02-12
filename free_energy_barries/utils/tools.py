import numpy as np
import jax.numpy as jnp
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.spherical_coords import sample_spherical_coords, spherical_to_cartesian


def sample_annuli(n_dim=3, n_samples=5000, r_low=0.66, r_high=1):
    """
    Samples from the annuli with specified radius

    Parameters:
        - n_dims : dimension of samples
        - n_samples : number of samples to be drawn
        - r_low : lower bound for radius (inner radius)
        - r_high: upper bound for radius (outer radius)
    """
    # Generate samples
    r, phis = sample_spherical_coords(n_dim, n_samples, r_low, r_high)

    # Convert to Cartesian coordinates
    samples = spherical_to_cartesian(r, phis)

    return samples


def deterministic_sample(N, scale=10):
    return scale * jnp.ones(N)


def sample_prior(key, mean, covariance, n_samples):
    return random.multivariate_normal(key, mean, covariance, shape=(n_samples,))

    return jnp.stack([theta_init for _ in range(self.n_chains)])


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    samples = sample_annuli()
    print(np.linalg.norm(samples, axis=1))
    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211, projection="3d")
    ax1.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.1)
    ax2 = fig.add_subplot(212)
    ax2.hist(np.linalg.norm(samples, axis=1), bins=100)
    ax2.set_xlabel("norm")
    ax2.set_ylabel("Samples")
    plt.title("Samples from Annuli")
    plt.tight_layout()
    plt.show()
