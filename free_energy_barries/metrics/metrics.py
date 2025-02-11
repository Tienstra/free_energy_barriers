import numpy as np
import jax.numpy as jnp


def norm(chain, axs=1):
    """
    Computes the norm of the chain for each iteration.

    Parameters:
        - chain (array): Array of samples of theta (n_iter x n_dims)
        - axs (int 0 or 1): Int for the axis to compute the norm
    """

    norms = np.linalg.norm(chain, axis=axs)
    return norms


def mean(samples, iter=-1):
    """
    Computes the posterior mean of samples.

    Parameters:
        - samples (array): Array of samples of theta (n_chains x n_iter x n_dims)
        - iter (int): Integer for which iteration to compute the mean

    """
    sample_mean = jnp.mean(samples[:, iter:, :], axis=(1, 0))
    return sample_mean


def sd(samples, iter=-1):
    """
    Computes the posterior pointwise sd of samples.

    Parameters:
        - samples (array): Array of samples of theta (n_chains x n_iter x n_dims)
        - iter (int): Integer for which iteration to compute the mean

    """
    samples_sd = jnp.std(samples[:, iter:, :], axis=(1, 0))
    return samples_sd
