import numpy as np
import jax.numpy as jnp


def norm(samples):
    """
    Computes the norm of the chain for each iteration.

    Parameters:
        - samples (array): Array of samples of theta (n_chains x n_iter x n_dims)
    """

    norms = jnp.mean(jnp.linalg.norm(samples, axis=2), axis=0)
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
