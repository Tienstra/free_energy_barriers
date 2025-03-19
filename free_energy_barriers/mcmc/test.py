import jax.numpy as jnp
from jax import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.regression import DummyModel
from samplers.local import MALA


def test_prior_sampling():
    # Test parameters
    D = 100
    n_steps = 1000
    n_chains = 20
    epsilon = 0.005

    # Initialize with DummyModel to effectively remove likelihood
    model = DummyModel(D)
    y_observed = jnp.zeros(D)

    # Create MALA sampler
    mala_sampler = MALA(
        model,
        y=y_observed,
        D=D,
        sigma_noise=1.0,
        epsilon=epsilon,
        n_steps=n_steps,
        n_chains=n_chains,
        initializer="sample_prior",
    )

    # Sample
    theta_chains = mala_sampler.sample(thin=100)

    # Analysis
    final_samples = theta_chains[:, -1, :]  # Shape: (n_chains, D)

    # Compute statistics
    norms = jnp.linalg.norm(final_samples, axis=1)
    mean_norm = jnp.mean(norms)
    std_norm = jnp.std(norms)

    # Component-wise statistics
    mean_component = jnp.mean(final_samples)
    std_component = jnp.std(final_samples)

    print(f"Expected norm (theory): 1.0")
    print(f"Actual mean norm: {mean_norm:.3f} ± {std_norm:.3f}")
    print(f"Component mean: {mean_component:.3f}")
    print(f"Component std: {std_component:.3f}")
    print(f"Expected component std (theory): {1/jnp.sqrt(D):.3f}")
    print(f"Acceptance ratio: {mala_sampler.acceptance_ratio:.3f}")

    # Check mixing
    early_norms = jnp.linalg.norm(theta_chains[:, 1, :], axis=1)
    mid_norms = jnp.linalg.norm(theta_chains[:, len(theta_chains[0]) // 2, :], axis=1)
    late_norms = norms

    print("\nMixing check (norms at different times):")
    print(f"Early: {jnp.mean(early_norms):.3f}")
    print(f"Middle: {jnp.mean(mid_norms):.3f}")
    print(f"Late: {jnp.mean(late_norms):.3f}")

    return theta_chains


if __name__ == "__main__":
    chains = test_prior_sampling()
