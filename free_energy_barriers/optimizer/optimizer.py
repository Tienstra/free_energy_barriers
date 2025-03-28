# optimizer.py

import jax.numpy as jnp
from jax import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.tools import sample_annuli, sample_stdnorm_prior


class Optimizer:
    """Optimizer that takes in a kernel for gradient-based optimization"""

    def __init__(
            self,
            kernel,
            D=100,
            n_steps=1000,
            n_chains=2,
            initializer=None,
            init_args=[],
            seed=42,
    ):
        self.kernel = kernel
        self.D = D
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.sigma_prior = 1 / jnp.sqrt(self.D)
        self.key = random.PRNGKey(seed)

        # Initialize chains
        self.theta_init = self._initialize_chains(initializer, init_args)
        self.final_states = None

    def _initialize_chains(self, initializer, init_args):
        if initializer == "sample_annuli" and init_args:
            print("initialized with sample annuli")
            return sample_annuli(self.D, self.n_chains, init_args)
        elif initializer == "sample_prior":
            print("initialized with sample std prior")
            return (
                    random.normal(self.key, shape=(self.n_chains, self.D))
                    * self.sigma_prior
            )
        else:
            print("initialized at 0")
            return jnp.zeros(shape=(self.n_chains, self.D))

    def optimize(self):
        """
        Run optimization for n_steps, using the provided kernel.
        Returns the final optimized states.
        """
        final_states = []
        objective_values = []

        for chain_idx in range(self.n_chains):
            key = random.fold_in(self.key, chain_idx)
            theta_current = self.theta_init[chain_idx]

            # Run optimization for n_steps
            for step in range(self.n_steps):
                key, subkey = random.split(key)
                theta_current, _ = self.kernel.step(subkey, theta_current)

            # Store the final optimized state
            final_states.append(theta_current)

            # Calculate final objective value
            final_objective = self.kernel.log_posterior(theta_current)
            objective_values.append(final_objective)
            print(f"Chain {chain_idx} final objective: {final_objective}")

        self.final_states = jnp.stack(final_states)
        self.final_objectives = jnp.array(objective_values)

        return self.final_states