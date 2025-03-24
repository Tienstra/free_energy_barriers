# mcmc/sampler.py

import jax.numpy as jnp
from jax import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.tools import sample_annuli, sample_stdnorm_prior


class MCMC:
    """MCMC sampler that takes in a kernel"""

    def __init__(
            self,
            kernel,
            D=100,
            n_steps=1000,
            n_chains=2,
            initializer=None,
            init_args=[],
            seed=42
    ):
        self.kernel = kernel
        self.D = D
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.sigma_prior = 1 / jnp.sqrt(self.D)
        self.key = random.PRNGKey(seed)

        # Initialize chains
        self.theta_init = self._initialize_chains(initializer, init_args)
        self.acceptance_ratio = 0

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

    def sample(self, thin=None):
        if thin is None:
            thin = int(self.D / 10)

        chains = []
        accept_counts = []

        for chain_idx in range(self.n_chains):
            key = random.fold_in(self.key, chain_idx)
            theta_current = self.theta_init[chain_idx]
            chain = [theta_current]
            accept_count = 0

            for step in range(self.n_steps):
                key, subkey = random.split(key)
                theta_current, accepted = self.kernel.step(subkey, theta_current)
                if step % thin == 0:
                    chain.append(theta_current)
                accept_count += accepted

            chains.append(jnp.stack(chain))
            accept_counts.append(accept_count / self.n_steps)

        self.acceptance_ratio = jnp.mean(jnp.array(accept_counts))
        return jnp.stack(chains)