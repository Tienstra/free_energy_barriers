import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plots.plotter import TracePlot, NormPlot
from models.regression import DummyModel, StepRegression
from utils.tools import sample_annuli, sample_stdnorm_prior
from metrics.metrics import mean, sd, norm


class MALASampleNormal:
    def __init__(
        self,
        D=100,
        sigma_prior=None,
        epsilon=0.0001,
        n_steps=1000,
        n_chains=2,
        initializer=None,
        args=[],
    ):
        self.D = D
        self.sigma_prior = 1 / jnp.sqrt(self.D) if sigma_prior is None else sigma_prior
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.key = random.PRNGKey(42)

        # Initialize chains
        self.theta_init = self._initialize_chains(initializer, args)
        self.acceptance_ratio = 0

    def _initialize_chains(self, initializer, args):
        if initializer == "sample_annuli" and args:
            print("initialized with sample annuli")
            # You'll need to make sure sample_annuli is properly imported
            return sample_annuli(self.D, self.n_chains, args)
        elif initializer == "sample_prior":
            print("initialized with sample std prior")
            return (
                random.normal(self.key, shape=(self.n_chains, self.D))
                * self.sigma_prior
            )
        else:
            print("initialized at 0")
            return jnp.zeros(shape=(self.n_chains, self.D))

    @partial(jit, static_argnums=(0,))
    def log_prior(self, theta):
        return -0.5 * jnp.sum(theta**2) / self.sigma_prior**2

    @partial(jit, static_argnums=(0,))
    def log_posterior(self, theta):
        # In this case, posterior = prior (no likelihood)
        return self.log_prior(theta)

    @partial(jit, static_argnums=(0,))
    def mala_step(self, key, theta_current):
        key1, key2 = random.split(key)

        # Compute gradient and propose new theta
        grad_log_post = grad(self.log_posterior)(theta_current)
        noise = random.normal(key1, shape=theta_current.shape)
        theta_proposed = (
            theta_current + 0.5 * self.epsilon**2 * grad_log_post + self.epsilon * noise
        )

        # Compute log probabilities
        log_post_current = self.log_posterior(theta_current)
        log_post_proposed = self.log_posterior(theta_proposed)

        # Compute acceptance ratio
        log_accept_ratio = log_post_proposed - log_post_current

        # Accept/reject
        accept = jnp.log(random.uniform(key2)) < log_accept_ratio
        theta_new = jnp.where(accept, theta_proposed, theta_current)

        return theta_new, accept

    def sample(self):
        chains = []
        accept_counts = []

        for chain_idx in range(self.n_chains):
            key = random.fold_in(self.key, chain_idx)
            theta_current = self.theta_init[chain_idx]
            chain = [theta_current]
            accept_count = 0

            for step in range(self.n_steps):
                key, subkey = random.split(key)
                theta_current, accepted = self.mala_step(subkey, theta_current)
                chain.append(theta_current)
                accept_count += accepted

            chains.append(jnp.stack(chain))
            accept_counts.append(accept_count / self.n_steps)

        self.acceptance_ratio = jnp.mean(jnp.array(accept_counts))
        return jnp.stack(chains)
