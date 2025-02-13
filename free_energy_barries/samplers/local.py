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


class MALA:
    def __init__(
        self,
        regression_model,
        y,
        D=100,
        sigma_noise=1.0,
        epsilon=0.0001,
        n_steps=1000,
        n_chains=2,
        initializer=None,
        args=[],
    ):
        self.regression_model = regression_model
        self.y = jnp.asarray(y)
        self.D = D
        self.sigma_noise = sigma_noise
        self.sigma_prior = 1 / jnp.sqrt(self.D)
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.key = random.PRNGKey(42)

        # Initialize chains
        self.theta_init = self._initialize_chains(initializer, args)
        self.acceptance_ratio = 0

    def _initialize_chains(self, initializer, args):
        if initializer == "sample_annuli" and args:
            return sample_annuli(self.D, self.n_chains, args)
        return random.normal(self.key, shape=(self.n_chains, self.D)) * self.sigma_prior

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, theta):
        y_pred = self.regression_model.evaluate(theta)
        residuals = self.y - y_pred
        return -0.5 * jnp.sum(residuals**2) / self.sigma_noise**2

    @partial(jit, static_argnums=(0,))
    def log_prior(self, theta):
        return -0.5 * jnp.sum(theta**2) / self.sigma_prior**2

    @partial(jit, static_argnums=(0,))
    def log_posterior(self, theta):
        return self.log_likelihood(theta) + self.log_prior(theta)

    def gradient_log_posterior(self, theta):
        return grad(self.log_posterior)(theta)

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


if __name__ == "__main__":
    # Test code
    key = random.PRNGKey(42)
    D = 100
    reg_model = StepRegression(D)
    theta_true = jnp.zeros(D)
    y_observed = random.normal(key, shape=(D,)) * 0.5

    mala_sampler = MALA(
        reg_model,
        y=y_observed,
        D=D,
        sigma_noise=1.0,
        epsilon=0.005,
        n_steps=5,
        n_chains=20,
        initializer="sample_annuli",
        args=[0.66, 1],
    )

    theta_chains = mala_sampler.sample()
    print("Acceptance ratio:", mala_sampler.acceptance_ratio)
    # Display posterior estimates
    print(f"{mean(theta_chains) = }")
    print(f"{sd(theta_chains) = }")
    print(f"{norm(theta_chains) = }")
   
