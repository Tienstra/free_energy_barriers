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
            print("initialized with sample annuli")
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
    def log_likelihood(self, theta):
        """
        Log-likelihood for Gaussian likelihood.
        """
        y_pred = self.regression_model.evaluate(theta)
        residuals = self.y - y_pred
        return -0.5 * jnp.sum(residuals**2) / self.sigma_noise**2

    @partial(jit, static_argnums=(0,))
    def log_prior(self, theta):
        """
        Log-prior for a Gaussian prior on theta.
        """
        return -0.5 * jnp.sum(theta**2) / self.sigma_prior**2

    @partial(jit, static_argnums=(0,))
    def log_posterior(self, theta):
        """
        Log-posterior: Sum of log-likelihood and log-prior.
        """
        return self.log_likelihood(theta) + self.log_prior(theta)

    @partial(jit, static_argnums=(0,))
    def gradient_log_posterior(self, theta):
        """
        Gradient of the log-posterior with respect to theta.
        """
        return grad(self.log_posterior)(theta)

    @partial(jit, static_argnums=(0,))
    def proposal_density_q(self, x_prime, x):
        """
        Proposal density q(x' | x) using the MALA formula:
        q(x' | x) ∝ exp(-1 / 4τ * ||x' - x - τ ∇log π(x)||^2)

        Parameters:
        - x_prime: The proposed state.
        - x: The current state.
        - tau: The step size.

        Returns:
        - log_q: The log of the proposal density q(x' | x).
        """
        grad_log_pi_x = grad(self.log_posterior)(x)
        delta = x_prime - x - 0.5 * (self.epsilon**2) * grad_log_pi_x
        return -jnp.sum(delta**2) / (2 * self.epsilon**2)

    @partial(jit, static_argnums=(0,))
    def mala_step(self, rng_key, theta_current):
        """
        Single MALA step: Propose a new theta using Langevin dynamics and accept/reject it.
        """
        # Compute gradient of log-posterior at current theta
        grad_log_post = self.gradient_log_posterior(theta_current)

        # Propose a new theta
        noise = random.normal(rng_key, shape=theta_current.shape)
        theta_proposed = (
            theta_current + 0.5 * self.epsilon**2 * grad_log_post + self.epsilon * noise
        )

        # Compute log-posterior for current and proposed thetas
        log_post_current = self.log_posterior(theta_current)
        log_post_proposed = self.log_posterior(theta_proposed)

        # Compute proposal densities q(x' | x) and q(x | x')
        log_q_current_given_proposed = self.proposal_density_q(
            theta_current, theta_proposed
        )
        log_q_proposed_given_current = self.proposal_density_q(
            theta_proposed, theta_current
        )

        # Compute acceptance ratio
        log_accept_ratio = (log_post_proposed + log_q_current_given_proposed) - (
            log_post_current + log_q_proposed_given_current
        )

        # Compute acceptance ratio (simplified for symmetric proposal)
        # log_accept_ratio = log_post_proposed - log_post_current

        # Accept or reject the proposal
        accept = jnp.log(random.uniform(rng_key)) < log_accept_ratio
        theta_new = jnp.where(accept, theta_proposed, theta_current)

        return theta_new, accept

    def sample(self, thin=5000):
        chains = []
        accept_counts = []

        # Initialize storage for samples (adjusted for thinning)

        for chain_idx in range(self.n_chains):
            key = random.fold_in(self.key, chain_idx)
            theta_current = self.theta_init[chain_idx]
            chain = [theta_current]
            accept_count = 0

            for step in range(self.n_steps):
                key, subkey = random.split(key)
                theta_current, accepted = self.mala_step(subkey, theta_current)
                if step % thin == 0:
                    chain.append(theta_current)
                accept_count += accepted

            chains.append(jnp.stack(chain))
            accept_counts.append(accept_count / self.n_steps)

        self.acceptance_ratio = jnp.mean(jnp.array(accept_counts))
        return jnp.stack(chains)
