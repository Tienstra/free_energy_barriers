import jax.numpy as jnp
from jax import random, grad
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plots.plotter import TracePlot, NormPlot
from models.regression import DummyModel, StepRegression
from utils.tools import sample_annuli, deterministic_sample, sample_prior


class MALA:
    def __init__(
        self,
        regression_model,
        y,
        N=100,
        sigma_noise=1.0,
        epsilon=0.0001,
        n_steps=1000,
        n_chains=2,
        initializer=None,
        *args,
    ):
        self.regression_model = regression_model
        self.y = y
        self.N = N
        self.sigma_noise = sigma_noise
        self.sigma_prior = 1 / jnp.sqrt(self.N)
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.initializer = initializer
        self.init_ags = None
        self.theta_init = self.initialize_chains()
        self.key = random.PRNGKey(42) if key is None else key
        self.acceptance_ratio = 0
        self.rhat = 0

    def initialize_chains(self, *args, **kwargs):
        if self.initializer is not None:
            if self.init_ags is None:
                return self.initializer()
            else:
                return self.initializer(self.init_ags)
        else:
            # Default initialization (e.g., multivariate normal)
            return random.multivariate_normal(
                self.key, jnp.zeros(self.N), jnp.eye(self.N), shape=(self.n_chains,)
            )

    def log_likelihood(self, theta):
        """
        Log-likelihood for Gaussian likelihood.
        """
        y_pred = self.regression_model.evaluate(theta)
        residuals = self.y - y_pred
        return -0.5 * jnp.sum(residuals**2) / self.sigma_noise**2

    def log_prior(self, theta):
        """
        Log-prior for a Gaussian prior on theta.
        """
        return -0.5 * jnp.sum(theta**2) / self.sigma_prior**2

    def log_posterior(self, theta):
        """
        Log-posterior: Sum of log-likelihood and log-prior.
        """
        return self.log_likelihood(theta) + self.log_prior(theta)

    def gradient_log_posterior(self, theta):
        """
        Gradient of the log-posterior with respect to theta.
        """
        return grad(self.log_posterior)(theta)

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

    def sample(self):
        # Prepare to store all chains
        theta_chains = []
        acceptance_rates = []

        for chain_idx in range(self.n_chains):
            # Initialize for this chain with a unique key
            chain_key = random.split(self.key, self.n_chains)[chain_idx]
            theta_current = self.theta_init[
                chain_idx
            ]  # Use different starting points for each chain
            theta_chain = []
            accept_count = 0

            for i in range(self.n_steps):
                chain_key, subkey = random.split(chain_key)
                theta_current, accepted = self.mala_step(subkey, theta_current)
                theta_chain.append(theta_current)
                accept_count += accepted
            acceptance_rate = accept_count / self.n_steps
            acceptance_rates.append(acceptance_rate)

            # Convert chain list to a JAX array and append to the overall chains list
            theta_chains.append(jnp.array(theta_chain))
        self.acceptance_ratio = jnp.mean(acceptance_rate)
        # Convert the chains into a single JAX array
        # print(jnp.array(theta_chains).shape)
        # chains x itterations x dim theta
        return jnp.array(theta_chains)


if __name__ == "__main__":

    # Create some synthetic data for a Bayesian linear regression model
    key = random.PRNGKey(42)
    N = 100
    x = random.uniform(key, shape=(N,), minval=0.0, maxval=1.0)  # Input data
    reg_model = StepRegression(N)
    theta_true = jnp.zeros(N)
    y_observed = random.normal(key, shape=(N,)) * 0.5  # Add noise to the data

    # Initialize the MALA sampler
    mala_sampler = MALA(
        reg_model,
        y=y_observed,
        N=N,
        sigma_noise=1.0,
        epsilon=0.005,
        n_steps=50000,
        n_chains=2,
        key=key,
    )
    theta_chains = mala_sampler.sample()
    print("Acceptance ratio:", mala_sampler.acceptance_ratio)

    # Display posterior estimates
    theta_mean = jnp.mean(theta_chains[:, -2000:, :], axis=(1, 0))
    print("Posterior mean for theta:", theta_mean)
    theta_std = jnp.std(theta_chains[:, -2000:, :], axis=(1, 0))
    print(f"{theta_std = }")
    print("True theta values:", theta_true)
    # Generate trace plots
    # plotter = TracePlot(theta_chains)
    # plotter.plot_traces()
    # Create an instance of the NormPlotter with the sampled chains
    norm_plotter = NormPlot(theta_chains)

    # Plot the norm of the parameter vector at each iteration
    norm_plotter.plot_norm()
