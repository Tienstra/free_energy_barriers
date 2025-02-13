import jax.numpy as jnp
from jax import random, grad
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
        self.y = y
        self.D = D
        self.sigma_noise = sigma_noise
        self.sigma_prior = 1 / jnp.sqrt(self.D)
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.initializer = initializer
        self.args = args
        self.theta_init = self.initialize_chains()
        self.key = random.PRNGKey(42) 
        self.acceptance_ratio = 0
        self.rhat = 0

    def initialize_chains(self):
        if self.initializer == "sample_annuli" and self.args != []:
                parms_str = str(self.args)
                print("Intialised with :" + self.initializer)
                print("Parameters :" + parms_str)
                return sample_annuli(self.D, self.n_chains, self.args)
        else:
            # Default initialization (sample prior)
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
    D = 3
    x = random.uniform(key, shape=(D,), minval=0.0, maxval=1.0)  # Input data
    reg_model = StepRegression(D)
    theta_true = jnp.zeros(D)
    y_observed = random.normal(key, shape=(D,)) * 0.5  # Add noise to the data

    # Initialize the MALA sampler
    mala_sampler = MALA(
        reg_model,
        y=y_observed,
        D=D,
        sigma_noise=1.0,
        epsilon=0.005,
        n_steps=5,
        n_chains=2,
        initializer=sample_annuli,
        args=[0.66, 1],
    )
    theta_chains = mala_sampler.sample()
    print("Acceptance ratio:", mala_sampler.acceptance_ratio)
    # Display posterior estimates
    print(f"{mean(theta_chains) = }")
    print(f"{sd(theta_chains) = }")
    print(f"{norm(theta_chains) = }")
    print("True theta values:", theta_true)
