import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial


class Kernel:
    """Base class for MCMC kernels"""

    def __init__(self, log_posterior):  # Tienstra
        self.log_posterior = log_posterior


    def step(self, rng_key, theta_current):
        """Perform one step of the kernel"""
        raise NotImplementedError("Subclasses must implement this method")


class MALAKernel(Kernel):
    """Metropolis-Adjusted Langevin Algorithm kernel"""

    def __init__(self, log_posterior, epsilon=0.0001):
        super().__init__(log_posterior)
        self.epsilon = epsilon


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
        delta = x_prime - x - 0.5 * (self.epsilon ** 2) * grad_log_pi_x
        return -jnp.sum(delta ** 2) / (2 * self.epsilon ** 2)

    def step(self, rng_key, theta_current):
        """
        Single MALA step: Propose a new theta using Langevin dynamics and accept/reject it.
        """

        key1, key2 = random.split(rng_key)
        # Compute gradient of log-posterior at current theta
        grad_log_post = self.gradient_log_posterior(theta_current)

        # Propose a new theta
        noise = random.normal(key1, shape=theta_current.shape)
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

        # Accept or reject the proposal
        accept = jnp.log(random.uniform(key2)) < log_accept_ratio
        theta_new = jnp.where(accept, theta_proposed, theta_current)

        return theta_new, accept