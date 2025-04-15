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

    def __str__(self):
        return "Kernel"
    def __repr__(self):
        return self.__str__()


class MALAKernel(Kernel):
    """Metropolis-Adjusted Langevin Algorithm kernel"""

    def __str__(self):
        return "MALA"

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
        delta = x_prime - x - 0.5 * (self.epsilon**2) * grad_log_pi_x
        return -jnp.sum(delta**2) / (2 * self.epsilon**2)

    @partial(jit, static_argnums=(0,))
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

class GradientDescentKernel(Kernel):
    """
    This kernel implements deterministic gradient-based optimization to find the mode
    of the target distribution by maximizing the log-posterior.
    """

    def __init__(self, log_posterior, step_size=0.01, decay_rate=0.999):
        super().__init__(log_posterior)
        self.step_size = step_size
        self.decay_rate = decay_rate
        self._step_count = 0

    def __str__(self):
        return f"GradientDescent(step_size={self.step_size}, decay_rate={self.decay_rate})"

    @partial(jit, static_argnums=(0,))
    def _compute_gradient(self, theta):
        """Compute gradient of log-posterior with respect to theta."""
        return grad(self.log_posterior)(theta)

    @partial(jit, static_argnums=(0,))
    def step(self, rng_key, theta_current):
        """
        Single gradient descent step.

        Args:
            rng_key: JAX random key (unused for deterministic updates but kept for API consistency)
            theta_current: Current parameter values

        Returns:
            theta_new: Updated parameter values
            accepted: Always True for optimization kernels
        """
        # Compute gradient of log-posterior at current theta
        gradient = self._compute_gradient(theta_current)

        # Compute adaptive step size with decay
        effective_step_size = self.step_size * (self.decay_rate ** self._step_count)

        # Update theta using gradient ascent (since we're maximizing log_posterior)
        theta_new = theta_current + effective_step_size * gradient

        # Increment step counter
        self._step_count += 1

        # No acceptance criterion for pure optimization so it returns True and hence will have accept-rate =1
        return theta_new, True


class pCNKernel(Kernel):
    """
    Preconditioned Crank-Nicolson (pCN) kernel
    """

    def __init__(self, log_posterior, beta=0.1, prior_std=1.0):
        super().__init__(log_posterior)
        # Step size parameter (controls how far proposals can go)
        self.beta = beta  # beta in pCN literature
        # Standard deviation of the Gaussian prior
        self.prior_std = prior_std

    def __str__(self):
        return f"pCN(step_size={self.beta}, prior_std={self.prior_std})"

    @partial(jit, static_argnums=(0,))
    def step(self, rng_key, theta_current):
        """
        Single pCN step

        Args:
            rng_key: JAX random key
            theta_current: Current parameter values

        Returns:
            theta_new: Updated parameter values
            accepted: Boolean indicating whether the proposal was accepted
        """
        key_proposal, key_accept = random.split(rng_key)

        # Generate standard normal noise
        xi = random.normal(key_proposal, shape=theta_current.shape)

        # pCN scaling beta parameters
        sqrt_beta = jnp.sqrt(1 - self.beta ** 2)
        theta_proposed = sqrt_beta * theta_current + self.beta * xi

        # Compute log-posterior for current and proposed thetas
        log_post_current = self.log_posterior(theta_current)
        log_post_proposed = self.log_posterior(theta_proposed)

        # For pCN, the acceptance ratio simplifies to the likelihood ratio
        # (prior terms cancel out exactly)
        log_accept_ratio = log_post_proposed - log_post_current


        # Accept or reject the proposal
        accept = random.uniform(key_accept) < jnp.exp(jnp.minimum(0.0, log_accept_ratio))
        theta_new = jnp.where(accept, theta_proposed, theta_current)

        return theta_new, accept




