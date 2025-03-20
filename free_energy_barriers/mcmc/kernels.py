import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial


class Kernel:
    """Base class for MCMC kernels"""

    def __init__(self, log_posterior_fn, gradient_fn=None):  # Tienstra
        self.log_posterior_fn = log_posterior_fn

        # Create the gradient function if not provided
        if gradient_fn is not None:
            # Store the provided gradient function
            self.gradient_fn = gradient_fn
        else:
            # Create and JIT the gradient function
            self.gradient_fn = jit(grad(log_posterior_fn))

    def step(self, rng_key, theta_current):
        """Perform one step of the kernel"""
        raise NotImplementedError("Subclasses must implement this method")


class MALAKernel(Kernel):
    """Metropolis-Adjusted Langevin Algorithm kernel"""

    def __init__(self, log_posterior_fn, gradient_fn=None, epsilon=0.0001):
        super().__init__(log_posterior_fn, gradient_fn)
        self.epsilon = epsilon

    @partial(jit, static_argnums=(0,))
    def proposal_density_q(self, x_prime, x):
        """Compute proposal density
        Proposal density q(x' | x)
        q(x' | x) ∝ exp(-1 / 4τ * ||x' - x - τ ∇log π(x)||^2)

        Parameters:
        - x_prime: The proposed state.
        - x: The current state.
        - tau: The step size.

        Returns:
        - log_q: The log of the proposal density q(x' | x).
        """

        grad_log_pi_x = self.gradient_fn(x)
        delta = x_prime - x - 0.5 * (self.epsilon ** 2) * grad_log_pi_x
        return -jnp.sum(delta ** 2) / (2 * self.epsilon ** 2)

    @partial(jit, static_argnums=(0,))
    def step(self, rng_key, theta_current):
        """Single MALA step"""
        grad_log_post = self.gradient_fn(theta_current)

        noise = random.normal(rng_key, shape=theta_current.shape)
        theta_proposed = (
                theta_current + 0.5 * self.epsilon ** 2 * grad_log_post + self.epsilon * noise
        )

        log_post_current = self.log_posterior_fn(theta_current)
        log_post_proposed = self.log_posterior_fn(theta_proposed)

        log_q_current_given_proposed = self.proposal_density_q(
            theta_current, theta_proposed
        )
        log_q_proposed_given_current = self.proposal_density_q(
            theta_proposed, theta_current
        )

        log_accept_ratio = (log_post_proposed + log_q_current_given_proposed) - (
                log_post_current + log_q_proposed_given_current
        )

        accept = jnp.log(random.uniform(rng_key)) < log_accept_ratio
        theta_new = jnp.where(accept, theta_proposed, theta_current)

        return theta_new, accept