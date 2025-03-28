# optimization_kernels.py

import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

class Kernel:
    """Base class for MCMC kernels"""

    def __init__(self, log_posterior):
        self.log_posterior = log_posterior

    def step(self, rng_key, theta_current):
        """Perform one step of the kernel"""
        raise NotImplementedError("Subclasses must implement this method")

    def __str__(self):
        return "Kernel"
    def __repr__(self):
        return self.__str__()


class GradientDescentKernel(Kernel):
    """Gradient Descent optimization kernel"""

    def __str__(self):
        return f"GradientDescent(step_size={self.step_size})"

    def __init__(self, log_posterior, step_size=0.01):
        super().__init__(log_posterior)
        self.step_size = step_size

    @partial(jit, static_argnums=(0,))
    def step(self, rng_key, theta_current):
        """
        Single gradient descent step.
        Maximizes the log_posterior (which is equivalent to minimizing negative log_posterior).
        """
        # Compute gradient of log-posterior at current theta
        gradient = grad(self.log_posterior)(theta_current)

        # Update theta using gradient ascent (since we're maximizing log_posterior)
        theta_new = theta_current + self.step_size * gradient

        # No acceptance criterion for pure optimization
        # Return True for "accepted" to maintain API consistency
        return theta_new, True


class SGDKernel(Kernel):
    """Stochastic Gradient Descent optimization kernel with optional momentum"""

    def __str__(self):
        return f"SGD(step_size={self.step_size}, momentum={self.momentum})"

    def __init__(self, log_posterior, step_size=0.01, momentum=0.9, use_momentum=True):
        super().__init__(log_posterior)
        self.step_size = step_size
        self.momentum = momentum
        self.use_momentum = use_momentum
        self.velocity = None

    @partial(jit, static_argnums=(0,))
    def step(self, rng_key, theta_current):
        """
        Single stochastic gradient descent step with optional momentum.
        """
        # Compute gradient of log-posterior at current theta
        gradient = grad(self.log_posterior)(theta_current)

        # Add some noise to the gradient to make it stochastic
        key1, _ = random.split(rng_key)
        noise = random.normal(key1, shape=theta_current.shape) * 0.01
        noisy_gradient = gradient + noise

        # Apply momentum if enabled
        if self.use_momentum:
            # Initialize velocity if this is the first step
            if self.velocity is None:
                self.velocity = jnp.zeros_like(theta_current)

            # Update velocity with momentum
            self.velocity = self.momentum * self.velocity + self.step_size * noisy_gradient

            # Update theta using the velocity
            theta_new = theta_current + self.velocity
        else:
            # Simple SGD update without momentum
            theta_new = theta_current + self.step_size * noisy_gradient

        # No acceptance criterion for pure optimization
        return theta_new, True


class AdamKernel(Kernel):
    """Adam optimizer kernel"""

    def __str__(self):
        return f"Adam(step_size={self.step_size})"

    def __init__(self, log_posterior, step_size=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(log_posterior)
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    @partial(jit, static_argnums=(0,))
    def step(self, rng_key, theta_current):
        """
        Single Adam optimization step.
        """
        # Compute gradient
        gradient = grad(self.log_posterior)(theta_current)

        # Initialize moment estimates if this is the first step
        if self.m is None:
            self.m = jnp.zeros_like(theta_current)
            self.v = jnp.zeros_like(theta_current)

        # Update timestep
        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        theta_new = theta_current + self.step_size * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

        return theta_new, True