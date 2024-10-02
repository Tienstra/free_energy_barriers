import jax
import jax.numpy as jnp
from jax import random, grad
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Plots.plotter import TracePlot
from Models.regression import DummyModel

class MALA:
    def __init__(self, regression_model, y, sigma_noise=1.0, sigma_prior=1.0, epsilon=0.05, n_steps=1000, key=None):
        """
        Initializes the MALA sampler for Bayesian linear regression.
        
        Parameters:
        X (array): Design matrix (n_samples x n_features)
        y (array): Response variable (n_samples)
        sigma_noise (float): Noise standard deviation
        sigma_prior (float): Prior standard deviation
        epsilon (float): Step size for MALA updates
        n_steps (int): Number of MCMC steps
        key (jax.random.PRNGKey): Random key for reproducibility
        """
        self.regression_model = regression_model
        self.y = y
        self.sigma_noise = sigma_noise
        self.sigma_prior = sigma_prior
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.key = random.PRNGKey(42) if key is None else key

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

    def mala_step(self, rng_key, theta_current):
        """
        Single MALA step: Propose a new theta using Langevin dynamics and accept/reject it.
        """
        # Compute gradient of log-posterior at current theta
        grad_log_post = self.gradient_log_posterior(theta_current)

        # Propose a new theta
        noise = random.normal(rng_key, shape=theta_current.shape)
        theta_proposed = theta_current + 0.5 * self.epsilon**2 * grad_log_post + self.epsilon * noise

        # Compute log-posterior for current and proposed thetas
        log_post_current = self.log_posterior(theta_current)
        log_post_proposed = self.log_posterior(theta_proposed)

        # Compute acceptance ratio (simplified for symmetric proposal)
        log_accept_ratio = log_post_proposed - log_post_current

        # Accept or reject the proposal
        accept = jnp.log(random.uniform(rng_key)) < log_accept_ratio
        theta_new = jnp.where(accept, theta_proposed, theta_current)

        return theta_new, accept

    def sample(self, theta_init):
        """
        Run MALA for a specified number of iterations.
        
        Parameters:
        theta_init (array): Initial value of theta (parameters)
        
        Returns:
        theta_chain (array): Chain of sampled theta values
        """
        theta_chain = []
        theta_current = theta_init
        accept_count = 0

        # Iterate for a specified number of steps
        for i in range(self.n_steps):
            self.key, subkey = random.split(self.key)
            theta_current, accepted = self.mala_step(subkey, theta_current)
            theta_chain.append(theta_current)
            accept_count += accepted

        acceptance_rate = accept_count / self.n_steps
        print(f"Acceptance rate: {acceptance_rate:.4f}")
        
        return jnp.array(theta_chain)


if __name__ == "__main__":

    # Create some synthetic data for a Bayesian linear regression model
    key = random.PRNGKey(42)
    x = jnp.linspace(0, 10, 100)  # Input data
    reg_model = DummyModel(x) 
    theta_true = jnp.zeros(100)
    y_observed = random.normal(key, shape=(100,)) * 0.5  # Add noise to the data

    # Initialize the MALA sampler
    mala_sampler = MALA(reg_model, y=y_observed, sigma_noise=1.0, sigma_prior=1.0, epsilon=0.05, n_steps=5000, key=key)

    # Run MALA with an initial guess for theta
    theta_init = jnp.zeros(100)
    theta_chain = mala_sampler.sample(theta_init)

    # Display posterior estimates
    theta_mean = jnp.mean(theta_chain, axis=0)
    print("Posterior mean for theta:", theta_mean)
    print("True theta values:", theta_true)
    # Generate trace plots
    plotter = TracePlot(theta_chain)
    plotter.plot_traces()
