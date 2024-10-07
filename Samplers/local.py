import jax
import jax.numpy as jnp
from jax import random, grad
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Plots.plotter import TracePlot, NormPlot
from Models.regression import DummyModel, StepRegression

class MALA:
    def __init__(self, regression_model, y, N=100, sigma_noise=1.0, sigma_prior=1.0, epsilon=0.05, n_steps=1000, n_chains=2,key=None):
        self.regression_model = regression_model
        self.y = y
        self.N = N
        self.sigma_noise = sigma_noise
        self.sigma_prior = sigma_prior
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.n_chains = n_chains 
        self.theta_init = self.initialize_chains()
        self.key = random.PRNGKey(42) if key is None else key
        self.acceptance_ratio = 0
        self.rhat = 0

    def initialize_chains(self, scale=10, key=random.PRNGKey(42)):
        # Split the random key into n_chains subkeys
        keys = random.split(key, self.n_chains)

        # Initialize an empty list to store the initial points for each chain
        theta_init_list = []

        # Loop over each key to generate the random starting points
        for i in range(self.n_chains):
            # Generate a random direction for the i-th chain
            # Mean vector of zeros
            mean = jnp.zeros(self.N)
            covariance = (jnp.sqrt(self.N)) * jnp.eye(self.N)
            theta_init = random.multivariate_normal(keys[i], mean, covariance)

            # Append the initial point to the list
            theta_init_list.append(theta_init)

        # Convert the list of initial points to a JAX array
        theta_init = jnp.stack(theta_init_list)

        return theta_init

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

    def sample(self):
        # Prepare to store all chains
        theta_chains = []
        acceptance_rates = []  

        for chain_idx in range(self.n_chains):
            # Initialize for this chain with a unique key
            chain_key = random.split(self.key, self.n_chains)[chain_idx]
            theta_current = self.theta_init[chain_idx]  # Use different starting points for each chain
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
        self.acceptance_ratio = jnp.mean(jnp.array(accepted))
        # Convert the chains into a single JAX array
        return jnp.array(theta_chains)


if __name__ == "__main__":

    # Create some synthetic data for a Bayesian linear regression model
    key = random.PRNGKey(42)
    x = random.uniform(key, shape=(100,), minval=0.0, maxval=1.0)  # Input data
    reg_model = StepRegression(100) 
    theta_true = jnp.zeros(100)
    y_observed = random.normal(key, shape=(100,)) * 0.5  # Add noise to the data

    # Initialize the MALA sampler
    mala_sampler = MALA(reg_model, y=y_observed, sigma_noise=1.0, sigma_prior=1.0, epsilon=0.05, n_steps=5000, n_chains=1,key=key)
    theta_chain = mala_sampler.sample()
    print('Acceptance ratio:', mala_sampler.acceptance_ratio)

    # Display posterior estimates
    theta_mean = jnp.mean(theta_chain, axis=(0,1))
    print("Posterior mean for theta:", theta_mean)
    print("True theta values:", theta_true)
    # Generate trace plots
    plotter = TracePlot(theta_chain)
    plotter.plot_traces()
    # Create an instance of the NormPlotter with the sampled chains
    norm_plotter = NormPlot(theta_chain)

    # Plot the norm of the parameter vector at each iteration
    norm_plotter.plot_norm()
