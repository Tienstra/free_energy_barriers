import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

from mcmc.kernels import MALAKernel, Kernel
from mcmc.sampler import MCMC
from models.regression import LogisticRegression
from utils.tools import generate_synthetic_data, create_log_posterior

# Set up plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="bright", rc=custom_params)


class GradientDescentKernel(Kernel):
    """Gradient Descent optimization kernel with gradient clipping"""

    def __str__(self):
        return f"GradientDescent(step_size={self.step_size})"

    def __init__(self, log_posterior, step_size=0.01, max_grad_norm=10.0):
        super().__init__(log_posterior)
        self.step_size = step_size
        self.max_grad_norm = max_grad_norm

    @partial(jit, static_argnums=(0,))
    def step(self, rng_key, theta_current):
        """
        Single gradient descent step with gradient clipping.
        Maximizes the log_posterior (which is equivalent to minimizing negative log_posterior).
        """
        # Compute gradient of log-posterior at current theta
        gradient = grad(self.log_posterior)(theta_current)

        # Clip gradients to avoid explosion
        grad_norm = jnp.linalg.norm(gradient)
        # Avoid division by zero or very small values
        safe_norm = jnp.maximum(grad_norm, 1e-8)

        # Only clip if norm is greater than threshold
        gradient = jnp.where(
            grad_norm > self.max_grad_norm,
            gradient * self.max_grad_norm / safe_norm,
            gradient,
        )

        # Check for any NaN in gradient and replace with zeros
        gradient = jnp.nan_to_num(gradient, nan=0.0)

        # Update theta using gradient ascent (since we're maximizing log_posterior)
        theta_new = theta_current + self.step_size * gradient

        # No acceptance criterion for pure optimization
        # Return True for "accepted" to maintain API consistency
        return theta_new, True


def run_mala(dim, n_steps, n_chains):
    """Run MALA sampling for logistic regression"""
    print(f"\n===== Running MALA Sampling (D={dim}, steps={n_steps}) =====")

    # Generate synthetic data
    n_samples = dim
    n_features = dim
    X, y, true_beta = generate_synthetic_data(n_samples, n_features)

    # Convert to JAX arrays
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    # Initialize model
    model = LogisticRegression(N=n_samples, p=n_features)

    # Create log posterior
    sigma_prior = 1 / jnp.sqrt(dim)
    log_posterior_fn = create_log_posterior(model, X_jax, y_jax, sigma_prior)

    # Initialize kernel
    epsilon = 1 / dim
    mala_kernel = MALAKernel(log_posterior_fn, epsilon=epsilon)

    # Initialize sampler
    mcmc = MCMC(
        kernel=mala_kernel,
        D=n_features,
        n_steps=n_steps,
        n_chains=n_chains,
        initializer="sample_prior",
        init_args=[],
        seed=45,
    )

    # Run sampling and time it
    start_time = time.time()
    samples = mcmc.sample(thin=1)  # Save all samples for tracking convergence
    end_time = time.time()

    # Calculate runtime
    runtime = end_time - start_time

    # Find MAP estimate (sample with highest log posterior value)
    log_posts = jnp.array([log_posterior_fn(state) for state in samples[0]])
    map_idx = jnp.argmax(log_posts)
    map_estimate = samples[0, map_idx]

    # Calculate parameter norms for each iteration
    param_norms = jnp.linalg.norm(samples[0], axis=1)

    # Set model parameters to MAP estimate for prediction
    model.beta = map_estimate

    # Compute prediction metrics on training data
    y_pred = model.predict(X_jax.T)
    accuracy = jnp.mean(y_pred == y_jax)

    # Calculate prediction error at each iteration
    prediction_errors = []
    for i in range(
        0, samples.shape[1], max(1, samples.shape[1] // 100)
    ):  # Sample ~100 points to avoid too many calculations
        model.beta = samples[0, i]
        y_pred = model.predict(X_jax.T)
        error = 1 - jnp.mean(y_pred == y_jax)  # Error rate = 1 - accuracy
        prediction_errors.append((i, error))

    prediction_error_indices = [p[0] for p in prediction_errors]
    prediction_error_values = [p[1] for p in prediction_errors]

    print(f"MALA runtime: {runtime:.2f} seconds")
    print(f"MALA MAP log posterior: {log_posts[map_idx]:.4f}")
    print(f"MALA training accuracy: {accuracy:.4f}")
    print(f"MALA MAP estimate norm: {jnp.linalg.norm(map_estimate):.4f}")

    return {
        "map_estimate": map_estimate,
        "param_norms": param_norms,
        "prediction_error_indices": prediction_error_indices,
        "prediction_error_values": prediction_error_values,
        "runtime": runtime,
        "log_posterior": log_posts[map_idx],
        "accuracy": accuracy,
        "true_beta": true_beta,
    }


def run_gradient_descent(dim, n_steps, n_chains):
    """Run gradient descent optimization for logistic regression"""
    print(f"\n===== Running Gradient Descent (D={dim}, steps={n_steps}) =====")

    # Generate synthetic data - use same seed as MALA
    n_samples = dim
    n_features = dim
    X, y, true_beta = generate_synthetic_data(n_samples, n_features)

    # Convert to JAX arrays
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    # Initialize model
    model = LogisticRegression(N=n_samples, p=n_features)

    # Create log posterior
    sigma_prior = 1 / jnp.sqrt(dim)
    log_posterior_fn = create_log_posterior(model, X_jax, y_jax, sigma_prior)

    # Initialize GD kernel
    step_size = 0.05  # Can be tuned
    gd_kernel = GradientDescentKernel(log_posterior_fn, step_size=step_size)

    # Use MCMC framework for fair comparison with same initialization
    optimizer = MCMC(
        kernel=gd_kernel,
        D=n_features,
        n_steps=n_steps,
        n_chains=n_chains,
        initializer="sample_prior",
        init_args=[],
        seed=45,
    )

    # Run optimization and time it
    start_time = time.time()
    samples = optimizer.sample(thin=1)  # Save all samples for tracking convergence
    end_time = time.time()

    # Calculate runtime
    runtime = end_time - start_time

    # Find MAP estimate (sample with highest log posterior value)
    log_posts = jnp.array([log_posterior_fn(state) for state in samples[0]])
    map_idx = jnp.argmax(log_posts)
    map_estimate = samples[0, map_idx]

    # Calculate parameter norms for each iteration
    param_norms = jnp.linalg.norm(samples[0], axis=1)

    # Set model parameters to MAP estimate for prediction
    model.beta = map_estimate

    # Compute prediction metrics on training data
    y_pred = model.predict(X_jax.T)
    accuracy = jnp.mean(y_pred == y_jax)

    # Calculate prediction error at each iteration
    prediction_errors = []
    for i in range(
        0, samples.shape[1], max(1, samples.shape[1] // 100)
    ):  # Sample ~100 points to avoid too many calculations
        model.beta = samples[0, i]
        y_pred = model.predict(X_jax.T)
        error = 1 - jnp.mean(y_pred == y_jax)  # Error rate = 1 - accuracy
        prediction_errors.append((i, error))

    prediction_error_indices = [p[0] for p in prediction_errors]
    prediction_error_values = [p[1] for p in prediction_errors]

    print(f"GD runtime: {runtime:.2f} seconds")
    print(f"GD MAP log posterior: {log_posts[map_idx]:.4f}")
    print(f"GD training accuracy: {accuracy:.4f}")
    print(f"GD MAP estimate norm: {jnp.linalg.norm(map_estimate):.4f}")

    return {
        "map_estimate": map_estimate,
        "param_norms": param_norms,
        "prediction_error_indices": prediction_error_indices,
        "prediction_error_values": prediction_error_values,
        "runtime": runtime,
        "log_posterior": log_posts[map_idx],
        "accuracy": accuracy,
        "true_beta": true_beta,
    }


def plot_map_comparison(mala_results, gd_results):
    """Plot simplified comparison focusing only on parameter norms over iterations"""
    # Create a single figure for parameter norms
    plt.figure(figsize=(10, 6))

    # Plot parameter norms over iterations
    plt.plot(
        mala_results["param_norms"][-100:], label="MALA Parameter Norm", linewidth=2
    )
    plt.plot(gd_results["param_norms"][-100:], label="GD Parameter Norm", linewidth=2)
    plt.ylim((-1, 2))
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Parameter Norm", fontsize=12)
    plt.title("Parameter Norm Over Iterations", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("mala_vs_gd_norm_comparison.png", dpi=300)
    plt.show()


def main():
    # Parameters
    dim = 1000  # Dimensionality (features and samples)
    n_steps = 50000  # Number of steps
    n_chains = 1  # Only need 1 chain for MAP comparison

    # Run MALA sampling
    mala_results = run_mala(dim, n_steps, n_chains)

    # Run Gradient Descent
    gd_results = run_gradient_descent(dim, n_steps, n_chains)

    # Plot comparison
    plot_map_comparison(mala_results, gd_results)

    print(mala_results["param_norms"][-1])
    print(gd_results["param_norms"][-1])

    # Print summary comparison
    print("\n===== MAP Estimate Comparison =====")
    print(
        f"MALA runtime: {mala_results['runtime']:.2f}s, GD runtime: {gd_results['runtime']:.2f}s"
    )
    print(
        f"MALA accuracy: {mala_results['accuracy']:.4f}, GD accuracy: {gd_results['accuracy']:.4f}"
    )
    print(
        f"MALA log posterior: {mala_results['log_posterior']:.4f}, GD log posterior: {gd_results['log_posterior']:.4f}"
    )

    # Calculate solution difference
    solution_diff = jnp.linalg.norm(
        mala_results["map_estimate"] - gd_results["map_estimate"]
    )
    print(f"MAP solution difference (L2 norm): {solution_diff:.4f}")


if __name__ == "__main__":
    main()
