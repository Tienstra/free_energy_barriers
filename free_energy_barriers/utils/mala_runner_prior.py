import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.experiment_manager import (
    ExperimentManager,
    ExperimentConfig,
)
from mcmc.sampler import MCMC
from mcmc.kernels import MALAKernel
from models.regression import StepRegression, LogisticRegression
from utils.tools import create_log_posterior, generate_synthetic_data
import jax.numpy as jnp


def run_mala_experiments(dim, n_steps=1000):

    # Initialize experiment manager
    manager = ExperimentManager(project_root=".")

    # Set up base configuration
    config = ExperimentConfig(
        epsilon=(1 / dim),
        kernel=MALAKernel,
        D=dim,
        n_steps=n_steps,
        n_chains=2,
        model_type="LogisticRegression",
        init_method="sample_prior",
        args=[],
        dtype="float16",
        description="MALA experiment init w prior",
    )

    # Generate synthetic data
    n_samples = config.D
    n_features = config.D
    X, y, true_beta = generate_synthetic_data(n_samples, n_features)

    # Convert to JAX arrays
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    # Initialize model and sampler
    model = LogisticRegression(N=n_samples, p=n_features)
    log_posterior_fn = create_log_posterior(
        model, X_jax, y_jax, sigma_prior=1.0
    )
    # Initialize kernel
    mala_kernel = MALAKernel(log_posterior_fn, epsilon=config.epsilon)

    # Initialize sampler
    mcmc = MCMC(
        kernel=mala_kernel,
        D=n_features,
        n_steps=50000,
        n_chains=5,
        initializer="sample_prior",
        init_args=[],
        seed=45,
    )

    # Run sampling
    samples = mcmc.sample()

    results = {
        "init_norm": jnp.linalg.norm(samples[:,0,:]),
        "theta_chains": samples,
        "acceptance_ratio": mcmc.acceptance_ratio,
        "average norm": jnp.linalg.norm(samples[:,-1,:]),
        "escaped": jnp.mean(jnp.linalg.norm(samples[:, -1, :], axis=1) < 0.33),
        "norm_mean": jnp.linalg.norm(jnp.mean(samples[:, -1, :], axis=0)),
    }

    # Save results
    exp_dir = manager.create_experiment_dir(config)
    manager.save_config(config, exp_dir)
    manager.save_results(results, exp_dir, config)

    print(f"Experiment saved with ID: {config.experiment_id}")
    print(f"Acceptance ratio: {mcmc.acceptance_ratio:.3f}")


if __name__ == "__main__":
    run_mala_experiments(dim=10, n_steps=100)
