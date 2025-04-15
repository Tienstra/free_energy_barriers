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
from utils.tools import create_log_posterior, generate_synthetic_data, generate_bounds
import jax.numpy as jnp


def run_mala_experiments(dim, n_steps=1000):

    # Initialize experiment manager
    manager = ExperimentManager(project_root=".")

    # Set up base configuration
    base_config = ExperimentConfig(
        epsilon=(1 / dim),
        kernel=MALAKernel,
        D=dim,
        n_steps=n_steps,
        n_chains=2,
        model_type="StepRegression",
        init_method="sample_annuli",
        args=[],
        dtype="float16",
        description="MALA experiment init w annuli",
    )

    # Generate synthetic data
    n_samples = base_config.D
    n_features = base_config.D
    X, y, true_beta = generate_synthetic_data(n_samples, n_features)

    # Convert to JAX arrays
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    # Initialize model and sampler
    model = StepRegression(N=n_features)
    sigma_prior = 1/jnp.sqrt(base_config.D)
    log_posterior_fn = create_log_posterior(
        model, y_jax, sigma_prior
    )
    # Initialize kernel
    mala_kernel = MALAKernel(log_posterior_fn, epsilon=base_config.epsilon)

    r_bounds = generate_bounds(start=0, stop=1, length=(1 / 9))

    print(r_bounds)

    for r_lowerupper in r_bounds:
        config_dict = base_config.__dict__.copy()
        config_dict.update(
            {
                "args": r_lowerupper,
                "description": f"MALA experiment with radi upper and lower bounds={r_lowerupper}",
                "timestamp": None,
                "experiment_id": None,
            }
        )
        config = ExperimentConfig(**config_dict)

        # Initialize sampler
        mcmc = MCMC(
            kernel=mala_kernel,
            D=n_features,
            n_steps=base_config.n_steps,
            n_chains=base_config.n_chains,
            initializer=base_config.init_method,
            init_args=config.args,
            seed=45,
        )

        print(f"\nRunning experiment with radius={r_lowerupper}")
        # Run sampling
        print(config.args)
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
    run_mala_experiments(dim=1000, n_steps=10000)
