import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.experiment_manager import (
    ExperimentManager,
    ExperimentConfig,
)
from samplers.local import MALA
from models.regression import StepRegression
from utils.tools import generate_bounds, sample_annuli
from metrics.metrics import mean, sd, norm
from jax import random
import jax.numpy as jnp


def run_mala_experiments():

    # Initialize experiment manager
    manager = ExperimentManager(project_root=".")

    # Set up base configuration
    base_config = ExperimentConfig(
        D=100,
        sigma_noise=1.0,
        epsilon=0.005,
        n_steps=1000,
        n_chains=50,
        model_type="StepRegression",
        init_method="sample_annuli",
        args=[],
        dtype="float16",
        description="MALA experiment with different init sampling from different annuli",
    )

    # Create synthetic data (same for all experiments)
    key = random.PRNGKey(42)
    y_observed = random.normal(key, shape=(base_config.D,)) * 0.5

    # Run experiments with different epsilon values
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

        # Initialize model and sampler
        model = StepRegression(config.D)
        mala = MALA(
            model,
            y=y_observed,
            D=config.D,
            sigma_noise=config.sigma_noise,
            epsilon=config.epsilon,
            n_steps=config.n_steps,
            n_chains=config.n_chains,
            initializer=config.init_method,
            args=config.args,
        )

        print(f"\nRunning experiment with epsilon={r_lowerupper}")

        # Run sampling
        theta_chains = mala.sample()

        results = {
            "init_norm": norm(theta_chains)[0],
            "theta_chains": theta_chains,
            "acceptance_ratio": mala.acceptance_ratio,
            "theta_mean": mean(theta_chains),
            "theta_std": sd(theta_chains),
            "average norm": norm(theta_chains)[-1],
            "norm_of_each_sample": jnp.linalg.norm(theta_chains[:, -1, :], axis=0),
            "escaped": jnp.mean(jnp.linalg.norm(theta_chains[:, -1, :], axis=1) < 0.33),
            "norm_mean": jnp.linalg.norm(jnp.mean(theta_chains[:, -1, :], axis=0)),
        }

        # Save results
        exp_dir = manager.create_experiment_dir(config)
        manager.save_config(config, exp_dir)
        manager.save_results(results, exp_dir, config)

        print(f"Experiment saved with ID: {config.experiment_id}")
        print(f"Acceptance ratio: {mala.acceptance_ratio:.3f}")

    # List all experiments
    print("\nAll experiments:")
    for exp in manager.list_experiments():
        print(f"ID: {exp['id']}")
        print(f"Description: {exp['description']}")
        print(f"Acceptance ratio: {exp['acceptance_ratio']:.3f}")
        print(f"Storage: {exp['storage_info']['memory_mb']:.2f} MB")
        print()


if __name__ == "__main__":
    run_mala_experiments()
