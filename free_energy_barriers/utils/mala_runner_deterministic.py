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


def run_mala_experiments(dim, n_steps=1000):

    # Initialize experiment manager
    manager = ExperimentManager(project_root=".")

    # Set up base configuration
    config = ExperimentConfig(
        D=dim,
        sigma_noise=1.0,
        epsilon=(1 / dim),
        n_steps=n_steps,
        n_chains=50,
        model_type="StepRegression",
        init_method=None,
        args=[],
        dtype="float16",
        description="MALA experiment intit at zero",
    )

    # Create synthetic data (same for all experiments)
    key = random.PRNGKey(42)
    y_observed = random.normal(key, shape=(config.D,)) * 0.5

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


if __name__ == "__main__":
    run_mala_experiments(dim=10000, n_steps=50000)
