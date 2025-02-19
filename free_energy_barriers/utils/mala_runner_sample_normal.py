import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.experiment_manager import (
    ExperimentManager,
    ExperimentConfig,
)
from samplers.local import MALA
from models.regression import StepRegression, DummyModel
from utils.tools import generate_bounds, sample_annuli
from metrics.metrics import mean, sd, norm
from jax import random
import jax.numpy as jnp


def run_mala_experiments():

    # Initialize experiment manager
    manager = ExperimentManager(project_root=".")

    # Set up base configuration
    config = ExperimentConfig(
        D=1000,
        sigma_noise=1.0,
        epsilon=(1 / 1000),
        n_steps=1000,
        n_chains=50,
        model_type="DummyModel",
        init_method="sample_prior",
        args=[],
        dtype="float16",
        description="MALA experiment with different init sampling from different annuli",
    )

    # Create synthetic data (same for all experiments)
    key = random.PRNGKey(42)
    y_observed = jnp.zeros(config.D)

    # Initialize model and sampler
    model = DummyModel(config.D)
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

    print(f"\nRunning experiment to sample normal")

    # Run sampling
    theta_chains = mala.sample()

    results = {
        "init_norm": norm(theta_chains)[0],
        "theta_chains": theta_chains,
        "acceptance_ratio": mala.acceptance_ratio,
        "average norm": norm(theta_chains)[-1],
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
    run_mala_experiments()
