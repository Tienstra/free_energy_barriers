from free_energy_barriers.utils.experiment_manager import ExperimentManager, ExperimentConfig
from free_energy_barriers.samplers.local import MALA
from free_energy_barriers.models.regression import StepRegression
from jax import random
import jax.numpy as jnp

def run_mala_experiments():
    # Initialize experiment manager
    manager = ExperimentManager(project_root=".")
    
    # Set up base configuration
    base_config = ExperimentConfig(
        N=100,
        sigma_noise=1.0,
        epsilon=0.005,
        n_steps=10000,
        n_chains=100,
        model_type="StepRegression",
        init_method="random",
        init_params={},
        dtype='float16',
        description="MALA experiment with different epsilons"
    )
    
    # Create synthetic data (same for all experiments)
    key = random.PRNGKey(42)
    y_observed = random.normal(key, shape=(base_config.N,)) * 0.5
    
    # Run experiments with different epsilon values
    epsilon_values = [0.001, 0.005, 0.01]
    
    for epsilon in epsilon_values:
        # Update config for this experiment
        config = ExperimentConfig(
            **{**base_config.__dict__,
               'epsilon': epsilon,
               'description': f"MALA experiment with epsilon={epsilon}"
            }
        )
        
        # Initialize model and sampler
        model = StepRegression(config.N)
        mala = MALA(
            model,
            y=y_observed,
            N=config.N,
            sigma_noise=config.sigma_noise,
            epsilon=config.epsilon,
            n_steps=config.n_steps,
            n_chains=config.n_chains,
        )
        
        print(f"\nRunning experiment with epsilon={epsilon}")
        
        # Run sampling
        theta_chains = mala.sample()
        
        # Calculate summary statistics
        theta_mean = jnp.mean(theta_chains[:, -2000:, :], axis=(1, 0))
        theta_std = jnp.std(theta_chains[:, -2000:, :], axis=(1, 0))
        
        results = {
            "theta_chains": theta_chains,
            "acceptance_ratio": mala.acceptance_ratio,
            "theta_mean": theta_mean,
            "theta_std": theta_std
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
