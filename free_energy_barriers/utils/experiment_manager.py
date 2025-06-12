from symtable import Class

import yaml
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import uuid

import sys
import os
import matplotlib.colors as mcolors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import your modules
from models.regression import LogisticRegression
from mcmc.kernels import Kernel, MALAKernel
from mcmc.sampler import MCMC

# Define README template
README_TEMPLATE = """# Experiments Directory

This directory contains saved experimental results from running the MALA sampler.

## Directory Structure
```
experiments/
├── mala_runs/           # MALA sampler experiments
│   ├── YYYYMMDD_HHMMSS_[ID]/  # Individual experiment directories
│   │   ├── config.yaml         # Experiment configuration
│   │   ├── results.npz         # Compressed chain data
│   │   └── metadata.json       # Experiment metadata
└── README.md            # This file
```

## Experiment Naming Convention
Each experiment is stored in a directory named with the format: `YYYYMMDD_HHMMSS_[ID]`
- YYYYMMDD: Date
- HHMMSS: Time
- ID: Unique identifier

## File Descriptions
- `config.yaml`: Contains all parameters and settings used for the experiment
- `results.npz`: Compressed numpy archive containing chain data
- `metadata.json`: Contains summary statistics and experiment results

## Storage Information
- Chain data is stored using {dtype} precision
- Compression: numpy compressed format (npz)
"""


@dataclass
class ExperimentConfig:
    # MALA kernel parameters
    epsilon: float  # step size should be 1/L*D

    # MCMC sampler parameters
    kernel: Kernel  # type of transition kernel that implements the step
    D: int  # dim of parameter
    n_steps: int  # step size
    n_chains: int

    # Initialization parameters
    init_method: str  # e.g. sample_annuli, sample_prior
    args: list

    # Model
    model_type: str  # e.g., "StepRegression", "DummyModel",
    # N: int #numeber of samples / scaling factor for likelihood

    # Storage parameters
    dtype: str = "float16"  # Options: 'float16', 'float32', 'float64'

    # Experiment metadata
    experiment_id: str = None
    timestamp: str = None
    description: str = ""

    def __post_init__(self):
        if self.experiment_id is None:
            self.experiment_id = str(uuid.uuid4())[:8]
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self):
        """Convert to dictionary with proper string representation of complex types."""
        d = asdict(self)
        d["args"] = repr(self.args)  # Convert list to string representation
        return d


class ExperimentManager:
    def __init__(self, project_root=None):
        """
        Initialize ExperimentManager with project root directory.
        If not provided, assumes current directory is project root.
        """
        if project_root is None:
            project_root = Path.cwd()
        else:
            project_root = Path(project_root)

        # Create experiments directory structure
        self.experiments_dir = project_root / "experiments"
        self.mala_runs_dir = self.experiments_dir / "mala_runs"

        # Create directories
        self.experiments_dir.mkdir(exist_ok=True)
        self.mala_runs_dir.mkdir(exist_ok=True)

        # Initialize README if it doesn't exist
        self.initialize_readme()

    def initialize_readme(self):
        """Create README.md if it doesn't exist."""
        readme_path = self.experiments_dir / "README.md"
        if not readme_path.exists():
            with open(readme_path, "w") as f:
                f.write(README_TEMPLATE.format(dtype="float16"))

    def create_experiment_dir(self, config):
        """Create a directory for the experiment with a unique name."""
        exp_dir = self.mala_runs_dir / f"{config.timestamp}_{config.experiment_id}"
        exp_dir.mkdir(exist_ok=True)
        return exp_dir

    def save_config(self, config, exp_dir):
        """Save experiment configuration to YAML file."""
        config_path = exp_dir / "config.yaml"

        # Create a copy of the config dictionary
        config_dict = asdict(config)

        # Convert args list to string representation
        config_dict["args"] = repr(config_dict["args"])

        # Save to YAML
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Also save a brief README for this specific experiment
        exp_readme = exp_dir / "README.md"
        with open(exp_readme, "w") as f:
            f.write(
                f"""# Experiment {config.experiment_id}


## Description
{config.description}

## Key Parameters
- Model: {config.model_type}
- Kernel: {config.kernel}
- Chains: {config.n_chains}
- Steps: {config.n_steps}
- Epsilon: {config.epsilon}
- D: {config.D}


Detailed configuration can be found in `config.yaml`
"""
            )

    def save_results(self, results, exp_dir, config):
        """Save experiment results with specified precision."""
        results_path = exp_dir / "results.npz"
        metadata_path = exp_dir / "metadata.json"

        # Convert JAX arrays to numpy and change dtype
        theta_chains = np.array(results["theta_chains"], dtype=config.dtype)

        # Calculate and store memory usage
        memory_mb = theta_chains.nbytes / (1024 * 1024)

        # Save chain data
        np.savez_compressed(
            results_path,
            theta_chains=theta_chains,
            acceptance_ratio=results["acceptance_ratio"],
        )

        # Save metadata
        metadata = {
            "norm of init:": results["init_norm"].tolist(),
            "acceptance_ratio": float(results["acceptance_ratio"]),
            "average_norm_of_last": results["average norm"].tolist(),
            "escaped": float(results["escaped"]),
            "norm_mean": float(results["norm_mean"]),
            "storage_info": {
                "dtype": config.dtype,
                "memory_mb": memory_mb,
                "shape": theta_chains.shape,
            },
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update experiment README with results
        exp_readme = exp_dir / "README.md"
        with open(exp_readme, "a") as f:
            f.write(
                f"""
## Results
- Acceptance Ratio: {metadata['acceptance_ratio']:.3f}
- Storage Size: {memory_mb:.2f} MB
- Chain Shape: {theta_chains.shape}
"""
            )

    def load_experiment(self, experiment_id):
        """Load experiment data by ID."""
        # Find the experiment directory
        exp_dir = next(self.mala_runs_dir.glob(f"*_{experiment_id}"), None)
        if exp_dir is None:
            raise ValueError(f"No experiment found with ID {experiment_id}")

        # Load configuration
        with open(exp_dir / "config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)  # Load into a dict
            config = ExperimentConfig(**config_dict)  # Create the dataclass instance
            config.args = eval(config.args)  # Convert the string back to a list

        # Load results
        results = np.load(exp_dir / "results.npz")
        with open(exp_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        return config, dict(results), metadata

    def list_experiments(self):
        """List all experiments with their basic information."""
        experiments = []
        for exp_dir in self.mala_runs_dir.glob("*"):
            if exp_dir.is_dir():
                with open(exp_dir / "config.yaml", "r") as f:
                    config = yaml.safe_load(f)
                with open(exp_dir / "metadata.json", "r") as f:
                    metadata = json.load(f)
                experiments.append(
                    {
                        "id": config["experiment_id"],
                        "timestamp": config["timestamp"],
                        "description": config["description"],
                        "acceptance_ratio": metadata["acceptance_ratio"],
                        "storage_info": metadata.get("storage_info", {}),
                    }
                )
        return experiments


def estimate_storage(n_chains, n_steps, D, dtype="float16"):
    """Estimate storage requirements for an experiment."""
    dtype_sizes = {"float16": 2, "float32": 4, "float64": 8}
    bytes_per_value = dtype_sizes[dtype]
    total_values = n_chains * n_steps * D
    total_bytes = total_values * bytes_per_value
    return {"mb": total_bytes / (1024 * 1024), "gb": total_bytes / (1024 * 1024 * 1024)}
