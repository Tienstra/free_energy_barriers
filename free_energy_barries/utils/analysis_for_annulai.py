import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import seaborn as sns
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.experiment_manager import ExperimentManager
from plots.plotter import NormPlot


def analyze_mala_experiments_detailed():
    # Initialize experiment manager
    manager = ExperimentManager(project_root=".")

    figs_dir = Path("free_energy_barries/figs")

    # Get timestamp for file naming
    timestamp = manager.list_experiments()[0][
        "timestamp"
    ]  # Using first experiment's timestamp

    # Get list of all experiments
    experiments = manager.list_experiments()

    # Sort experiments by lower bound of annulus
    experiments.sort(key=lambda x: eval(x["config"]["args"])[0])

    # Create figure for norm plots (3x3 grid for 9 annuli)
    fig_all, axes_all = plt.subplots(3, 3, figsize=(20, 20))
    axes_all = axes_all.flatten()

    # Create containers for R² values
    r_bounds = []
    r_squared_values = []

    # Load and analyze each experiment
    for idx, exp in enumerate(experiments):
        # Load the experiment data
        config, results, metadata = manager.load_experiment(exp["id"])

        # Extract the annuli bounds
        r_bounds.append(config.args)

        # Get theta chains
        theta_chains = results["theta_chains"]

        # Create a subplot for this annulus
        plt.sca(axes_all[idx])

        # Use your NormPlot class for this subplot
        norm_plotter = NormPlot(theta_chains)
        norm_plotter.plot_norm()

        # Customize the subplot
        axes_all[idx].set_title(f"Annulus [{config.args[0]:.2f}, {config.args[1]:.2f}]")

        # Calculate R²
        final_samples = theta_chains[:, -1:, :]
        mean_final = np.mean(final_samples, axis=(0, 1))

        # Calculate total sum of squares and residual sum of squares
        y_true = np.zeros_like(mean_final)  # Assuming true parameters are zero
        ss_res = np.sum((y_true - mean_final) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        r_squared_values.append(r_squared)
