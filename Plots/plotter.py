import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
class TracePlot:
    def __init__(self, theta_chain):
        self.theta_chain = theta_chain
        self.n_steps, self.n_features = theta_chain.shape

    def plot_traces(self):
        fig, axes = plt.subplots(self.n_features, 2, figsize=(12, 2 * self.n_features))

        # If there's only one feature, make sure to handle the axes as a 2D array
        if self.n_features == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(self.n_features):
            # Plot the trace (sample values over iterations)
            axes[i, 0].plot(self.theta_chain[:, i])
            axes[i, 0].set_title(f"Trace plot for theta[{i}]")
            axes[i, 0].set_ylabel(f"theta[{i}]")
            axes[i, 0].set_xlabel("Iteration")
            
            # Perform KDE using scipy's gaussian_kde
            density = gaussian_kde(self.theta_chain[:, i])
            x_vals = np.linspace(np.min(self.theta_chain[:, i]), np.max(self.theta_chain[:, i]), 1000)
            y_vals = density(x_vals)

            # Plot the KDE
            axes[i, 1].plot(x_vals, y_vals, linestyle="--")
            axes[i, 1].set_title(f"Density plot for theta[{i}]")
            axes[i, 1].set_ylabel("Density")
            axes[i, 1].set_xlabel(f"theta[{i}]")
        
        
        plt.tight_layout()
        plt.show()