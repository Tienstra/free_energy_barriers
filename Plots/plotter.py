import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
# from sklearn.neighbors import KernelDensity
# from sklearn.decomposition import PCA

class TracePlot:
    def __init__(self, theta_chains):
        self.theta_chains = theta_chains
        self.n_chains, self.n_steps, self.n_features = theta_chains.shape

    def plot_traces(self):
        fig, axes = plt.subplots(self.n_features, 2, figsize=(12, 2 * self.n_features), sharex=False)

        # If there's only one feature, we treat axes as a 2D array
        if self.n_features == 1:
            axes = [axes]

        # Loop over each feature (parameter)
        for i in range(self.n_features):
            # Plot the trace plot for each chain (all chains in one subplot, different colors)
            for chain_idx in range(self.n_chains):
                axes[i, 0].plot(self.theta_chains[chain_idx, :, i], label=f"Chain {chain_idx + 1}")

            axes[i, 0].set_title(f"Trace plot for theta[{i}]")
            axes[i, 0].set_ylabel(f"theta[{i}]")
            axes[i, 0].set_xlabel("Iteration")
            axes[i, 0].legend(loc='upper right')

            # Plot the KDE for each chain (all chains in one subplot, different colors)
            # for chain_idx in range(self.n_chains):

            #     #PCA
            #     pca = PCA(10)
            #     transformed_data = pca.fit_transform(self.theta_chains[chain_idx, :, i])
            #     # Step 2: Fit KDE using Scikit-learn
            #     kde = KernelDensity(kernel='gaussian', bandwidth=0.5)  
            #     kde.fit(transformed_data)

            #     # Step 3: Evaluate the KDE
            #     x_vals = np.linspace(np.min(pca_data), np.max(pca_data), 1000).reshape(-1, 1)  # Reshape to match sklearn's requirements
            #     log_density = kde.score_samples(x_vals)  # This returns log(density), so we need to exponentiate it
            #     density = np.exp(log_density)

            #     # Perform KDE using scipy's gaussian_kde
            #     # density = gaussian_kde(self.theta_chains[chain_idx, :, i])
            #     # x_vals = np.linspace(np.min(self.theta_chains[chain_idx, :, i]), np.max(self.theta_chains[chain_idx, :, i]), 1000)
            #     # y_vals = density(x_vals)
                
            #     # Plot the KDE for the chain
            #     axes[i, 1].plot(x_vals, density, linestyle="--", label=f"Chain {chain_idx + 1}")

            # axes[i, 1].set_title(f"KDE for theta[{i}]")
            # axes[i, 1].set_ylabel("Density")
            # axes[i, 1].set_xlabel(f"theta[{i}]")
            # axes[i, 1].legend()
        plt.tight_layout()
        plt.savefig("TracePlots.png")
        plt.show()
     

class NormPlot:
    def __init__(self, theta_chains):
        """
        Initializes the NormPlotter class with the sampled chains.
        
        Parameters:
        - theta_chain (array): Array of sampled theta values (n_steps x n_features)
        """
        self.theta_chains = theta_chains
        self.n_chains, self.n_steps, self.n_features = theta_chains.shape

    def plot_norm(self):
        # Plot norms for each chain
        for chain_idx in range(self.n_chains):
            norms = np.linalg.norm(self.theta_chains[chain_idx], axis=1)  # Norm for each iteration in the chain
            plt.plot(norms, lw=1, label=f"Chain {chain_idx + 1}")

        plt.title("Norm of the parameter vector at each iteration (across chains)")
        plt.xlabel("Iteration")
        plt.ylabel("Norm of theta")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("NormPlot.png")
        plt.show()
        print(norms)

if __name__ == '__main__': 
    theta_chains = np.random.randn(3, 100, 2)
    # Create an instance and plot
    trace_plotter = TracePlot(theta_chains)
    trace_plotter.plot_traces()
    norm_plotter = NormPlot(theta_chains)
    norm_plotter.plot_norm()
