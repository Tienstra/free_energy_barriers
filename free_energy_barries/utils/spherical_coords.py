import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def sample_spherical_coords(n_dim: int, n_samples: int, r_low: float = 0.0, r_high: float = 1.0) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate uniform samples in n-dimensional spherical coordinates.
    
    Args:
        n_dim: Number of dimensions
        n_samples: Number of samples to generate
        r_low: Lower bound for radial coordinate
        r_high: Upper bound for radial coordinate
    
    Returns:
        Tuple containing:
        - Array of radial coordinates
        - List of arrays containing angular coordinates
    """
    # Sample radial coordinate uniformly
    r = np.random.uniform(r_low, r_high, n_samples)
    
    # Initialize list to store angular coordinates
    phis = []
    
    # Generate n-1 angular coordinates
    for i in range(n_dim - 1):
        if i == n_dim - 2:
            # Last angle φ_(n-1) ranges from [0, 2π)
            phi = np.random.uniform(0, 2*np.pi, n_samples)
        else:
            # Other angles φ_i range from [0, π]
            phi = np.random.uniform(0, np.pi, n_samples)
        phis.append(phi)
    
    return r, phis

def spherical_to_cartesian(r: np.ndarray, phis: List[np.ndarray]) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        r: Array of radial coordinates
        phis: List of arrays containing angular coordinates
    
    Returns:
        Array of shape (n_samples, n_dim) containing Cartesian coordinates
    """
    n_samples = len(r)
    n_dim = len(phis) + 1
    
    # Initialize output array
    x = np.zeros((n_samples, n_dim))
    
    # First coordinate
    x[:, 0] = r * np.cos(phis[0])
    
    # Middle coordinates
    for i in range(1, n_dim - 1):
        x[:, i] = r
        for j in range(i):
            x[:, i] *= np.sin(phis[j])
        x[:, i] *= np.cos(phis[i])
    
    # Last coordinate
    x[:, -1] = r
    for i in range(n_dim - 1):
        x[:, -1] *= np.sin(phis[i])
    
    return x



# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_dim = 3  # Number of dimensions
    n_samples = 10000  # Number of samples
    r_low = 0.9  # Lower bound for radial coordinate
    r_high = 1.0  # Upper bound for radial coordinate
    
    # Generate samples
    r, phis = sample_spherical_coords(n_dim, n_samples, r_low, r_high)
    
    # Convert to Cartesian coordinates
    x = spherical_to_cartesian(r, phis)
    
    # Plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:,0],x[:,1],x[:,2], alpha=0.1)
    plt.show()
