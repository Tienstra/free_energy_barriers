import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.spherical_coords import sample_spherical_coords, spherical_to_cartesian


def sample_annulai(n_dims=3, n_samples=5000, r_low=.66, r_high=1):
    """
    Samples from the annulai with specified radius

    Parameters: 
        - n_dims : dimension of samples 
        - n_samples : number of samples to be drawn
        - r_low : lower bound for radius (inner radius)
        - r_high: upper bound for radius (outer radius)
    """
    # Generate samples
    r, phis = sample_spherical_coords(n_dim, n_samples, r_low, r_high)

    # Convert to Cartesian coordinates
    samples = spherical_to_cartesian(r, phis)

    return samples 
