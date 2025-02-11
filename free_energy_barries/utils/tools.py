import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.spherical_coords import sample_spherical_coords, spherical_to_cartesian


def sample_annulai():
    # Parameters
    n_dim = 3  # Number of dimensions
    n_samples = 10000  # Number of samples
    r_low = 0.9  # Lower bound for radial coordinate
    r_high = 1.0  # Upper bound for radial coordinate

    # Generate samples
    r, phis = sample_spherical_coords(n_dim, n_samples, r_low, r_high)

    # Convert to Cartesian coordinates
    x = spherical_to_cartesian(r, phis)
