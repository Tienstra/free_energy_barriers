from typing import Tuple, Sequence
from jax import jit,vmap, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial


@jit
def _sample_spherical_coords_inner(
    key: random.PRNGKey,
    keys: random.PRNGKey,
    n_samples: int,
    r_low: float,
    r_high: float,
    is_last_angle: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Inner function for sampling a single angle"""
    r = random.uniform(key, shape=(n_samples,), minval=r_low, maxval=r_high)
    max_val = jnp.where(is_last_angle, 2.0 * jnp.pi, jnp.pi)
    phi = random.uniform(keys, shape=(n_samples,), minval=0.0, maxval=max_val)
    return r, phi

@partial(jit, static_argnames=['n_dim', 'n_samples'])
def sample_spherical_coords(
    key: random.PRNGKey,
    n_dim: int,
    n_samples: int,
    r_low: float = 0.0,
    r_high: float = 1.0,
) -> Tuple[jnp.ndarray, Sequence[jnp.ndarray]]:
    """
    Generate uniform samples in n-dimensional spherical coordinates using JAX.

    Args:
        key: JAX random number generator key
        n_dim: Number of dimensions
        n_samples: Number of samples to generate
        r_low: Lower bound for radial coordinate
        r_high: Upper bound for radial coordinate

    Returns:
        Tuple containing:
        - Array of radial coordinates
        - Sequence of arrays containing angular coordinates
    """
     # Split the key for different random operations
    keys = random.split(key, n_dim)

    # Sample radial coordinate uniformly
    r = random.uniform(keys[0], shape=(n_samples,), minval=r_low, maxval=r_high)

    # Generate angular coordinates using vmap
    def sample_phi(key, is_last):
        max_val = jnp.where(is_last, 2.0 * jnp.pi, jnp.pi)
        return random.uniform(key, shape=(n_samples,), minval=0.0, maxval=max_val)

    # Create a list of boolean flags indicating if the angle is the last one
    is_last = jnp.arange(n_dim - 1) == (n_dim - 2)
    phis = vmap(sample_phi)(keys[1:], is_last)

    return r, phis


@jit
def spherical_to_cartesian(r: jnp.ndarray, phis: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates using JAX.

    Args:
        r: Array of radial coordinates with shape (n_samples,)
        phis: Sequence of arrays containing angular coordinates,
              each with shape (n_samples,)

    Returns:
        Array of shape (n_samples, n_dim) containing Cartesian coordinates
    """
    n_samples = r.shape[0]
    n_dim = len(phis) + 1

    # Compute the Cartesian coordinates using vectorized operations
    sin_phis = jnp.sin(phis)
    cos_phis = jnp.cos(phis)

    # Initialize the Cartesian coordinates with the first coordinate
    x = r * cos_phis[0]

    # Compute the middle coordinates
    for i in range(1, n_dim - 1):
        x = jnp.column_stack((x, r * jnp.prod(sin_phis[:i], axis=0) * cos_phis[i]))

    # Compute the last coordinate
    last_coord = r * jnp.prod(sin_phis, axis=0)
    x = jnp.column_stack((x, last_coord))

    return x


if __name__ == "__main__":
    # Set random seed for reproducibility
    key = random.PRNGKey(42)

    # Parameters
    n_dim = 3  # Number of dimensions
    n_samples = 10  # Number of samples
    r_low = 0.9  # Lower bound for radial coordinate
    r_high = 1.0  # Upper bound for radial coordinate

    # Generate samples
    r, phis = sample_spherical_coords(key, n_dim, n_samples, r_low, r_high)

    # Convert to Cartesian coordinates
    x = spherical_to_cartesian(r, phis)
    if n_dim <=3:
        # Convert JAX arrays to NumPy for plotting
        x_np = jnp.asarray(x).copy()

        # Create the 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Create scatter plot
        scatter = ax.scatter(
            x_np[:, 0], x_np[:, 1], x_np[:, 2], alpha=0.1, c="blue", marker="."
        )

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Spherical Coordinates Sampling")

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Show the plot
        plt.show()
    else:
        print("Sampling and conversion completed successfully!")
