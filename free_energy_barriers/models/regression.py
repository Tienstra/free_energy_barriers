from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random, grad, jit
import numpy as np
from functools import partial
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt

# sets plotting format
import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="bright", rc=custom_params)


class Regression(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def log_likelihood(self, theta, y, sigma_noise=1.0):
        """
        Compute log-likelihood assuming Gaussian noise.

        Args:
            theta: Model parameters
            y: Observed data
            sigma_noise: Standard deviation of observation noise

        Returns:
            Log-likelihood value
        """
        y_pred = self.evaluate(theta)
        residuals = y - y_pred
        return -0.5 * jnp.sum(residuals ** 2) / sigma_noise ** 2

class DummyModel(Regression):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def evaluate(self):
        return jnp.zeros(self.N)


class StepRegression(Regression):
    def __init__(self, N, args=[0.5, 15, 2, 1]):
        super().__init__()
        self.t = args[0]
        self.L = args[1]
        self.T = args[2]
        self.rho = args[3]
        self.N = N

    @partial(jit, static_argnums=(0,))
    def w(self, r):
        t = self.t
        L = self.L
        T = self.T
        rho = self.rho


        w_case1 = 4 * (T * r) ** 2
        w_case2 = (T * t) ** 2 + T * (r - (t / 2))
        w_case3 = (T * t) ** 2 + (T * (t / 2)) + rho * (r - t)
        w_case4 = (T * t) ** 2 + (T * (t / 2)) + rho * (L - t)

        # Use where to select appropriate cases
        result = jnp.where(
            r <= t / 2,
            w_case1,
            jnp.where(r < t, w_case2, jnp.where(r < L, w_case3, w_case4)),
        )

        return result

    @partial(jit, static_argnums=(0,))
    def evaluate(self, theta):
        key = random.PRNGKey(42)
        r = jnp.linalg.norm(theta)
        w_theta = self.w(r)
        X = random.uniform(key, shape=(self.N,))
        return jnp.sqrt(w_theta) * X


class LogisticRegression(Regression):
    def __init__(self, N, p):
        super().__init__()
        self.N = N #number of observations
        self.p = p #number of parameters beta_j's
        self.beta = jnp.zeros(p)

    @partial(jit, static_argnums=(0,))
    def evaluate(self, t):
        """
        Compute the sigmoid function: p(t) = 1 / (1 + e^(-t))
        """
        return 1 / (1 + jnp.exp(-t))

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, X, y):

        n = X.shape[0]

        return -1 / n * jnp.sum(y * jnp.log(self.evaluate(X@self.beta)) + (1 - y) * jnp.log(1 - self.evaluate(X@self.beta)))

    @partial(jit, static_argnums=(0,))
    def gradient(self, X, y):
        """
        Compute the gradient of the loglik function:
        """
        n = X.shape[0]

        return 1 / n * jnp.dot(X.T, (self.evaluate(X@self.beta) - y))


    def predict(self, X, threshold=0.5):
        """
        Make binary predictions using the fitted model
        Args:
            X: Input features
            threshold: Classification threshold (default: 0.5)
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.evaluate(X.T@self.beta)
        return (probabilities >= threshold).astype(int)


if __name__ == "__main__":
    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    model = LogisticRegression(N=100, p=100)
    model.beta = jnp.array([1.0])  # Set a simple weight for testing

    # Compute predictions
    predictions = model.evaluate(X)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(X, predictions)
    plt.title("Logistic Regression Evaluation Function")
    plt.xlabel("X@Beta")
    plt.ylabel("p(X@Beta)")
    plt.grid(True)
    plt.legend()
    plt.show()
