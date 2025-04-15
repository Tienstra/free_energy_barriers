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
        return -0.5 * jnp.sum(residuals**2) / sigma_noise**2


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

        # Define the different cases
        w_case1 = 4 * (T * r) ** 2  # for r in [0, t/2]
        w_case2 = (T * t) ** 2 + T * (r - t / 2)  # for r in (t/2, t)
        w_case3 = (T * t) ** 2 + (T * t / 2) + rho * (r - t)  # for r in [t, L)
        w_case4 = (T * t) ** 2 + (T * t / 2) + rho * (L - t)  # for r in [L, ∞)

        # Use nested where to select appropriate cases
        result = jnp.where(
            r <= t / 2,
            w_case1,
            jnp.where(r < t, w_case2, jnp.where(r < L, w_case3, w_case4)),
        )

        return result

    @partial(jit, static_argnums=(0,))
    def evaluate(self, theta):
        r = jnp.linalg.norm(theta)
        w_r = self.w(r)
        X = jnp.ones(self.N)
        return jnp.sqrt(w_r) * X


class LogisticRegression(Regression):
    def __init__(self, N, p):
        super().__init__()
        self.N = N  # number of observations
        self.p = p  # number of parameters beta_j's
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

        return (
            -1
            / n
            * jnp.sum(
                y * jnp.log(self.evaluate(X @ self.beta))
                + (1 - y) * jnp.log(1 - self.evaluate(X @ self.beta))
            )
        )

    @partial(jit, static_argnums=(0,))
    def gradient(self, X, y):
        """
        Compute the gradient of the loglik function:
        """
        n = X.shape[0]

        return 1 / n * jnp.dot(X.T, (self.evaluate(X @ self.beta) - y))

    def predict(self, X, threshold=0.5):
        """
        Make binary predictions using the fitted model
        Args:
            X: Input features
            threshold: Classification threshold (default: 0.5)
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.evaluate(X.T @ self.beta)
        return (probabilities >= threshold).astype(int)


def log_posterior_fn(theta, X, y):
    log_like = regression_model.log_likelihood(X, y)
    log_prior = -0.5 * jnp.sum(theta**2) / sigma_prior**2
    return log_like + log_prior


def log_lik(w_r):
    y = np.random.randn(1000)

    return -0.5 * (np.linalg.norm(w_r - y)) ** 2


if __name__ == "__main__":
    rs = np.linspace(0, 0.6, 1000)

    model = StepRegression(N=1000)
    w_rs = model.w(rs)
    log_prior = 1000 / 2 * np.linalg.norm(rs) ** 2

    log_lik_wr = list(map(log_lik, w_rs))

    log_post = log_lik_wr + log_prior

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    ax1.plot(rs, model.w(rs))
    ax1.set_title("w(r)")
    ax1.set_xlabel("r")
    ax1.set_ylabel("w(r)")
    ax1.grid(True)

    ax2.plot(rs, log_lik_wr)
    ax2.set_title("log likelihood")
    ax2.set_xlabel("r")
    ax2.set_ylabel("G(r)")
    # ax2.set_yscale('log')
    ax2.grid(True)

    ax3.plot(rs, log_lik_wr)
    ax3.set_title("log post")
    ax3.set_xlabel("r")
    ax3.set_ylabel("G(r)")
    # ax2.set_yscale('log')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
