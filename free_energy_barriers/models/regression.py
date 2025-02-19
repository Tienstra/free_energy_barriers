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
    def evaluate():
        pass


class DummyModel(Regression):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def evaluate(self, theta):
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

        # Replace if/else with JAX's where
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
    def __init__(self, N, learning_rate=0.01, max_iter=1000):
        super().__init__()
        self.N = N
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.theta = None

    @partial(jit, static_argnums=(0,))
    def sigmoid(self, t):
        """
        Compute the sigmoid function: σ(t) = 1 / (1 + e^(-t))
        """
        return 1 / (1 + jnp.exp(-t))

    @partial(jit, static_argnums=(0,))
    def evaluate(self, X):
        """
        Compute σ(θᵀx)
        Args:
            X: Input features of shape (N, num_features)
        Returns:
            Predicted probabilities of shape (N,)
        """
        z = jnp.dot(X, self.theta)
        return self.sigmoid(z)

    @partial(jit, static_argnums=(0,))
    def cost_function(self, X, y):
        """
        Compute the cost function J(θ):
        J(θ) = -1/m ∑[y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))]
        """
        m = X.shape[0]
        h = self.evaluate(X)
        return -1 / m * jnp.sum(y * jnp.log(h) + (1 - y) * jnp.log(1 - h))

    @partial(jit, static_argnums=(0,))
    def gradient(self, X, y):
        """
        Compute the gradient of the cost function:
        ∇J(θ) = 1/m X^T(h_θ(X) - y)
        """
        m = X.shape[0]
        h = self.evaluate(X)
        return 1 / m * jnp.dot(X.T, (h - y))

    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent
        Args:
            X: Training features of shape (N, num_features)
            y: Binary target values of shape (N,)
        """
        # Initialize parameters
        key = random.PRNGKey(0)
        self.theta = random.normal(key, shape=(X.shape[1],))

        # Gradient descent
        for i in range(self.max_iter):
            grad_value = self.gradient(X, y)
            self.theta = self.theta - self.learning_rate * grad_value

            # Optional: add convergence check here
            if jnp.all(jnp.abs(grad_value) < 1e-5):
                break

    def predict(self, X, threshold=0.5):
        """
        Make binary predictions using the fitted model
        Args:
            X: Input features
            threshold: Classification threshold (default: 0.5)
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.evaluate(X)
        return (probabilities >= threshold).astype(int)


if __name__ == "__main__":
    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    model = LogisticRegression(N=100)
    model.theta = jnp.array([1.0])  # Set a simple weight for testing

    # Compute predictions
    predictions = model.evaluate(X)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(X, predictions, label="Sigmoid Function")
    plt.title("Logistic Regression Evaluation Function")
    plt.xlabel("z = θᵀx")
    plt.ylabel("σ(z)")
    plt.grid(True)
    plt.legend()
    plt.show()
