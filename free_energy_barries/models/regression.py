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
    def __init__(self, x):
        super().__init__()
        self.x = 0

    def evaluate(self, theta):
        return 1


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


# if __name__ == "__main__":
# Create synthetic observed data with noise
# key = random.PRNGKey(42)
# N=100
# true_theta = np.linspace(0,10,N) # True parameters
# y_observed = random.normal(key, shape=(N,)) # Adding noise
# forward_model = StepRegression(100).plot_w()
# y = forward_model.evaluate(true_theta)
# plt.plot(y)
# plt.show()
