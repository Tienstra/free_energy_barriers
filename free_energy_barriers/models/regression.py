from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random, grad, jit
import numpy as np
from functools import partial
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

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
    def __init__(self, N, args=[0.5, 5, 2, 1]):
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
        # w_case2 = (T * t) ** 2 + T * (r - t / 2)  # for r in (t/2, t)
        w_case2 = (T * t) ** 2
        w_case3 = (T * t) ** 2
        w_case4 = (T * t) ** 2
        # w_case3 = (T * t) ** 2 + (T * t / 2) + rho * (r - t)  # for r in [t, L)
        # w_case4 = (T * t) ** 2 + (T * t / 2) + rho * (L - t)  # for r in [L, ∞)

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
        self.X = jnp.array(np.random.randn(N, p))

    @partial(jit, static_argnums=(0,))
    def evaluate(self, t):
        """
        Compute the sigmoid function: p(t) = 1 / (1 + e^(-t))
        """
        return 1 / (1 + jnp.exp(-t))

    # @partial(jit, static_argnums=(0,))
    def log_likelihood(self, theta, y, sigma_noise=1.0):


        logits = self.X @ theta
        probs = self.evaluate(logits)
        return (jnp.sum(y * jnp.log(probs)
                          + (1 - y) * jnp.log(1 - probs)))

    def predict(self, theta, threshold=0.5):
        """
        Make binary predictions using the fitted model
        """
        probabilities = self.evaluate(self.X @ theta)
        return (probabilities >= threshold).astype(int)


def log_lik(w_r, y):

    return -0.5 * (np.linalg.norm(w_r - y)) ** 2


def create_interactive_plot():
    # Initialize parameters
    rs = np.linspace(0, 1, 1000)
    D_init = 1000
    N_init = 100
    model = StepRegression(N=N_init)
    w_rs = model.w(rs)
    y = np.random.randn(N_init)
    log_lik_obs = partial(log_lik, y=y)

    # Create the figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.45)  # Make room for sliders and text boxes

    # Initial values

    # Setup tick locations for all axes
    tick_locs = np.linspace(0, 1, 19)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"{x:.2f}" for x in tick_locs], rotation=90)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.grid(True)

    # Create initial plots with initial D and N values
    log_prior = (-D_init / 2) * rs**2
    log_lik_base = list(map(log_lik_obs, w_rs))
    log_lik_base = jnp.array(log_lik_base)

    (line1,) = ax1.plot(rs, w_rs)
    ax1.set_title("w(r)")
    ax1.set_xlabel("r")
    ax1.set_ylabel("w(r)")

    log_lik_wr = N_init * log_lik_base
    (line2,) = ax2.plot(rs, log_lik_wr)
    ax2.set_title("log likelihood")
    ax2.set_xlabel("r")
    ax2.set_ylabel("G(r)")

    log_post = log_lik_wr + log_prior
    (line3,) = ax3.plot(rs, log_post)
    ax3.set_title("log post")
    ax3.set_xlabel("r")
    ax3.set_ylabel("log post")

    (line4,) = ax4.plot(rs, np.gradient(log_post))
    ax4.set_title("grad log post")
    ax4.set_xlabel("r")
    ax4.set_ylabel("grad")

    # Create sliders
    ax_slider_N = plt.axes([0.2, 0.25, 0.6, 0.03])
    slider_N = Slider(ax_slider_N, "N", 10, 10000, valinit=N_init, valfmt="%d")

    ax_slider_D = plt.axes([0.2, 0.2, 0.6, 0.03])
    slider_D = Slider(ax_slider_D, "D", 10, 1000, valinit=D_init, valfmt="%d")

    # Create text boxes
    ax_textbox_N = plt.axes([0.2, 0.12, 0.1, 0.04])
    textbox_N = TextBox(ax_textbox_N, "N: ", initial=str(N_init))

    ax_textbox_D = plt.axes([0.4, 0.12, 0.1, 0.04])
    textbox_D = TextBox(ax_textbox_D, "D: ", initial=str(D_init))

    # Update function
    def update(val):
        N = int(slider_N.val)
        D = int(slider_D.val)

        # Update model with new D
        model_new = StepRegression(D)
        w_rs_new = model_new.w(rs)

        # Update y data if D changed
        if D != len(y):
            y_new = np.random.randn(D)
            log_lik_obs_new = partial(log_lik, y=y_new)
        else:
            log_lik_obs_new = log_lik_obs

        # Recalculate all values
        log_prior_new = (-D / 2) * rs**2
        log_lik_base_new = list(map(log_lik_obs_new, w_rs_new))
        log_lik_base_new = jnp.array(log_lik_base_new)
        log_lik_wr_new = N * log_lik_base_new
        log_post_new = log_lik_wr_new + log_prior_new
        grad_log_post_new = np.gradient(log_post_new)

        # Update plots
        line1.set_ydata(w_rs_new)
        line2.set_ydata(log_lik_wr_new)
        line3.set_ydata(log_post_new)
        line4.set_ydata(grad_log_post_new)

        # Rescale y-axes
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()
        ax4.relim()
        ax4.autoscale_view()

        fig.canvas.draw_idle()

    # Text box update functions
    def update_N(text):
        try:
            N_val = int(float(text))
            if 1 <= N_val <= 1000:
                slider_N.set_val(N_val)
        except:
            pass

    def update_D(text):
        try:
            D_val = int(float(text))
            if 100 <= D_val <= 5000:
                slider_D.set_val(D_val)
        except:
            pass

    # Connect sliders and text boxes
    slider_N.on_changed(update)
    slider_D.on_changed(update)
    textbox_N.on_submit(update_N)
    textbox_D.on_submit(update_D)

    plt.show()


if __name__ == "__main__":
    create_interactive_plot()
