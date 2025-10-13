import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Generate synthetic data (1D for simplicity)
np.random.seed(42)
X = np.random.randn(100, 1)  # 100 samples, 1 feature
true_theta = np.array([2.0])
y = (X.dot(true_theta) + 0.1 * np.random.randn(100) > 0).astype(int)

# Define log-likelihood function for logistic regression
def log_likelihood(theta, X, y):
    z = X.dot(theta)
    log_sig = -np.log(1 + np.exp(-z))
    log_one_minus_sig = -np.log(1 + np.exp(z))
    return np.sum(y * log_sig + (1 - y) * log_one_minus_sig)

# Evaluate log-likelihood along a range of theta norms
theta_norms = np.linspace(0, 10, 100)
log_liks = []
for norm in theta_norms:
    theta = np.array([norm])  # Direction aligned with true_theta (for simplicity)
    log_liks.append(log_likelihood(theta, X, y))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(theta_norms, log_liks, color='blue', label='Log-likelihood')
plt.axvline(x=np.linalg.norm(true_theta), color='red', linestyle='--', label='True theta')
plt.xlabel(r'$\|\theta\|$', fontsize=12)
plt.ylabel('Log-likelihood', fontsize=12)
plt.title('Log-Likelihood vs. Norm of Theta (Logistic Regression)')
plt.legend()
plt.grid(True)
plt.show()