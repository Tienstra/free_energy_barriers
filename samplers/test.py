import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt


samples = norm.rvs(size=100, scale=1/np.sqrt(100))

print(np.linalg.norm(samples))