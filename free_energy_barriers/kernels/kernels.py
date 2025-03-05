from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random, grad, jit
import numpy as np
from functools import partial
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt



