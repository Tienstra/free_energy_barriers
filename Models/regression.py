from abc import ABC, abstractmethod 
import jax.numpy as jnp
from jax import random
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt


class Regression(ABC):
    def __init__(self):
        pass 
    @abstractmethod
    def evaluate():
        pass 

class DummyModel(Regression):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def evaluate(self, theta):
        return 0*self.x

class StepRegression(Regression):
    def __init__(self,N, args=[5, 15, 2, 1]):
        super().__init__()

        self.t = args[0]
        self.L = args[1]
        self.T = args[2]
        self.rho = args[3]
        self.N = N 

    def w(self,r_array):
        t = self.t
        L = self.L
        T = self.T
        rho = self.rho
        w_values = np.zeros_like(r_array)  # Initialize an array of the same size as r_array
        
        # Case 1: r <= t / 2
        mask1 = (r_array <= t / 2)
        w_values[mask1] = 4 * (T * r_array[mask1])**2
        
        # Case 2: t / 2 <= r < t
        mask2 = (r_array >= t / 2) & (r_array < t)
        w_values[mask2] = (T * t)**2 + T * (r_array[mask2] - (t / 2))
        
        # Case 3: t < r < L
        mask3 = (r_array >= t) & (r_array < L)
        w_values[mask3] = (T * t)**2 + (T * (t / 2)) + rho * (r_array[mask3] - t)
        
        # Case 4: r >= L
        mask4 = (r_array >= L)
        w_values[mask4] = (T * t)**2 + (T * (t / 2)) + rho * (L - t)
        
        return w_values
    
    def evaluate(self, theta):
            theta_norm = jnp.linalg.norm(theta)
            w_theta = self.w(theta_norm)  # Call the w function with the norm of theta
            # Initialize the random key
            key = random.PRNGKey(42)
            # Generate N samples uniformly from the interval [0, 1]
            X = random.uniform(key, shape=(self.N,))
            
            return jnp.sqrt(w_theta)*X 

         



if __name__ == "__main__":
    # Create synthetic observed data with noise
    key = random.PRNGKey(42)
    N=100
    true_theta = jnp.zeros(N) # True parameters
    y_observed = random.normal(key, shape=(N,)) # Adding noise
    forward_model = StepRegression(N)
    # y = forward_model.evaluate(true_theta)
    # plt.plot(y)
    # plt.show()
    

 

