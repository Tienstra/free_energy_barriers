from abc import ABC, abstractmethod 
import jax.numpy as jnp
from jax import random
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
#sets plotting format
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks",palette='bright', rc=custom_params)



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
    def __init__(self,N, args=[0.5, 15, 2, 1]):
        super().__init__()

        self.t = args[0]
        self.L = args[1]
        self.T = args[2]
        self.rho = args[3]
        self.N = N 
    


    def w(self,r):
        t = self.t
        L = self.L
        T = self.T
        rho = self.rho
        
        # Case 1: r <= t / 2
        if r <= t / 2:
            return 4 * (T * r)**2
        
        # Case 2: t / 2 <= r < t
        elif r >= t / 2 and r< t :
            return (T * t)**2 + T * (r - (t / 2))
        
        # Case 3: t < r < L
        elif r >= t and r < L:
            return (T * t)**2 + (T * (t / 2)) + rho * (r - t)
        
        # Case 4: r >= L
        else:
            return  (T * t)**2 + (T * (t / 2)) + rho * (L - t)
        
        
    
    def evaluate(self,theta):
            w_theta = self.w(jnp.linalg.norm(theta))  # Call the w function with the norm of theta
            # Initialize the random key
            key = random.PRNGKey(42)
            # Generate N samples uniformly from the interval [0, 1]
            X = random.uniform(key, shape=(self.N,))
            
            return jnp.sqrt(w_theta)*X 

         



if __name__ == "__main__":
    # Create synthetic observed data with noise
    key = random.PRNGKey(42)
    N=100
    true_theta = np.linspace(0,10,N) # True parameters
    y_observed = random.normal(key, shape=(N,)) # Adding noise
    forward_model = StepRegression(N)
    y = forward_model.evaluate(true_theta)
    plt.plot(y)
    plt.show()
    

 

