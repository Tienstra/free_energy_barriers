import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt


#samples = norm.rvs(size=100, scale=1/np.sqrt(100))

#print(np.linalg.norm(samples))

import numpy as np
r_min=0.9
r_max = 1
num_samples=200
d=5
samples =[]
for n in range(num_samples):
    print(n)
    r = np.random.uniform(r_min, r_max,  1) 
    phis= list(np.random.uniform(0, np.pi,d-2))
    phis.append(np.random.uniform(0, 2*np.pi,1))
    print(phis)
    coords = [r*np.cos(phis[-1])]
    for i in range(1, len(phis)): 
        print(i)
        print(phis[0:i])
        print(phis[i])
        coords.append(r*np.prod(np.sin(phis[0:(i)]))*np.cos(phis[i]))
    samples.append(coords)

samples = np.array(samples)

plt.scatter(samples[:,0], samples[:,2])
plt.show()




