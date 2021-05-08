import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn import datasets 

# Load the Iris Data
iris = datasets.load_iris()
Xt = iris.data[:, :4] 

bandwidth = 1.06 * Xt.std() * len(Xt) ** (-1 / 5.)
x = iris.data[:, :4] 

k = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x)
# Grab the data, and compute the support (sampling points)
support = np.linspace(2, 8, len(x))

# Create the KDE, and return the support values.
y = k.score_samples(x)

# Plot the results including underlying histogram
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(x,  alpha=1, density=True )
ax.plot(support, np.exp(y))
plt.title('Kernel Density Plot')

# Make plots
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(x, alpha=1,  density=True, label=' ')

# Gaussian KDE with varying bandwidths
for bandwidth in np.linspace(0.1, 1, 5):
    k = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x)
    y = k.score_samples(x)
    ax.plot(support, np.exp(y), label='bw = {0:3.2f}'.format(bandwidth))

# Decorate plot
plt.title('Kernel Density Plot')
plt.legend()
