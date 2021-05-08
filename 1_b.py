import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load the Iris Data
iris = datasets.load_iris()
Xt = iris.data[:, :2] 

x, y = iris.data, iris.target
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print("accuracy: " ,metrics.accuracy_score(y_test,y_pred))

bandwidth = 1.06 * Xt.std() * len(Xt) ** (-1 / 5.)
x = iris.data[:, :2] 

k = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x)
# Grab the data, and compute the support (sampling points)
support = np.linspace(2, 8, len(x))

# Create the KDE, and return the support values.
y = k.score_samples(x)

# Plot the results including underlying histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(x,  alpha=0.5, density=True )
ax.plot(support, np.exp(y))
plt.title('Kernel Density Plot')

# Make plots
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(x, alpha=0.5,  density=True, label='')

# Gaussian KDE with varying bandwidths
for bandwidth in np.linspace(0.1, 0.9, 5):
    k = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x)
    y = k.score_samples(x)
    ax.plot(support, np.exp(y), label='bw = {0:3.2f}'.format(bandwidth))

# Decorate plot
plt.title('Kernel Density Plot')
plt.legend()

