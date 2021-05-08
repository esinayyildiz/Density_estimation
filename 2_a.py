# -*- coding: utf-8 -*-
"""

@author: Esin Ayyıldız
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.mixture import GaussianMixture 
from matplotlib.colors import LogNorm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# load the iris dataset 
iris = load_iris()
x, y = iris.data, iris.target
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print("accuracy: ",metrics.accuracy_score(y_test,y_pred))

# select first two columns  
Xt = iris.data[:, :4] 
  
# turn it into a dataframe 
array = pd.DataFrame(Xt) 
  
# plot the data 
plt.scatter(array[0], array[1],array[2], array[3]) 

gaus_mix = GaussianMixture(n_components = 1, covariance_type='full') 
  

gaus_mix.fit(array) 
x = np.linspace(3., 10.)
y = np.linspace(1., 6.)
X, Y = np.meshgrid(x, y)
x_new = np.array([X.ravel(), Y.ravel()]).T
Z = -gaus_mix.score_samples(x_new)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=2.0, vmax=10000.0),
                 levels=np.logspace(0, 3, 10))
# Assign a label to each sample 
labels = gaus_mix.predict(array) 
array['labels']= labels 
array0 = array[array['labels']== 0] 
array1 = array[array['labels']== 1] 
array2 = array[array['labels']== 2] 
  
# plot three clusters in same plot 
plt.scatter(array0[0], array0[1], c ='blue',marker='*') 
plt.scatter(array1[0], array1[1], c ='pink',marker='*') 
plt.scatter(array2[0], array2[1], c ='black',marker='*') 