# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:52:01 2020

@author: ntruo
"""

#%% Importing the libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
scalery = StandardScaler()
X = scalerX.fit_transform(X)
y = scalery.fit_transform(y)

#%% Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)


#%% Predicting a new result
scalery.inverse_transform(regressor.predict(scalerX.transform([[6.5]])))




#%% Visualising the SVR results
y_predict = scalery.inverse_transform(regressor.predict(X))
X = scalerX.inverse_transform(X)
y = scalery.inverse_transform(y)


plt.scatter(X,y, color = 'blue', label='Data')
plt.plot(X,y_predict, color = 'red', label = 'Result')
plt.legend();
plt.xlabel('level')
plt.ylabel('Salary')
plt.title('SVR regression')

plt.show()


#%% Visualising the SVR results (for higher resolution and smoother curve)

#%%
