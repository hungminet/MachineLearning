# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 09:43:17 2020

@author: ntruo
"""





#%% import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%% importing data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


#%% Build Decision Tree model on whole data set
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=60,random_state=0)
regressor.fit(X,y)
 


#%% Predict a new result
y_predict = regressor.predict(X)



#%% Visualising the Decision Tree Regression Model

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red', label = 'Data')
y_predict = regressor.predict(X_grid)
plt.plot(X_grid,y_predict, color = 'blue', label = 'Predict')
plt.title('Random Forest Regression')
plt.legend()
plt.show()
