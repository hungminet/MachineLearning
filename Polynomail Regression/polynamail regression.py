# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 19:00:14 2020

@author: ntruo
"""


#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
#%% Training the Linear Regression model on the whole dataset

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X,y)
#%% Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_reg = PolynomialFeatures(degree=4)
X_poly = polynomial_reg.fit_transform(X)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_poly,y)

#%% Visualising the Linear Regression results

plt.scatter(X,y,color = 'red')
plt.plot(X,linear_reg.predict(X),color='blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Linear Regression)')
plt.show()
#%% Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#%% predict linear
print(linear_reg.predict([[6.5]]))

#%% poly prediction

print(linear_regressor2.predict(polynomial_reg.fit_transform([[6.5]])))



