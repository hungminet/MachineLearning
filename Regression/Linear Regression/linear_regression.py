# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:57:45 2020

@author: ntruo
"""
 
#%% import lib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 

#%% loaddata

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#%% split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=1)



#%% build model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#%% predting

y_pre = regressor.predict([[2]])


#%%  Virtualization 
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = "blue")
plt.title('Salary by Experience (Training set)')
plt.xlabel('years')
plt.ylabel('Salary')
plt.show()


#%% result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = "blue")
plt.title('Salary by Experience (Test set)')
plt.xlabel('Year')
plt.ylabel('Salary')
plt.show()

