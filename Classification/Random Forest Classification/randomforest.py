# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 07:04:55 2020

@author: ntruo
"""
#%% importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%% importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
#%% splitting the dataset to training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
#%% feauture scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#%% training the randomforest model on training set
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)

#%% predict a value

#%% valuate model
y_pred = classifier.predict(X_test)
result = np.concatenate((y_pred.reshape(len(y_pred),1),
                         (y_test.reshape(len(y_test),1))),1)
print(result)
#%% confusion matrix/ error matrix
from sklearn.metrics import confusion_matrix,accuracy_score
matrix = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print(matrix,score)
