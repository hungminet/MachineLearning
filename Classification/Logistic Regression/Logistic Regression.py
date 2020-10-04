# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 16:19:21 2020

@author: ntruo
"""
#%% importing library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values



#%% splitting the dataset to training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


#%% feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#%% training the Logistic regression model on training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)




#%% predict a value
print(classifier.predict(sc.transform([[48,33000]])))




#%% valuate model

y_pred = classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



#%% confusion matrix/ error matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred
                       )
print(score)





