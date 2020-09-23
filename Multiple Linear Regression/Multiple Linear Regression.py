# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:07:31 2020

@author: ntruo
"""
#%% IMPORT LIBRARY

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% LOADING DATA
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#%% ENCODING CATEGORICAL DATA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transform_tool = ColumnTransformer(transformers = 
                                   [('encoder', OneHotEncoder(),[3])],
                                   remainder = 'passthrough')
X = np.array(transform_tool.fit_transform(X))





#%% SPLITTING THE DATASET TO TRAINING SET AND TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2, 
                                                    random_state=1)




#%%







#%%



