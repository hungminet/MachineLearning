# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:41:45 2020

@author: ntruo
"""

#%% importing lib
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
#%% importing data
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
#%% find optimal number of clusters by using the elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    model = KMeans(n_clusters=i, init='k-means++',random_state=20)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(np.arange(1,11),wcss,label = 'WCSS')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.title('WCSS')
plt.show()

#%% training model on dataset
model = KMeans(n_clusters=5, init='k-means++',random_state=20)
y_kmeans = model.fit_predict(X)
#%%# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 150, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()