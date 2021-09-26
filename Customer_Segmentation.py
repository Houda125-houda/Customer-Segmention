# -*- coding: utf-8 -*-
"""


@author: LENOVO
"""


###############################
#    Customer segmentation   #
##############################
# Import all the dependencies 
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

Customer_data = pd.read_csv("C:/Users/LENOVO/Documents/GitHub/Customer-Segmention/Mall_Customers.csv")

Customer_data.head()
# finding the number of rows and columns 
Customer_data.shape
# getting information about the dataset
Customer_data.info()
# finding the number of rows and columns 
Customer_data.dtypes
# checking the missing values 
Customer_data.isna().sum()

# select the necessary columns except the ID_customer
# choosing the annual income column and spending score 
X = Customer_data.iloc[:,[3,4]].values  # selection only the 2 columns using their indexes or with loc using the name of columns "dataframe.loc[:,['column1','column2']]"
print(X)

# Choosing the number of clusters basing dataset using for a example a parameter like WCSS = within clusters sum of squares 
# WCSS is the sum of squared distance between each point and the centroid in a cluster. When we plot the WCSS with the K value, the plot looks like an Elbow. As the number of clusters increases, the WCSS value will start to decrease.
#finding  wcss value for different clusters

wcss = []
for i in range(1,11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) # the minimum number of clusters should be 4 
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)  #  kmeans.inertia_  will give us the wcss of each clusters and this value will be stored in the list above []

# Plotting the elbow graph 
sns.set() # it will give the basic themes and parameters for the graph 
plt.plot(range(1, 11), wcss)
plt.title("The Elbow point graph")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()
# the optimum number of clusters is 5 ~## “init” argument is the method for initializing the centroid.
kmeans = KMeans(n_clusters = 5, init = "k-means++", random_state = 0)
# return each label for each data point based on their cluster 
y = kmeans.fit_predict(X)
print(y)
plt.scatter(X[y == 0, 0], X[y == 0, 1], s = 50, c = 'red', label = 'Cluster1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], s = 50, c = 'blue', label = 'Cluster2')
plt.scatter(X[y == 2, 0], X[y == 2, 1], s = 50, c = 'blue', label = 'Cluster3')
plt.scatter(X[y == 3, 0], X[y == 3, 1], s = 50, c = 'violet', label = 'Cluster4')
plt.scatter(X[y == 4, 0], X[y == 4, 1], s = 50, c = 'yellow', label = 'Cluster5') 
# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend() 
plt.show()





