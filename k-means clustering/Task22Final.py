#Importing the various libraries.
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#A function used to import the data from the csv file into python.
k_means_data = pd.read_csv('dataBoth.csv')

#Converting the data to an array and setting the birth rate as the x-axis
#and the life expectancy as the y-axis.
x = np.asarray(k_means_data['BirthRate(Per1000)'])
y = np.asarray(k_means_data['LifeExpectancy'])
countries = np.asarray(k_means_data['Countries'])

#A function used to calculate the Eucaladein distance. 
def calc_distance(p1, p2):
    d = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return d

#A function to seperate the countries into their respective clusters.
def country_cluster(clust,d_points):
    country = []
    count_numb = 0
    for d in d_points:
        if d in clust:
            country.append(countries[count_numb])
        count_numb += 1

    return country

def mean_values (cluster):
    x = [x_p[0] for x_p in cluster]
    y = [y_p[1] for y_p in cluster]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    return [x_mean,y_mean]

#datapoints for the birthrate and life expectancy. 
datapoints = [(x[inter], y[inter]) for inter in range(len(x))]

#Requriing the user to enter the number of clusters.
number_clusters = int(input("Please enter the number of clusters: "))

#sample the starting points clusters
cluster_points = random.sample(datapoints, number_clusters)

#Iterating over the cluster points.
clusters = [[] for cp in range(len(cluster_points))]

#Iterating over the data points and then appending them to their specific cluster.
for point in datapoints:
    dista = []
    for c in range(number_clusters):
        dista.append(calc_distance(cluster_points[c],point))

    #The smallest value is the minimum of the distance points.
    smallest_value = min(dista)

    #Appending the clusters with the data points. 
    for cl in range(len(clusters)):
        if smallest_value == dista[cl]:
            clusters[cl].append(point)
            break


#Implementing the k-means algorithm on the data. 
k_means_clustering = KMeans(n_clusters=2, n_init = 6, init = "random", random_state = 42)
k_means_clustering.fit(datapoints)
y_means = k_means_clustering.predict(datapoints)
centers = k_means_clustering.cluster_centers_

#Plotting the k-means algorithm on the scatter plot. 
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha = 0.6);

#The mean for both the life expectancy and birth rate.
x_mean = np.mean(x)
y_mean = np.mean(y)

#Displaying the mean to the user. 
print(x_mean)
print(y_mean)

for c1 in range(number_clusters):
    print(clusters[c1])
    countries_clusters = country_cluster(clusters[c1],datapoints)
    print(countries_clusters)
    print("mean")
    print(mean_values(clusters[c1]))

#Plotting the x and y axis.
plt.ylabel("Life Expectancy")
plt.xlabel("Birth Rate")


#Giving the cluster points colors as well as plotting the scatter plot. 
list_colurs =['g', 'b', 'r','c']
for clust in range(len(clusters)):
    x_points = [p[0] for p in clusters[clust]]
    y_points = [p[1] for p in clusters[clust]]
    plt.scatter(x_points, y_points, c= list_colurs[clust])

#Showing the scatter plot. 
plt.show()
