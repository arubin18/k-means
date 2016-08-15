# file: k_means.py
# ----------------
# Unsupervised learning algorithm that assigns data points to clusters.

import numpy as np
import matplotlib.pyplot as plt
import random
import math

def k_means (data, k):
	""" Returns updated data with classifiers for k clusters 
	and a centroid for each clusters. """

	instances = data.shape[0]
	labels = np.ones((instances, 1)) # classifiers

	us = assign_centroids(data, k) 
	centroids = np.array(us)  # numpy array to hold centroids

	for i in range(10):
		labels = assignment(data, us, labels, k)

		if empty(labels, k):
			return k_means(data, k)

		us = update(data, labels, us, k)  # updates centroids
		centroids = np.vstack((centroids, us))  # appends new centroids

	data = np.c_[data, labels]  # adds the labels column to the data array

	path_plot(centroids, k)  # plots centroids
	plot_data(data, k) 

	print "CLUSTERING..."

	return data, us

def assign_centroids(data, k):
	""" Assigns k centroids to points in the data. """

	attributes = data.shape[1]
	us = np.ones((1, attributes))
	
	for i in range(k):
		index = random.randint(0, len(data) - 1)
		u_new = np.array(data[index])  # new centroid 
		us = np.vstack((us, u_new))
		data = np.delete(data, index, 0)  # removes data point
	
	us = us[1:] # removes ones row
	
	return us

def assignment(data, us, labels, k):
	""" Assign data points to a cluster and returns the updated labels.
	Data points will be assigned to the cluster whose centroid has the smallest distance
	to that point. """

	d = lambda p1, p2: math.sqrt(sum((p2 - p1)**2)) # distance function

	for i in range(len(data)):
		point = data[i] # data point
		distances = [d(point, us[j]) for j in range(k)]
		min_distance = min(distances)
		cluster = distances.index(min_distance)
		labels[i] = cluster  # assign label to cluster

	return labels

def update(data, labels, us, k):
	""" Updates centroids for the clusters and returns the updated centroids. """

	counters = {} # holds sum of all the points that belong to each cluster

	# setting counters to 0 for each centroid
	for i in range(k):
		counters[i] = 0

	# Updating centroids by averaging all points that belong to that cluster
	for i in range(len(labels)):
		cluster = int(labels[i])
		us[cluster] = np.add(us[cluster], data[i])
		counters[cluster] += 1

	for i in range(k):
		us[i] = us[i] / counters[i]

	return us

def empty(labels, k):
	""" Returns true if a cluster was not assigned to any of the data points. """

	for i in range(k):
		count = 0
		for j in range(len(labels)):
			if labels[j] == i:
				count += 1
		if count == 0:
			return True
	return False

def path_plot (centroids, k):
	""" Plots the paths of the centroids. """

	xs = {}
	ys = {}

	for cluster in range(k):
		xs[cluster] = []
		ys[cluster] = []

	for i in range(len(centroids)):
		for j in range(k):
			if i % k == j:
				x, y = centroids[i, :2]

				xs[j] += [x]
				ys[j] += [y]

				plt.plot(x, y, 'ro')
				break

	for cluster in range(k):
		plt.plot(xs[cluster], ys[cluster], 'r')

def color_generator(k):
	""" Yields k unique colors. """

	for i in range(k):
		r = lambda: random.randint(0,255)
		c = '#%02X%02X%02X' % (r(),r(),r())
		yield c

def plot_data(data, k):
	""" Assigns a marker to each data point based on its label.
	Returns a scatter plot of the data. """

	m, n = data.shape
	labels = data[:, n-1]

	colors = list(color_generator(k))

	for i in range(0, m):
		label = labels[i]
		for j in range(k):
			if label == j:
				plt.scatter(data[i,0], data[i,1], color=colors[j])

	plt.show()
