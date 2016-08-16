import numpy as np
from k_means import k_means

def total_clusters(labels):
	""" Returns the total numbers of clusters by counting 
	the number of classifiers in the labels array. """

	clusters = []

	for i in labels:
		if i not in clusters:
			clusters.append(i)

	k = len(clusters)
	return k

def main():

	a = np.genfromtxt('iris.txt', dtype = str, delimiter=',')
	n = a.shape[1]
	labels = a[:, n-1]  # labels on the far right column
	data = a[:, 0:n-1]  # removes labels
	data = data.astype(float)
	k = total_clusters(labels)

	data, us = k_means(data, k)
	n = a.shape[1]
	new_labels = data[:, n-1]

	print data
	print "CENTROIDS:"
	print us

if __name__ == "__main__":
	main()
