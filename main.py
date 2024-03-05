from datasets_generators import DataGen
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal, uniform
from sklearn.neighbors import NearestNeighbors

# Make sure we only get one type of rows

def generate_sparse_unique_vectors(rows, dimension, sampling_function):

    seen = set()

    while len(seen) < rows:
        vector = sampling_function(dimension)
        seen.add(tuple(vector))

    return np.array(list(seen))

def generate_pairwise_equal_labels(unique_vectors, labels):
    
    neighbours = NearestNeighbors(n_neighbors=2).fit(unique_vectors)
    distances, indices = neighbours.kneighbors(unique_vectors)


def unique_information(num_vecs):
    return np.log2(num_vecs)

def histogram_information(labels):
    hist = np.bincount(labels-np.min(labels)) / len(labels)
    return -np.sum(hist * np.log2(hist))

def generate_random_labels(unique_vectors, labels):
    pass


#x = np.array([[1, 2, 1],
#              [2, 3, 1]])

#y = np.array([2, 2, 2, 2, 3])

#print(unique_information(len(x)), histogram_information(y))

# We have 2 classes 0 and 1. In a d-dimensional space we can separate 2d points with a hyperplane

# For a 2D space we can expect to separate 4 points with a line

# For Nearest Neighbours we can say that each vector is a parameter in our model?

# We would expect 4 classes to be classified correctly using only 2 of the points

d = 2
k = 3
r = d*k

print("Rows:", 2*r)

points = np.random.randint(0, 2**8, (2*r, d)).astype(np.uint8)
labels = np.array(r * [0] + r * [1]).reshape(-1, 1)

reference_points = np.vstack((points[:r//2], points[r:2*r-r//2]))
reference_labels = np.vstack((labels[:r//2], labels[r:2*r-r//2]))

indices = NearestNeighbors(n_neighbors=1).fit(reference_points).kneighbors(points, return_distance=False).flatten()
predicted_labels = reference_labels[indices]

#print(predicted_labels, indices, reference_labels)

#print(predicted_labels, labels)

correct_predictions = np.sum(predicted_labels == labels)

print(correct_predictions, correct_predictions/r)
#print(rand_points, reference_points, d, 2*d-d//2)

    
#data_points = 4

#x_full = generate_sparse_unique_vectors(data_points, 2, lambda x: uniform(0, 10, 2).astype(int))
#y_full = generate_pairwise_equal_labels(x_full, labels=np.arange(data_points))

##x_known = x_full[:data_points//2]
##x_hidden = x_full[data_points//2:]

