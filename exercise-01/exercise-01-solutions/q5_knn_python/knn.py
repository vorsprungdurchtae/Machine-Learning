import numpy as np
import math

def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    pos = np.arange(-5, 5.0, 0.1)

    if len(samples.shape) > 1:

        D = samples.shape[1]

    else:

        D = 1

    N = samples.shape[0]

    #the distances to the data point from the interval [-5 5]
    #to the samples
    dist = np.abs(pos[np.newaxis, :] - samples[:, np.newaxis])

    #sort the distances
    #extract k-1 th element from the sorted diatances
    sorted_dist = np.sort(dist, axis = 0)[k-1, :]
    densities = k / (sorted_dist * 2 * N)

    estDensity = np.column_stack((pos, densities))

    return estDensity
