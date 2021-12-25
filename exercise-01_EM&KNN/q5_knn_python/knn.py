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

    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector

    density_dict = {}

    for i in range(len(pos)):

        current_center = pos[i]
        neighbor_distances = []
        density_dict[current_center] = 0
        neighbor_distances_k = []

        for j in range(len(samples)):

            current_neighbor = samples[j]

            current_distance = math.sqrt(abs(math.pow(current_center,2) - math.pow(current_neighbor,2)))

            neighbor_distances.append(current_distance)

        neighbor_distances.sort(reverse = False)

        neighbor_distances_k = neighbor_distances[:k]

        v = max(neighbor_distances_k)

        for l in range(len(neighbor_distances)):

            current_distance = neighbor_distances[l]

            if current_distance <= v:

                density_dict[current_center] += 1

            else :

                continue

        density_dict[current_center] = density_dict[current_center]/k

    print(density_dict.keys())

    print(density_dict.values())

    estDensity = np.array([list(density_dict.keys()), list(density_dict.values())])

    return estDensity
