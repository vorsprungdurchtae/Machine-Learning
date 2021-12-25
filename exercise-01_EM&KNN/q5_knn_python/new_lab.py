import numpy as np
import math

pos = np.arange(-5, 5.0, 0.1)

samples = np.random.normal(0, 1, 100)

current_center = pos[0]

all_neighbor_distances = []

neighbor_distances_k = []

k_n = 0

for j in range(len(samples)):

    current_neighbor = samples[j]

    current_distance = math.sqrt(math.pow(current_center, 2) - math.pow(current_neighbor, 2))

    all_neighbor_distances.append(current_distance)

    all_neighbor_distances.sort(reverse = False)

neighbor_distances_k = all_neighbor_distances[:k]

v = max(neighbor_distances_k)

for k in all_neighbor_distances:

    current_distance = all_neighbor_distances[k]

    if k <= v:

        k_n += 1

    else :

        continue




#neighbor_distances.sort(reverse = False)

#

#print(max(neighbor_distances))