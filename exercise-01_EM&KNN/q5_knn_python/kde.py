import numpy as np
import math

def kernel_method(u):

    if abs(u) <= 0.5:

        return 1

    else :

        return 0


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    # D = dimenstion = sample.shape[0]
    #
    #generated  100 data points

    density_dict = {}
    pos = np.arange(-5, 5.0, 0.1)

    D = samples.shape[0]
    N = len(samples)

    #a column containing the density values estimated at the corresponding positions from pos
    pos_kde = []

    for i in range(len(pos)):

        current_center = pos[i]

        kernel_sum = 0

        #density_dict[current_center] = 0

        for j in range(len(samples)):

            kernel_sum += kernel_method((current_center - samples[j]/h))

        density_dict[current_center] = kernel_sum/(math.pow(h, D)*N)

    estDensity = np.array([list(density_dict.keys()),list(density_dict.values())])

    return estDensity


