import numpy as np
import math

def kde(samples, h):
    print("kde start")
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel, h = 0.5 * the length of edge
    # Output
    #  estDensity : estimated density in the range of [-5,5]
    # c

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created

    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Compute probability density of data points within samples
    (data point x - sample)
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    #data points
    pos = np.arange(-5, 5.0, 0.1)

    if len(samples.shape) > 1 :

        D = samples.shape[1]

    else :

        D = 1

    N = samples.shape[0]

    """
    
    To programm the kde such that any form of density
    can be estimated
    
    K/NV = 1/N * sum(1/sqrt(2*pi)*h) * exp(|data point - sample|^2 / s*h^2 )
    N = samples
    
    Approach
    1. create the array
    2. use l2 norm
    
    """

    densities = np.zeros((N,1))
    denominator = pow(math.pi * 2, 1/2) * N * h

    #loop start first the data points

    for i in range(len(pos)):
        """
        row wise l2 norm
        np.sum(np.abs(x)**2,axis=-1)**(1./2)
        """
        densities[i] = np.sum(np.exp(-(np.abs(pos[i] - samples) ** 2)/(2 * h ** 2)))/denominator


    estDensity = np.column_stack((pos, densities))

    return estDensity
