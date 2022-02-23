import numpy as np
from numpy.random import choice
from simpleClassifier import simpleClassifier
from simpleClassifier import simpleClassifierTae

def adaboostSimpleTae(X, Y, K, nSamples):
    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK     : voting weights (K x 1) - for each round
    # para        : parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta

    """
    1. Initialize the weights of each datapoint w_n i.g. weights = (1/N, 1/N, ... , 1/N)
        where N is the number of input data

    2. Sample input data according to the weight

    3. Use the simple classifier, and label all the data

    4. count the error rate e_m

    5. get the updated weight a_m = ln( (1 - e_m) / e_m )

    6. adjust the weight w_n of each data point

        6.1 by w_n ^(m+1) = w_n * exp{ a_m * I(h_m(x_n) ≠ t_n) }

    """

    # N : # of data input data points
    # D : Dimension of the input
    N, D = X.shape

    # this weights means the probability of being chosen for each input
    weights = (np.ones(N) / N).reshape(N, 1)

    alphaK = np.zeros(K)

    j = np.zeros(K)
    theta = np.zeros(K)

    for i in range(K):

        # choose the indices of to be sampled input data
        random_idx = choice(N, nSamples, True, weights.ravel())

        # sample input data from X
        sample_X = X[random_idx, :]

        # sample target data from Y
        sample_Y = Y[random_idx]

        # get the weak classifier given the input and output with sampled index
        # new_dim : the index of the dimension to see
        # new_th : the new threshold to decide +1 or -1
        new_dim, new_th = simpleClassifierTae(sample_X, sample_Y)

        # store these in para
        j[i] = new_dim
        theta[i] = new_th

        # make the +1 / -1 array based on the new_dim and new_threshold

        # pseudo target array to compare
        pseudo_Y = (np.ones(N) * -1).reshape(N, 1)

        # set 1 every points at the dimension greater than new_th
        pseudo_Y[X[:, int(j[i]-1)] > theta[i]] = 1

        # compare the target labels with the pseudo_Y, then catch the position of zeros
        positions = np.where([Y[n] + pseudo_Y[n] == 0 for n in range(N)], 1, 0).reshape(N, 1)

        # count the error rate w.r.t weight
        error_weights = np.sum(positions * weights)

        # compute the error_rate
        if error_weights < 1.0e-01:

            alphaK[i] = 1
            break

        alphaK[i] = 0.5 * np.log((1 - error_weights) / error_weights)

        # now update the weights
        weights = np.exp(-alphaK[i] * pseudo_Y * Y)

        # final weights
        weights = weights / np.sum(weights)

    para = np.stack((j, theta), axis = 1)

    print("\n\n alphaK")
    print(alphaK)
    print("\n\n para")
    print(para)

    return alphaK, para


def adaboostSimple(X, Y, K, nSamples):
    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK     : voting weights (K x 1) - for each round
    # para        : parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta

    N, _ = X.shape

    j = np.zeros(K)
    theta = np.zeros(K)

    # weight is the factor used to regulate the choosing probability of each data point
    w = (np.ones(N) / N).reshape(N, 1)

    alpha = np.zeros(K)

    for k in range(K):

        index = choice(N, nSamples, True, w.ravel())

        sX = X[index, :]
        sY = Y[index]

        j[k], theta[k] = simpleClassifierTae(sX, sY)

        cY = (np.ones(N) * -1).reshape(N, 1)
        cY[X[:, int(j[k] - 1)] > theta[k]] = 1

        temp = np.where([Y[i] != cY[i] for i in range(N)], 1, 0).reshape(N, 1)

        ek = np.sum(temp * w)

        if ek < 1.0e-01 :

            alpha[k] = 1
            break

        alpha[k] = 0.5*np.log((1-ek)/ek)
        w = w * np.exp(-alpha[k] * cY * Y)
        w = w/np.sum(w)#normalization, otherwise, the weights grow exponentially

    alphaK = alpha
    para = np.stack((j, theta), axis = 1)

    print("\n\n alphaK")
    print(alphaK)
    print("\n\n para")
    print(para)

    return alphaK, para
