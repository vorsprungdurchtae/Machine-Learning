import numpy as np
from numpy.random import choice
from leastSquares import leastSquares

def adaboostLSLC(X, Y, K, nSamples):
    # Adaboost with least squares linear classifier as weak classifier
    # for a D-dim dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (iteration number of Adaboost) (scalar)
    # nSamples  : number of data which are weighted sampled (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of least square classifier (K x 3)
    #             For a D-dim dataset each least square classifier has D+1 parameters
    #             w0, w1, w2........wD

    #####Insert your code here for subtask 1e#####

    """
    least squares classifier returns weight w and bias b
    such that y = w*x + b
    """
    N, D = X.shape

    new_dim = D+1

    # weight is the factor used to regulate the choosing probability of each data point
    w = (np.ones(N) / N).reshape(N, 1)

    alphaK = np.zeros(K)
    para = np.zeros((K, new_dim))

    #X = np.insert(X, 0, 1, axis = 1)

    for k in range(K):

        index = choice(N, nSamples, True, w.ravel())

        sX = X[index, :]
        sY = Y[index]

        weight, bias = leastSquares(sX, sY)

        para[k, :] = [bias, weight[0], weight[1]]

        cX = np.sign(X.dot(weight) + bias).reshape(N, 1)

        cX = cX*Y

        mask = [item[0] for sublist in [cX < 0] for item in sublist]
        # Compute weighted error of classifier
        I = np.zeros(N)
        I[mask] = 1

        ek = max(I.dot(w), 0.001)

        alphaK[k] = 0.5 * np.log((1 - ek) / ek)
        w = w * np.exp((-alphaK[k]) * cX)
        w = w / np.sum(w)  # normalization, otherwise, the weights grow exponentially

    return [alphaK, para]
