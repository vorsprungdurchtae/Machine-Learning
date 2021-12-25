import numpy as np
from getLogLikelihood import getLogLikelihood
from getLogLikelihood import llh


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    # check whether it is 1-D array
    # s.t. the zero array resembles the input
    if len(X.shape) == 1:

        # if yes, N = 1, dimension is same
        N = 1
        D = X.shape[0]

    # otherwise it just resembles the shape of the input
    else:

        N, D = X.shape

    # get the number of mixtures
    K = len(weights)

    #zero gamma array
    gamma = np.zeros((N, K))

    #contains the weighted normal distribution of each point for each given
    #j-th mean, covariance, and weight
    #e.sum(axis=1)[:,None]
    llhd = llh(means, weights, covariances, X, N, D, K)
    gamma = llhd/llhd.sum(axis = 1)[:,None]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    #####Insert your code here for subtask 6b#####
    return [logLikelihood, gamma]
