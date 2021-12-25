import numpy as np
from math import pow

#check whether it is 1-D array
#s.t. the zero array resembles the input
def getshape(X):

    if len(X.shape) == 1:

        #if yes, N = 1, dimension is same
        N = 1
        D = X.shape[0]

    #otherwise it just resembles the shape of the input
    else :

        N = X.shape[0]
        D = X.shape[1]

    return N, D

#this function returns the NxK Array
def llh(means, weights, covariances, X, N, D, K):

    logLikelihood_array = np.zeros(N)

    # zero array
    llhd = np.zeros((N, K))

    # iterate from 0 to N-1 th row
    for i in range(N):

        for j in range(K):
            # get the j- the mean for the i-th datapoint
            m = X[i] - means[j]

            # denominator to compute the normal distribution of given the i-th datapoint for the
            # j-th covariance and mean
            denom = pow((np.pi * 2), 0.5 * D) * pow(np.linalg.det(covariances[:, :, j]), 0.5)

            # sum these up for all the j-th mixture components
            llhd[i, j] = weights[j] * np.exp(
                -0.5 * np.matmul(np.matmul(m, np.linalg.inv(covariances[:, :, j])), m.T)) / denom

    return llhd

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    N, D = getshape(X)

    logLikelihood_array = np.zeros(N)

    #get the number of mixtures
    K = len(weights)

    #NxK array
    logLikelihood = np.sum(np.log(np.sum(llh(means, weights, covariances, X, N, D, K), axis = 1)), axis = 0)

    #####End Subtask#####
    return logLikelihood