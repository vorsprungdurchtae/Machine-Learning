import numpy as np
from getLogLikelihood import getLogLikelihood, getshape

def newmeans(gamma, softnums, X, N, K, D):
    # new means KxD
    means = np.zeros((K, D))

    for j in range(K):

        for i in range(N):

            means[j, :] += gamma[i, j] * X[i]

        means[j, :] = means[j, :] / softnums[j]

    return means

def newcovs(X, means, gamma, softnums, N, D, K):

    #new covariances DxDxK
    covariances = np.zeros((D, D, K))

    #iterate over K
    for k in range(K):

        auxSigma = np.zeros((D, D))

        #iterate over all the datapoints
        for i in range(N):

            #difference between the given datapoint and mean(KxD)
            m = X[i] - means[k]
            #gamma is NxK, for each row
            #gamma j in K for datapoint n in N
            auxSigma = auxSigma + gamma[i, k] * np.outer(m.T, m)

        covariances[:, :, k] = auxSigma / softnums[k]

    return covariances

def MStepTae(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    N, D = getshape(X)

    _, K = getshape(gamma)

    #soft number, 1xK
    softnums = gamma.sum(axis=0)

    #new weights pi
    weights = softnums // N

    means = newmeans(gamma, softnums, X, N, K, D)

    covariances = newcovs(X, means, gamma, softnums, N, D, K)

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    #####Insert your code here for subtask 6c#####
    return weights, means, covariances, logLikelihood

#From solution
def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Start Subtask 6c#####

    # Get the sizes
    n_training_samples, dim = X.shape
    K = gamma.shape[1]

    # Create matrices
    means = np.zeros((K, dim))
    covariances = np.zeros((dim, dim, K))

    # Compute the weights
    Nk = gamma.sum(axis=0)
    weights = Nk / n_training_samples

    # for i in range(K):
    #     auxMean = np.zeros(dim)
    #     for j in range(n_training_samples):
    #         auxMean += gamma[j, i] * X[j]
    #     means[i] = auxMean / Nk[i]
    # newmeans(gamma, softnums, X, N, K, D)
    #
    #means = np.divide(gamma.T.dot(X), Nk[:, np.newaxis])
    # assert means_new.all() == means.all()

    """
    for i in range(K):
        auxSigma = np.zeros((dim, dim))
        
        for j in range(n_training_samples):
            meansDiff = X[j] - means[i]
            auxSigma = auxSigma + gamma[j, i] * np.outer(meansDiff.T, meansDiff)
        covariances[:, :, i] = auxSigma/Nk[i]
    """
    means = newmeans(gamma, Nk, X, n_training_samples, K, dim)
    covariances = newcovs(X, means, gamma, Nk, n_training_samples, dim, K)

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    #####End Subtask#####
    return weights, means, covariances, logLikelihood
