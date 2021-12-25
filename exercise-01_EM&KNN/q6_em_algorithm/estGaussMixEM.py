import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov
from getLogLikelihood import getshape


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians
    # logLikelihood  : log-likelihood of the data given the model

    N, D = getshape(data)

    kmeans = KMeans(n_clusters=K, n_init=10).fit(data)
    cluster_idx = kmeans.labels_

    means = kmeans.cluster_centers_

    covariances = np.zeros((D,D,K))

    weights = np.ones(K) / K
    # Create initial covariance matrices
    for j in range(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf

        for i in range(K):
            # compute sum of distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))

            if dist < min_dist:

                min_dist = dist

        covariances[:, :, j] += np.eye(D) * min_dist

    for idx in range(n_iters):

        oldLogLi, gamma = EStep(means, covariances, weights, data)
        weights, means, covariances, newLogli = MStep(gamma, data)

        # regularize covariance matrix
        for j in range(K):
            covariances[:, :, j] = regularize_cov(covariances[:, :, j], epsilon)

        # termination criterion
        if abs(oldLogLi - newLogli) < 1:
            break

    #####Insert your code here for subtask 6e#####
    return [weights, means, covariances]
