import numpy as np
from numpy.random import choice
from simpleClassifier import simpleClassifier
from eval_adaBoost_simpleClassifier import eval_adaBoost_simpleClassifier

def adaboostCross(X, Y, K, nSamples, percent):
    # Adaboost with an additional cross validation routine
    #
    # INPUT:
    # X         : training examples (numSamples x numDims )
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier)
    # nSamples  : number of training examples which are selected in each round. (scalar)
    #             The sampling needs to be weighted!
    # percent   : percentage of the data set that is used as test data set (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of simple classifier (K x 2)
    # testX     : test dataset (numTestSamples x numDim)
    # testY     : test labels  (numTestSamples x 1)
    # error        : error rate on validation set after each of the K iterations (K x 1)

    #####Insert your code here for subtask 1d#####
    # Randomly sample a percentage of the data as test data set
    """

    textX = sampled data with the train_prop
    textY = sampled data with the train_prop

    TODO
    extract the instances of train_prop with exact indices

    """

    N, _ = X.shape

    train_prop = 1 - percent

    train_N = round(N*train_prop)

    pos = choice(N, round(N*percent), False)

    allpos = range(N)

    restpos = np.setdiff1d(allpos, pos)

    train_X = X[restpos, :]
    train_Y = Y[restpos]

    testX = X[pos]
    testY = Y[pos]
    error = np.zeros(K)

    j = np.zeros(K) * (-1)
    theta = np.zeros(K) * (-1)

    # weight is the factor used to regulate the choosing probability of each data point
    w = (np.ones(train_N) / train_N).reshape(train_N, 1)

    alpha = np.zeros(K)

    for k in range(K):

        index = choice(train_N, nSamples, True, w.ravel())

        sX = train_X[index, :]
        sY = train_Y[index]

        j[k], theta[k] = simpleClassifier(sX, sY)

        cY = (np.ones(train_N) * -1).reshape(train_N, 1)
        cY[train_X[:, int(j[k] - 1)] > theta[k]] = 1

        temp = np.where([Y[i] != cY[i] for i in range(train_N)], 1, 0).reshape(train_N, 1)

        ek = np.sum(temp * w)

        if ek < 1.0e-01:
            alpha[k] = 1
            break

        alpha[k] = 0.5 * np.log((1 - ek) / ek)
        w = w * np.exp(-alpha[k] * cY * train_Y)
        w = w / np.sum(w)  # normalization, otherwise, the weights grow exponentially

        para = np.stack((j[:k + 1], theta[:k + 1]), axis=1)

        classlabels, _ = eval_adaBoost_simpleClassifier(testX, alpha[:k+1],
                                                        para[:k+1])
        classlabels = classlabels.reshape(len(classlabels), 1)

        error[k] = len(classlabels[classlabels != testY]) / len(testY)

    alphaK = alpha
    para = np.stack((j, theta), axis=1)

    return alphaK, para, testX, testY, error

