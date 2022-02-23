import numpy as np
from numpy.random import choice
from leastSquares import leastSquares
from eval_adaBoost_leastSquare import eval_adaBoost_leastSquare


def adaboostUSPS(X, Y, K, nSamples, percent):
    # Adaboost with least squares linear classifier as weak classifier on USPS data
    # for a high dimensional dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (scalar)
    # nSamples  : number of data points obtained by weighted sampling (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (1 x k)
    # para      : parameters of simple classifier (K x (D+1))
    #             For a D-dim dataset each simple classifier has D+1 parameters
    # error     : training error (1 x k)

    N, D = X.shape

    #this number will be used for the number of  training data
    index_number = round(percent * N)

    #this indicates the number of validation dataset
    test_number = N - index_number

    index = choice(N, index_number, False)
    allpos = range(N)
    restpos = np.setdiff1d(allpos, index)

    trainingX = X[index, :]
    trainingY = Y[index]

    testX = X[restpos, :]
    testY = Y[restpos]

    w = (np.ones(index_number)/index_number).reshape(index_number, 1)
    error = np.zeros(K)
    para = np.zeros((K, D+1))
    alphaK = np.zeros(K)

    for k in range(K):

        #choose the index randomly
        idx = choice(index_number, nSamples, True, w.ravel())
        #samples from the training data according to the idx
        sX = trainingX[idx, :]
        #sampled target label according to the idx
        sY = trainingY[idx]

        weight, bias = leastSquares(sX, sY)

        para[k, :] = np.append(weight, [bias])
        ones = np.ones(len(trainingX)).reshape(len(trainingX), 1)
        cX = np.sign(np.append(ones, trainingX, axis = 1).dot(para[k].T)).T

        cY = np.where([cX[i] != trainingY[i] for i in range(len(cX))], 1, 0).reshape(len(cX), 1)

        ek = np.sum(cY*w)

        if ek < 1.0e-01:
            alphaK[k] = 1
            break

        alphaK[k] = 0.5 * np.log((1 - ek) / ek)
        w = w * np.exp((-alphaK[k]) * (trainingY*(cX.reshape(len(cX),1))))
        w = w / np.sum(w)  # normalization, otherwise, the weights grow exponentially

        classlabel, _ = eval_adaBoost_leastSquare(testX, alphaK[:k+1], para[:k+1])

        classlabel = classlabel.reshape(len(classlabel), 1)

        error[k] = np.sum(np.where([classlabel[i] != testY[i] for i in range(len(testY))], 1, 0))/len(testY)

    return [alphaK, para, error]
