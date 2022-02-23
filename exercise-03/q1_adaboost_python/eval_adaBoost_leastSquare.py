import numpy as np

def eval_adaBoost_leastSquare(X, alphaK, para):
    # INPUT:
    # para        : parameters of simple classifier (K x (D +1))
    #           : dimension 1 is w0
    #           : dimension 2 is w1
    #           : dimension 3 is w2
    #             and so on
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (scalar)

    #####Insert your code here for subtask 1e#####
    K = para.shape[0]  # number of classifiers
    N, D = X.shape  # number of test points and dimension
    result = np.zeros(N)  # prediction for each test point
    X = np.insert(X, 0, 1, axis = 1)

    for k in range(K):
        # Initialize temporary labels for given j and theta
        cY = np.ones(N) * (-1)
        # Classify
        cY[X.dot(para[k]) >= 0] = 1

        result += cY.T.dot(alphaK[k]) # update results with weighted prediction

    classLabels = np.sign(result)  # class-predictions for each test point

    return [classLabels, result]

