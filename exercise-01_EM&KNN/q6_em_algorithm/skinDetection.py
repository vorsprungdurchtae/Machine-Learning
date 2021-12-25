import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood, getshape

import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Start Subtask 1g#####
    print('creating GMM for non-skin')
    weight_nonskin, means_nonskin, cov_nonskin = estGaussMixEM(ndata, K, n_iter, epsilon)
    print('GMM for non-skin completed')
    print('creating GMM for skin')
    weight_skin, means_skin, cov_skin = estGaussMixEM(sdata, K, n_iter, epsilon)
    print('GMM for skin completed')

    height, width, _ = img.shape

    noSkin = np.ndarray((height, width))
    skin = np.ndarray((height, width))

    for h in range(height):
        for w in range(width):
            noSkin[h, w] = np.exp(
                getLogLikelihood(means_nonskin, weight_nonskin, cov_nonskin, np.array([img[h, w, 0], img[h, w, 1],
                                                                                       img[h, w, 2]])))
            skin[h, w] = np.exp(
                getLogLikelihood(means_skin, weight_skin, cov_skin, np.array([img[h, w, 0], img[h, w, 1],
                                                                              img[h, w, 2]])))


    # calculate ration and threshold
    result = skin / noSkin
    result = np.where(result > theta, 1, 0)
    #####End Subtask#####
    return result


def skinDetectionTae(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    height, width, _ = img.shape

    #data from the skin data
    s_weights, s_means, s_covariances = estGaussMixEM(sdata, K, n_iter, epsilon)

    #data from the non skin data
    ns_weights, ns_means, ns_covariances = estGaussMixEM(ndata, K, n_iter, epsilon)

    skin_array = np.ndarray((height, width))
    non_skin_array = np.ndarray((height, width))

    for h in range(height):

        for w in range(width):

            skin_array[h,w] = np.exp(
                                    getLogLikelihood(s_means,
                                                     s_weights,
                                                     s_covariances,
                                                     np.array([img[h, w, 0],
                                                               img[h, w, 1],
                                                               img[h, w, 2]])
                                                     )
                                    )

            non_skin_array[h, w] = np.exp(
                                    getLogLikelihood(ns_means,
                                                     ns_weights,
                                                     ns_covariances,
                                                     np.array([img[h, w, 0],
                                                               img[h, w, 1],
                                                               img[h, w, 2]])
                                                    )
                                    )

    result = skin_array / non_skin_array
    result = np.where(result > theta, 1, 0)

    #####Insert your code here for subtask 1g#####
    return result
