import numpy as np

def simpleClassifierTae(X, Y):
    # Select a simple classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    #
    # OUTPUT:
    # theta     : threshold value for the decision (scalar)
    # j         : the dimension to "look at" (scalar)

    #goal : finding a classifier(value within the X that splits the samples with minimum error rate)
    # HOWTO?
    
    """

    PLAN
    1. shift the Y 1 index
    2. add them each other
    3. get the positions where the sum of Y values result in 0
    4. get the indices of X sorted (aufsteigend)
    5.

    1. order the X and get the ordered index after value
    2. set the Ys like the order from 1.
    3. shift them one position
    4. add them and get the position of 0s
    5.

    """

    N, D = X.shape

    le = 1  # minimum error
    j = 1  # the dimension to look at -> scalar
    theta = 0

    for jj in range(D):

        # order the X and get the ordered index after value
        sorted_X = np.sort(X[:, jj])

        idx = np.argsort(X[:, jj])

        # Y[sorted_X] : re-position the y values after the achieved index above
        # shift the rep_Y each one position, add them, and get the positions where == 0
        # indices of X values for the criteria of Y being -1 and 1
        variations = np.where(np.roll(Y[idx], -1) + Y[idx] == 0)[0]

        # get the average between them
        transitions = (sorted_X[variations[variations < len(X) - 1]] + sorted_X[variations[variations < len(X) - 1] + 1])/2

        # based on these position, we now predict the y values
        # and will get the position where the least error rate occurs
        # theta should be the average between the X value occurring -1 and 1

        # this loop creates the -1 and 1 array based on the transitions
        # set the values 1 with x > t, otherwise, -1

        # here the array of -1
        error = np.zeros(len(transitions))

        for t in range(len(transitions)):

            cY = np.ones(N)*-1

            cY[X[:, jj] > transitions[t]] = 1

            error[t] = sum([Y[i] != cY[i] for i in range(N)])

        le1 = min(error/N)
        ind1 = np.argmin(error/N)

        le2 = min(1-error/N)
        ind2 = np.argmin(1-error/N)

        le0 = min([le1, le2, le])

        if le0 == le:

            continue

        else:

            le = le0
            j = jj+1

            if le1 == le:

                theta = transitions[ind1]

            else :

                theta = transitions[ind2]

    #####Insert your code here for subtask 1b#####
    return j, theta











def simpleClassifier(X, Y):
    # Select a simple classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    #
    # OUTPUT:
    # theta     : threshold value for the decision (scalar)
    # j         : the dimension to "look at" (scalar)

    N, D = X.shape

    le = 1 # minimum error
    j = 1 # the dimension to look at -> scalar
    theta = 0

    for jj in range(D):

        val = X[:, jj]

        # sorted X values in the dimension j aufsteigend
        sVal = np.sort(val)

        # sorted indices from sVal aufsteigend
        idx = np.argsort(val)

        # change points of Y(-1 -> 1 or 1 -> -1)
        # return values
        # for example
        #  Y = [-1, -1, -1, ... -1, 1, 1, 1, ...]
        # when we shift just one index and sum them up
        # some point appears with the value zero
        # get the indices of values fulfilling condition np.roll(Y[idx], -1) + Y[idx] == 0
        # Y values are randomly distributed i.e. 1,-1,1,1,-1,-1, 1, -1, 1,1, -1, -1, -1, ...
        # what we need is to see where is the change in an incremental way so where can the change exist
        changes = np.where(np.roll(Y[idx], -1) + Y[idx] == 0)[0]
        # the thresholds are value between the + and - value
        th = (sVal[changes[changes < len(X) - 1]] + sVal[changes[changes < len(X) - 1]+1])/2
        #
        error = np.zeros(len(th))

        for t in range(len(th)):

            cY = np.ones(N)*-1

            cY[X[:, jj] > th[t]] = 1

            error[t] = sum([Y[i] != cY[i] for i in range(N)])

            print('J = {0} \t Theta = {1} \t Error = {2}\n'.format(jj, th[t], error[t]))

        print(error)

        le1 = min(error/N)
        ind1 = np.argmin(error/N)
        le2 = min(1-error/N)
        ind2 = np.argmin(1-error/N)
        print([le1, le2, le])
        le0 = min([le1, le2, le])

        if le == le0:
            continue
        else:
            le = le0
            j = jj + 1  # Set theta to current value of threshold
            # Choose theta and parity for minimum error
            if le1 == le:
                theta = th[ind1]
            else:
                theta = th[ind2]

    return j, theta
