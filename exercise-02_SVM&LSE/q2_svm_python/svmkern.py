import numpy as np
from kern import kern
import cvxopt
import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
# import os
# os.environ["PATH"] += os.pathsep + ...
from cvxopt import solvers as cvxopt_solvers
from cvxopt import matrix



def svmkern(X, t, C, p):
    # Non-Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                        (num_samples x dim)
    # t        : labeling                           (num_samples x 1)
    # C        : penalty factor the slack variables (scalar)
    # p        : order of the polynom               (scalar)
    #
    # OUTPUT:
    # sv       : support vectors (boolean)          (1 x num_samples)
    # b        : bias of the classifier             (scalar)
    # slack    : points inside the margin (boolean) (1 x num_samples)

    #####Insert your code here for subtask 2d#####

    N = X.shape[0]

    q = (-1) * np.ones(N)

    H = np.zeros((N, N))

    for i in range(N):

        for j in range(N):
            H[i, j] = t[i] * t[j] * kern(X[i], X[j], p)

    P = matrix(H)
    q = matrix((-1) * np.ones(N))
    n = H.shape[1]
    # vstack because it should represent 0 < a < C
    # we want to determine a and in the optimizer, we need
    #  Gx < h x-> a, h-> C, 0 some value before or after <,> operator
    G = matrix(np.vstack((-np.eye(n), np.eye(n))))
    # Ax = b
    # t.T*a = 0
    # b -> 0, A -> t.T
    A = matrix(t.reshape(1, -1))
    b = matrix(np.zeros(1))
    h = matrix(np.hstack((np.zeros(N), np.ones(N) * C)))

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    # Hisamatrix[N×N]withelementsH(n,m)=tntm⟨xn,xm⟩,
    """
    P : constant matrix for minimizing problem of 0.5 part
    q : constant matrix for minimizing problem of behind part(-1)
    G : constant matrix for the standard form that has equality with the less-than sign, which lays right side x * h  ≤ G
    We do not have any inequality relation, thus exchange with np.vstack([-np.eye(n), np.eye(n)])
    h : constant matrix for the standard form that has equality with the less-than sign, which lays left side
    A : constant matrix for equality wrt the lagrangian multiplier
    b : constant for equality wrt the lagrangian multiplier

    Assumption
    tn*y(xn) = 1 - slack_n
    -> slack_n = abs(tn - y(xn))
    https://xavierbourretsicotte.github.io/SVM_implementation.html
    """

    # a vector
    alpha = np.array(sol['x']).reshape(-1, )
    # weight vector

    sv = np.where(alpha > 1e-6, True, False)
    if ~sv.any():

        raise ValueError("No support vectors found")

    else:

        slack = np.where(alpha > C - 1e-6, True, False)
        w = (alpha[sv] * t[sv]).dot(X[sv])
        b = np.mean(t[sv] - w.dot(X[sv].T))
        result = X.dot(w) + b

    return alpha, sv, b, result, slack
