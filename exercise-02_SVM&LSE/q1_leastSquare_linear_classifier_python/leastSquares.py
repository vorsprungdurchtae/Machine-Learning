import numpy as np
from io import StringIO

load_data = np.loadtxt('lc_train_data.dat')

load_label = np.loadtxt('lc_train_label.dat')
#insert 1 at the first position of each list


data = []

for i in range(len(load_data.T)):

    current_column = load_data.T[i]
    current_column = np.insert(current_column, 0, 1)

    data.append(current_column)


#X_Tilde
data_tilde = load_data.T
data_extended = np.array(data)
#print(data)
#print(data_extended)
"""
0. X vector is X = [x_1, x_2]
0.1 convert X data to Tilde X -> [x_1 T
                                  x_2 T]
1. In die X vector jeweils das erste Element 1 hinzufügen für w_0
2. Inverse of (transpose of Tilde_X inner product Tilde_X) Inner product transpose of Tilde_X Inner product of T
3. Bias is each of the first element(w_0)
"""
#print(load_label)
def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    # x = 38
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)

    num_samples, dim = data.shape
    data_tilde = np.ones((num_samples, dim + 1))
    data_tilde[:, 1:] = data

    weight = np.linalg.pinv(data_tilde) @ label
    bias = weight[0]
    weight = np.delete(weight, 0, axis=0)

    return weight, bias

#w,b = leastSquares(load_data, load_label)

#print(w, b)

