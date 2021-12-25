import numpy as np
from parameters import parameters
import matplotlib.pyplot as plt
import math

h, k = parameters()

def gauss1D(m, v, N, w):
    pos = np.arange(-w, w - w / N, 2 * w / N)
    insE = -0.5 * ((pos - m) / v) ** 2
    norm = 1 / (v * np.sqrt(2 * np.pi))
    res = norm * np.exp(insE)
    realDensity = np.stack((pos, res), axis=1)
    return realDensity

realDensity = gauss1D(0, 1, 100, 5)

print(realDensity[0][0])

plt.subplot(2, 1, 1)
plt.plot(realDensity[:, 0], realDensity[:, 1], 'b', linewidth=1.5, label='Real Distribution')
plt.legend()
plt.show()

#h = 0.3
#k