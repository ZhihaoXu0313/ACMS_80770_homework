"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 4: Programming assignment
Problem 1
"""
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)


class kernel:
    def __init__(self, K, R, d, J, lamb_max):
        # -- filter properties
        self.R = float(R)
        self.J = J
        self.K = K
        self.d = np.array(d)
        self.lamb_max = torch.tensor(lamb_max)

        # -- Half-Cosine kernel
        self.a = self.R * torch.log(self.lamb_max) / (self.J - self.R + 1)
        self.g_hat = lambda lamb: sum([self.d[k] * torch.cos(2 * np.pi * k * (lamb / self.a + 1 / 2))
                                       for k in range(self.K + 1)]) * (-lamb >= 0) * (-lamb <= self.a)

    def wavelet(self, lamb, j):
        """
        constructs wavelets ($j\in [2, J]$).
        :param lamb: eigenvalue (analogue of frequency).
        :param j: filter index in the filter bank.
        :return: filter response to input eigenvalues.
        """
        return self.g_hat(torch.log(lamb) - self.a / self.R * j)

    def scaling(self, lamb):
        """
        constructs scaling function (j=1).
        :param lamb: eigenvalue (analogue of frequency).
        :return: filter response to input eigenvalues.
        """
        norm_square = np.array([self.wavelet(lamb, i) ** 2 for i in range(1, self.J)])
        eps = self.R * self.d[0] ** 2 + 0.5 * self.R * sum(self.d[1:] ** 2) - sum(norm_square)
        eps[eps < 0] = 0
        return torch.sqrt(eps)


# -- define filter-bank
lamb_max = 2
J = 8
filter_bank = kernel(K=1, R=3, d=[0.5, 0.5], J=J, lamb_max=lamb_max)

# -- plot filters
lamb = torch.tensor(np.arange(0.01, lamb_max, 0.01))

for j in range(J):
    if j == 0:
        sf = filter_bank.scaling(lamb)
    else:
        sf = filter_bank.wavelet(lamb, j)
    plt.plot(lamb, sf)

plt.xlim([0, lamb_max])
plt.xlabel("lambda")
plt.show()
