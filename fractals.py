from pylab import *
import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import *
from numpy import linalg as LA


class Fractal2D:

    def __init__(self, f, deriv=None):
        """
        The constructor take two arguments, the first is f the function, while the second is derivative of this function
        :param f: tow dim array, which contain two functions
        :param deriv: partial derivative
        """
        self.f = f
        self.deriv = deriv
        self.zeros = np.array([[]])

    def newtons_method(self, x0, tol=1e-6, maxiter=20):
        x0.shape = (2, 1)
        f_1 = self.function[0, 0]  # takeout the functions
        f_2 = self.function[1, 0]

        d_11 = self.deriv[0, 0]  # takeout the derivative
        d_12 = self.deriv[0, 1]
        d_21 = self.deriv[1, 0]
        d_22 = self.deriv[1, 1]

        for i in range(maxiter):
            x1 = x0[0]
            x2 = x0[1]

            jacobean = array([[d_11(x1, x2), d_12(x1, x2)],  # create jacobean.
                              [d_21(x1, x2), d_22(x1, x2)]])
            jacobean.shape(2, 2)

            jacobean_inv = solve(jacobean, eye(2, 2))  # find jacobean inverse.

            f = array([[f_1(x1, x2)], [f_2(x1, x2)]])
            f.shape(2, 1)  # put together the function.

            x = x0 - dot(jacobean_inv, f)  # run the iteration
            if np.abs(x[0] - x0[0] < tol) and np.abs(x[1] - x0[1] < tol):
                self.zeros = np.append(x)
                self.zeros.shape(2, 1)
                x0 = x
                # run to last iteration.
            else:
                self.zeroes['divergence']
                # when no solution found

        return self.zeros
        # return zeros
