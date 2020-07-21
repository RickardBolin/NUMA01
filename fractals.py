from pylab import *


class Fractal2D:

    def __init__(self, func, deriv=None):
        """
        The constructor take two arguments, the first is f the function, while the second is derivative of this function
        :param f: tow dim array, which contain two functions
        :param deriv: partial derivative
        """
        self.func = func
        self.deriv = deriv
        self.zeros = np.array([[]])

    def newtons_method(self, x0, tol=1e-6, maxiter=20):

        for i in range(maxiter):

            jacobean = self.deriv([x0[0], x0[1]])

            jacobean_inv = solve(jacobean, eye(2, 2))  # find jacobean inverse.

            f = self.func([x0[0], x0[1]])

            x = x0 - dot(jacobean_inv, f)  # run the iteration
            if np.abs(x[0] - x0[0]) < tol and np.abs(x[1] - x0[1]) < tol:
                self.zeros = x

                # run to last iteration.
            else:
                self.zeros = None
                # when no solution found
            x0 = x
        return self.zeros
        # return zeros






if __name__ == '__main__':
    # Define the vector function
    f1 = lambda x: np.array([x[0] ** 3 - 3 * x[0] * (x[1] ** 2) - 1, 3 * (x[0] ** 2) * x[1] - x[1] ** 3])
    # Define the Jacobian
    j1 = lambda x: np.array(
        [[3 * (x[0] ** 2 - x[1] ** 2), -6 * x[0] * x[1]], [6 * x[0] * x[1], 3 * (x[0] ** 2 - x[1] ** 2)]])

    fractal = Fractal2D(f1, j1)
    x0 = np.array([-1, 1])
    print(fractal.newtons_method(x0))
