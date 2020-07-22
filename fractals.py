import numpy as np



class Fractal2D:

    def __init__(self, f, deriv=None):
        """
        The constructor take two arguments, the first is f the function, while the second is derivative of this function
        :param f: tow dim array, which contain two functions
        :param deriv: partial derivative
        """
        self.f = f
        self.deriv = deriv
        self.zeros = []

    def newtons_method(self, x0, tol=1e-6, maxiter=20):
        for i in range(maxiter):
            x = x0
            for i in range(maxiter):
                x = x + np.linalg.solve(self.deriv(x), -self.f(x))
                if np.linalg.norm(self.f(x)) < tol:
                    return x, i
            return None, maxiter


if __name__ == '__main__':
    # Define the vector function
    f1 = lambda x: np.array([x[0] ** 3 - 3 * x[0] * (x[1] ** 2) - 1, 3 * (x[0] ** 2) * x[1] - x[1] ** 3])
    # Define the Jacobian
    j1 = lambda x: np.array(
        [[3 * (x[0] ** 2 - x[1] ** 2), -6 * x[0] * x[1]], [6 * x[0] * x[1], 3 * (x[0] ** 2 - x[1] ** 2)]])

    fractal = Fractal2D(f1, j1)
    x0 = np.array([-1, 1])
    print(fractal.newtons_method(x0))
