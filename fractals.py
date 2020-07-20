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

# Hej! Jag har lagt till en main-funktion där jag definierar en vektorvärd funktion och Jacobian. Ser att det finns ett 
# flertal problem i koden som kommer att ge errors, du kommer se vart de är när du försöker köra koden. Har också skrivit 
# några rader som feedback på koden nedan, kolla gärna på det! 

# Du har skrivit f som en matris, det räcker med en vektor (som exemplet i main-funktionen).

# Man brukar undvika "import *"-uttryck för att göra det tydligt vilka moduler som används till vad. Till den kod du 
# skrivit räcker det med en enda import, "import numpy as np".

# Shape är ett attribut som numpy-arrayer har som beskriver hur vektorn ser ut (alltså ingenting som behöver definieras
# av dig). Om man skulle vilja ändra shape använder man "reshape", men i detta fallet behövs detta inte så du redan
# när du skapar arrayerna har rätt form på dem.

# Du behöver inte packa upp x0 till x1 och x2, det skulle dessutom skapa problem om man hade velat använda koden
# på fler än två dimensioner. På det sättet som jag definierat j1 i main-funktionen kan du helt enkelt skriva j1(x).
# Det finns också skalningsproblem där du jämför med toleransen, testa att använda np.linalg.norm för en generell 
# lösning!

# Uppdateringsformeln bör nog vara dx = solve(J, -f) -> x = x + dx, det finns ingen anledning att räkna ut inversen 
# explicit som du gör i "np.linalg.solve(jacobean, np.eye(2, 2))"

# Kolla i WolframAlpha/Matlab/liknande vilka nollställen som du borde hitta och testa med olika x0 för att se 
# så att allt stämmer! Skriv gärna till mig (Rickard) på Discord om du undrar över någon kommentar, vill 
# diskutera någon lösning eller vad som helst! 

if __name__ == '__main__':
    # Define the vector function
    f1 = lambda x: np.array([x[0]**3 - 3*x[0]*(x[1]**2) - 1, 3*(x[0]**2)*x[1] - x[1]**3])
    # Define the Jacobian
    j1 = lambda x: np.array([[3*(x[0]**2 - x[1]**2), -6*x[0]*x[1]], [6*x[0]*x[1], 3*(x[0]**2 - x[1]**2)]])
    
    fractal = Fractal2D(f1, j1)
    x0 = np.array([-1, 1])
    print(fractal.newtons_method(x0))
