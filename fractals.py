#from pylab import *  # Detta brukar undvikas, man brukar vara tydlig med att skriva vilken modul det är man importerar
import numpy as np # <-- Kör på denna istället, så blir det tydlig vad som ligger i np.linalg och vad som ligger i np.

class Fractal2D:

    def __init__(self, f, deriv=None):
        """
        The constructor take two arguments, the first is f the function, while the second is derivative of this function
        :param f: tow dim array, which contain two functions
        :param deriv: partial derivative
        """
        self.f = f # Eric och Mårten har skrivit sin kod med "self.f", så blir lättast om vi behåller det som f!
        self.deriv = deriv
        self.zeros = [] # Räcker att detta är en lista, behöver inte vara en matris

    def newtons_method(self, x0, tol=1e-6, maxiter=20):
        for i in range(maxiter):
            
            # Dessa fyra raderna kan kortas ner till en rad, där vi dessutom slipper beräkna en dyr invers! 
            
            jacobean = self.deriv([x0[0], x0[1]])
            jacobean_inv = solve(jacobean, eye(2, 2))  # find jacobean inverse.
            f = self.func([x0[0], x0[1]])
            x = x0 - dot(jacobean_inv, f)  # run the iteration
            
            # fyra rader ovan samma sak som  x = x + np.linalg.solve(self.deriv(x), -self.f(x)) 
        
             
            # Här går alternativt att göra "if np.linalg.norm(self.f(x)) < tol:", vilket blir ett lite annorlunda stopping 
            # criteria jämfört med det du använt, med fördelen att vi slipper spara det förgående värdet. Båda bör dock ge samma resultat.
            
           
            if np.abs(x[0] - x0[0]) < tol and np.abs(x[1] - x0[1]) < tol:
                self.zeros = x

                # run to last iteration.
            else:
                self.zeros = None
                # when no solution found
            x0 = x
        return self.zeros
    
    # self.zeros är tänkt att användas i en annan funktion senare, Newton's method ska returnera det nollställe som hittades samt
    # hur många iterationer det tog att hitta nollstället (verkar skumt, men blir snygga plots av det, testa själv med Erics och Mårtens kod!)

    # Alltså skulle jag föreslå att det istället borde vara:
    
    #def newtons_method(self, x0, tol=1e-6, maxiter=20):
    #x = x0
    #for i in range(maxiter)
        #x = x + np.linalg.solve(self.deriv(x), -self.f(x)) 
        #if np.linalg.norm(self.f(x)) < tol:
            #return x, i
    #return None, maxiter
    
    # Testa gärna och säg till om du tycker att jag skrivit något konstigt eller att något kan skrivas bättre! 






if __name__ == '__main__':
    # Define the vector function
    f1 = lambda x: np.array([x[0] ** 3 - 3 * x[0] * (x[1] ** 2) - 1, 3 * (x[0] ** 2) * x[1] - x[1] ** 3])
    # Define the Jacobian
    j1 = lambda x: np.array(
        [[3 * (x[0] ** 2 - x[1] ** 2), -6 * x[0] * x[1]], [6 * x[0] * x[1], 3 * (x[0] ** 2 - x[1] ** 2)]])

    fractal = Fractal2D(f1, j1)
    x0 = np.array([-1, 1])
    print(fractal.newtons_method(x0))
