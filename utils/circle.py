import numpy as np
from plot.plotter import PlotObject


class Circle(PlotObject):
    """
    This class is a container for a circle, which is a geometric object in the complex plane
    """
    def __init__(self, center, radius):
        """
        defines a circle in the complex plane
        :param center:
        :param radius:
        """
        self.r = radius
        self.c = center

    def visualize(self):
        phi = np.linspace(start=0,stop=np.pi*2,num=100)
        z = np.cos(phi)*self.r+1j*np.sin(phi)*self.r+self.c
        return np.real(z),np.imag(z)

    def area(self):
        return np.pi*self.r**2

    def circumference(self):
        return np.pi*self.r*2


def random_circle():
    r = np.random.random()*10-5
    c = np.random.random()*10-5+1j*(np.random.random()*10-5)
    return Circle(c,r)