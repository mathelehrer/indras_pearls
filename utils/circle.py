import numpy as np
from plot.plotter import PlotObject


class Circle(PlotObject):
    """
    This class is a container for a circle, which is a geometric object in the complex plane
    :param center:
    :param radius:
    """
    def __init__(self, center, radius,p1=None,p2=None):
        """
        defines a circle in the complex plane
        :param center:
        :param radius:
        """
        self.r = radius
        self.c = center
        self.p1 = p1
        self.p2 = p2

    def visualize(self):
        phi = np.linspace(start=0,stop=np.pi*2,num=100)
        if self.r<np.inf:
            # if it's a circle
            z = np.cos(phi)*self.r+1j*np.sin(phi)*self.r+self.c
        else:
            # if it's a line
            x = -1+phi/np.pi
            z = self.p1+(self.p2-self.p1)*x
        return np.real(z),np.imag(z)

    def area(self):
        return np.pi*self.r**2

    def circumference(self):
        return np.pi*self.r*2


def random_circle():
    r = np.random.random()*10-5
    c = np.random.random()*10-5+1j*(np.random.random()*10-5)
    return Circle(c,r)