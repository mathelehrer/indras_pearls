import unittest

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import det

from plot.plotter import Plotter
from utils import mymath
from utils.circle import Circle, random_circle
from utils.function import Function
from utils.mymath import moebius_on_point, moebius_on_circle


class mymath_test(unittest.TestCase):
    def test_cxroot(self):
        """
        take the square root of 1000 random complex numbers
        :return:
        """
        for i in range(0,1000):
            z = (5-np.random.random()*10)+1j*(5-np.random.random()*10)
            root = mymath.cxsqrt(z)
            self.assertAlmostEqual(z,root*root)

    def test_moebius_on_point(self):
        """
        calculate 1000 random Moebius transformations and their inverses
        Check that the Moebius transformation on a point is compatible with the inversion of the matrix
        :return:
        """
        m=np.array([[1,0],[0,1]])
        for i in range(0,1000):
            for r in range(0,2):
                for c in range(0,2):
                    m[r][c]=5-np.random.random()*10

            if det(m)!=0:
                z =  (5-np.random.random()*10)+1j*(5-np.random.random()*10)
                mi = np.linalg.inv(m)
                z2 = mymath.moebius_on_point(mi,mymath.moebius_on_point(m,z))
                self.assertAlmostEqual(z,z2)

        z = np.inf
        self.assertAlmostEqual(moebius_on_point(m,z),m[0][0]/m[1][0])

    def test_plot(self):
        parabola = Function(lambda x:x**2,-2,2,100)
        Plotter.plot(parabola)

    def test_circle(self):
        circle = Circle(1+1j,1)
        Plotter.plot(circle)

    def test_moebius_on_circle(self):
        """
        :return:
        """
        # random moebius transformation
        m = np.array([[1, 0], [0, 1]])

        for r in range(0, 2):
            for c in range(0, 2):
                m[r][c] = 3 - np.random.random() * 6

        # random circle
        circles=[random_circle()]

        for i in range(100):
            circles.append(moebius_on_circle(m,circles[-1]))
        Plotter.plot(*circles)