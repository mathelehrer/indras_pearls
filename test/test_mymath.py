import unittest

import numpy as np
from numpy.linalg import det

from utils import mymath
from utils.circle import Circle, random_circle
from utils.mymath import moebius_on_point, plot_function, moebius_on_circle


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
        plot_function(lambda x:x**2,-2,2,100)

    def test_circle(self):
        circle = Circle(1+1j,1)
        circle.visualize()


    def test_moebius_on_circle(self):
        """
        :return:
        """

        # random moebius transformation
        m = np.array([[1, 0], [0, 1]])
        for i in range(0, 1000):
            for r in range(0, 2):
                for c in range(0, 2):
                    m[r][c] = 5 - np.random.random() * 10

        # random circle

        circle = random_circle()
        circle.visualize()
        circle2 = moebius_on_circle(m,circle)
        circle2.visualize()