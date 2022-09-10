import unittest

import numpy as np
from matplotlib import pyplot as plt
from numpy import eye
from numpy.linalg import det

from plot.plotter import Plotter
from shottky_dance.breadth_first_search import BreadFirstSearch
from utils import mymath
from utils.circle import Circle, random_circle
from utils.function import Function
from utils.mymath import moebius_on_point, moebius_on_circle


class breadth_first_search_test(unittest.TestCase):
    def test_initialize(self):
        bfs = BreadFirstSearch(theta=np.pi/6,level_max=4)
        identity = np.eye(2)
        self.assertTrue(np.allclose(np.dot(bfs.a,bfs.A),identity))
        self.assertTrue(np.allclose(np.dot(bfs.b,bfs.B),identity))

        cols = ['r','g','b','y']
        colors=['y','b','r','g']
        for tag in bfs.tags:
            colors.append(cols[tag])
        Plotter.plot(*bfs.output(),colors=colors)
