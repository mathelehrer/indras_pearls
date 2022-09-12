import unittest

import numpy as np
from matplotlib import pyplot as plt
from numpy import eye
from numpy.linalg import det

from plot.plotter import Plotter
from shottky_dance.breadth_first_search import BreadFirstSearch
from shottky_dance.depth_first_search import DepthFirstSearch
from utils import mymath
from utils.circle import Circle, random_circle
from utils.function import Function
from utils.mymath import moebius_on_point, moebius_on_circle


class depth_first_search_test(unittest.TestCase):
    def test_initialize(self):
        dfs = DepthFirstSearch(theta=np.pi/4,level_max=5)
        identity = np.eye(2)
        # check inverses A = a**(-1)
        # check inverses B = b**(-1)
        # check inverses a = A**(-1)
        # check inverses b = B**(-1)
        [self.assertTrue(np.allclose(np.dot(dfs.gens[i],dfs.gens[dfs.inv[i]]),identity)) for i in range(0,4)]
        # check go_forward and turn
        dfs.search()