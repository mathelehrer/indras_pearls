import unittest

import numpy as np
from matplotlib import pyplot as plt

from shottky_dance.depth_first_search import DepthFirstSearch



def get_index_or_last(i,list):
    if len(list)>i:
        return list[i]
    else:
        return list[-1]


def plot_lists(lists, colors=['r', 'r']):
    for i,list in enumerate(lists):
        for l in list:
            x,y=l.visualize()
            plt.plot(x,y,get_index_or_last(i,colors))
    plt.gca().set_aspect('equal')
    plt.show()


class depth_first_search_test(unittest.TestCase):
    def test_initialize(self):
        dfs = DepthFirstSearch(theta=np.pi/4,level_max=5)
        identity = np.eye(2)
        # check inverses A = a**(-1)
        # check inverses B = b**(-1)
        # check inverses a = A**(-1)
        # check inverses b = B**(-1)
        [self.assertTrue(np.allclose(np.dot(dfs.gens[i],dfs.gens[dfs.inv[i]]),identity)) for i in range(0,4)]

    def test_circles_at_levels(self):
        circles_at_levels=[]
        for level_max in range(0,5):
            dfs = DepthFirstSearch(theta=np.pi/4,level_max=level_max)
            dfs.search()

            circles_at_levels.append(dfs.circs)

        colors=['r','g','b','y','m']
        plot_lists(circles_at_levels,colors)


