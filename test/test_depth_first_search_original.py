import unittest

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from plot.plotter import Plotter
from shottky_dance.depth_first_search import DepthFirstSearch, DepthFirstSearch2
from shottky_dance.depth_first_search_original import DepthFirstSearchOriginal, KissingSchottky, ApollonianGasket

class depth_first_search_original_test(unittest.TestCase):
    def test_initialize(self):
        dfs = DepthFirstSearchOriginal(KissingSchottky, y=1, k=1, eps=0.1)
        for gen in dfs.gens:
            print(gen)
            print(dfs.fixed_point_of(gen))

    def test_end_check(self):
        dfs = DepthFirstSearchOriginal(KissingSchottky, y=1, k=0.1, eps=0.1)
        dfs.setup_start(begin_tag=[0,1,2,3],end_tag=[1,0,3,2])
        print(dfs.check_end(test=[2,3,1]),dfs.breaking_length)
        print(dfs.check_end(test=[1]),dfs.breaking_length)
        print(dfs.check_end(test=[1,1]),dfs.breaking_length)
        print(dfs.check_end(test=[1,0,3]),dfs.breaking_length)
        print(dfs.check_end(test=[1,0,3,1]),dfs.breaking_length)
        print(dfs.check_end(test=[1,0,3]),dfs.breaking_length)
        print(dfs.check_end(test=[1,0,4]),dfs.breaking_length)
        print(dfs.check_end(test=[1,0]),dfs.breaking_length)

    def test_kissing_schottky(self):
        dfs = DepthFirstSearchOriginal(KissingSchottky, y=0.5, k=0.1, eps=0.0001)
        dfs.setup_start(end_tag=[1, 0, 3, 2])
        dfs.run()
        # plt.scatter(np.real(dfs.points), np.imag(dfs.points), s=0.5, marker='.')
        plt.clf()
        plt.plot(np.real(dfs.points), np.imag(dfs.points))
        plt.gca().set_aspect('equal')
        plt.show()

    def test_sequence(self):
        plt.gca().set_aspect('equal')
        fig = plt.gcf()
        fig.set_size_inches(34, 14)
        for i in range(100,200):
            k=0.5-0.495*(i-100)/100
            dfs = DepthFirstSearchOriginal(KissingSchottky, y=0.875, k=k, eps=0.001)
            dfs.run(close_curve=True)
            # plt.scatter(np.real(dfs.points), np.imag(dfs.points), s=0.5, marker='.')
            plt.clf()
            plt.plot(np.real(dfs.points),np.imag(dfs.points))
            fig.savefig("/home/jmartin/PycharmProjects/indras_pearls/tmp/image"+str(i)+".png",dpi=100)
            #plt.show()
            # Plotter.plot(*dfs.circs) # only used this for level_max<6

    def test_sequence2(self):
        plt.gca().set_aspect('equal')
        fig = plt.gcf()
        fig.set_size_inches(34, 14)
        for i in range(100,200):
            y=3*(1-(i-100)/100)
            dfs = DepthFirstSearchOriginal(KissingSchottky, y=y, k=0.1, eps=0.001)
            dfs.run(close_curve=True)
            # plt.scatter(np.real(dfs.points), np.imag(dfs.points), s=0.5, marker='.')
            plt.clf()
            plt.plot(np.real(dfs.points),np.imag(dfs.points))
            fig.savefig("/home/jmartin/PycharmProjects/indras_pearls/tmp/figure2_"+str(i)+"_"+str(np.round(y*10)/10)+".png",dpi=100)
            #plt.show()
            #Plotter.plot(*dfs.circs) # only used this for level_max<6

    def test_simple(self):
        dfs = DepthFirstSearchOriginal(KissingSchottky, y=0.5, k=0.1, eps=0.001)
        dfs.run(close_curve=True)
        plt.clf()
        plt.plot(np.real(dfs.points),np.imag(dfs.points))
        plt.gca().set_aspect('equal')
        plt.show()
        # Plotter.plot(*dfs.circs) # only used this for level_max<6

    def test_Apollonian(self):
        dfs = DepthFirstSearchOriginal(ApollonianGasket,eps=0.05)
        dfs.run(close_curve=True)
        plt.clf()
        plt.gca().set_aspect('equal')
        plt.plot(np.real(dfs.points),np.imag(dfs.points))
        plt.show()
