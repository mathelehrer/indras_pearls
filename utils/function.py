import numpy as np

from plot.plotter import PlotObject


class Function(PlotObject):
    def __init__(self,mapping,x_min=0,x_max=10,resolution=10):
        self.mapping = mapping
        self.range=[x_min,x_max]
        self.resolution=resolution

    def visualize(self):
        x = np.linspace(start=self.range[0], stop=self.range[1], num=self.resolution)
        y = self.mapping(x)
        return x,y