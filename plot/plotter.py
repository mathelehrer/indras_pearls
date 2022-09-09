from matplotlib import pyplot as plt


class PlotObject:

    def visualize(self):
        xs=[]
        ys=[]
        return xs,ys


class Plotter:
    def plot(*plot_objects):
        for plot_object in plot_objects:
            x,y = plot_object.visualize()
            plt.plot(x,y)
        plt.gca().set_aspect('equal')
        plt.show()