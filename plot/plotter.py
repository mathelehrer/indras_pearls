from matplotlib import pyplot as plt


class PlotObject:

    def visualize(self):
        xs = []
        ys = []
        return xs, ys


class Plotter:
    def plot(*plot_objects, colors=None):
        for i, plot_object in enumerate(plot_objects):
            col = 'r'
            if colors:
                if len(colors) > i:
                    col = colors[i]
                else:
                    col = colors[-1]

            x, y = plot_object.visualize()
            plt.plot(x, y, col)
        plt.gca().set_aspect('equal')
        plt.show()




