import numpy as np

from utils.circle import Circle
from utils.mymath import moebius_on_circle


class BreadFirstSearch:
    def __init__(self, theta=np.pi / 4, level_max=3):
        self.level_max=level_max
        self.gens = []
        self.group = []
        self.circles = []
        self.level = 0
        self.tags = []
        self.inv = []
        self.num = []
        self.a = None
        self.A = None
        self.b = None
        self.B = None

        self.initialize(theta=theta)

        for i in range(0, 4):
            self.group.append(self.gens[i])
            self.tags.append(i)

        self.num.append(0)  # index, where level 0 begins
        self.num.append(4)  # index, where level 1 begins

        self.generate_levels()

    def initialize(self, theta=np.pi / 4):
        i = 1j
        cos = np.cos(theta)
        sin = np.sin(theta)
        self.gens.append(1 / sin * np.array([[1, i * cos], [-i * cos, 1]]))
        self.a = self.gens[0]
        self.gens.append(1 / sin * np.array([[1, cos], [cos, 1]]))
        self.b = self.gens[1]
        self.gens.append(np.linalg.inv(self.a))
        self.A = self.gens[2]
        self.gens.append(np.linalg.inv(self.b))
        self.B = self.gens[3]

        self.inv.append(2)  # setup index of the corresponding inverse transformation
        self.inv.append(3)
        self.inv.append(0)
        self.inv.append(1)

        center = 1 / np.cos(theta)
        radius = np.tan(theta)
        self.circles.append(Circle(i * center, radius))
        self.circles.append(Circle(center, radius))
        self.circles.append(Circle(-i * center, radius))
        self.circles.append(Circle(-center, radius))


    def generate_levels(self):
        for level in range(1, self.level_max):
            i_new = self.num[level]
            for i_old in range(self.num[level - 1], self.num[level]):
                for j in range(0, 4):
                    if self.inv[self.tags[i_old]] != j:
                        self.group.append(np.dot(self.group[i_old], self.gens[j]))
                        self.tags.append(j)
                        i_new += 1
            self.num.append(i_new)

    def output(self):
        circles = [*self.circles]
        for i in range(0, self.num[self.level_max-1]):
            for j in range(0, 4):
                if self.inv[self.tags[i]] != j:
                    circles.append(moebius_on_circle(self.group[i], self.circles[j]))
        return circles


if __name__ == '__main__':
    BreadFirstSearch()
