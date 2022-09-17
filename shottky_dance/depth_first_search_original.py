import numpy as np

from utils.circle import Circle
from utils.mymath import moebius_on_circle, moebius_on_point


class theta_family:
    def __init__(self, theta=np.pi / 4):
        self.theta = theta

    def get_fixed_points(self):
        return [1j, 1, -1j, -1]

    def get_generators(self):
        gens = []
        sin = np.sin(self.theta)
        cos = np.cos(self.theta)
        i = 1j
        gens.append(1 / sin * np.array([[1, i * cos], [-i * cos, 1]]))
        a = gens[-1]
        gens.append(1 / sin * np.array([[1, cos], [cos, 1]]))
        b = gens[-1]
        gens.append(np.linalg.inv(a))
        A = gens[-1]
        gens.append(np.linalg.inv(b))
        B = gens[-1]
        return [a, b, A, B]

    def get_circles(self):
        c = 1 / np.cos(self.theta)
        r = np.tan(self.theta)
        i = 1j
        return [
            Circle(i * c, r),
            Circle(c, r),
            Circle(-i * c, r),
            Circle(-c, r)
        ]


class DepthFirstSearchOriginal:
    def __init__(self, family, eps=0.1, **kwargs):
        self.lev_max = None  # defined in run()
        self.gens = family(**kwargs).get_generators()
        self.fixed_points = family(**kwargs).get_fixed_points()
        self.circles = family(**kwargs).get_circles()

        self.inv = [2, 3, 0, 1]

        self.word = []
        self.tags = []
        self.epsilon = eps

        self.lev = 0
        self.tags.append(0)
        self.word.append(self.gens[0])

        self.points = []  # field to store the limit points

    def run(self, lev_max=4):
        self.lev_max = lev_max
        while self.lev != -1 or self.tags[0] != 1:
            while not self.branch_termination():
                self.go_forward()
                # print(self.tags)
            while True:  # do ... while loop
                self.go_backward()
                if self.lev == -1 or self.available_turn():
                    break
            self.turn_and_go_forward()

    def go_forward(self):
        self.lev += 1
        new_tag = (self.tags[
                       self.lev - 1] + 1) % 4  # start with the rightmost branch: enter with a-> b,a,B; enter with b -> A,b,a; ...
        if len(self.tags) > self.lev:
            self.tags[self.lev] = new_tag
        else:
            self.tags.append(new_tag)

        new_word = np.dot(self.word[self.lev - 1], self.gens[self.tags[self.lev]])
        if len(self.word) > self.lev:
            self.word[self.lev] = new_word
        else:
            self.word.append(new_word)

    def go_backward(self):
        self.lev -= 1

    def available_turn(self):
        if self.tags[self.lev + 1] - 1 == (self.tags[self.lev] + 2) % 4:
            return False
        else:
            return True

    def turn_and_go_forward(self):
        self.tags[self.lev + 1] = (self.tags[self.lev + 1] - 1) % 4  # go to the next left branch
        if self.lev == -1:
            self.word[0] = self.gens[self.tags[0]]
        else:
            self.word[self.lev + 1] == np.dot(self.word[self.lev], self.gens[self.tags[self.lev + 1]])
        self.lev += 1

    def branch_termination(self):
        new_circ = moebius_on_circle(self.word[self.lev - 1], self.circles[self.tags[self.lev]])
        if self.lev == self.lev_max or new_circ.r < self.epsilon:
            self.points.append(moebius_on_point(self.word[self.lev], self.fixed_points[self.tags[self.lev]]))
            print("point added for tag: ",self.tags)
            return True
        else:
            return False


if __name__ == '__main__':
    dfs = DepthFirstSearchOriginal(theta_family, theta=np.pi / 4)
    dfs.run(1)
    print(dfs.points)
