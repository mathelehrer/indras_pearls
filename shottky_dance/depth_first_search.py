import numpy as np

from utils.circle import Circle
from utils.mymath import moebius_on_circle, moebius_on_point
from utils.strings import tab_strings, underscore_strings


class DepthFirstSearch:
    def __init__(self, theta=np.pi / 4, level_max=4,eps=0.001):
        self.level = None
        self.word = []
        self.tags = []
        self.level_max = level_max
        self.labels = ['a', 'b', 'A', 'B']
        self.colors = ['r','g','b','y']
        self.gens = []
        self.circles = []
        self.circs = []
        self.cols =[]
        self.inv = []
        self.num = []
        self.fixed_points = [1j, 1,-1j, -1]
        self.a = None
        self.A = None
        self.b = None
        self.B = None
        self.points = []
        self.initialize(theta=theta)

        self.epsilon = eps

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
        self.inv.append(2)
        self.inv.append(3)
        self.inv.append(0)
        self.inv.append(1)
        [print(self.labels[i], self.gens[i]) for i in range(0, 4)]
        print("inverses: ", self.inv)

        # setup initial circles
        center = 1 / np.cos(theta)
        radius = np.tan(theta)
        self.circles.append(Circle(i * center, radius))
        self.circles.append(Circle(center, radius))
        self.circles.append(Circle(-i * center, radius))
        self.circles.append(Circle(-center, radius))

        print("Created ", len(self.circles), " circles:")
        [print("C[(", c.c, "),", c.r, "]") for c in self.circles]

        print("Fixed points:")
        [print(self.labels[i], ": ", self.fixed_points[i]) for i in range(0, 4)]
        print("Finished initialization")

    def recursive_search(self):
        id = np.eye(2)
        for k in range(0, 4):
            self.circs.append(self.circles[k])
            self.cols.append(self.colors[k])
            self.explore_tree(id,k) # start with the identity matrix

    def explore_tree(self, gen, k):
        for i in range(0,3):
            index = (k + 3 + i) % 4
            local_gen = np.dot(gen, self.gens[index])
            termination = True

            for j, circle in enumerate(self.circles):
                if j != index:
                    new_circle = moebius_on_circle(local_gen, circle)
                    self.circs.append(new_circle)
                    self.cols.append(self.colors[j])
                    if new_circle.r < self.epsilon:
                        self.points.append(
                            moebius_on_point(local_gen, self.fixed_points[index]))
                    else:
                        termination = False
            if not termination:
                self.explore_tree(local_gen, index)

    def search(self):
        if self.level_max == -1:
            # nothing to do, no new fixedpoints generated
            self.circs = self.circles
            self.points = self.fixed_points
        else:
            for i in range(4):
                # do the first step into the tree
                # self.level = 0
                # self.word = [self.gens[i]]
                # self.tags = [i]
                # # self.output()
                self.level = -1
                first = True
                while first or self.level >= 0:
                    if first:
                        first = False
                        nextTurn = i
                    while self.go_forward(nextTurn):
                        # go deep down the tree until the max level is reached or the circle is smaller than the resolution
                        nextTurn = None  # step down the tree with autopilot, the generators are cycled down cyclically.
                        # When the node was entered with 'a', the order of turns is 'B', 'a', 'b'.
                        # When the node was entered with 'b', the order of turns is 'a', 'b', 'A'.
                    nextTurn = self.go_backward_until_turn_available()  # step down the tree after turn

    def go_forward(self, gen=None):
        self.level += 1
        if gen is None:
            self.tags.append((self.tags[-1] + 3) % 4)  # automatic forward
        else:
            self.tags.append(gen)  # directed forward after turn
        if len(self.word) > 0:
            self.word.append(np.dot(self.word[-1], self.gens[self.tags[-1]]))
        else:
            self.word.append(self.gens[self.tags[-1]])
        self.output()
        # print("word: ", self.word, " at level ", self.level)
        return not self.branch_termination()

    def branch_termination(self):
        new_circle = moebius_on_circle(self.word[self.level], self.circles[
            self.tags[self.level]])  # the value of the level fits the position of the word and tag in the array
        if self.level == self.level_max or new_circle.r < self.epsilon:
            self.points.append(moebius_on_point(self.word[self.level], self.fixed_points[self.tags[self.level]]))
            self.circs.append(new_circle)
            # print("terminated: ", new_circle.r, self.level)
            return True
        else:
            return False

    def go_backward_until_turn_available(self):
        while self.level >= 0:
            self.level -= 1
            if (self.tags[self.level + 1] + 1) % 4 != (self.tags[self.level] + 2) % 4:
                nextTurn = (self.tags[self.level + 1] + 1) % 4
                self.tags.pop()  # level is finished remove
                self.word.pop()
                return nextTurn
            else:
                self.tags.pop()  # level is finished remove
                self.word.pop()

    def output(self):
        print(underscore_strings(self.level), self.labels[self.tags[-1]])
