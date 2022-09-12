import numpy as np

from utils.circle import Circle
from utils.mymath import moebius_on_circle, moebius_on_point


class DepthFirstSearch:
    def __init__(self, theta=np.pi / 4, level_max=4):
        self.level_max = level_max
        self.labels=['a','b','A','B']
        self.gens = []
        self.circles=[]
        self.inv = []
        self.num = []
        self.fixed_points = [1, -1, 1j, -1j]
        self.a = None
        self.A = None
        self.b = None
        self.B = None
        self.points = []
        self.initialize(theta=theta)

        # do the first step into the tree
        self.level = 0
        self.word = [self.gens[0]]
        self.tags = [0]
        print("tags: ",self.tags)
        print("word: ", self.word, " at level ", self.level)

        self.epsilon = 0.0001

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
        [print(self.labels[i],self.gens[i]) for i in range(0,4)]
        print("inverses: ",self.inv)

        #setup initial circles
        center = 1 / np.cos(theta)
        radius = np.tan(theta)
        self.circles.append(Circle(i * center, radius))
        self.circles.append(Circle(center, radius))
        self.circles.append(Circle(-i * center, radius))
        self.circles.append(Circle(-center, radius))

        print("Created ",len(self.circles)," circles:")
        [print("C[(",c.c,"),",c.r,"]") for c in self.circles]

        print("Fixed points:")
        [print(self.labels[i],": ",self.fixed_points[i]) for i in range(0,4)]
        print("Finished initialization")

    def search(self):
        # start with stepping down the tree in direction a
        # self.level = 1
        # self.tags.append(0 + 3)  # plus 2 makes a->A (going into the node with a corresponds to going out the node with A, start cycle of turns with B(3), then a(0), then b(1). Stop at A(2)
        # print("tags: ",self.tags)
        # print("word: ", self.word, " at level ", self.level)
        # self.word.append(self.gens[0])
        while self.level >= 0 or self.tags[0] != 2:
            while self.go_forward():
                pass
            if self.go_backward_and_find_possible_turn():
                self.turn_and_go_forward()

    def go_forward(self):
        print("entered go_forward:")
        self.level += 1
        self.tags.append((self.tags[-1] + 3) % 4)
        self.word.append(np.dot(self.word[-1], self.gens[self.tags[-1]]))
        print("tag: ",self.tags)
        print("word: ", self.word, " at level ", self.level)
        return not self.branch_termination()

    def branch_termination(self):
        new_circle = moebius_on_circle(self.word[self.level - 1], self.circles[self.tags[self.level-1]])
        if self.level == self.level_max or new_circle.r < self.epsilon:
            self.points.append(moebius_on_point(self.word[self.level], self.fixed_points[self.tags[self.level]]))
            return True
        else:
            return False

    def go_backward_and_find_possible_turn(self):
        while self.level>=0:
            self.level -= 1
            if self.turn_available():
               return True
        return False

    def turn_available(self):
        if (self.tags[self.level+1]+1)%4==(self.tags[self.level]+2)%4:
            return False
        else:
            return True

    def turn_and_go_forward(self):
        print("turn:")
        self.tags[self.level+1]+=1
        self.tags[self.level+1]%=4
        print("tag: ", self.tags)
        print("forward:")
        if self.level==0:
            self.word[0]=self.gens[self.tags[0]]
        else:
            self.word[self.level+1]=np.dot(self.word[self.level],self.gens[self.tags[self.level+1]])
            self.level+=1
        print("word: ",self.word," at level ",self.level)




