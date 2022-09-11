import numpy as np


class DepthFirstSearch:
    def __init__(self, theta=np.pi / 4, level_max=4):
        self.level_max=level_max
        self.level = 0
        self.gens = []
        self.inv = []
        self.word=[]
        self.tags = []
        self.num = []
        self.fixed_points=[]
        self.a = None
        self.A = None
        self.b = None
        self.B = None

        self.initialize(theta=theta)

    def initialize(self,theta=np.pi/4):
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

    def search(self):
        # start with stepping down the tree in direction a
        self.level =1
        self.tags.append(0+2) # plus 2 makes a->A (going into the node with a corresponds to going out the node with A, start cycle of turns with B(3), then a(0), then b(1). Stop at A(2)
        self.word.append(self.gens[0])
        while self.level>0 or self.tags[0]==2:
            while self.go_forward():
                pass
            self.go_backward()

    def go_forward(self):
        self.level+=1
        self.tags.append((self.tags[-1]+1)%4)
        self.word.append(np.dot(self.word[-1],self.gens(self.tags[-1])))
        return True
        return False