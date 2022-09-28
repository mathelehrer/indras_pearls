import numpy as np
from matplotlib import pyplot as plt

from plot.plotter import Plotter
from utils.circle import Circle
from utils.mymath import moebius_on_circle, moebius_on_point, flatten, cx_sqrt


class ThetaFamily:
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


def nice(array):
    return flatten(np.round(array, 1))


class KissingSchottky:
    def __init__(self, y=1, k=1):
        self.y = y
        self.k = k

    def get_fixed_points(self):
        return [1j * self.k, 1, -1j * self.k, -1]

    def get_generators(self):
        gens = []

        i = 1j
        y = self.y
        k = self.k
        x = np.sqrt(1 + y * y)
        v = 2 / y / (k + 1 / k)
        u = np.sqrt(1 + v * v)

        gens.append(np.array([[u, i * k * v], [-i * v / k, u]]))
        a = gens[-1]
        gens.append(np.array([[x, y], [y, x]]))
        b = gens[-1]
        gens.append(np.linalg.inv(a))
        A = gens[-1]
        gens.append(np.linalg.inv(b))
        B = gens[-1]

        return [a, b, A, B]

    def get_circles(self):
        i = 1j
        y = self.y
        k = self.k
        x = np.sqrt(1 + y * y)
        v = 2 / y / (k + 1 / k)
        u = np.sqrt(1 + v * v)

        return [
            Circle(i * k * u / v, k / v),
            Circle(x / y, 1 / y),
            Circle(-i * k * u / v, k / v),
            Circle(-x / y, 1 / y)
        ]


class ApollonianGasket:
    def __init__(self):
        pass

    def get_fixed_points(self):
        i = 1j
        return [0,-i,0,-i]

    def get_generators(self):
        gens = []
        i = 1j
        gens.append(np.array([[1, 0], [-2*i, 1]]))
        a = gens[-1]
        gens.append(np.array([[1-i,1], [1, 1+i]]))
        b = gens[-1]
        gens.append(np.linalg.inv(a))
        A = gens[-1]
        gens.append(np.linalg.inv(b))
        B = gens[-1]

        return [a, b, A, B]

    def get_circles(self):
        i = 1j
        return [Circle(np.inf,np.inf,-1, 1),Circle(1-i,1),Circle(-i/4,1/4),Circle(-1-i,1)]


class DepthFirstSearchOriginal:
    def __init__(self, family, eps=0.1, **kwargs):
        self.lev = None
        self.old_point = None

        self.gens = family(**kwargs).get_generators()
        self.fixed_points = family(**kwargs).get_fixed_points()
        self.circles = family(**kwargs).get_circles()

        self.begin_pt_generators = self.create_begin_point_generators()
        self.end_pt_generators = self.create_end_point_generators()

        self.inv = [2, 3, 0, 1]

        self.word = None
        self.tags = None
        self.epsilon = eps

        self.setup_start()
        self.points = []  # field to store the limit points
        self.circs = []  # field to store all circles

        self.breaking_length = None

    def setup_start(self,
                    begin_tag=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                    end_tag=[1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2]):
        '''
        This function can be called before run() to setup the begin_tag and end_tag
        :param begin_tag: [0,0,0,1,2]
        :param end_tag: [0,3,3,0,1]
        :return:
        '''

        self.tags = []
        self.word = []
        for t in begin_tag:
            self.tags.append(t)
            if len(self.word) == 0:
                self.word.append(self.gens[t])
            else:
                self.word.append(np.dot(self.word[-1], self.gens[t]))
        self.lev = len(begin_tag) - 1
        self.old_point = moebius_on_point(self.word[-1], self.begin_pt_generators[self.tags[-1]])

        self.end_tag = end_tag

    def run(self, close_curve=False):
        while self.lev != -1 or not self.check_end():
            while not self.branch_termination() and not self.check_end():
                self.go_forward()
            while True:  # do ... while loop
                self.go_backward()
                if self.lev == -1 or self.available_turn():
                    break
            self.turn_and_go_forward()
        if close_curve:
            self.points.append(self.points[0])

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
        if (self.tags[self.lev + 1] + 3) % 4 == (self.tags[self.lev] + 2) % 4:
            self.tags.pop()
            return False
        else:
            return True

    def turn_and_go_forward(self):
        if (self.lev != -1 or self.tags[
            0] != 1) and not self.check_end():  # stop turning when the zeroth level has cycled all four values
            self.tags[self.lev + 1] = (self.tags[self.lev + 1] - 1) % 4  # go to the next left branch
            if self.lev == -1:
                self.word[0] = self.gens[self.tags[0]]
            else:
                self.word[self.lev + 1] = np.dot(self.word[self.lev], self.gens[self.tags[self.lev + 1]])
            self.lev += 1

    def check_end(self, test=None):
        '''
        if self.tags is shorter than end_tag and starts with the same sequence as end_tag,
        it is returned false until it reached a shorter version that deviates

        :param test:
        :return:
        '''
        if test is not None:
            self.tags = test
        for a, b in zip(self.tags, self.end_tag):
            if a != b:
                if self.breaking_length is None:
                    return False
                elif len(self.tags) > self.breaking_length:
                    return False

        # check the cases with equal start
        # easy case, both tags agree
        if len(self.tags) == len(self.end_tag):
            return True
        # self.tag is longer, but starts with the self.end_tag
        if len(self.tags) > len(self.end_tag):
            return True
        # self.tag is shorter than self.end_tag but they have a common start
        # breaking_length captures the length of this common start
        else:
            # self.tags equals to the beginning of self.end_tag
            if self.breaking_length is None:
                self.breaking_length = len(self.tags)
                return False
            # self.tags has become shorter than the previous agreement
            if self.breaking_length > len(self.tags):
                return True
            else:
                # the common start has grown longer
                self.breaking_length < len(self.tags)
                self.breaking_length = len(self.tags)
                return False

    def branch_termination_1(self):
        new_circ = moebius_on_circle(self.word[self.lev - 1], self.circles[
            self.tags[self.lev]])  # this is rather smart, the last tag is used to cycle through the relevant discs
        self.circs.append(new_circ)
        if new_circ.r < self.epsilon:
            self.points.append(moebius_on_point(self.word[self.lev - 1], self.fixed_points[self.tags[self.lev]]))
            # print("point added for tag: ", self.tags)
            return True
        else:
            return False

    def branch_termination_2(self):
        new_circ = moebius_on_circle(self.word[self.lev - 1], self.circles[
            self.tags[self.lev]])  # this is rather smart, the last tag is used to cycle through the relevant discs
        self.circs.append(new_circ)
        if new_circ.r < self.epsilon:
            self.points.append(moebius_on_point(self.word[self.lev], self.fixed_points[self.tags[self.lev]]))
            # print("point added for tag: ", self.tags)
            return True
        else:
            return False

    def branch_termination_3(self):
        """
        this branch_termination terminates, when there is not enough change between different levels of the tree
        :return:
        """
        new_point = moebius_on_point(self.word[self.lev], self.fixed_points[self.tags[self.lev]])

        if np.abs(new_point - self.old_point) < self.epsilon:
            self.points.append(new_point)
            self.old_point = new_point
            return True
        else:
            self.old_point = new_point
            return False

    def branch_termination(self):
        """
        now only images of commutator fixed points are plotted.
        They can be approached symmetrically from either side.
        This creates the most symmetric plots of the limit set

        :return: True, when the required accuracy is reached.
        """
        new_point = moebius_on_point(self.word[self.lev], self.end_pt_generators[self.tags[self.lev]])
        # print(self.tags2string(), new_point, self.old_point)
        # if self.tags == [1, 0, 3, 2]:
        #     i = 0
        #     i = i + 1
        if np.abs(new_point - self.old_point) < self.epsilon:
            self.points.append(new_point)
            self.old_point = new_point
            return True
        else:
            return False

    def fixed_point_of_commutator(self, a, b, c, d):
        m = np.dot(np.dot(np.dot(self.gens[a], self.gens[b]), self.gens[c]), self.gens[d])
        return self.fixed_point_of(m)

    def fixed_point_of(self, m):
        '''
        return the attractive fixed point of a matrix
        :param m:
        :return:
        '''
        a = m[0][0]
        b = m[0][1]
        c = m[1][0]
        d = m[1][1]
        z1 = 1 / 2 / c * ((a - d) + np.sqrt((a - d) ** 2 + 4 * c * b))
        z2 = 1 / 2 / c * ((a - d) - cx_sqrt((a - d) ** 2 + 4 * c * b))

        # check for attractiveness
        z = z1 * 1.1
        z_img = moebius_on_point(m, moebius_on_point(m, moebius_on_point(m, moebius_on_point(m, z))))
        if np.abs(z - z_img) < 0.1:
            return z1
        return z2

    def create_begin_point_generators(self):
        end_points = [self.fixed_point_of_commutator(1, 2, 3, 0),
                      self.fixed_point_of_commutator(2, 3, 0, 1),
                      self.fixed_point_of_commutator(3, 0, 1, 2),
                      self.fixed_point_of_commutator(0, 1, 2, 3)]
        return end_points

    def create_end_point_generators(self):
        begin_points = [self.fixed_point_of_commutator(3, 2, 1, 0),
                        self.fixed_point_of_commutator(0, 3, 2, 1),
                        self.fixed_point_of_commutator(1, 0, 3, 2),
                        self.fixed_point_of_commutator(2, 1, 0, 3)]
        return begin_points

    def tags2string(self):
        out = ''
        for t in self.tags:
            if t == 0:
                out += 'a'
            elif t == 1:
                out += 'b'
            elif t == 2:
                out += 'A'
            else:
                out += 'B'
        return out


if __name__ == '__main__':
    # dfs = DepthFirstSearchOriginal(ThetaFamily, theta=np.pi / 4,eps=0.000001)

    dfs = DepthFirstSearchOriginal(KissingSchottky, y=0.5, k=0.1, eps=0.0001)
    dfs.setup_start(end_tag=[1,0,3,2])
    dfs.run()
    # plt.scatter(np.real(dfs.points), np.imag(dfs.points), s=0.5, marker='.')
    plt.clf()
    plt.plot(np.real(dfs.points), np.imag(dfs.points))
    plt.gca().set_aspect('equal')
    plt.show()

    # for i in range(100,200):
    #     k=1-(i-100)/100
    #     dfs = DepthFirstSearchOriginal(KissingShottky, y=0.5,k=k,eps=0.00000001)
    #     dfs.run(8)
    #     # plt.scatter(np.real(dfs.points), np.imag(dfs.points), s=0.5, marker='.')
    #     plt.clf()
    #     plt.plot(np.real(dfs.points),np.imag(dfs.points))
    #     plt.gca().set_aspect('equal')
    #     plt.savefig("/home/jmartin/figure"+str(i)+".png")
    #     #plt.show()
    #     # Plotter.plot(*dfs.circs) # only used this for level_max<6

    # for i in range(100,200):
    #     y=10*(1-(i-100)/100)
    #     dfs = DepthFirstSearchOriginal(KissingShottky, y=y,k=0.1,eps=0.00000001)
    #     dfs.run(8)
    #     # plt.scatter(np.real(dfs.points), np.imag(dfs.points), s=0.5, marker='.')
    #     plt.clf()
    #     plt.plot(np.real(dfs.points),np.imag(dfs.points))
    #     plt.gca().set_aspect('equal')
    #     plt.savefig("/home/jmartin/figure2"+str(i)+".png")
    #     #plt.show()
    # Plotter.plot(*dfs.circs) # only used this for level_max<6
