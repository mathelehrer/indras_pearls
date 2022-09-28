import math

import numpy as np
from utils.circle import Circle


def flatten(matrix):
    return [v for row in matrix for v in row]


def cx_sqrt(z):
    x = np.real(z)
    y = np.imag(z)
    u2 = 0.5 * (x + np.sqrt(x * x + y * y))

    # u2 is zero when z=-|x|, then the square root is purely imaginary
    if u2 == 0:
        return 1j * np.sqrt(np.abs(x))
    else:
        u = np.sqrt(u2)
        v = y / 2 / u
        return u + 1j * v


def moebius_on_point(m, z):
    """
    :param m: matrix representing a Moebius transformation
    :param z: complex number
    :return:
    """
    if z == np.inf:
        if m[1][0] != 0:
            return m[0][0] / m[1][0]
        else:
            return np.inf
    else:
        return (m[0][0] * z + m[0][1]) / (m[1][0] * z + m[1][1])


def moebius_on_circle(m, circle):
    if circle.c==np.inf:
        # convert line to circle
        return circle_from_three_points(
            moebius_on_point(m,circle.p1),
            moebius_on_point(m,circle.p2),
            moebius_on_point(m,np.inf),
        )
    else:
        denominator = np.conj(m[1][1] / m[1][0] + circle.c)
        if np.abs(denominator)!=0:
            z = circle.c - circle.r ** 2 / denominator
            if np.abs(m[1][0] * z + m[1][1]) != 0:
                cen = moebius_on_point(m, z)
                rad = np.abs(cen - moebius_on_point(m, circle.c + circle.r))
                return Circle(cen, rad)
            else:
                return Circle(np.inf, np.inf, p1=moebius_on_point(m, circle.c + circle.r),
                              p2=moebius_on_point(m, circle.c + 1j * circle.r))
        else:
            cen = m[0][0]/m[1][0]
            rad = np.abs(cen - moebius_on_point(m, circle.c + circle.r))
            return Circle(cen,rad)


def circle_from_three_points(x,y,z)->Circle:
    w = (z-x)/(y-x)
    if np.isclose(np.imag(w),0):
        # circle is a line
        return Circle(np.inf,np.inf,x,y)
    else:
        c = (y-x)*(w-np.conj(w)*w)/(2j*w.imag)+x
        r = np.abs(x-c)
        return Circle(c,r)

