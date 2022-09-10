import numpy as np
from utils.circle import Circle


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
    z = circle.c - circle.r ** 2 / np.conj(m[1][1] / m[1][0] + circle.c)
    cen = moebius_on_point(m, z)
    rad = np.abs(cen - moebius_on_point(m, circle.c + circle.r))
    return Circle(cen, rad)
