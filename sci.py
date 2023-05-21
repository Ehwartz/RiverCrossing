import scipy
import numpy as np
from scipy import optimize


def f(H, L, u, v):
    return (np.sqrt((H * H + L * L) * u * u - H * H * v * v) - L * v) / (u * u - v * v)


def f_total(L2):
    return f(600, L2, 1.5, 2.1) + 2 * f(300, (1000 - L2) / 2, 1.5, 0.875)


def f1():
    return f(1200, 1000, 1.5, 1.2)


if __name__ == '__main__':
    # res_x = optimize.fminbound(f, 589.8776, 1000)
    # print(res_x)

    res = optimize.fmin(f_total, 589.8776)
    print('L2 = ', res)
    print('Min Time = ', f_total(res))