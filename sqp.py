import numpy as np

EPS = 0.001

def f(x):
    return 7 * (x[0] ** 2) - 14 * x[0] * x[1] + 8 * (x[1] ** 2) - 8 * x[0] + 6 * x[1]


def h(x):
    return 12 * x[0] ** 2 + 16 * x[0] * x[1] + 6 * x[1] ** 2 - 9


def grad_f(x):
    return np.array([14 * x[0] - 14 * x[1] - 8, -14 * x[0] + 16 * x[1] + 6])


def grad_h(x):
    return np.array([24 * x[0] + 16 * x[1], 16 * x[0] + 12 * x[1]])


def hessian_f(x):
    return np.array([[14, -14], [-14, 16]])


def hessian_h(x):
    return np.array([[24, 16], [16, 12]])


def SQP(x0, l0):
    x = x0
    l = l0
    while True:
        g = grad_f(x)
        A = grad_h(x)
        D = hessian_f(x) + l * hessian_h(x)
        r = h(x)
        left = np.array([
            [D[0][0], D[0][1], A[0]],
            [D[1][0], D[1][1], A[1]],
            [A[0],       A[1],    0]
        ])
        right = [-g[0],
                 -g[1],
                 -r]
        solution = np.linalg.inv(left).dot(right)
        p, u = solution[:2], solution[2]
        if np.sqrt(np.sum(p ** 2)) <= EPS:
            return x, l

        x = x + p
        l = u


x0 = np.array([3, 3])
l0 = 3
x, l = SQP(x0, l0)
print("min f", f(x))
print("min x", x)
