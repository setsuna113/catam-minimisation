"""
Program that defines the minimisation program
"""

import numpy as np


def gradient(f: np.function, x: np, array) -> np.array:
    "This should somehow calculate the gradient of f(x)"
    return g


def step(f: np.function, x0: np.array, s: np.array) -> np.float64:
    g = gradient(f, x0)
    assert s * g < 0
    count: np.float64 = 0.0

    # Suppose to find (non-negative?) _ to minimise f(x + _ * s)
    for _ in range(0, 100, 0.01):
        if f(x0 + _ * s) < f(x0 + count * s):
            count = _
    return count


def recurse(time: int, f: np.function, x0: np.array, s: np.array):
    k = step(f, x0, s)
    X = [x0]
    for i in range(time):
        X.append[x0 + k]
        k = step(f.X[-1], s)
