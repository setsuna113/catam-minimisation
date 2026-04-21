"""Test functions, analytic gradients, and numerical gradient checker."""

from __future__ import annotations

from typing import Callable

import numpy as np


def f_bedpan(x: np.ndarray) -> float:
    """bedpan function - eqn 4"""
    x1, x2 = x[0], x[1]
    return x1 + x2 + x1**2 / 4.0 - x2**2 + (x2**2 - x1 / 2.0) ** 2


def grad_bedpan(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of the bedpan function"""
    x1, x2 = x[0], x[1]
    df_dx1 = 1.0 + x1 / 2.0 - (x2**2 - x1 / 2.0)
    df_dx2 = 1.0 - 2.0 * x2 + 4.0 * x2 * (x2**2 - x1 / 2.0)
    return np.array([df_dx1, df_dx2])


def f_rosen(x: np.ndarray) -> float:
    """Rosenbrock function - eqn 5"""
    x1, x2 = x[0], x[1]
    return (1.0 - x1) ** 2 + 80.0 * (x2 - x1**2) ** 2


def grad_rosen(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of f_rosen"""
    x1, x2 = x[0], x[1]
    df_dx1 = -2.0 * (1.0 - x1) - 320.0 * x1 * (x2 - x1**2)
    df_dx2 = 160.0 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])


def f_quad3(x: np.ndarray) -> float:
    """Quadratic function - eqn 6"""
    x1, x2, x3 = x[0], x[1], x[2]
    return 0.4 * x1**2 + 0.2 * x2**2 + x3**2 + x1 * x3


def grad_quad3(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of f_quad3"""
    x1, x2, x3 = x[0], x[1], x[2]
    df_dx1 = 0.8 * x1 + x3
    df_dx2 = 0.4 * x2
    df_dx3 = 2.0 * x3 + x1
    return np.array([df_dx1, df_dx2, df_dx3])


# Inverse Hessian of f_quad3 - eqn 10
HESS_QUAD3_INV = np.array(
    [
        [10.0 / 3.0, 0.0, -5.0 / 3.0],
        [0.0, 2.5, 0.0],
        [-5.0 / 3.0, 0.0, 4.0 / 3.0],
    ]
)


def numerical_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Central-difference gradient for checking hand-derived gradients."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2.0 * eps)
    return g
