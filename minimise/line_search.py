"""Line-search helpers for choosing lambda*"""

from __future__ import annotations
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np


def plot_phi(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    s: np.ndarray,
    lam_range: tuple[float, float] = (-0.2, 2.0),
    n: int = 400,
    ax=None,
):
    """phi(lambda) = f(x0 + lambda*s)"""
    lams = np.linspace(lam_range[0], lam_range[1], n)
    vals = np.array([f(x0 + lam * s) for lam in lams])
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(lams, vals)
    ax.axvline(0.0, color="k", linewidth=0.5)
    kmin = int(np.argmin(vals))
    ax.plot(lams[kmin], vals[kmin], "ro", label=f"min @ λ≈{lams[kmin]:.3g}")
    ax.set_xlabel("λ")
    ax.set_ylabel("φ(λ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def line_search_auto(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    s: np.ndarray,
    bracket: tuple[float, float] = (0.0, 2.0),
    n_points: int = 201,
) -> float:
    """Automated lambda* via grid search (precision to ~2 decimal places)"""
    lams = np.linspace(bracket[0], bracket[1], n_points)
    vals = np.array([f(x0 + lam * s) for lam in lams])
    kmin = int(np.argmin(vals))
    return float(lams[kmin])


def ask_lambda(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    s: np.ndarray,
    lam_range: tuple[float, float] = (-0.2, 2.0),
) -> float:
    """Plot phi(lambda) then prompt the user for lambda* (in manual mode)"""
    plot_phi(f, x0, s, lam_range=lam_range)
    plt.show()
    raw = input(f"λ* (2 sig figs is fine; default = auto): ").strip()
    if not raw:
        return line_search_auto(
            f, x0, s, bracket=(max(lam_range[0], 0.0), lam_range[1])
        )
    return float(raw)
