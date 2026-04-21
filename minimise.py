"""
CATAM 7.3 — Minimisation Methods.

three descent algorithms (Steepest Descents, Conjugate
Gradients, DFP) applied to the three test functions in the project.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# 1. Test functions and gradients
# ---------------------------------------------------------------------------


# Equation (4) — the "bedpan" function of two variables.
def f_bedpan(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return x1 + x2 + x1**2 / 4.0 - x2**2 + (x2**2 - x1 / 2.0) ** 2


def grad_bedpan(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of f_bedpan.  Derive this yourself and fill in."""
    raise NotImplementedError("derive df/dx and df/dy of equation (4) by hand")


# Equation (5) — Rosenbrock-like two-variable function.
def f_rosen(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return (1.0 - x1) ** 2 + 80.0 * (x2 - x1**2) ** 2


def grad_rosen(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of f_rosen.  Derive this yourself and fill in."""
    raise NotImplementedError("derive df/dx and df/dy of equation (5) by hand")


# Equation (6) — three-variable quadratic used to exhibit DFP properties.
def f_quad3(x: np.ndarray) -> float:
    x1, x2, x3 = x[0], x[1], x[2]
    return 0.4 * x1**2 + 0.2 * x2**2 + x3**2 + x1 * x3


def grad_quad3(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of f_quad3.  Derive this yourself and fill in."""
    raise NotImplementedError("derive the three partial derivatives of equation (6)")


# Inverse Hessian of f_quad3 (equation 10) — target for the DFP H matrix in Q6.
HESS_QUAD3_INV = np.array(
    [
        [10.0 / 3.0, 0.0, -5.0 / 3.0],
        [0.0, 2.5, 0.0],
        [-5.0 / 3.0, 0.0, 4.0 / 3.0],
    ]
)


# ---------------------------------------------------------------------------
# 2. Gradient sanity check
# ---------------------------------------------------------------------------


def numerical_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Central-difference gradient.  For checking hand-derived gradients."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2.0 * eps)
    return g


# ---------------------------------------------------------------------------
# 3. Line-search helpers for choosing lambda*
# ---------------------------------------------------------------------------


def plot_phi(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    s: np.ndarray,
    lam_range: tuple[float, float] = (-0.2, 2.0),
    n: int = 400,
    ax=None,
):
    """Plot phi(lambda) = f(x0 + lambda*s).  Useful for picking lambda* by eye."""
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
) -> float:
    """Automated lambda* via scipy's bounded scalar minimiser."""
    res = minimize_scalar(lambda lam: f(x0 + lam * s), bounds=bracket, method="bounded")
    return float(res.x)


def ask_lambda(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    s: np.ndarray,
    lam_range: tuple[float, float] = (-0.2, 2.0),
) -> float:
    """Plot phi(lambda) then prompt the user for lambda*.  Used by "manual" mode."""
    plot_phi(f, x0, s, lam_range=lam_range)
    plt.show()
    raw = input(f"λ* (2 sig figs is fine; default = auto): ").strip()
    if not raw:
        return line_search_auto(
            f, x0, s, bracket=(max(lam_range[0], 0.0), lam_range[1])
        )
    return float(raw)


# ---------------------------------------------------------------------------
# 4. Search-direction rules
# ---------------------------------------------------------------------------


class SteepestDescent:
    """s = -g at every step."""

    name = "Steepest Descents"

    def start(self, x0: np.ndarray) -> None:
        pass

    def direction(self, k: int, g: np.ndarray) -> np.ndarray:
        return -g

    def update(
        self,
        k: int,
        x_new: np.ndarray,
        g_new: np.ndarray,
        s: np.ndarray,
        lam: float,
        g_old: np.ndarray,
    ) -> None:
        pass

    def snapshot(self) -> dict:
        return {}


class ConjugateGradient:
    """
    Conjugate gradients with restart every N steps (N = dim(x)).

    k = 0 (and every Nth thereafter): s = -g.
    Otherwise:                        s = -g + beta * s_prev,
                                      beta = (g . g) / (g_prev . g_prev).   (eqn 7)
    """

    name = "Conjugate Gradients"

    def __init__(self, restart_every: int | None = None):
        # None -> use dim(x0) when start() is called.
        self._restart_every = restart_every
        self._n = 0
        self._s_prev: np.ndarray | None = None
        self._g_prev: np.ndarray | None = None

    def start(self, x0: np.ndarray) -> None:
        if self._restart_every is None:
            self._restart_every = int(x0.size)
        self._n = self._restart_every
        self._s_prev = None
        self._g_prev = None

    def direction(self, k: int, g: np.ndarray) -> np.ndarray:
        if k % self._n == 0 or self._s_prev is None or self._g_prev is None:
            return -g
        beta = float(np.dot(g, g) / np.dot(self._g_prev, self._g_prev))
        return -g + beta * self._s_prev

    def update(
        self,
        k: int,
        x_new: np.ndarray,
        g_new: np.ndarray,
        s: np.ndarray,
        lam: float,
        g_old: np.ndarray,
    ) -> None:
        self._s_prev = s
        self._g_prev = g_old

    def snapshot(self) -> dict:
        return {}


class DFP:
    """
    Davidon-Fletcher-Powell variable-metric method.

    H starts as the identity.  s = -H g.  After each step:
        H* = H - (H p pᵀ H) / (pᵀ H p) + (q qᵀ) / (pᵀ q)       (eqn 8)
    with p = g_new - g_old, q = lambda* s = x_new - x_old.       (eqn 9)
    """

    name = "DFP"

    def __init__(self):
        self.H: np.ndarray | None = None

    def start(self, x0: np.ndarray) -> None:
        self.H = np.eye(x0.size)

    def direction(self, k: int, g: np.ndarray) -> np.ndarray:
        assert self.H is not None
        return -self.H @ g

    def update(
        self,
        k: int,
        x_new: np.ndarray,
        g_new: np.ndarray,
        s: np.ndarray,
        lam: float,
        g_old: np.ndarray,
    ) -> None:
        assert self.H is not None
        p = g_new - g_old
        q = lam * s
        Hp = self.H @ p
        pHp = float(p @ Hp)
        pq = float(p @ q)
        if abs(pHp) < 1e-14 or abs(pq) < 1e-14:
            # Degenerate update; leave H unchanged rather than producing NaNs.
            return
        self.H = self.H - np.outer(Hp, Hp) / pHp + np.outer(q, q) / pq

    def snapshot(self) -> dict:
        return {"H": None if self.H is None else self.H.copy()}


# ---------------------------------------------------------------------------
# 5. Core iterator
# ---------------------------------------------------------------------------


@dataclass
class Step:
    k: int  # iteration index (0 = starting point)
    x: np.ndarray  # point after this step (or x0 at k=0)
    f: float  # f(x)
    g: np.ndarray  # grad f at x
    s: np.ndarray | None  # search direction that led to this x (None at k=0)
    lam: float | None  # lambda* used for this step    (None at k=0)
    df: float | None  # f_k - f_{k-1}                 (None at k=0)
    H: np.ndarray | None  # DFP: H used to pick s at step k (None otherwise)


LamSource = (
    str
    | Sequence[float]
    | Callable[[Callable[[np.ndarray], float], np.ndarray, np.ndarray], float]
)


def _resolve_lambda(
    lam_source: LamSource,
    k: int,
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    s: np.ndarray,
) -> float:
    if isinstance(lam_source, str):
        if lam_source == "manual":
            return ask_lambda(f, x, s)
        if lam_source == "auto":
            return line_search_auto(f, x, s)
        raise ValueError(f"unknown lam_source string: {lam_source!r}")
    if callable(lam_source):
        return float(lam_source(f, x, s))
    # Assume a sequence of presets, one per step (k starts at 1 for the step
    # that produces x_k from x_{k-1}).
    return float(lam_source[k - 1])


def minimise(
    algorithm,
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_iter: int,
    lam_source: LamSource,
    verbose: bool = True,
) -> list[Step]:
    """
    Run `n_iter` steps of `algorithm` on `f`.

    Returns a history of `Step` records (including the starting point as k=0).
    """
    x = np.asarray(x0, dtype=float).copy()
    algorithm.start(x)

    g = grad(x)
    snap = algorithm.snapshot()
    history: list[Step] = [
        Step(
            k=0,
            x=x.copy(),
            f=float(f(x)),
            g=g.copy(),
            s=None,
            lam=None,
            df=None,
            H=snap.get("H"),
        )
    ]
    if verbose:
        print(f"k=0  x={x}  f={history[0].f:.6g}  |g|={np.linalg.norm(g):.3g}")

    for k in range(1, n_iter + 1):
        s = algorithm.direction(k - 1, g)
        # Sanity: s should be a descent direction.
        gs = float(np.dot(g, s))
        if gs >= 0.0 and verbose:
            print(f"  [warn] g·s = {gs:.3g} >= 0 at step {k}; not a descent direction")

        lam = _resolve_lambda(lam_source, k, f, x, s)

        x_new = x + lam * s
        g_new = grad(x_new)
        f_new = float(f(x_new))
        df = f_new - history[-1].f

        algorithm.update(k, x_new, g_new, s, lam, g_old=g)

        snap = algorithm.snapshot()
        history.append(
            Step(
                k=k,
                x=x_new.copy(),
                f=f_new,
                g=g_new.copy(),
                s=s.copy(),
                lam=lam,
                df=df,
                H=snap.get("H"),
            )
        )

        if verbose:
            print(
                f"k={k}  λ*={lam:.4g}  x={x_new}  f={f_new:.6g}  Δf={df:+.3g}"
                f"  |g|={np.linalg.norm(g_new):.3g}"
            )

        x = x_new
        g = g_new

    return history


# ---------------------------------------------------------------------------
# 6. Output: tables and plots
# ---------------------------------------------------------------------------


def print_table(history: list[Step], show_H: bool = False) -> None:
    """Tabular summary of an iteration history, suitable for pasting into a writeup."""
    dim = history[0].x.size
    x_hdrs = "  ".join(f"x{i + 1:<10}" for i in range(dim))
    print(f"{'k':>2}  {x_hdrs}  {'f(x)':>12}  {'Δf':>11}  {'λ*':>9}  {'|g|':>9}")
    for st in history:
        xs = "  ".join(f"{v:>+10.6f}" for v in st.x)
        df = f"{st.df:+11.3e}" if st.df is not None else " " * 11
        lam = f"{st.lam:9.4g}" if st.lam is not None else " " * 9
        gn = np.linalg.norm(st.g)
        print(f"{st.k:>2}  {xs}  {st.f:>12.6g}  {df}  {lam}  {gn:9.3g}")
    if show_H:
        for st in history:
            if st.H is not None:
                print(f"\nH at k={st.k}:\n{st.H}")


def plot_trajectory(
    f: Callable[[np.ndarray], float],
    history: list[Step],
    xlim: tuple[float, float] = (-1.5, 1.5),
    ylim: tuple[float, float] = (-1.5, 1.5),
    levels: int | Sequence[float] = 30,
    ax=None,
    label: str | None = None,
):
    """Contour of a 2-variable f with iteration points joined by line segments."""
    assert history[0].x.size == 2, "plot_trajectory is for 2-variable functions only"
    xs = np.linspace(xlim[0], xlim[1], 200)
    ys = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda a, b: f(np.array([a, b])))(X, Y)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.contour(X, Y, Z, levels=levels, linewidths=0.5, colors="0.7")
    path = np.array([st.x for st in history])
    ax.plot(path[:, 0], path[:, 1], "-o", markersize=4, label=label)
    for st in history:
        ax.annotate(
            str(st.k), xy=st.x, xytext=(4, 4), textcoords="offset points", fontsize=8
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    if label is not None:
        ax.legend()
    return ax


def plot_surface(
    f: Callable[[np.ndarray], float],
    xlim: tuple[float, float] = (-1.5, 1.5),
    ylim: tuple[float, float] = (-1.5, 1.5),
    n: int = 80,
):
    """3D surface plot for a 2-variable f (used in Q1)."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda a, b: f(np.array([a, b])))(X, Y)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    return ax
