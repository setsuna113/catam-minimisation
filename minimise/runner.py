"""Core iteration driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from line_search import ask_lambda, line_search_auto


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
