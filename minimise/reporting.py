"""Tables and contour/surface plots for iteration histories."""

from __future__ import annotations

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .runner import Step


def print_table(history: list[Step], show_H: bool = False) -> None:
    """Tabular summary of an iteration history"""
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
    levels: int | Sequence[float] | np.ndarray = 30,
    ax=None,
    label: str | None = None,
):
    """Contour of a 2-variable function"""
    if history:
        assert history[0].x.size == 2, "plot_trajectory is for 2-variable functions only"
    
    xs = np.linspace(xlim[0], xlim[1], 200)
    ys = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda a, b: f(np.array([a, b])))(X, Y)
    
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.contour(X, Y, Z, levels=levels, linewidths=0.5, colors="0.7")
        
    if history:
        path = np.array([st.x for st in history])
        ax.plot(path[:, 0], path[:, 1], "-o", markersize=4, label=label)
        for st in history:
            ax.annotate(
                str(st.k), xy=(float(st.x[0]), float(st.x[1])), xytext=(4, 4), textcoords="offset points", fontsize=8
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
    """3D surface plot for a 2-variable function"""
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


if __name__ == "__main__":
    from .functions import f_bedpan, f_rosen

    print("PLotting function 4: bedpan...")
    plot_surface(f_bedpan, xlim=(-2, 2.5), ylim=(-2, 1.5))
    plot_trajectory(f_bedpan, history=[], xlim=(-2, 2.5), ylim=(-2, 1.5), levels=40)

    print("Plottingh function 5: Rosenbrock...")
    plot_surface(f_rosen, xlim=(-1.5, 2), ylim=(-1, 3))
    
    plot_trajectory(
        f_rosen, history=[], xlim=(-1.5, 2), ylim=(-1, 3), levels=40
    )

    plt.show()
