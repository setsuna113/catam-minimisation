"""Tables and contour/surface plots for iteration histories."""

from __future__ import annotations

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from minimise.runner import Step


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
    from pathlib import Path

    from minimise.functions import f_bedpan, f_rosen

    xs = np.linspace(-1.5, 1.5, 300)
    ys = np.linspace(-1.5, 1.5, 300)
    X, Y = np.meshgrid(xs, ys)
    Z4 = np.vectorize(lambda a, b: f_bedpan(np.array([a, b])))(X, Y)
    Z5 = np.vectorize(lambda a, b: f_rosen(np.array([a, b])))(X, Y)

    x4_star, y4_star = 2 ** (-2 / 3) - 1, -(2 ** (-1 / 3))
    f4_star = float(f_bedpan(np.array([x4_star, y4_star])))
    x5_star, y5_star, f5_star = 1.0, 1.0, 0.0

    fig = plt.figure(figsize=(11, 10))

    # Top-left: contour of function (4)
    ax_c4 = fig.add_subplot(2, 2, 1)
    cs4 = ax_c4.contour(X, Y, Z4, levels=18, linewidths=0.6, colors="0.35")
    ax_c4.clabel(cs4, inline=True, fontsize=7, fmt="%.2g")
    ax_c4.plot(x4_star, y4_star, marker="*", color="red", markersize=14, linestyle="none")
    ax_c4.set_title("Function (4): contour")
    ax_c4.set_xlabel("x"); ax_c4.set_ylabel("y")
    ax_c4.set_xlim(-1.5, 1.5); ax_c4.set_ylim(-1.5, 1.5)
    ax_c4.set_aspect("equal", adjustable="box")

    # Top-right: contour of function (5)
    ax_c5 = fig.add_subplot(2, 2, 2)
    cs5 = ax_c5.contour(X, Y, Z5, levels=18, linewidths=0.6, colors="0.35")
    ax_c5.clabel(cs5, inline=True, fontsize=7, fmt="%.2g")
    ax_c5.plot(x5_star, y5_star, marker="*", color="red", markersize=14, linestyle="none")
    ax_c5.set_title("Function (5): contour")
    ax_c5.set_xlabel("x"); ax_c5.set_ylabel("y")
    ax_c5.set_xlim(-1.5, 1.5); ax_c5.set_ylim(-1.5, 1.5)
    ax_c5.set_aspect("equal", adjustable="box")

    # Bottom-left: surface of function (4)
    ax_s4 = fig.add_subplot(2, 2, 3, projection="3d")
    ax_s4.plot_surface(X, Y, Z4, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax_s4.scatter([x4_star], [y4_star], [f4_star], color="red", s=60, marker="*")  # type: ignore[arg-type]
    ax_s4.set_title("Function (4): surface")
    ax_s4.set_xlabel("x"); ax_s4.set_ylabel("y"); ax_s4.set_zlabel("f")

    # Bottom-right: surface of function (5); clip the tall walls so the valley is visible
    Z5_clip = np.minimum(Z5, 40.0)
    ax_s5 = fig.add_subplot(2, 2, 4, projection="3d")
    ax_s5.plot_surface(X, Y, Z5_clip, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax_s5.scatter([x5_star], [y5_star], [f5_star], color="red", s=60, marker="*")  # type: ignore[arg-type]
    ax_s5.set_title("Function (5): surface (clipped at f=40)")
    ax_s5.set_xlabel("x"); ax_s5.set_ylabel("y"); ax_s5.set_zlabel("f")

    fig.suptitle(r"Contour and surface views on $[-1.5,1.5]^2$; red $\star$ = analytic minimum")
    fig.tight_layout()

    out = Path("figures")
    out.mkdir(exist_ok=True)
    fig.savefig(out / "q1_contours.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q1_contours.pdf'}")
