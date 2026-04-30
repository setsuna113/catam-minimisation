"""Tables and contour/surface plots for iteration histories."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from algorithms import ConjugateGradient, DFP, SteepestDescent
from functions import (
    HESS_QUAD3_INV,
    f_bedpan,
    f_quad3,
    f_rosen,
    grad_bedpan,
    grad_quad3,
    grad_rosen,
    hess_bedpan,
    hess_rosen,
)
from line_search import line_search_auto
from runner import Step, minimise


def print_table(
    history: list[Step],
    show_H: bool = False,
    step_types: list[str] | None = None,
) -> None:
    """Tabular summary of an iteration history"""
    dim = history[0].x.size
    x_hdrs = "  ".join(f"x{i + 1:<10}" for i in range(dim))
    step_hdr = f"  {'step':>6}" if step_types is not None else ""
    print(f"{'k':>2}  {x_hdrs}  {'f(x)':>12}  {'Δf':>11}  {'λ*':>9}  {'|g|':>9}{step_hdr}")
    for st in history:
        xs = "  ".join(f"{v:>+10.6f}" for v in st.x)
        df = f"{st.df:+11.3e}" if st.df is not None else " " * 11
        lam = f"{st.lam:9.4g}" if st.lam is not None else " " * 9
        gn = np.linalg.norm(st.g)
        extra = f"  {step_types[st.k]:>6}" if step_types is not None else ""
        print(f"{st.k:>2}  {xs}  {st.f:>12.6g}  {df}  {lam}  {gn:9.3g}{extra}")
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
    annotate: bool = True,
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
        if annotate:
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

    # --- Question 2: Steepest Descents on f_bedpan from (-1.0, -1.3) ---
    print("\n--- Q2: Steepest Descents on f_bedpan from (-1.0, -1.3) ---")
    x0 = np.array([-1.0, -1.3])
    history_q2 = minimise(
        SteepestDescent(), f_bedpan, grad_bedpan, x0, n_iter=10,
        lam_source="auto", verbose=False,
    )
    print_table(history_q2)

    ax_q2 = plot_trajectory(
        f_bedpan, history_q2,
        xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
        label="SD path",
    )
    ax_q2.set_title("Steepest descents on function (4) from $(-1.0, -1.3)$")
    ax_q2.figure.savefig(out / "q2_steepest_function4_path.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q2_steepest_function4_path.pdf'}")

    x_final = history_q2[-1].x
    f_final = history_q2[-1].f
    print(f"final: x = ({x_final[0]:+.6f}, {x_final[1]:+.6f}),  f = {f_final:.6g}")

    # --- Question 3: Steepest Descents on f_rosen from (0.676, 0.443) ---
    print("\n--- Q3: Steepest Descents on f_rosen from (0.676, 0.443) ---")
    x0_r = np.array([0.676, 0.443])
    n_iter_r = 12
    history_q3 = minimise(
        SteepestDescent(), f_rosen, grad_rosen, x0_r, n_iter=n_iter_r,
        lam_source="auto", verbose=False,
    )
    print_table(history_q3)

    xlim_r = (0.4, 1.05)
    ylim_r = (0.3, 1.05)
    levels_r = np.geomspace(1e-3, 5.0, 20)

    ax_q3 = plot_trajectory(
        f_rosen, history_q3,
        xlim=xlim_r, ylim=ylim_r, levels=levels_r,
        label="SD path",
    )
    ax_q3.plot(1.0, 1.0, marker="*", color="red", markersize=12, linestyle="none")
    ax_q3.set_title("Steepest descents on function (5) from $(0.676, 0.443)$")
    ax_q3.figure.savefig(out / "q3_steepest_function5_path.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q3_steepest_function5_path.pdf'}")
    x_final = history_q3[-1].x
    print(f"final: x = ({x_final[0]:+.6f}, {x_final[1]:+.6f}),  f = {history_q3[-1].f:.6g}")

    # --- Question 4: Conjugate Gradients on f_bedpan from (-1.0, -1.3) ---
    print("\n--- Q4: Conjugate Gradients on f_bedpan from (-1.0, -1.3) ---")
    history_q4 = minimise(
        ConjugateGradient(), f_bedpan, grad_bedpan, x0, n_iter=10,
        lam_source="auto", verbose=False,
    )
    # For a 2-D problem with restart_every = N = 2: odd k => SD (or restart), even k => CG.
    step_types_q4 = ["--"] + ["SD" if k % 2 == 1 else "CG" for k in range(1, len(history_q4))]
    print_table(history_q4, step_types=step_types_q4)

    ax_q4 = plot_trajectory(
        f_bedpan, history_q4,
        xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
        label="CG path",
    )
    ax_q4.set_title("Conjugate gradients on function (4) from $(-1.0, -1.3)$")
    ax_q4.figure.savefig(out / "q4_cg_function4_path.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q4_cg_function4_path.pdf'}")
    x_final = history_q4[-1].x
    print(f"final: x = ({x_final[0]:+.6f}, {x_final[1]:+.6f}),  f = {history_q4[-1].f:.6g}")

    # --- Question 5: Conjugate Gradients on f_rosen from (0.676, 0.443) ---
    print("\n--- Q5: Conjugate Gradients on f_rosen from (0.676, 0.443) ---")
    history_q5 = minimise(
        ConjugateGradient(), f_rosen, grad_rosen, x0_r, n_iter=n_iter_r,
        lam_source="auto", verbose=False,
    )
    step_types_q5 = ["--"] + ["SD" if k % 2 == 1 else "CG" for k in range(1, len(history_q5))]
    print_table(history_q5, step_types=step_types_q5)

    ax_q5 = plot_trajectory(
        f_rosen, history_q5,
        xlim=xlim_r, ylim=ylim_r, levels=levels_r,
        label="CG path",
    )
    ax_q5.plot(1.0, 1.0, marker="*", color="red", markersize=12, linestyle="none")
    ax_q5.set_title("Conjugate gradients on function (5) from $(0.676, 0.443)$")
    ax_q5.figure.savefig(out / "q5_cg_function5_path.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q5_cg_function5_path.pdf'}")
    x_final = history_q5[-1].x
    print(f"final: x = ({x_final[0]:+.6f}, {x_final[1]:+.6f}),  f = {history_q5[-1].f:.6g}")

    # --- Sensitivity sub-runs: at each step re-optimise lambda*, then scale by (1+/-rel) ---
    # Online (re-optimise at each step) rather than pre-recorded, because Rosenbrock is so
    # nonlinear that a fixed pre-recorded schedule applied to a drifted path overshoots
    # and blows up. Online perturbation keeps each path self-consistent and directly
    # answers "how much does a small mis-estimate of each lambda* shift the trajectory?".
    def _first_divergence(base: list[Step], other: list[Step], tol: float = 1e-2) -> int | None:
        """Iteration index at which |x_other - x_base| first exceeds tol."""
        for st_b, st_o in zip(base, other):
            if float(np.linalg.norm(st_o.x - st_b.x)) > tol:
                return st_b.k
        return None

    def _make_perturbed_auto(factor: float):
        def _wrapper(f, x, s):
            lam = line_search_auto(f, x, s)
            return max(0.0, lam * factor)
        return _wrapper

    def _run_sensitivity(algo_factory, f, grad, x0, n_iter, rel: float = 0.05):
        """Overlay baseline + per-step-perturbed runs and save.

        Returns the list of histories (baseline, +rel, -rel) and their labels.
        """
        histories: list[list[Step]] = []
        labels = [
            "λ* (baseline)",
            f"λ* × {1.0 + rel:.2f} per step",
            f"λ* × {1.0 - rel:.2f} per step",
        ]
        for factor in (1.0, 1.0 + rel, 1.0 - rel):
            hist = minimise(
                algo_factory(), f, grad, x0, n_iter=n_iter,
                lam_source=_make_perturbed_auto(factor), verbose=False,
            )
            histories.append(hist)
        return histories, labels

    # Q3 sensitivity (SD on f_rosen) — +/-5% per-step perturbation of each λ*
    print("\n--- Q3 sensitivity: per-step +/-5% perturbations of λ* ---")
    hists_q3s, labels_q3s = _run_sensitivity(
        SteepestDescent, f_rosen, grad_rosen, x0_r, n_iter_r,
    )
    _, ax = plt.subplots(figsize=(6, 6))
    xs_g = np.linspace(xlim_r[0], xlim_r[1], 200)
    ys_g = np.linspace(ylim_r[0], ylim_r[1], 200)
    Xg, Yg = np.meshgrid(xs_g, ys_g)
    Zg = np.vectorize(lambda a, b: f_rosen(np.array([a, b])))(Xg, Yg)
    ax.contour(Xg, Yg, Zg, levels=levels_r, linewidths=0.5, colors="0.7")
    for hist, label in zip(hists_q3s, labels_q3s):
        plot_trajectory(f_rosen, hist, xlim=xlim_r, ylim=ylim_r,
                        levels=levels_r, ax=ax, label=label, annotate=False)
    ax.plot(1.0, 1.0, marker="*", color="red", markersize=12, linestyle="none")
    ax.set_title(r"Q3 sensitivity: SD on function (5), $\lambda^*\times(1\pm 0.05)$ per step")
    ax.figure.savefig(out / "q3_steepest_function5_sensitivity.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q3_steepest_function5_sensitivity.pdf'}")
    base = hists_q3s[0]
    for hist, label in zip(hists_q3s, labels_q3s):
        xf = hist[-1].x
        diverge = _first_divergence(base, hist)
        print(f"  {label}: final ({xf[0]:+.4f}, {xf[1]:+.4f})  f={hist[-1].f:.6g}  "
              f"diverges from baseline at k={diverge}")

    # Q5 sensitivity (CG on f_rosen) — +/-5% per-step perturbation of each λ*
    print("\n--- Q5 sensitivity: per-step +/-5% perturbations of λ* ---")
    hists_q5s, labels_q5s = _run_sensitivity(
        ConjugateGradient, f_rosen, grad_rosen, x0_r, n_iter_r,
    )
    _, ax = plt.subplots(figsize=(6, 6))
    ax.contour(Xg, Yg, Zg, levels=levels_r, linewidths=0.5, colors="0.7")
    for hist, label in zip(hists_q5s, labels_q5s):
        plot_trajectory(f_rosen, hist, xlim=xlim_r, ylim=ylim_r,
                        levels=levels_r, ax=ax, label=label, annotate=False)
    ax.plot(1.0, 1.0, marker="*", color="red", markersize=12, linestyle="none")
    ax.set_title(r"Q5 sensitivity: CG on function (5), $\lambda^*\times(1\pm 0.05)$ per step")
    ax.figure.savefig(out / "q5_cg_function5_sensitivity.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q5_cg_function5_sensitivity.pdf'}")
    base = hists_q5s[0]
    for hist, label in zip(hists_q5s, labels_q5s):
        xf = hist[-1].x
        diverge = _first_divergence(base, hist)
        print(f"  {label}: final ({xf[0]:+.4f}, {xf[1]:+.4f})  f={hist[-1].f:.6g}  "
              f"diverges from baseline at k={diverge}")

    # --- Question 6: DFP on f_quad3 from (1, 1, 1), 3 iterations with preset λ* ---
    print("\n--- Q6: DFP on f_quad3 from (1, 1, 1), 3 iterations, λ* = [0.3942, 2.5522, 4.2202] ---")
    x0_q = np.array([1.0, 1.0, 1.0])
    lam_q6 = [0.3942, 2.5522, 4.2202]
    history_q6 = minimise(
        DFP(), f_quad3, grad_quad3, x0_q, n_iter=3,
        lam_source=lam_q6, verbose=False,
    )
    print_table(history_q6, show_H=True)
    H3 = history_q6[-1].H
    print(f"\ninv(Hess) (eqn 10) =\n{HESS_QUAD3_INV}")
    print(f"H_3 - inv(Hess) =\n{H3 - HESS_QUAD3_INV}")
    print(f"||H_3 - inv(Hess)||_F = {float(np.linalg.norm(H3 - HESS_QUAD3_INV, 'fro')):.3e}")

    # Sensitivity: perturb each λ* one at a time by ±5%, plus a uniform +1%.
    print("\nQ6 sensitivity to λ* (3 iterations from (1,1,1)):")
    perturbations_q6 = [
        ("baseline",   list(lam_q6)),
        ("λ_1 × 1.05", [lam_q6[0] * 1.05, lam_q6[1],         lam_q6[2]]),
        ("λ_1 × 0.95", [lam_q6[0] * 0.95, lam_q6[1],         lam_q6[2]]),
        ("λ_2 × 1.05", [lam_q6[0],         lam_q6[1] * 1.05, lam_q6[2]]),
        ("λ_2 × 0.95", [lam_q6[0],         lam_q6[1] * 0.95, lam_q6[2]]),
        ("λ_3 × 1.05", [lam_q6[0],         lam_q6[1],         lam_q6[2] * 1.05]),
        ("λ_3 × 0.95", [lam_q6[0],         lam_q6[1],         lam_q6[2] * 0.95]),
        ("all   × 1.01", [v * 1.01 for v in lam_q6]),
    ]
    print(f"  {'label':<13}  {'x_3':<36}  {'f_3':>12}  {'||H_3 - invH||_F':>17}")
    for label, lams in perturbations_q6:
        h = minimise(DFP(), f_quad3, grad_quad3, x0_q, n_iter=3,
                     lam_source=lams, verbose=False)
        x3, f3 = h[-1].x, h[-1].f
        nrm = float(np.linalg.norm(h[-1].H - HESS_QUAD3_INV, 'fro'))
        print(f"  {label:<13}  ({x3[0]:+8.5f}, {x3[1]:+8.5f}, {x3[2]:+8.5f})  "
              f"{f3:12.6e}  {nrm:17.3e}")

    # --- Question 7: DFP on f_bedpan from (-1.0, -1.3) ---
    print("\n--- Q7: DFP on f_bedpan from (-1.0, -1.3), 10 iterations ---")
    history_q7 = minimise(
        DFP(), f_bedpan, grad_bedpan, x0, n_iter=10,
        lam_source="auto", verbose=False,
    )
    print_table(history_q7, show_H=True)

    ax_q7 = plot_trajectory(
        f_bedpan, history_q7,
        xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
        label="DFP path",
    )
    ax_q7.set_title("DFP on function (4) from $(-1.0, -1.3)$")
    ax_q7.figure.savefig(out / "q7_dfp_function4_path.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q7_dfp_function4_path.pdf'}")

    x_final = history_q7[-1].x
    H_final = history_q7[-1].H
    Hinv_at_final = np.linalg.inv(hess_bedpan(x_final))
    print(f"final: x = ({x_final[0]:+.6f}, {x_final[1]:+.6f}),  f = {history_q7[-1].f:.6g}")
    print(f"H_10 (DFP) =\n{H_final}")
    print(f"inv(Hess(x_10)) at ({x_final[0]:+.4f}, {x_final[1]:+.4f}) =\n{Hinv_at_final}")
    print(f"||H_10 - inv(Hess(x_10))||_F = "
          f"{float(np.linalg.norm(H_final - Hinv_at_final, 'fro')):.3e}")

    # --- Question 8: DFP on f_rosen from (0.676, 0.443) ---
    print("\n--- Q8: DFP on f_rosen from (0.676, 0.443), 12 iterations ---")
    history_q8 = minimise(
        DFP(), f_rosen, grad_rosen, x0_r, n_iter=n_iter_r,
        lam_source="auto", verbose=False,
    )
    print_table(history_q8, show_H=True)

    ax_q8 = plot_trajectory(
        f_rosen, history_q8,
        xlim=xlim_r, ylim=ylim_r, levels=levels_r,
        label="DFP path",
    )
    ax_q8.plot(1.0, 1.0, marker="*", color="red", markersize=12, linestyle="none")
    ax_q8.set_title("DFP on function (5) from $(0.676, 0.443)$")
    ax_q8.figure.savefig(out / "q8_dfp_function5_path.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q8_dfp_function5_path.pdf'}")

    x_final = history_q8[-1].x
    H_final = history_q8[-1].H
    Hinv_at_final = np.linalg.inv(hess_rosen(x_final))
    Hinv_at_min = np.linalg.inv(hess_rosen(np.array([1.0, 1.0])))
    print(f"final: x = ({x_final[0]:+.6f}, {x_final[1]:+.6f}),  f = {history_q8[-1].f:.6g}")
    print(f"H_12 (DFP) =\n{H_final}")
    print(f"inv(Hess(x_12)) at ({x_final[0]:+.4f}, {x_final[1]:+.4f}) =\n{Hinv_at_final}")
    print(f"||H_12 - inv(Hess(x_12))||_F = "
          f"{float(np.linalg.norm(H_final - Hinv_at_final, 'fro')):.3e}")
    print(f"inv(Hess(1, 1)) =\n{Hinv_at_min}")
    print(f"||H_12 - inv(Hess(1, 1))||_F = "
          f"{float(np.linalg.norm(H_final - Hinv_at_min, 'fro')):.3e}")

    # --- Question 9: synthesis of SD, CG, DFP on functions (4) and (5) ---
    print("\n--- Q9: Synthesis: SD, CG, DFP on functions (4) and (5) ---")
    x4_star_arr = np.array([2 ** (-2 / 3) - 1, -(2 ** (-1 / 3))])
    f4_star_val = float(f_bedpan(x4_star_arr))
    x5_star_arr = np.array([1.0, 1.0])
    f5_star_val = 0.0

    def _summary_row(fn_label, method_label, history, x_star):
        n_iters = history[-1].k
        f_final = history[-1].f
        x_final = history[-1].x
        dx = float(np.linalg.norm(x_final - x_star))
        gn = float(np.linalg.norm(history[-1].g))
        print(f"  {fn_label:<3}  {method_label:<22}  {n_iters:>3d}  {f_final:+13.6e}  "
              f"{dx:11.3e}  {gn:11.3e}  ({x_final[0]:+8.5f}, {x_final[1]:+8.5f})")

    print(f"  {'fn':<3}  {'method':<22}  {'n':>3}  {'f_n':>13}  "
          f"{'|x_n-x*|':>11}  {'|g_n|':>11}  endpoint")
    _summary_row("(4)", "Steepest descents",   history_q2, x4_star_arr)
    _summary_row("(4)", "Conjugate gradients", history_q4, x4_star_arr)
    _summary_row("(4)", "DFP",                 history_q7, x4_star_arr)
    _summary_row("(5)", "Steepest descents",   history_q3, x5_star_arr)
    _summary_row("(5)", "Conjugate gradients", history_q5, x5_star_arr)
    _summary_row("(5)", "DFP",                 history_q8, x5_star_arr)

    # Convergence figure: |f_n - f*| vs n on log scale, three methods per function.
    floor = 1e-16
    fig_q9, axes_q9 = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        (axes_q9[0], (history_q2, history_q4, history_q7), f4_star_val,
         "Function (4): bedpan, $f^{*}\\approx -1.0953$"),
        (axes_q9[1], (history_q3, history_q5, history_q8), f5_star_val,
         "Function (5): Rosenbrock-like, $f^{*}=0$"),
    ]
    method_styles = (
        ("Steepest descents",   "o", "-"),
        ("Conjugate gradients", "s", "--"),
        ("DFP",                 "^", "-."),
    )
    for ax, hists, f_star, title in panels:
        for hist, (label, marker, ls) in zip(hists, method_styles):
            ks = [st.k for st in hist]
            err = [max(abs(st.f - f_star), floor) for st in hist]
            ax.semilogy(ks, err, marker=marker, linestyle=ls, markersize=4, label=label)
        ax.set_xlabel("iteration $n$")
        ax.set_ylabel(r"$|f_n - f^{*}|$")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig_q9.tight_layout()
    fig_q9.savefig(out / "q9_convergence.pdf", bbox_inches="tight")
    print(f"wrote {out / 'q9_convergence.pdf'}")
