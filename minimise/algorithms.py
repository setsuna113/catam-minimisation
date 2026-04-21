"""Search-direction strategies: Steepest Descents, Conjugate Gradients, DFP"""

from __future__ import annotations
import numpy as np


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
