from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Grid1D:
    z: np.ndarray

    @property
    def dz(self) -> float:
        if self.z.size < 2:
            return 0.0
        return float(self.z[1] - self.z[0])


def linspace_grid(z0: float, z1: float, n: int) -> Grid1D:
    if n < 2:
        raise ValueError("n must be >= 2")
    if not np.isfinite(z0) or not np.isfinite(z1):
        raise ValueError("z0 and z1 must be finite")
    if z1 <= z0:
        raise ValueError("z1 must be > z0")
    return Grid1D(z=np.linspace(z0, z1, int(n), dtype=float))


def derivative_central(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Numerical derivative dy/dx using numpy.gradient (2nd order interior)."""
    return np.gradient(y, x)


def integrate_simpson(y: np.ndarray, x: np.ndarray) -> float:
    """Composite Simpson's rule on an (almost) uniform grid.

    Falls back to trapezoid if the number of points is too small.
    """
    if y.size != x.size:
        raise ValueError("x and y must have same length")
    if y.size < 3:
        return float(np.trapz(y, x))

    n = y.size
    if n % 2 == 0:
        # Simpson requires odd number of points; drop last point for stability.
        y = y[:-1]
        x = x[:-1]

    h = float(x[1] - x[0])
    # Check near-uniform spacing
    if not np.allclose(np.diff(x), h, rtol=1e-3, atol=1e-9):
        return float(np.trapz(y, x))

    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])
    return float(h * s / 3.0)


def integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def safe_clip_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def sample_function_on_grid(func: Callable[[np.ndarray], np.ndarray], grid: Grid1D) -> np.ndarray:
    y = np.asarray(func(grid.z), dtype=float)
    if y.shape != grid.z.shape:
        raise ValueError("Function must return an array with same shape as grid")
    if not np.all(np.isfinite(y)):
        raise ValueError("Function returned non-finite values")
    return y
