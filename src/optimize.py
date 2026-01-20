from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .geometry import area_from_profile, volume_from_profile_disk
from .math_utils import Grid1D, integrate_simpson


@dataclass(frozen=True)
class CylinderOptimum:
    radius_cm: float
    height_cm: float
    area_cm2: float


def cylinder_optimum_closed(volume_cm3: float) -> CylinderOptimum:
    """Closed cylinder optimum via Lagrange multipliers.

    Minimize A = 2πRH + 2πR^2 subject to V = πR^2H = V0.
    Result: H = 2R and R^3 = V0 / (2π).
    """
    if volume_cm3 <= 0:
        raise ValueError("volume_cm3 must be > 0")

    R = float((volume_cm3 / (2.0 * np.pi)) ** (1.0 / 3.0))
    H = 2.0 * R
    area = float(2.0 * np.pi * R * H + 2.0 * np.pi * R * R)
    return CylinderOptimum(radius_cm=R, height_cm=H, area_cm2=area)


@dataclass(frozen=True)
class ProfileOptResult:
    success: bool
    message: str
    knots_r: np.ndarray
    area_cm2: float
    volume_cm3: float


def optimize_knots_radii(
    knots_z: np.ndarray,
    initial_knots_r: np.ndarray,
    target_volume_cm3: float,
    grid: Grid1D,
    include_caps: bool,
    r_min: float,
    r_max: float,
    smoothness_weight: float = 0.02,
    maxiter: int = 200,
) -> ProfileOptResult:
    """Optimize knot radii to minimize surface area with fixed volume.

    Variables: knot radii at fixed knot positions along z.
    Constraint: V(knots_r) == target_volume.
    Bounds: r_min <= r_i <= r_max.
    Regularization: smoothness penalty on second differences.
    """
    z_knots = np.asarray(knots_z, dtype=float)
    r0 = np.asarray(initial_knots_r, dtype=float)
    if z_knots.ndim != 1 or r0.ndim != 1 or z_knots.size != r0.size:
        raise ValueError("knots_z and initial_knots_r must be 1D arrays of same length")
    if target_volume_cm3 <= 0:
        raise ValueError("target_volume_cm3 must be > 0")

    def r_of_z_from_knots(r_knots: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return lambda z: np.interp(np.asarray(z, dtype=float), z_knots, r_knots)

    def smoothness_penalty(r_knots: np.ndarray) -> float:
        if r_knots.size < 3:
            return 0.0
        d2 = r_knots[:-2] - 2.0 * r_knots[1:-1] + r_knots[2:]
        # approximate integral of squared curvature-like term
        return float(smoothness_weight * np.sum(d2 * d2))

    def objective(r_knots: np.ndarray) -> float:
        r_of_z = r_of_z_from_knots(r_knots)
        area_total, _, _ = area_from_profile(r_of_z, grid, include_caps=include_caps)
        return float(area_total + smoothness_penalty(r_knots))

    def constraint_volume(r_knots: np.ndarray) -> float:
        r_of_z = r_of_z_from_knots(r_knots)
        v = volume_from_profile_disk(r_of_z, grid)
        return float(v - target_volume_cm3)

    bounds = [(float(r_min), float(r_max)) for _ in range(r0.size)]

    # Initial projection into bounds
    x0 = np.clip(r0, r_min, r_max)

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": constraint_volume}],
        options={"maxiter": int(maxiter), "ftol": 1e-8},
    )

    r_opt = np.clip(np.asarray(res.x, dtype=float), r_min, r_max)
    r_of_z = r_of_z_from_knots(r_opt)
    area_total, _, _ = area_from_profile(r_of_z, grid, include_caps=include_caps)
    vol = volume_from_profile_disk(r_of_z, grid)

    message = str(res.message)
    return ProfileOptResult(
        success=bool(res.success),
        message=message,
        knots_r=r_opt,
        area_cm2=float(area_total),
        volume_cm3=float(vol),
    )


def dimensionless_efficiency(area_cm2: float, volume_cm3: float) -> float:
    """Compute A / V^(2/3), useful for comparing across different volumes."""
    if volume_cm3 <= 0:
        return float("nan")
    return float(area_cm2 / (volume_cm3 ** (2.0 / 3.0)))
