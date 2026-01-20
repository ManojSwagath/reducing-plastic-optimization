from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .math_utils import Grid1D, derivative_central, integrate_simpson, safe_clip_nonnegative


@dataclass(frozen=True)
class GeometryResult:
    volume_cm3: float
    area_cm2: float
    lateral_area_cm2: float
    caps_area_cm2: float
    volume_shells_cm3: float


def volume_from_profile_disk(r_of_z: Callable[[np.ndarray], np.ndarray], grid: Grid1D) -> float:
    """Volume by disk method: V = pi * ∫ r(z)^2 dz."""
    r = safe_clip_nonnegative(np.asarray(r_of_z(grid.z), dtype=float))
    return float(np.pi * integrate_simpson(r * r, grid.z))


def volume_from_profile_shells(r_of_z: Callable[[np.ndarray], np.ndarray], grid: Grid1D, n_r: int = 240) -> float:
    """Volume via cylindrical shells (numerical).

    For a shape of revolution around the z-axis with radius profile r(z),
    define the shell radius ρ and shell height h(ρ) = measure of z where r(z) >= ρ.

    Then V = ∫_0^{rmax} 2π ρ h(ρ) dρ.

    This is slower than the disk method but is a useful educational cross-check.
    """
    z = grid.z
    r = safe_clip_nonnegative(np.asarray(r_of_z(z), dtype=float))
    r_max = float(np.max(r))
    if r_max <= 0:
        return 0.0

    rho = np.linspace(0.0, r_max, int(n_r), dtype=float)
    dz = np.diff(z)
    # For each rho_k, approximate shell height by counting segments where r >= rho_k.
    # Use midpoint indicator on segments.
    r_mid = 0.5 * (r[:-1] + r[1:])

    heights = np.empty_like(rho)
    for k, rk in enumerate(rho):
        mask = r_mid >= rk
        heights[k] = float(np.sum(dz[mask]))

    integrand = 2.0 * np.pi * rho * heights
    return float(integrate_simpson(integrand, rho))


def area_from_profile(r_of_z: Callable[[np.ndarray], np.ndarray], grid: Grid1D, include_caps: bool) -> tuple[float, float, float]:
    """Surface area of revolution.

    Lateral area: A_lat = 2π ∫ r(z) * sqrt(1 + (r'(z))^2) dz
    Caps: A_caps = π r(0)^2 + π r(H)^2
    """
    z = grid.z
    r = safe_clip_nonnegative(np.asarray(r_of_z(z), dtype=float))
    dr_dz = derivative_central(r, z)

    lateral = float(2.0 * np.pi * integrate_simpson(r * np.sqrt(1.0 + dr_dz * dr_dz), z))

    caps = 0.0
    if include_caps:
        caps = float(np.pi * (r[0] ** 2 + r[-1] ** 2))

    total = lateral + caps
    return total, lateral, caps


def compute_geometry(
    r_of_z: Callable[[np.ndarray], np.ndarray],
    grid: Grid1D,
    include_caps: bool,
    shells_n_r: int = 240,
) -> GeometryResult:
    vol_disk = volume_from_profile_disk(r_of_z, grid)
    vol_shells = volume_from_profile_shells(r_of_z, grid, n_r=shells_n_r)
    area_total, area_lat, area_caps = area_from_profile(r_of_z, grid, include_caps=include_caps)
    return GeometryResult(
        volume_cm3=vol_disk,
        area_cm2=area_total,
        lateral_area_cm2=area_lat,
        caps_area_cm2=area_caps,
        volume_shells_cm3=vol_shells,
    )


def scale_radius_to_target_volume(
    r_of_z: Callable[[np.ndarray], np.ndarray],
    grid: Grid1D,
    target_volume_cm3: float,
) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Uniformly scales radius by factor s so that volume matches target.

    Since V ∝ r^2 under uniform scaling, s = sqrt(V_target / V_current).
    """
    if target_volume_cm3 <= 0:
        raise ValueError("target_volume_cm3 must be > 0")
    current = volume_from_profile_disk(r_of_z, grid)
    if current <= 0:
        return r_of_z, 1.0

    s = float(np.sqrt(target_volume_cm3 / current))
    return (lambda z: s * np.asarray(r_of_z(z), dtype=float)), s
