from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


ShapeType = Literal[
    "cylinder",
    "frustum",
    "bulge",
    "waist",
    "bottle",
    "knots",
    "image",
]


@dataclass(frozen=True)
class ProfileParams:
    height: float


@dataclass(frozen=True)
class CylinderParams(ProfileParams):
    radius: float


@dataclass(frozen=True)
class FrustumParams(ProfileParams):
    r0: float
    r1: float


@dataclass(frozen=True)
class BulgeParams(ProfileParams):
    radius: float
    amplitude: float  # 0..0.8 typical


@dataclass(frozen=True)
class WaistParams(ProfileParams):
    radius: float
    amplitude: float


@dataclass(frozen=True)
class KnotsParams(ProfileParams):
    knots_z: np.ndarray
    knots_r: np.ndarray


def profile_radius_function(shape: ShapeType, params: ProfileParams) -> Callable[[np.ndarray], np.ndarray]:
    H = float(params.height)
    if H <= 0:
        raise ValueError("height must be > 0")

    if shape == "cylinder":
        p = params  # type: ignore[assignment]
        assert isinstance(p, CylinderParams)
        R = float(p.radius)
        return lambda z: np.full_like(z, R, dtype=float)

    if shape == "frustum":
        p = params  # type: ignore[assignment]
        assert isinstance(p, FrustumParams)
        r0, r1 = float(p.r0), float(p.r1)
        return lambda z: r0 + (r1 - r0) * (np.asarray(z, dtype=float) / H)

    if shape == "bulge":
        p = params  # type: ignore[assignment]
        assert isinstance(p, BulgeParams)
        R = float(p.radius)
        a = float(p.amplitude)
        # Bulge: max radius at middle.
        return lambda z: R * (1.0 + a * np.sin(np.pi * (np.asarray(z, dtype=float) / H)))

    if shape == "waist":
        p = params  # type: ignore[assignment]
        assert isinstance(p, WaistParams)
        R = float(p.radius)
        a = float(p.amplitude)
        # Waist: min radius at middle.
        return lambda z: R * (1.0 - a * np.sin(np.pi * (np.asarray(z, dtype=float) / H)))

    # Knot-based profiles: used directly ("knots") and also as backing for
    # higher-level presets like "bottle" and "image".
    if shape in ("knots", "bottle", "image"):
        p = params  # type: ignore[assignment]
        assert isinstance(p, KnotsParams)
        z_knots = np.asarray(p.knots_z, dtype=float)
        r_knots = np.asarray(p.knots_r, dtype=float)
        if z_knots.ndim != 1 or r_knots.ndim != 1 or z_knots.size != r_knots.size:
            raise ValueError("knots_z and knots_r must be 1D arrays of same length")
        if z_knots.size < 2:
            raise ValueError("Need at least 2 knots")
        if not np.all(np.isfinite(z_knots)) or not np.all(np.isfinite(r_knots)):
            raise ValueError("Knots must be finite")
        if np.any(np.diff(z_knots) <= 0):
            raise ValueError("knots_z must be strictly increasing")

        # Piecewise-linear interpolation.
        return lambda z: np.interp(np.asarray(z, dtype=float), z_knots, r_knots)

    raise ValueError(f"Unknown shape: {shape}")


def default_knots(height: float, n_knots: int = 7, base_radius: float = 3.0) -> KnotsParams:
    H = float(height)
    z = np.linspace(0.0, H, int(n_knots), dtype=float)
    r = np.full_like(z, float(base_radius), dtype=float)
    # Gentle bottle-like silhouette by default
    if r.size >= 5:
        mid = r.size // 2
        r[mid] *= 1.15
        r[1] *= 0.95
        r[-2] *= 0.95
    return KnotsParams(height=H, knots_z=z, knots_r=r)
