from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ExtractedProfile:
    z_unit: np.ndarray  # 0..1
    r_unit: np.ndarray  # 0..1
    preview_mask: np.ndarray  # uint8 0/255


def extract_profile_from_image(
    image_bytes: bytes,
    threshold: int = 140,
    invert: bool = False,
    smooth_window: int = 7,
) -> ExtractedProfile:
    """Extract a radius profile from a silhouette-like image.

    Assumptions (v1):
    - Bottle silhouette is the foreground.
    - We scan each horizontal row and measure the foreground width.
    - Radius is half the width.

    Returns a unit profile (z in [0,1], r in [0,1]) and a preview binary mask.
    """
    img = Image.open(BytesIO(image_bytes)).convert("L")
    arr = np.asarray(img, dtype=np.uint8)

    # Binary mask: foreground = 1
    mask = arr < int(threshold)
    if invert:
        mask = ~mask

    # Remove obvious border noise by zeroing a 1px frame
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False

    h, w = mask.shape
    widths = np.zeros(h, dtype=float)

    for i in range(h):
        row = mask[i]
        idx = np.flatnonzero(row)
        if idx.size >= 2:
            widths[i] = float(idx[-1] - idx[0])
        else:
            widths[i] = 0.0

    # Keep rows where we have some shape
    valid = widths > 0
    if np.count_nonzero(valid) < max(10, h // 20):
        raise ValueError("Could not detect a clear silhouette. Try adjusting threshold/invert.")

    y = np.flatnonzero(valid).astype(float)
    r = (widths[valid] / 2.0).astype(float)

    # Normalize
    r_max = float(np.max(r))
    if r_max <= 0:
        raise ValueError("Detected silhouette has zero radius.")

    r_unit = r / r_max

    # Map y (image row) to z (0..1), bottom at z=0
    y0, y1 = float(y.min()), float(y.max())
    z_unit = (y1 - y) / (y1 - y0)

    # Sort by z increasing
    order = np.argsort(z_unit)
    z_unit = z_unit[order]
    r_unit = r_unit[order]

    # Smooth radius a bit (moving average)
    smooth_window = int(max(1, smooth_window))
    if smooth_window >= 3:
        k = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        pad = k // 2
        padded = np.pad(r_unit, (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=float) / float(k)
        r_unit = np.convolve(padded, kernel, mode="valid")

    preview = (mask.astype(np.uint8) * 255)
    return ExtractedProfile(z_unit=z_unit, r_unit=r_unit, preview_mask=preview)


def sample_knots_from_profile(
    z_unit: np.ndarray,
    r_unit: np.ndarray,
    height_cm: float,
    n_knots: int,
    base_radius_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a unit profile into knots and scale to (cm)."""
    z_unit = np.asarray(z_unit, dtype=float)
    r_unit = np.asarray(r_unit, dtype=float)
    if z_unit.ndim != 1 or r_unit.ndim != 1 or z_unit.size != r_unit.size:
        raise ValueError("z_unit and r_unit must be 1D arrays of same length")

    n_knots = int(n_knots)
    if n_knots < 3:
        raise ValueError("n_knots must be >= 3")

    z_knots = np.linspace(0.0, float(height_cm), n_knots, dtype=float)
    zq = z_knots / float(height_cm)
    rq = np.interp(zq, z_unit, r_unit)

    # Scale radius so max radius roughly equals base_radius_cm
    r_knots = float(base_radius_cm) * rq
    return z_knots, r_knots
