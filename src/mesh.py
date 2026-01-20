from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .math_utils import Grid1D, safe_clip_nonnegative


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) int
    vertex_normals: np.ndarray | None = None


def revolve_profile_to_mesh(
    r_of_z: Callable[[np.ndarray], np.ndarray],
    grid: Grid1D,
    n_theta: int = 128,
    caps: bool = True,
    compute_normals: bool = True,
) -> TriangleMesh:
    """Create a triangle mesh by revolving the profile r(z) around the z-axis."""
    z = grid.z
    r = safe_clip_nonnegative(np.asarray(r_of_z(z), dtype=float))

    n_z = int(z.size)
    n_theta = int(n_theta)
    if n_z < 2:
        raise ValueError("grid must have at least 2 points")
    if n_theta < 8:
        raise ValueError("n_theta must be >= 8")

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=float)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Vertices: for each (i, j)
    xs = (r[:, None] * cos_t[None, :]).astype(float)
    ys = (r[:, None] * sin_t[None, :]).astype(float)
    zs = np.repeat(z[:, None], n_theta, axis=1).astype(float)

    vertices = np.column_stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)])

    def vid(i: int, j: int) -> int:
        return i * n_theta + j

    faces: list[tuple[int, int, int]] = []

    # Side faces
    for i in range(n_z - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            v00 = vid(i, j)
            v01 = vid(i, j2)
            v10 = vid(i + 1, j)
            v11 = vid(i + 1, j2)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    # Caps (optional)
    if caps:
        # bottom
        v_center_bottom = vertices.shape[0]
        vertices = np.vstack([vertices, np.array([[0.0, 0.0, float(z[0])]], dtype=float)])
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            faces.append((v_center_bottom, vid(0, j2), vid(0, j)))

        # top
        v_center_top = vertices.shape[0]
        vertices = np.vstack([vertices, np.array([[0.0, 0.0, float(z[-1])]], dtype=float)])
        top_ring_i = n_z - 1
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            faces.append((v_center_top, vid(top_ring_i, j), vid(top_ring_i, j2)))

    faces_arr = np.asarray(faces, dtype=np.int32)

    normals = None
    if compute_normals:
        normals = _vertex_normals(vertices, faces_arr)

    return TriangleMesh(vertices=vertices, faces=faces_arr, vertex_normals=normals)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = vertices
    f = faces

    n = np.zeros_like(v)
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    # Accumulate
    for k in range(3):
        np.add.at(n, f[:, k], face_normals)

    # Normalize
    lengths = np.linalg.norm(n, axis=1)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    n = n / lengths[:, None]
    return n
