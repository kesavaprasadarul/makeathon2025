#!/usr/bin/env python3
"""find_positions_visualize.py

Generate synthetic 3-D data, fit an **L² (least-squares) plane**, displace each
point 1 unit along the plane’s outward normal, and visualise everything –
including the fitted plane surface.

Run:
    python find_positions_visualize.py

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – activates 3-D projection

# ────────────────────────────── DATA GENERATION ──────────────────────────────

def _in_diamond(x: float) -> bool:
    """Return **True** iff (x, y) with y = –x + 1 satisfies |x| + |y| ≤ 3."""
    y = -x + 1.0
    return abs(x) + abs(y) <= 3.0


def generate_points(
    n_points: int = 200,
    noise_std: float = 0.25,
    z_low: float = 0.0,
    z_high: float = 30.0,
    z_offset: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Return *n_points* around the line y = –x + 1 inside |x| + |y| ≤ 3.

    Gaussian noise σ = *noise_std* is added in all directions and every z is
    shifted upward by *z_offset*.
    """

    rng = np.random.default_rng(seed)

    xs: list[float] = []
    while len(xs) < n_points:
        cand = rng.uniform(-2.0, 2.0)
        if _in_diamond(cand):
            xs.append(cand)

    x = np.asarray(xs)
    y = -x + 1.0
    z = rng.uniform(z_low, z_high, size=n_points) + z_offset

    pts = np.column_stack((x, y, z))
    pts += rng.normal(scale=noise_std, size=pts.shape)
    return pts

# ────────────────────────────── PLANE FITTING (L²) ───────────────────────────

def _fit_plane_L2(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane *n·x + d = 0* that minimises ∑dist² to *points*.

    **Algorithm (PCA):**
    1. Centre the cloud at its centroid.
    2. The eigenvector corresponding to the smallest eigenvalue of the
       covariance matrix is the normal.
    """

    # 1. Centroid and centred coordinates
    centroid = points.mean(axis=0)
    A = points - centroid

    # 2. SVD (more stable than eig on covariance)
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    normal = vh[-1]  # right-singular vector with smallest singular value
    normal /= np.linalg.norm(normal)

    # 3. Offset so plane passes through centroid:  n·c + d = 0  ⇒  d = –n·c
    d = -np.dot(normal, centroid)
    return normal, d

# ───────────────────────── FIND POSITIONS FUNCTION ───────────────────────────

def find_positions(
    segmented_points: np.ndarray,
    camera_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Displace each *segmented_point* by +1 unit along the plane normal.

    Returns ``(new_points, normal, d)``; *normal* is oriented so its dot with
    *(camera − centroid)* is positive (i.e. broadly facing the camera).
    """

    normal, d = _fit_plane_L2(segmented_points)

    # Orient the normal toward the camera if necessary
    if np.dot(normal, camera_point - segmented_points.mean(axis=0)) < 0:
        normal = -normal
        d = -d

    displaced = []
    for p in segmented_points:
        dir_vec = normal if np.dot(normal, camera_point - p) > 0 else -normal
        displaced.append(p + dir_vec)

    return np.vstack(displaced), normal, d

# ─────────────────────────────── PLOTTING HELPERS ───────────────────────────

def plane_mesh(
    normal: np.ndarray,
    d: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    density: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return meshgrid (X, Y, Z) for the plane within the xy-rectangle."""

    xx, yy = np.meshgrid(
        np.linspace(*x_range, density), np.linspace(*y_range, density)
    )
    nz = normal[2]
    if abs(nz) < 1e-9:
        raise ValueError("Plane normal almost parallel to Z axis – cannot mesh.")
    zz = -(normal[0] * xx + normal[1] * yy + d) / nz
    return xx, yy, zz

# ─────────────────────────────────── DEMO ────────────────────────────────────

def _demo() -> None:  # pragma: no cover
    """Run an end-to-end demonstration with the new L² fit."""

    pts = generate_points(seed=123)
    camera = np.array([-3.0, -7.0, 200.0])

    new_pts, normal, d = find_positions(pts, camera)

    # Build a mesh just large enough to cover the projected xy extent
    pad = 0.5
    xmin, xmax = pts[:, 0].min() - pad, pts[:, 0].max() + pad
    ymin, ymax = pts[:, 1].min() - pad, pts[:, 1].max() + pad
    plane_X, plane_Y, plane_Z = plane_mesh(normal, d, (xmin, xmax), (ymin, ymax))

    print(
        f"Plane equation: {normal[0]:.3f}·x + {normal[1]:.3f}·y + {normal[2]:.3f}·z + {d:.3f} = 0"
    )
    print("Unit normal:", normal)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    #ax.plot_surface(plane_X, plane_Y, plane_Z, alpha=0.3, linewidth=0)

    orig_handle = ax.scatter(*pts.T, label="Original", marker="o")
    disp_handle = ax.scatter(*new_pts.T, label="Displaced (+1)", marker="^", depthshade=False)
    cam_handle = ax.scatter(*camera, label="Camera", marker="*", s=140, color="k")

    for p, q in zip(pts, new_pts):
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], lw=0.8, alpha=0.7)

    plane_proxy = Patch(facecolor="gray", alpha=0.3, edgecolor="none", label="Fitted plane (L²)")
    ax.legend(handles=[orig_handle, disp_handle, cam_handle, plane_proxy])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Segmented points, displaced points, L²-fitted plane & camera")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":  # pragma: no cover
    _demo()
