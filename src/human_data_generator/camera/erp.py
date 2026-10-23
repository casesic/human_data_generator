"""Historical whole-sphere ERP projection variants, without resampling.

camera is CV camera-to-world [4,4], whose columns are right/down/forward axes.
For row points, camera coordinates are (world - translation) @ rotation.
Returned distance is Euclidean camera-to-point range in meters, NOT optical Z.
Pixels use caller-supplied width/height; +/-pi can yield x=0 or x=width.
Projection does not wrap x; the individual consumers own their wrap policy.
"""

import numpy as np


def project_erp(points: np.ndarray, camera: np.ndarray, width: int, height: int):
    """Project world points [N,3] to full ERP pixels [N,2] and ray range [N].

    Zero-range rays retain historical NaNs. No pixel-center or half-width shift."""
    q = (points - camera[:3, 3]) @ camera[:3, :3]
    distance = np.linalg.norm(q, axis=1)
    ray = q / distance[:, None]
    lon = np.arctan2(ray[:, 0], ray[:, 2])
    lat = np.arcsin(np.clip(-ray[:, 1], -1, 1))
    xy = np.column_stack(((lon / (2 * np.pi) + 0.5) * width, (0.5 - lat / np.pi) * height))
    return xy, distance


def erp_points(world: np.ndarray, camera: np.ndarray, width: int, height: int) -> np.ndarray:
    """Historical ERP-only projection; retain its original norm/array expression order."""
    q = (world - camera[:3, 3]) @ camera[:3, :3]
    ray = q / np.linalg.norm(q, axis=1, keepdims=True)
    lon = np.arctan2(ray[:, 0], ray[:, 2])
    lat = np.arcsin(np.clip(-ray[:, 1], -1, 1))
    return np.column_stack(((lon / (2 * np.pi) + 0.5) * width, (0.5 - lat / np.pi) * height))
