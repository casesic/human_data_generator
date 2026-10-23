"""Frozen DS geometry-domain and explicit-mask membership for world points.

Keep floor sampling, horizontal modulo and vertically clipped indices. This
policy differs from rounded delivery-mask diagnostics; do not unify them.
"""

import numpy as np
from .double_sphere import DS_20260320, CAMERA_EXTRINSIC, project_double_sphere
from .erp import project_erp


def point_in_view(points, camera, name, mask):
    """Return ERP xy, physical fisheye xy, ray range, DS domain and in-view mask.

    points is [N,3] world meters; camera is CV camera-to-world [4,4].
    ERP resolution comes from mask[H,W]; fisheye size is DS_20260320 calibration.
    Mask sampling uses floor/modulo, independently of RGB and visibility."""
    h, w = mask.shape
    xy, distance = project_erp(points, camera, w, h)
    q = (points - camera[:3, 3]) @ camera[:3, :3]
    fish, domain = project_double_sphere(q @ CAMERA_EXTRINSIC[name].T, DS_20260320[name])
    size = DS_20260320[name]
    finite = np.isfinite(xy).all(1) & np.isfinite(fish).all(1)
    ix = np.floor(np.nan_to_num(xy[:, 0])).astype(int) % w
    iy = np.clip(np.floor(np.nan_to_num(xy[:, 1])).astype(int), 0, h - 1)
    in_view = (
        finite
        & domain
        & (fish[:, 0] >= 0)
        & (fish[:, 0] < size.width)
        & (fish[:, 1] >= 0)
        & (fish[:, 1] < size.height)
        & mask[iy, ix]
    )
    return xy, fish, distance, domain, in_view
