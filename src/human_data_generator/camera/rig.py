"""Paired normal-ERP and physical Double Sphere camera-to-world poses.

The baseline is an explicit synthetic parameter in meters. It is not a measured
hardware baseline. Both ERP bases align; the physical DS optical axes oppose.
"""

import math
import numpy as np
from .double_sphere import CAMERA_EXTRINSIC


def rig_poses(origin, yaw_deg, baseline_m):
    """Return ERP camera-to-world and physical DS camera-to-world matrices.

    Both normal ERPs are rendered in the same axis orientation, at distinct
    centers. cam4's DS axis flip belongs to fisheye projection/remapping only.
    """
    if not np.isfinite(baseline_m) or baseline_m < 0:
        raise ValueError("baseline_m must be nonnegative and finite")
    angle = math.radians(yaw_deg)
    yaw = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cv_basis = yaw @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    poses, lenses = {}, {}
    for name, sign in [("cam2", 1), ("cam4", -1)]:
        matrix = np.eye(4)
        matrix[:3, :3] = cv_basis
        matrix[:3, 3] = np.array(origin) + yaw @ [0.0, sign * baseline_m / 2, 1.15]
        lens = matrix.copy()
        lens[:3, :3] = cv_basis @ CAMERA_EXTRINSIC[name]
        poses[name], lenses[name] = matrix, lens
    return poses, lenses, yaw
