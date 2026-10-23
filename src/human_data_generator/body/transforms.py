"""Body-relative SMPL-X axes to scene world coordinates, in meters.

Input +Y is up; world +Z is up. Conversion is (x, -z, y), lifted using the BODY
minimum Y, followed by world-Z yaw and location. Clothing and semantic points
must use the same body's vertices for the lift, not their own minimum height.
These NumPy operations do not validate Blender object assembly.
"""

import math
import numpy as np


def world_points(points: np.ndarray, vertices: np.ndarray, person: dict) -> np.ndarray:
    """Match the Blender body matrix in render_equirectangular_source.py."""
    lift = -float(vertices[:, 1].min())
    world = np.column_stack((points[:, 0], -points[:, 2], points[:, 1] + lift))
    angle = math.radians(float(person["yaw_deg"]))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    return world @ rotation.T + np.asarray(person["location"], dtype=np.float64)
