"""Double Sphere projection and the production ERP/fisheye coordinate mapping.

Parameters and axis transforms match the production inference input path,
``pano-stitch/src/densekp_delivery/camera/fisheye_equirect.py`` (20260320/new).
The older ``lib_pano.camera.fisheye_equirect`` uses different rotations and
does not match the prepared ERP images or valid masks used by DenseKP.
ERP coordinates use x=longitude and y increasing from latitude -pi/2 to +pi/2.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DoubleSphereIntrinsics:
    xi: float
    alpha: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 2880
    height: int = 2880


DS_20260320 = {
    "cam2": DoubleSphereIntrinsics(-0.12464126, 0.56540615, 681.49808088,
                                     682.11759351, 1445.73197064, 1444.25576211),
    "cam4": DoubleSphereIntrinsics(-0.15331803, 0.55745717, 656.78496421,
                                     655.70192322, 1440.93901552, 1443.33089179),
}

ROTATION_MATRIX = np.eye(3, dtype=np.float64)
CAMERA_EXTRINSIC = {
    "cam2": np.eye(3, dtype=np.float64),
    "cam4": np.diag([-1.0, 1.0, -1.0]),
}


def project_double_sphere(rays: np.ndarray, intrinsics: DoubleSphereIntrinsics):
    """Project DS-camera rays to pixels; return pixels and model-domain mask."""
    points = np.asarray(rays, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("rays must have shape [N,3]")
    x, y, z = points.T
    d1 = np.sqrt(x * x + y * y + z * z)
    z_shifted = intrinsics.xi * d1 + z
    d2 = np.sqrt(x * x + y * y + z_shifted * z_shifted)
    alpha = intrinsics.alpha
    w1 = (1 - alpha) / alpha if alpha > .5 else alpha / (1 - alpha)
    w2 = (w1 + intrinsics.xi) / np.sqrt(2 * w1 * intrinsics.xi + intrinsics.xi**2 + 1)
    valid = z_shifted > -w2 * d1
    denominator = alpha * d2 + (1 - alpha) * z_shifted
    safe = np.where(np.abs(denominator) > 0, denominator, 1)
    pixels = np.column_stack((intrinsics.fx * x / safe + intrinsics.cx,
                              intrinsics.fy * y / safe + intrinsics.cy))
    return pixels, valid & np.isfinite(pixels).all(axis=1)


def unproject_double_sphere(pixels: np.ndarray, intrinsics: DoubleSphereIntrinsics):
    """Invert DS pixels to unit camera rays and return its valid-domain mask."""
    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pixels must have shape [N,2]")
    mx = (points[:, 0] - intrinsics.cx) / intrinsics.fx
    my = (points[:, 1] - intrinsics.cy) / intrinsics.fy
    r2 = mx * mx + my * my
    alpha, xi = intrinsics.alpha, intrinsics.xi
    valid = np.isfinite(points).all(axis=1)
    if alpha > .5:
        valid &= r2 < 1 / (2 * alpha - 1)
    root = np.sqrt(np.maximum(1 - (2 * alpha - 1) * r2, 0))
    denominator = alpha * root + 1 - alpha
    mz = (1 - alpha * alpha * r2) / np.where(np.abs(denominator) > 0, denominator, 1)
    kroot = np.sqrt(np.maximum(mz * mz + (1 - xi * xi) * r2, 0))
    kdenominator = mz * mz + r2
    k = (mz * xi + kroot) / np.where(np.abs(kdenominator) > 0, kdenominator, 1)
    rays = np.column_stack((k * mx, k * my, k * mz - xi))
    norm = np.linalg.norm(rays, axis=1)
    valid &= norm > 0
    rays /= np.where(norm > 0, norm, 1)[:, None]
    return rays, valid


def build_erp_to_fisheye_lut(intrinsics: DoubleSphereIntrinsics, camera: str,
                             width: int, height: int, chunk_rows: int = 128):
    """Map each output ERP pixel to the fisheye source, like pano-stitch."""
    if camera not in CAMERA_EXTRINSIC or width < 1 or height < 1:
        raise ValueError("invalid camera or ERP size")
    map_x = np.full((height, width), -1, dtype=np.float32)
    map_y = np.full((height, width), -1, dtype=np.float32)
    lon = np.arange(width, dtype=np.float64) / width * (2 * np.pi) - np.pi
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    for start in range(0, height, chunk_rows):
        stop = min(start + chunk_rows, height)
        lat = np.arange(start, stop, dtype=np.float64) / height * np.pi - np.pi / 2
        cos_lat, sin_lat = np.cos(lat)[:, None], np.sin(lat)[:, None]
        ray = np.stack((cos_lat * sin_lon[None, :],
                        np.broadcast_to(sin_lat, (stop - start, width)),
                        cos_lat * cos_lon[None, :]), axis=-1).reshape(-1, 3)
        camera_ray = ray @ CAMERA_EXTRINSIC[camera].T @ ROTATION_MATRIX.T
        pixels, valid = project_double_sphere(camera_ray, intrinsics)
        x = map_x[start:stop].reshape(-1)
        y = map_y[start:stop].reshape(-1)
        x[valid] = pixels[valid, 0].astype(np.float32)
        y[valid] = pixels[valid, 1].astype(np.float32)
    return map_x, map_y


def build_fisheye_to_source_erp_lut(intrinsics: DoubleSphereIntrinsics, camera: str,
                                    source_width: int, source_height: int, chunk_rows: int = 128):
    """Map synthetic DS fisheye pixels back to a rendered whole-sphere ERP."""
    if camera not in CAMERA_EXTRINSIC:
        raise ValueError(f"unknown camera: {camera}")
    map_x = np.full((intrinsics.height, intrinsics.width), -1, dtype=np.float32)
    map_y = np.full_like(map_x, -1)
    u = np.arange(intrinsics.width, dtype=np.float64)
    for start in range(0, intrinsics.height, chunk_rows):
        stop = min(start + chunk_rows, intrinsics.height)
        v = np.arange(start, stop, dtype=np.float64)
        pixels = np.stack(np.broadcast_arrays(u[None, :], v[:, None]), axis=-1).reshape(-1, 2)
        camera_ray, valid = unproject_double_sphere(pixels, intrinsics)
        erp_ray = camera_ray @ ROTATION_MATRIX @ CAMERA_EXTRINSIC[camera]
        lon = np.arctan2(erp_ray[:, 0], erp_ray[:, 2])
        lat = np.arcsin(np.clip(erp_ray[:, 1], -1, 1))
        x = map_x[start:stop].reshape(-1)
        y = map_y[start:stop].reshape(-1)
        x[valid] = ((lon[valid] / (2 * np.pi) + .5) * source_width).astype(np.float32)
        y[valid] = ((lat[valid] / np.pi + .5) * source_height).astype(np.float32)
    return map_x, map_y
