"""Explicit rectangular render support in the full ERP pixel grid.

Physical valid masks and computed high-quality coverage are separate. Outside
ROI depth/IDs stay unknown. A crop producer must call require_crop_in_region;
the unchanged historical Dataset does not infer this contract from black RGB.
"""

import numpy as np


def checked_region(region_xyxy, image_size_wh):
    a = np.asarray(region_xyxy, dtype=float)
    size = np.asarray(image_size_wh, dtype=float)
    if (
        size.shape != (2,)
        or not np.isfinite(size).all()
        or (size <= 0).any()
        or (size != np.floor(size)).any()
    ):
        raise ValueError("invalid_full_image_size")
    if a.shape != (4,) or not np.isfinite(a).all() or (a != np.floor(a)).any():
        raise ValueError("invalid_region_coordinates")
    if (a[:2] < 0).any() or (a[2:] <= a[:2]).any() or (a[2:] > size).any():
        raise ValueError("region_outside_full_raster_or_empty")
    return tuple(a.astype(int))


def region_mask(region_xyxy, image_size_wh):
    x0, y0, x1, y1 = checked_region(region_xyxy, image_size_wh)
    mask = np.zeros((image_size_wh[1], image_size_wh[0]), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def embed_region(array, region_xyxy, image_size_wh, *, unknown_value):
    x0, y0, x1, y1 = checked_region(region_xyxy, image_size_wh)
    if array.shape[:2] != (y1 - y0, x1 - x0):
        raise ValueError("local_buffer_shape_mismatch")
    out = np.full(
        (image_size_wh[1], image_size_wh[0], *array.shape[2:]), unknown_value, dtype=array.dtype
    )
    out[y0:y1, x0:x1] = array
    return out


def require_crop_in_region(center, scale, region_xyxy, image_size_wh, *, interpolation_margin_px=2):
    region = np.array(checked_region(region_xyxy, image_size_wh))
    center = np.asarray(center, dtype=float)
    if (
        center.shape != (2,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("invalid_crop_geometry")
    if not np.isfinite(interpolation_margin_px) or interpolation_margin_px < 0:
        raise ValueError("invalid_crop_margin")
    half = float(scale) * 100
    lo = center - half - interpolation_margin_px
    hi = center + half + interpolation_margin_px
    if (lo < region[:2]).any() or (hi > region[2:]).any():
        raise ValueError("crop_outside_high_quality_support")
    return True


def require_point_neighborhoods(xy, region_xyxy, image_size_wh, *, radius_px=2):
    region = np.array(checked_region(region_xyxy, image_size_wh))
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("invalid_point_coordinates")
    if type(radius_px) is not int or radius_px < 0:
        raise ValueError("invalid_neighborhood_radius")
    pixels = np.floor(xy)
    if ((pixels - radius_px) < region[:2]).any() or ((pixels + radius_px) >= region[2:]).any():
        raise ValueError("visibility_neighborhood_outside_computed_region")
    return True


def apply_physical_mask(rgb, valid_mask):
    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or valid_mask.dtype != bool
        or valid_mask.shape != rgb.shape[:2]
    ):
        raise ValueError("invalid_physical_mask_or_image_shape")
    out = rgb.copy()
    out[~valid_mask] = 0
    return out
