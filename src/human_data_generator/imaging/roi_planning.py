"""Conservative continuous-frame ROI in the unchanged full ERP grid (pixels)."""

import numpy as np
from .regions import checked_region


def plan_native_crop_region(projected_parts, image_size_wh, *, margin_px=32):
    """Cover geometry and any deployment square of a sub-box within its envelope.

    Visible instance boxes can shift as the calibrated lens cuts the silhouette.
    For envelope width W/height H, any contained box's square lies within
    [xmin-H/2, ymin-W/2, xmax+H/2, ymax+W/2]. Two raster pixels expand the
    geometry bounds first; the final margin also protects visibility/filtering.
    No cyclic wrapping, clipping, random augmentation, or RGB validity inference.
    """
    if not np.isfinite(margin_px) or margin_px < 32:
        raise ValueError("ROI_margin_below_32_pixels")
    bounds = {}
    for name, xy in projected_parts.items():
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
            raise ValueError("invalid_projected_geometry")
        if np.ptp(xy[:, 0]) >= image_size_wh[0] / 2:
            raise ValueError("ERP_geometry_seam_crossing")
        bounds[name] = np.r_[xy.min(0), xy.max(0)].tolist()
    if not bounds:
        raise ValueError("empty_geometry")
    b = np.array(list(bounds.values()))
    lo = b[:, :2].min(0) - 2
    hi = b[:, 2:].max(0) + 2
    if hi[0] - lo[0] >= image_size_wh[0] / 2:
        raise ValueError("ERP_geometry_seam_crossing")
    extent = hi - lo
    pad = extent[::-1] / 2
    roi = np.r_[np.floor(lo - pad - margin_px), np.ceil(hi + pad + margin_px)].astype(int)
    checked_region(roi, image_size_wh)
    result = dict(
        roi_xyxy=roi.tolist(),
        geometry_bounds_full_erp=bounds,
        geometry_raster_allowance_px=2,
        margin_final_erp_px=margin_px,
        planning_method="conservative envelope of all deployment squares of contained instance boxes",
    )
    return result


def plan_person_crop_union(projected_parts_by_person, image_size_wh, *, margin_px=32):
    """One render region covering each person's complete native square crop.

    The space between separate people does not enlarge either person's square.
    Every individual region must fit the full ERP raster before the union.
    """
    if not projected_parts_by_person:
        raise ValueError("empty_person_geometry")
    per_person = {
        str(person_id): plan_native_crop_region(parts, image_size_wh, margin_px=margin_px)
        for person_id, parts in projected_parts_by_person.items()
    }
    boxes = np.asarray([item["roi_xyxy"] for item in per_person.values()], dtype=int)
    roi = [
        int(boxes[:, 0].min()),
        int(boxes[:, 1].min()),
        int(boxes[:, 2].max()),
        int(boxes[:, 3].max()),
    ]
    checked_region(roi, image_size_wh)
    return dict(
        roi_xyxy=roi,
        person_regions=per_person,
        planning_method="union_of_individual_deployment_square_regions",
    )


def border_coordinates(region, image_size_wh):
    x0, y0, x1, y1 = checked_region(region, image_size_wh)
    w, h = image_size_wh
    return [(x0 + 0.01) / w, (x1 + 0.01) / w, (h - y1 + 0.01) / h, (h - y0 + 0.01) / h]
