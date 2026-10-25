"""Conservative continuous-frame ROI in the unchanged full ERP grid (pixels)."""

import numpy as np
from .regions import checked_region


def plan_native_crop_region(projected_parts, image_size_wh, *, margin_px=32):
    """Cover geometry and any deployment square of a sub-box within its envelope.

    Visible instance boxes can shift as the calibrated lens cuts the silhouette.
    For envelope width W/height H, any contained box's square lies within
    [xmin-H/2, ymin-W/2, xmax+H/2, ymax+W/2]. Two raster pixels expand the
    geometry bounds first; the final margin also protects visibility/filtering.
    X is periodic. A crop needing pixels on both sides of the seam uses a
    full-width render strip; the target is unfolded only by the reader.
    Y is bounded by the ERP raster. The actual mask crop is checked against
    the rendered support by the consumer, so an over-conservative envelope
    need not reject a crop that fits the raster.
    """
    if not np.isfinite(margin_px) or margin_px < 32:
        raise ValueError("ROI_margin_below_32_pixels")
    bounds = {}
    width = int(image_size_wh[0])
    all_x = []
    for name, xy in projected_parts.items():
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
            raise ValueError("invalid_projected_geometry")
        all_x.append(np.mod(xy[:, 0], width))
    if not all_x:
        raise ValueError("empty_geometry")
    columns = np.sort(np.concatenate(all_x))
    gaps = np.diff(np.r_[columns, columns[0] + width])
    origin = float(columns[(int(np.argmax(gaps)) + 1) % len(columns)])
    for name, xy in projected_parts.items():
        unwrapped_x = np.mod(np.mod(xy[:, 0], width) - origin, width) + origin
        unwrapped = np.column_stack((unwrapped_x, xy[:, 1]))
        bounds[name] = np.r_[unwrapped.min(0), unwrapped.max(0)].tolist()
    b = np.array(list(bounds.values()))
    lo = b[:, :2].min(0) - 2
    hi = b[:, 2:].max(0) + 2
    extent = hi - lo
    pad = extent[::-1] / 2
    roi = np.r_[np.floor(lo - pad - margin_px), np.ceil(hi + pad + margin_px)].astype(int)
    if roi[0] < 0 or roi[2] > width:
        roi[0], roi[2] = 0, width
    roi[1] = max(0, roi[1])
    roi[3] = min(int(image_size_wh[1]), roi[3])
    checked_region(roi, image_size_wh)
    result = dict(
        roi_xyxy=roi.tolist(),
        geometry_bounds_full_erp=bounds,
        geometry_raster_allowance_px=2,
        margin_final_erp_px=margin_px,
        periodic_x_origin_erp_px=origin,
        planning_method="conservative envelope of all deployment squares of contained instance boxes",
    )
    return result


def plan_person_crop_union(projected_parts_by_person, image_size_wh, *, margin_px=32):
    """One render region covering each person's complete native square crop.

    The space between separate people does not enlarge either person's square.
    Every individual region is bounded by the full ERP raster before the union.
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
    return [
        0.0 if x0 == 0 else (x0 + 0.01) / w,
        1.0 if x1 == w else (x1 + 0.01) / w,
        (h - y1 + 0.01) / h,
        (h - y0 + 0.01) / h,
    ]
