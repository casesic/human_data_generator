"""Evidence-only neighbor occlusion diagnostics; never a supervision filter."""

import numpy as np


def three_stage_event(ratios, *, delta=0.15, consecutive=3):
    """First ordered, disjoint windows with a conservative rise and recovery.

    Every value in the high window exceeds every value in both low windows by
    delta. Missing denominators invalidate a window; difficult times stay in it.
    """
    values = np.array([np.nan if x is None else x for x in ratios], float)
    windows = [values[i : i + consecutive] for i in range(len(values) - consecutive + 1)]
    for middle in range(consecutive, len(windows) - consecutive):
        high = windows[middle]
        if not np.isfinite(high).all():
            continue
        before = next(
            (
                i
                for i in range(middle - consecutive + 1)
                if np.isfinite(windows[i]).all() and high.min() - windows[i].max() >= delta
            ),
            None,
        )
        if before is None:
            continue
        after = next(
            (
                i
                for i in range(middle + consecutive, len(windows))
                if np.isfinite(windows[i]).all() and high.min() - windows[i].max() >= delta
            ),
            None,
        )
        if after is not None:
            return dict(
                before=list(range(before, before + consecutive)),
                during=list(range(middle, middle + consecutive)),
                after=list(range(after, after + consecutive)),
                rise=float(high.min() - windows[before].max()),
                recovery=float(high.min() - windows[after].max()),
            )
    return None


def point_occlusion_evidence(
    xy,
    distances,
    visible,
    in_view,
    point_person_ids,
    known_person_ids,
    depth,
    instance,
    *,
    strict_coverage=True,
):
    """Evidence codes for arbitrary flattened points and explicit owners."""
    xy = np.asarray(xy)
    distances = np.asarray(distances)
    visible = np.asarray(visible)
    in_view = np.asarray(in_view)
    owners = np.asarray(point_person_ids)
    shape = distances.shape
    if (
        xy.shape != shape + (2,)
        or visible.shape != shape
        or in_view.shape != shape
        or owners.shape != shape
    ):
        raise ValueError("evidence_shape_conflict")
    codes = np.full(shape, 5, np.uint8)
    codes[~in_view] = 0
    codes[visible] = 1
    front_id = np.full(shape, -1, np.int32)
    front_range = np.full(shape, np.nan, np.float32)
    agreement = np.zeros(shape, np.uint8)
    gap = np.full(shape, np.nan, np.float32)
    support = np.zeros(shape, bool)
    known = set(known_person_ids)
    h, w = depth.shape
    for k, (u, v) in enumerate(xy):
        if not np.isfinite([u, v]).all():
            if strict_coverage:
                raise ValueError("evidence_neighborhood_outside_raster")
            codes[k] = 6
            continue
        x, y = int(np.floor(u)) % w, int(np.floor(v))
        if y < 1 or y + 1 >= h:
            if strict_coverage:
                raise ValueError("evidence_neighborhood_outside_raster")
            codes[k] = 6
            continue
        columns = (np.arange(x - 1, x + 2) % w)
        rows = np.arange(y - 1, y + 2)
        d = depth[np.ix_(rows, columns)]
        ids = instance[np.ix_(rows, columns)]
        if not np.isfinite(d).all() or not np.isfinite(ids).all():
            if strict_coverage:
                raise ValueError("evidence_neighborhood_unobserved")
            codes[k] = 6
            continue
        support[k] = True
        owner = int(np.rint(ids[1, 1]))
        front_id[k] = owner
        front_range[k] = d[1, 1]
        gap[k] = distances[k] - d[1, 1]
        hits = (np.rint(ids) == owner) & (d > 0) & (distances[k] - d > 0.04)
        agreement[k] = hits.sum()
        if not in_view[k] or visible[k] or not hits[1, 1] or hits.sum() < 5:
            continue
        codes[k] = 3 if owner == owners[k] else (2 if owner in known else 4)
    return dict(
        occlusion_evidence_code=codes,
        front_instance_id=front_id,
        front_range_m=front_range,
        gt_minus_front_range_m=gap,
        nearer_same_front_id_3x3_count=agreement,
        evidence_neighborhood_complete=support,
    )


def occlusion_evidence(xy, distances, visible, in_view, person_ids, depth, instance):
    """Conservative central-ray + 3x3 ID/depth agreement within existing 5x5.

    Codes: 0 outside, 1 visible, 2 neighbor, 3 own-instance foreground,
    4 scene foreground, 5 unresolved. Own instance includes clothing/groom.
    A nearer hit must beat GT range by >4cm (historical visibility tolerance).
    Center and >=5/9 neighbors must agree on ID and nearer range. This evidence
    is deliberately incomplete near contours; unresolved is not self-occlusion.
    """
    xy = np.asarray(xy)
    shape = xy.shape[:-1]
    if shape != np.asarray(distances).shape or shape != (len(person_ids), 138):
        raise ValueError("evidence_shape_conflict")
    flat = point_occlusion_evidence(
        xy.reshape(-1, 2),
        np.asarray(distances).reshape(-1),
        np.asarray(visible).reshape(-1),
        np.asarray(in_view).reshape(-1),
        np.repeat(person_ids, 138),
        person_ids,
        depth,
        instance,
        strict_coverage=True,
    )
    # Preserve the exact legacy fields and shapes; the new support field is
    # intentionally not added to historical annotations.
    result = {k: v.reshape(shape) for k, v in flat.items() if k != "evidence_neighborhood_complete"}
    result["front_range_m"] = np.nan_to_num(result["front_range_m"], nan=0.0)
    result["gt_minus_front_range_m"] = np.nan_to_num(result["gt_minus_front_range_m"], nan=0.0)
    return result
