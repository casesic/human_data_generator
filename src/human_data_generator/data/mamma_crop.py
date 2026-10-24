"""CPU check for A1's mask_bbox_inclusive_v1 source sampling support."""

import numpy as np
import cv2


def mask_native_support(mask, roi, full_size_wh):
    mask = np.asarray(mask)
    roi = np.asarray(roi, dtype=np.int64)
    full = np.asarray(full_size_wh, dtype=np.int64)
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all():
        raise ValueError("oracle_mask_not_binary")
    if (
        roi.shape != (4,)
        or full.shape != (2,)
        or list(roi[2:] - roi[:2]) != [mask.shape[1], mask.shape[0]]
    ):
        raise ValueError("invalid_roi_size")
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return None
    periodic_x = roi[0] == 0 and roi[2] == full[0]
    if periodic_x:
        columns = np.flatnonzero(np.any(mask > 0, axis=0))
        gaps = np.diff(np.r_[columns, columns[0] + full[0]])
        start = int(columns[(int(np.argmax(gaps)) + 1) % len(columns)])
        end = int(((columns - start) % full[0]).max()) + start
        x, w = start, end - start + 1
    if w == 1 or h == 1:
        raise ValueError("degenerate_person_mask_box")
    box = np.array(
        [roi[0] + x, roi[1] + y, roi[0] + x + w - 1, roi[1] + y + h - 1], dtype=np.float32
    )
    center = (box[:2] + box[2:]) / 2
    size = (box[2:] - box[:2]) * np.float32(1.2)
    if size[0] / size[1] < 384 / 512:
        size[0] = size[1] * np.float32(384 / 512)
    else:
        size[1] = size[0] * np.float32(512 / 384)
    downsampling = size.max() / 384 / 2
    sigma = (downsampling - 1) / 2 if downsampling > 1.1 else 0
    margin = int(np.ceil(4 * sigma)) + 3
    rect = np.r_[
        np.floor(center - size / 2).astype(int) - margin,
        np.ceil(center + size / 2).astype(int) + margin,
    ]
    if (
        np.any(roi[:2] < 0)
        or np.any(roi[2:] > full)
        or rect[1] < roi[1]
        or rect[3] > roi[3]
        or (periodic_x and rect[2] - rect[0] > full[0])
        or (not periodic_x and (rect[0] < roi[0] or rect[2] > roi[2]))
    ):
        raise ValueError("crop_filter_support_outside_roi")
    return dict(
        box_xyxy=box.astype(int).tolist(),
        sampling_rect_erp=rect.tolist(),
        gaussian_sigma=float(sigma),
        filter_margin_px=margin,
        source_sampling_supported=True,
        crop_policy_version="mask_bbox_periodic_v2" if periodic_x else "mask_bbox_inclusive_v1",
    )
