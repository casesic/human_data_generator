"""Store one RGB/instance ROI and per-person binary target masks."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def roi_metadata(roi,full_size,physical_mask_ref):
    x0,y0,x1,y1=map(int,roi)
    if not 0<=x0<x1<=full_size[0] or not 0<=y0<y1<=full_size[1]:
        raise ValueError('invalid_roi')
    return dict(full_image_size_wh=list(full_size),roi_xyxy=[x0,y0,x1,y1],
                stored_size_wh=[x1-x0,y1-y0],physical_mask_ref=str(physical_mask_ref))


def save_roi_observation(dest,rgb,instance,data,valid,meta):
    dest=Path(dest)
    stored=meta['stored_size_wh']
    if tuple(rgb.shape[:2][::-1])!=tuple(stored) or instance.shape!=valid.shape or valid.shape!=rgb.shape[:2]:
        raise ValueError('ROI_raster_size_conflict')
    Image.fromarray(rgb).save(dest/'roi.png')
    np.savez(dest/'annotations.npz',**data)
    for pid in data['person_ids']:
        mask=(np.rint(instance)==pid)&valid
        Image.fromarray(mask.astype(np.uint8)*255).save(dest/f'person_{pid}_mask.png')
    (dest/'raster_storage.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
