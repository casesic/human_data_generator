"""Historical visibility tests with distinct neighborhood and boundary rules.

ERP depth is Euclidean ray range in meters, with instance IDs and 138-row person
blocks; a periodic 5x5 neighborhood and strict 4cm range difference are used.
The legacy DS instance-only rule rounds pixels, samples 7x7 without seam wrap,
and does not compare depth. These policies intentionally remain separate.
"""

import numpy as np


def erp_visibility_points(
    xy: np.ndarray,
    distance: np.ndarray,
    depth: np.ndarray,
    index: np.ndarray,
    point_person_ids: np.ndarray,
    *,
    tolerance_m: float = 0.04,
    neighborhood_radius_px: int = 2,
):
    """Historical ERP range/ID visibility for any ordered point collection.

    ``point_person_ids`` maps every point to its rendered instance owner.  This
    removes the old implicit 138-row grouping while preserving its 5x5,
    periodic-X, floor-pixel and strict-distance rules by default.
    """
    xy=np.asarray(xy);distance=np.asarray(distance);owners=np.asarray(point_person_ids)
    if (xy.ndim!=2 or xy.shape[1]!=2 or distance.shape!=(len(xy),) or
            owners.shape!=(len(xy),) or depth.shape!=index.shape):
        raise ValueError('visibility_point_shape_conflict')
    if not np.isfinite(tolerance_m) or tolerance_m<=0 or type(neighborhood_radius_px) is not int or neighborhood_radius_px<0:
        raise ValueError('invalid_visibility_policy')
    height,width=depth.shape
    visible=np.zeros(len(xy),dtype=bool)
    valid=np.isfinite(xy).all(axis=1)&(xy[:,1]>=0)&(xy[:,1]<height)
    for j,(x,y) in enumerate(xy):
        if not valid[j]:continue
        ix,iy=int(np.floor(x))%width,int(np.floor(y))
        rows=np.arange(max(0,iy-neighborhood_radius_px),min(height,iy+neighborhood_radius_px+1))
        cols=np.arange(ix-neighborhood_radius_px,ix+neighborhood_radius_px+1)%width
        sample_depth=depth[np.ix_(rows,cols)];sample_index=index[np.ix_(rows,cols)]
        matched=(sample_depth>0)&(np.rint(sample_index)==owners[j])
        visible[j]=bool(matched.any() and np.min(np.abs(sample_depth[matched]-distance[j]))<tolerance_m)
    return valid,visible


def erp_visibility(
    xy: np.ndarray,
    distance: np.ndarray,
    depth: np.ndarray,
    index: np.ndarray,
    person_ids: list[int],
):
    """Return finite/vertical-valid and surface-visible masks for 138-row person blocks.

    depth[H,W] is ray range in meters; index[H,W] holds rounded instance IDs.
    Horizontal samples wrap; vertical neighbors clip. Eligibility is not supervision."""
    if len(xy)!=len(person_ids)*138:raise ValueError('legacy_visibility_requires_138_rows_per_person')
    owners=np.repeat(np.asarray(person_ids),138)
    return erp_visibility_points(xy,distance,depth,index,owners)


def visible_at_instance_points(
    xy: np.ndarray, instance_mask: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Legacy instance-only visibility: rounded center, nonperiodic 7x7 neighborhood.

    valid_mask gates the rounded center; no ray-distance or occlusion-depth test."""
    height, width = instance_mask.shape
    visible = np.zeros(len(xy), dtype=bool)
    for index, (x, y) in enumerate(xy):
        if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
            continue
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < width and 0 <= iy < height) or not valid_mask[iy, ix]:
            continue
        region = instance_mask[max(0, iy - 3) : iy + 4, max(0, ix - 3) : ix + 4]
        visible[index] = bool(region.any())
    return visible
