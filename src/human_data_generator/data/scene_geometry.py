"""Shared frozen-person geometry used by rig preflight and the scene editor."""
import json
from pathlib import Path

import numpy as np

from .dynamic_scene import geometry_paths


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def person_geometry(recipe,frames):
    people = []
    for p in recipe['people']:
        asset = json.loads(Path(p['asset_manifest']).read_text())
        paths = geometry_paths(asset)
        indices = {frame:recipe['person_timelines'][str(p['person_id'])][frame]['asset_frame'] for frame in frames}
        if any(index is None for index in indices.values()):raise ValueError('selected_frame_missing_geometry')
        groom = Path(asset['groom_reference'])
        active = np.repeat(np.load(groom / 'active_curve_mask.npy'),
                           np.diff(np.load(groom / 'curve_offsets.npy')))
        hair = np.load(groom / 'points.npy')[active]
        with np.load(asset['parameters']) as parameters:
            orientation = {frame:parameters['global_orient'][index] for frame,index in indices.items()}
        people.append(dict(person=p, asset=asset,
                           body={frame:np.load(paths['body_vertices'],mmap_mode='r')[index] for frame,index in indices.items()},
                           cloth={frame:np.load(paths['cloth_vertices'],mmap_mode='r')[index] for frame,index in indices.items()},
                           head={frame:np.load(paths['head_transforms'],mmap_mode='r')[index] for frame,index in indices.items()},
                           hair=hair, orientation=orientation))
    return people
