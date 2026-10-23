"""Validate a run and export one portable MAMMA observation inventory."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .mamma_crop import mask_native_support


def _read(path):
    return json.loads(Path(path).read_text())


def _ids(values):
    ids = [int(v) for v in values]
    if not ids or any(v <= 0 for v in ids) or len(ids) != len(set(ids)):
        raise ValueError('invalid_person_ids')
    return ids


def _relative(path, run):
    return os.path.relpath(Path(path).resolve(),run)


def export_existing(run):
    """Validate selected frames, all cameras and all targets; never rewrite pixels."""
    run = Path(run).resolve()
    config = _read(run/'run_config.json')
    recipe = _read(run/'recipe.json')
    spec = _read(run/'multirig.json')
    frames = config['selected_frames']
    if not frames or frames != sorted(set(frames)) or any(type(k) is not int or k < 0 or k >= recipe['frame_count'] for k in frames):
        raise ValueError('invalid_selected_frames')
    people = recipe['people']
    pids = _ids([p['person_id'] for p in people])
    rigs = spec['rigs']
    rig_ids = [r['rig_id'] for r in rigs]
    if not rig_ids or len(rig_ids) != len(set(rig_ids)):
        raise ValueError('invalid_rig_ids')
    rows = []
    cameras_seen = set()
    template_order = None
    template_name = None
    for rig in rigs:
        rid = rig['rig_id']
        root = Path(rig['root']).resolve()
        if root!=run/'rigs'/rid:raise ValueError('rig_root_conflict')
        rig_info=_read(root/'rig.json')
        if rig_info['rig_id']!=rid:raise ValueError('rig_identity_conflict')
        camera_ids = [c['id'] for c in rig_info['cameras']]
        if not camera_ids or len(camera_ids) != len(set(camera_ids)):
            raise ValueError('invalid_camera_ids')
        for frame in frames:
            for camera in camera_ids:
                cameras_seen.add(camera)
                folder = root/f'frame_{frame:03d}'/camera
                raster = _read(folder/'raster_storage.json')
                rgb_path = folder/'roi.png'
                label_path = folder/'mamma_512.npz'
                annotation_path = folder/'annotations.npz'
                if not all(p.is_file() for p in (rgb_path,label_path,annotation_path)):
                    raise ValueError('missing_required_camera_product')
                with np.load(label_path,allow_pickle=False) as gt, np.load(annotation_path,allow_pickle=False) as ann:
                    if (str(gt['schema'])!='mamma_512_label_v1' or
                        _ids(gt['person_ids']) != pids or _ids(ann['person_ids']) != pids or
                        int(gt['frame_index']) != frame or int(ann['frame_index']) != frame or
                        str(gt['camera_id']) != camera or str(gt['rig_id']) != rid or
                        str(gt['scene_episode_id']) != recipe['scene_episode_id'] or
                        str(ann['scene_episode_id']) != recipe['scene_episode_id'] or
                        str(ann['rig_id']) != rid or int(gt['time_tick']) != int(ann['time_tick']) or
                        not np.array_equal(gt['time_base'],ann['time_base'])):
                        raise ValueError('label_identity_time_conflict')
                    if (not np.array_equal(gt['asset_frame_index'],ann['person_asset_frame']) or
                        not np.array_equal(gt['source_frame_index'],ann['person_source_frame']) or
                        not np.array_equal(gt['render_coverage_xyxy'],raster['roi_xyxy']) or
                        not np.array_equal(gt['full_image_size_wh'],raster['full_image_size_wh'])):
                        raise ValueError('label_source_or_roi_conflict')
                    if (gt['xy_full_erp'].shape != (len(pids),512,2) or
                        gt['visibility_value'].shape != (len(pids),512) or
                        gt['ordered_vertex_ids'].shape != (512,) or
                        gt['geometry_defined'].shape != (len(pids),512) or
                        gt['physical_domain'].shape != (len(pids),512) or
                        gt['render_coverage'].shape != (len(pids),512) or
                        not np.isfinite(gt['xy_full_erp']).all() or
                        not np.isin(gt['visibility_value'],[-1,0,1]).all()):
                        raise ValueError('invalid_mamma_512_label')
                    if any(name in gt for name in
                           ('visibility_known','contact_value','contact_known',
                            'floor_contact_value','floor_contact_known','sidecar_policy')):
                        raise ValueError('obsolete_label_fields')
                    vertex_ids = gt['ordered_vertex_ids'].astype(int).tolist()
                    if (len(set(vertex_ids))!=512 or any(v<0 or v>=10475 for v in vertex_ids)):
                        raise ValueError('invalid_template_point_order')
                    if template_order is None:template_order=vertex_ids
                    elif vertex_ids!=template_order:raise ValueError('template_point_order_conflict')
                    template_id = str(gt['template_id'])
                    if template_id!='MAMMA_verts_512_ordered_v1':
                        raise ValueError('unsupported_template_id')
                    if template_name is None:template_name=template_id
                    elif template_id!=template_name:raise ValueError('template_id_conflict')
                    time_tick = int(gt['time_tick'])
                    time_base = gt['time_base'].astype(int).tolist()
                    asset_frames = gt['asset_frame_index'].astype(int).tolist()
                    source_frames = gt['source_frame_index'].astype(int).tolist()
                with Image.open(rgb_path) as image:
                    if list(image.size) != raster['stored_size_wh']:
                        raise ValueError('raster_image_size_conflict')
                for index, person in enumerate(people):
                    pid = pids[index]
                    mask_path = folder/f'person_{pid}_mask.png'
                    if not mask_path.is_file():
                        raise ValueError('missing_person_mask')
                    mask = np.asarray(Image.open(mask_path).convert('L'))
                    support = mask_native_support(mask,raster['roi_xyxy'],raster['full_image_size_wh'])
                    rows.append(dict(
                        observation_id=f"{recipe['scene_episode_id']}__{rid}__f{frame:06d}__{camera}__p{pid}",
                        episode=recipe['scene_episode_id'],rig=rid,frame_index=frame,camera_id=camera,
                        person_id=pid,person_index=index,status='observed' if support is not None else 'absent',
                        absent_reason=None if support is not None else 'empty_target_mask',
                        time_tick=time_tick,time_base=time_base,
                        asset_frame_index=asset_frames[index],source_frame_index=source_frames[index],
                        source_asset_id=person['asset_id'],
                        source_manifest_id=person.get('source_manifest_id',person['asset_id']),
                        full_image_size_wh=raster['full_image_size_wh'],roi_xyxy=raster['roi_xyxy'],
                        rgb=_relative(rgb_path,run),mask=_relative(mask_path,run),
                        labels=_relative(label_path,run),annotations=_relative(annotation_path,run),
                        crop_support=support))
    keys = [(r['episode'],r['rig'],r['frame_index'],r['camera_id'],r['person_id']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate_observation')
    return dict(schema='mamma_bundle_v1',episode=recipe['scene_episode_id'],
                frame_count=recipe['frame_count'],fps=recipe['fps'],
                scope='complete' if frames==list(range(recipe['frame_count'])) else 'selected',
                frames=frames,rig_ids=rig_ids,camera_ids=sorted(cameras_seen),
                template_id=template_name,ordered_vertex_ids=template_order,observations=rows)
