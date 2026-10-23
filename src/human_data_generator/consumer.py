"""MAMMA ROI consumer: RGB, target masks, ordered 512 GT and light metadata."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .data.dynamic_scene import frame_record
from .data.mamma_crop import mask_native_support
from .data.mamma_labels import load_world_body, build_evidence, policy_record
from .data.raster_storage import roi_metadata, save_roi_observation
from .imaging.output_modes import complete_output
from .imaging.regions import embed_region, apply_physical_mask
from .imaging.source import read_ds_source
from .runtime.frame_consumer import serve_frames
from .runtime.task_paths import run_config, task_path


def _read(path):return json.loads(Path(path).read_text())

def _write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def _recipe(root):
    root=Path(root).resolve();task_root=root.parent.parent
    recipe=_read(task_path(task_root,'recipe.json'))
    recipe['rig_id']=root.name
    return recipe

def _config(root):
    root=Path(root).resolve();task_root=root.parent.parent
    config=run_config(task_root)
    if root.name not in config['rig_ids'] or root!=task_path(task_root,f'rigs/{root.name}'):
        raise ValueError('consumer_rig_not_in_task')
    config['valid_masks']=_read(task_path(task_root,f'rigs/{root.name}/rig.json'))['valid_masks']
    return config


def _annotation(recipe, geometry, frame, camera, pose):
    timing=geometry['timing'];records=timing.get('person_records')
    if records is None:
        records=[dict(asset_frame=int(geometry['asset_frames'][j]),source_frame=int(geometry['source_frames'][j]),
            source_time_tick=int(timing.get('source_time_tick',frame)),source_time_base=timing['time_base'])
            for j in range(len(recipe['people']))]
    return dict(person_ids=geometry['person_ids'],camera_to_world_erp=np.asarray(pose,dtype=np.float64),
        frame_index=np.array(frame,np.int64),time_tick=np.array(timing['episode_time_tick'],np.int64),
        time_base=np.asarray(timing['time_base'],np.int64),
        person_asset_frame=geometry['asset_frames'],person_source_frame=geometry['source_frames'],
        person_source_time_tick=np.asarray([r['source_time_tick'] for r in records],np.int64),
        person_source_time_base=np.asarray([r['source_time_base'] for r in records],np.int64),
        scene_episode_id=np.array(recipe['scene_episode_id']),rig_id=np.array(recipe['rig_id']))


def consume_camera_arrays(root,frame,camera,local_rgb,local_range,local_instance,roi,pose,valid_full,geometry):
    """The same array boundary serves actual EXR input and tiny CPU fixtures."""
    root=Path(root).resolve();recipe=_recipe(root);config=_config(root)
    contract=config.get('generation_contract')
    if contract not in ('mamma_512_v1','preview_rgb_v1'):raise ValueError('unsupported_generation_contract')
    ids=[p['person_id'] for p in recipe['people']]
    if geometry['person_ids'].tolist()!=ids:raise ValueError('mamma_person_identity_conflict')
    size=tuple(config['full_image_size_wh']);x0,y0,x1,y1=roi
    if local_rgb.shape!=(y1-y0,x1-x0,3) or local_range.shape!=(y1-y0,x1-x0) or local_instance.shape!=local_range.shape:
        raise ValueError('source_roi_shape_conflict')
    if valid_full.shape!=(size[1],size[0]):raise ValueError('physical_mask_shape_conflict')
    evidence=None
    if contract=='mamma_512_v1':
        depth=embed_region(local_range,roi,size,unknown_value=np.nan)
        instance=embed_region(local_instance,roi,size,unknown_value=np.nan)
        evidence=build_evidence(geometry['world'],geometry['full_body'],geometry['faces'],ids,
            np.asarray(pose),camera,valid_full,depth,instance,roi)
        evidence.update(template_id=np.array(config['mamma_512_label']['template_id']),
            ordered_vertex_ids=geometry['vertex_ids'],asset_frame_index=geometry['asset_frames'],
            source_frame_index=geometry['source_frames'],frame_index=np.array(frame,np.int64),
            time_tick=np.array(geometry['timing']['episode_time_tick'],np.int64),
            time_base=np.asarray(geometry['timing']['time_base'],np.int64),
            render_coverage_xyxy=np.asarray(roi,np.int32),full_image_size_wh=np.asarray(size,np.int32),
            camera_id=np.array(camera),rig_id=np.array(recipe['rig_id']),
            scene_episode_id=np.array(recipe['scene_episode_id']))
        evidence.pop('visibility_known',None)
    dest=root/f'frame_{frame:03d}'/camera;dest.mkdir(exist_ok=False)
    rgb=apply_physical_mask(local_rgb,valid_full[y0:y1,x0:x1])
    meta=roi_metadata(roi,size,Path(config['valid_masks'][camera]).resolve())
    save_roi_observation(dest,rgb,local_instance,_annotation(recipe,geometry,frame,camera,pose),
        valid_full[y0:y1,x0:x1],meta)
    if evidence is not None:np.savez(dest/'mamma_512.npz',**evidence)
    checks=[]
    for pid in ids:
        mask=np.asarray(Image.open(dest/f'person_{pid}_mask.png'))
        crop=mask_native_support(mask,roi,size)
        checks.append(dict(person_id=pid,status='observed' if crop is not None else 'person_absent',
                           reason='mask_has_target' if crop is not None else 'empty_target_mask',
                           source_sampling_supported=None if crop is None else crop['source_sampling_supported']))
    return dict(camera_id=camera,frame_index=frame,people=checks,policy=contract)


def consume_frame(root,frame):
    root=Path(root).resolve();recipe=_recipe(root);config=_config(root)
    contract=config.get('generation_contract')
    if contract not in ('mamma_512_v1','preview_rgb_v1'):raise ValueError('unsupported_generation_contract')
    source=root/f'frame_{frame:03d}'/'_temporary_ds_source/source'
    manifest=_read(source/'source_manifest.json');plan=_read(source/'roi_plan.json')
    timing=frame_record(recipe,frame)
    if manifest['selected_time_record']!=timing:raise ValueError('render_consumer_time_conflict')
    if contract=='mamma_512_v1':
        label_cfg=config['mamma_512_label'];geometry=load_world_body(recipe,root.resolve().parent.parent,frame,label_cfg['template_path'])
        policy_path=root/'label_policy.json';policy=policy_record()
        if policy_path.exists():
            if _read(policy_path)!=policy:raise ValueError('label_policy_changed')
        else:_write(policy_path,policy)
    else:
        records=timing.get('person_records')
        geometry=dict(person_ids=np.asarray([p['person_id'] for p in recipe['people']],np.int32),
            asset_frames=np.asarray([r['asset_frame'] for r in records],np.int64),
            source_frames=np.asarray([r['source_frame'] for r in records],np.int64),timing=timing)
    reports=[]
    for camera,info in plan['cameras'].items():
        roi=info['roi_xyxy'];local=read_ds_source(source/'roi',camera)
        valid=np.asarray(Image.open(config['valid_masks'][camera]))>0
        reports.append(consume_camera_arrays(root,frame,camera,local.rgb,local.ray_range_m,
            local.instance,roi,info['camera_to_world'],valid,geometry))
    _write(root/f'frame_{frame:03d}'/'report.json',dict(frame_index=frame,cameras=reports,status='passed'))
    complete_output(root/f'frame_{frame:03d}','compact',source,
        task_root=root.resolve().parent.parent,task_id=config['task_id'],
        checks=dict(cameras=len(reports),people=len(recipe['people']),mamma_512=contract=='mamma_512_v1'))
    return reports


def finish(root):
    """Check selected camera products; video is a separate optional stage."""
    root=Path(root).resolve();recipe=_recipe(root);config=_config(root)
    frames=config['selected_frames'];cameras=[c['id'] for c in _read(root/'rig.json')['cameras']]
    for frame in frames:
        for camera in cameras:
            dest=root/f'frame_{frame:03d}'/camera
            if not all((dest/name).is_file() for name in ('roi.png','annotations.npz','mamma_512.npz','raster_storage.json')):
                raise ValueError('incomplete_mamma_camera_output')
            with np.load(dest/'mamma_512.npz',allow_pickle=False) as gt:
                if int(gt['frame_index'])!=frame or str(gt['camera_id'])!=camera or gt['xy_full_erp'].shape!=(len(recipe['people']),512,2):
                    raise ValueError('mamma_finish_identity_conflict')
    _write(root/'finish.json',dict(status='passed',frames=frames,cameras=cameras,contract='mamma_512_v1'))


def make_video(root):
    root=Path(root).resolve();recipe=_recipe(root);config=_config(root)
    frames=config['selected_frames'];cameras=[c['id'] for c in _read(root/'rig.json')['cameras']]
    if not (root/'finish.json').is_file():raise ValueError('data_finish_required_before_video')
    if len(cameras)!=2:raise ValueError('paired_cameras_required_for_video')
    import av
    from fractions import Fraction
    pack=root/'pack';pack.mkdir(exist_ok=False)
    video=av.open(str(pack/'dual_roi.mp4'),'w');stream=video.add_stream('libx264',rate=recipe['fps'])
    stream.width=1440;stream.height=360;stream.pix_fmt='yuv420p';stream.options={'crf':'24'}
    for frame in frames:
        canvas=Image.new('RGB',(1440,360))
        for i,camera in enumerate(cameras):
            with Image.open(root/f'frame_{frame:03d}'/camera/'roi.png') as image:
                canvas.paste(image.convert('RGB').resize((720,360)),(720*i,0))
        sample=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24');sample.pts=frame;sample.time_base=Fraction(1,recipe['fps'])
        for packet in stream.encode(sample):video.mux(packet)
    for packet in stream.encode():video.mux(packet)
    video.close()
    _write(pack/'finish.json',dict(status='passed',frames=len(frames),cameras=cameras,contract='mamma_512_v1'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['frame','serve','finish','video'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--frame',type=int)
    args=parser.parse_args()
    if args.stage=='serve':raise SystemExit(serve_frames(args.root,consume_frame))
    if args.stage=='frame':
        if args.frame is None:parser.error('--frame required')
        consume_frame(args.root,args.frame)
    elif args.stage=='finish':finish(args.root)
    else:make_video(args.root)

if __name__=='__main__':main()
