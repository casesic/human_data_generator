"""Render one paired ERP ROI rig from the shared in-memory scene."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import resource
import bpy
import numpy as np
from human_data_generator.config import PROJECT_ROOT
ROOT=PROJECT_ROOT
from human_data_generator.render import assembly
def scene_settings(s):
    return dict(view_transform=s.view_settings.view_transform,look=s.view_settings.look,exposure=s.view_settings.exposure,gamma=s.view_settings.gamma,
        display_device=s.display_settings.display_device,view_use_curve_mapping=s.view_settings.use_curve_mapping,
        world=s.world.name if s.world else None,
        bounces={k:getattr(s.cycles,k) for k in ['max_bounces','diffuse_bounces','glossy_bounces','transmission_bounces','volume_bounces','transparent_max_bounces']},
        png_color_mode=s.render.image_settings.color_mode,png_color_depth=s.render.image_settings.color_depth,png_compression=s.render.image_settings.compression,
        film_transparent=s.render.film_transparent,filter_type=s.cycles.pixel_filter_type,filter_width=s.cycles.filter_width)

from human_data_generator.rendering.blender.assembly import render_cameras
from human_data_generator.camera.erp import project_erp
from human_data_generator.imaging.roi_planning import plan_person_crop_union,border_coordinates
from human_data_generator.runtime.frame_consumer import FrameConsumer


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def render_rig_roi(scene,cam,node,folder,rig_root,recipe,cameras,world_parts,part_owners,config,
                   frame,call_started):
    """Render one rig from the *already updated* scene geometry."""
    begin=time.monotonic();ids={p['person_id'] for p in recipe['people']};plans={}
    for camera in cameras:
        pose=np.array(camera['camera_to_world']);parts={n:project_erp(p,pose,5760,2880)[0] for n,p in world_parts.items()}
        grouped={str(pid):{name:xy for name,xy in parts.items() if part_owners[name]==pid} for pid in ids}
        region=plan_person_crop_union(grouped,(5760,2880))
        plans[camera['id']]=dict(region,camera_to_world=camera['camera_to_world'])
        if 'max_union_roi_pixels' in recipe:
            x0,y0,x1,y1=region['roi_xyxy']
            if (x1-x0)*(y1-y0)>recipe['max_union_roi_pixels']:raise ValueError('union_ROI_pixel_budget')
    plan_seconds=time.monotonic()-begin
    write(folder/'roi_plan.json',dict(frame_index=frame,rig_id=recipe.get('rig_id'),cameras=plans,planning_seconds=plan_seconds))
    records=[];tasks=[('roi',5760,2880,64)] if config.get('storage_mode')=='roi_only' else [('background',1440,720,8),('roi',5760,2880,64)]
    for kind,w,h,samples in tasks:
        dest=folder/kind;dest.mkdir(exist_ok=False)
        for camera in cameras:
            roi=plans[camera['id']]['roi_xyxy'] if kind=='roi' else None
            scene.render.resolution_x=w;scene.render.resolution_y=h;scene.render.resolution_percentage=100;scene.cycles.samples=samples
            scene.render.use_border=roi is not None;scene.render.use_crop_to_border=roi is not None
            if roi:
                coords=border_coordinates(roi,(w,h))
                scene.render.border_min_x,scene.render.border_max_x,scene.render.border_min_y,scene.render.border_max_y=coords
            if hasattr(node,'directory'):node.directory=str(dest)
            else:node.base_path=str(dest)
            call=call_started(frame,camera['id'],kind,roi,[w,h],samples)
            t=time.monotonic();record=render_cameras(scene,cam,node,rig_root/'camera_configs'/f'{camera["id"]}.json',dest)[0]
            dt=time.monotonic()-t;record.update(kind=kind,full_grid=[w,h],samples=samples,roi_xyxy=roi,seconds=dt)
            records.append(record)
            call_started(frame,camera['id'],kind,roi,[w,h],samples,completed_call=call,seconds=dt)
    return dict(frame_index=frame,rig_id=recipe.get('rig_id'),roi_planning_seconds=plan_seconds,records=records)

