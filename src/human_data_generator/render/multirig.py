"""One scene update per frame, followed by a frozen list of paired ROI rigs."""
import json
import os
from pathlib import Path
import resource
import sys
import time

import bpy
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from human_data_generator.config import PROJECT_ROOT
ROOT=PROJECT_ROOT
from human_data_generator.camera.erp import project_erp
from human_data_generator.data.dynamic_scene import frame_record
from human_data_generator.imaging.output_modes import prepare_source_workspace
from human_data_generator.runtime.frame_consumer import FrameConsumer
from human_data_generator.render import assembly
from human_data_generator.render.roi import render_rig_roi,scene_settings,write
from human_data_generator.runtime.task_paths import run_config, task_path


def main(root=None, *, reuse_session=False):
    root=Path(sys.argv[sys.argv.index('--root')+1] if root is None else root).resolve()
    config=run_config(root)
    recipe=json.loads(task_path(root,'recipe.json').read_text())
    rigs=[]
    for rig_id in config['rig_ids']:
        rig_root=task_path(root,f'rigs/{rig_id}')
        rig_recipe=dict(recipe,rig_id=rig_id,roi_planning_policy='per_person_square_union_v1')
        rig_info=json.loads(task_path(root,f'rigs/{rig_id}/rig.json').read_text())
        if rig_info['rig_id']!=rig_id:raise ValueError('rig_identity_conflict')
        rigs.append((rig_id,rig_root,rig_recipe,rig_info['cameras']))
    planned={rig_id:json.loads((rig_root/'planned_regions.json').read_text())
             for rig_id,rig_root,_,_ in rigs if (rig_root/'planned_regions.json').exists()}
    if not 1 <= len(rigs) <= 8 or len({r[0] for r in rigs})!=len(rigs):
        raise ValueError('invalid_rig_list')
    if not 1<=len(recipe['people'])<=5 or len({p['person_id'] for p in recipe['people']})!=len(recipe['people']):
        raise ValueError('invalid_shared_person_axis')
    for _,_,rig_recipe,cameras in rigs:
        if [c['id'] for c in cameras] != ['cam2','cam4']:
            raise ValueError('paired_camera_identity_conflict')
    start=time.monotonic();calls=0;updates=[];rendered={};signature=None;hair_cache={}
    ids={p['person_id'] for p in recipe['people']}
    def append(value):
        with (root/'render_calls.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    worker=FrameConsumer([config['consumer_python'],'-m','human_data_generator.consumer',
                          'serve','--root',str(root)],env=env,cwd=ROOT,log_path=root/'consumer_worker.log',
                         deadline=config['deadline_monotonic'])
    write(root/'consumer_worker_start.json',dict(pid=worker.process.pid,startup_seconds=worker.startup_seconds))
    def frame_source_dir(frame):
        for _,rig_root,_,_ in rigs:
            prepare_source_workspace(rig_root/f'frame_{frame:03d}','compact',
                                     task_root=root,task_id=config['task_id'])
        return rigs[0][1]/f'frame_{frame:03d}'/'_temporary_ds_source/source'
    def render(scene,cam,node,folder,*,frame_index):
        nonlocal calls,signature
        frame=frame_index
        if scene_settings(scene)!=config['scene_settings']:
            raise ValueError('scene_render_settings_changed')
        objects=[(o.name,o.as_pointer(),o.data.as_pointer() if o.data else None) for o in scene.objects]
        if signature is None:
            signature=objects
            write(root/'single_process_setup.json',dict(pid=os.getpid(),scene_load_count=1,
                object_signature=objects,setup_callback_seconds=time.monotonic()-start,
                persistent_data=bool(scene.render.use_persistent_data)))
        elif signature!=objects:
            raise ValueError('scene_or_person_objects_replaced')
        world_parts={};owners={}
        for obj in scene.objects:
            if obj.pass_index not in ids or obj.hide_render:continue
            if obj.type=='MESH':
                points=np.empty((len(obj.data.vertices),3),np.float32)
                obj.data.vertices.foreach_get('co',points.ravel())
            elif obj.type=='CURVES':
                if obj.name not in hair_cache:
                    points=np.empty((len(obj.data.points),3),np.float32)
                    obj.data.position_data.foreach_get('vector',points.ravel())
                    hair_cache[obj.name]=points
                points=hair_cache[obj.name]
            else:raise ValueError('unsupported_person_geometry')
            M=np.asarray(obj.matrix_world)
            world_parts[obj.name]=points@M[:3,:3].T+M[:3,3]
            owners[obj.name]=int(obj.pass_index)
        updates.append(dict(frame_index=frame,person_geometry_collections=1,world_part_names=sorted(world_parts)))
        rendered[frame]={}
        for rig_id,rig_root,rig_recipe,cameras in rigs:
            rig_folder=rig_root/f'frame_{frame:03d}'/'_temporary_ds_source/source'
            def call_started(frame_index,camera_id,kind,roi,resolution,samples,*,completed_call=None,seconds=None):
                nonlocal calls
                if completed_call is not None:
                    append(dict(event='completed',call=completed_call,frame_index=frame_index,rig_id=rig_id,
                                camera_id=camera_id,kind=kind,seconds=seconds,
                                rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                    return completed_call
                if calls>=config['max_camera_calls'] or time.monotonic()>=config['deadline_monotonic']:
                    raise RuntimeError('multirig_camera_or_time_budget_exhausted')
                calls+=1
                append(dict(event='started',call=calls,frame_index=frame_index,rig_id=rig_id,
                            camera_id=camera_id,kind=kind,roi_xyxy=roi,resolution=resolution,samples=samples))
                return calls
            result=render_rig_roi(scene,cam,node,rig_folder,root,rig_recipe,cameras,
                                  world_parts,owners,config,frame,call_started)
            if rig_id in planned:
                actual=json.loads((rig_folder/'roi_plan.json').read_text())['cameras']
                for camera in cameras:
                    c=camera['id']
                    if actual[c]['roi_xyxy']!=planned[rig_id][str(frame)][c]['roi_xyxy']:
                        raise ValueError(f'preflight_live_ROI_conflict:{frame}:{rig_id}:{c}')
            rendered[frame][rig_id]=result
            write(root/'render_progress.json',dict(render_calls=calls,geometry_updates=updates,
                  completed_render_units=[dict(frame_index=k,rig_id=r) for k,v in rendered.items() for r in v]))
        return [record for rig_id,_,_,_ in rigs for record in rendered[frame][rig_id]['records']]
    def consume(frame,folder,manifest):
        for rig_id,rig_root,rig_recipe,cameras in rigs:
            source=rig_root/f'frame_{frame:03d}'/'_temporary_ds_source/source'
            child=dict(manifest,sources=rendered[frame][rig_id]['records'],episode_recipe='recipe.json',
                       scene_episode_id=recipe['scene_episode_id'],rig_id=rig_id)
            write(source/'source_manifest.json',child)
            response=worker.consume(frame,rig_id)
            write(rig_root/f'frame_{frame:03d}'/'consumer_process.json',
                  dict(mode='persistent_multirig',rig_id=rig_id,worker_response=response))
            write(root/'unit_progress.json',dict(completed_units=[dict(frame_index=r['frame'],rig_id=r['rig_id'])
                                                        for r in worker.responses]))
    try:
        assembly.main(root=root,render_cameras_fn=render,frame_complete_fn=consume,
                      frame_source_dir_fn=frame_source_dir,reuse_session=reuse_session)
        write(root/'consumer_worker_result.json',worker.close())
    except BaseException:
        worker.abort()
        raise
    scene_reused=json.loads((root/'assembly_result.json').read_text()).get('scene_reused',False)
    write(root/'multirig_render_result.json',dict(scene_load_count=0 if scene_reused else 1,scene_reused=scene_reused,
        normal_path_geometry_updates=len(updates),
        camera_calls=calls,rig_order=[r[0] for r in rigs],consumer_process_count=1,
        elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
