"""Plan, prepare, render and export one current-format MAMMA scene."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from human_data_generator.config import PROJECT_ROOT,resource_path


def _write(path,value):
    path=Path(path)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def _task_path(name,preview=False):
    if not name or not name.replace('_','').isalnum() or len(name)>64:
        raise ValueError('invalid_task_name')
    return PROJECT_ROOT/'outputs'/('previews' if preview else 'runs')/name


def plan(scene_path,frames=None,rigs=None,preview=False):
    from human_data_generator.runtime.scene import read_scene
    scene=read_scene(scene_path)
    selected=list(range(scene['frame_count'])) if frames is None else list(frames)
    rig_ids=[r['rig_id'] for r in scene['rigs'] if rigs is None or r['rig_id'] in rigs]
    if not selected or selected!=sorted(set(selected)) or any(k<0 or k>=scene['frame_count'] for k in selected):
        raise ValueError('invalid_selected_frames')
    if not rig_ids or (rigs is not None and set(rig_ids)!=set(rigs)):
        raise ValueError('invalid_selected_rigs')
    if preview and (len(selected)!=1 or len(rig_ids)!=1):
        raise ValueError('preview_requires_one_frame_and_rig')
    from human_data_generator.editor.core import validate
    check=validate(scene,selected,rig_ids)
    if check['status']!='passed':raise ValueError(check['errors'])
    return dict(scene_id=scene['scene_id'],fps=scene['fps'],frame_count=scene['frame_count'],
                selected_frames=selected,rig_ids=rig_ids,person_ids=[p['person_id'] for p in scene['people']],
                camera_calls=len(selected)*len(rig_ids)*2,
                scope='complete' if selected==list(range(scene['frame_count'])) else 'selected')


def prepare(scene_path,name,frames=None,rigs=None,preview=False):
    from human_data_generator.runtime.multirig_prepare import prepare as prepare_run
    output=_task_path(name,preview)
    return output,prepare_run(output,scene_path=scene_path,frame_indices=frames,
                              selected_rig_ids=rigs,preview=preview)


def _resources(output,gpu_uuid):
    blender=resource_path('blender_executable')
    if not blender.is_file():raise FileNotFoundError(blender)
    queries=dict(gpus=['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv'],
                 compute=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],
                 memory=['free','-h'],disk=['df','-h',str(output),'/tmp'])
    evidence={key:subprocess.check_output(cmd,text=True) for key,cmd in queries.items()}
    gpu=next((row for row in evidence['gpus'].splitlines()[1:] if gpu_uuid in row),None)
    if gpu is None or int(gpu.split(',')[2].split()[0])>512 or int(gpu.split(',')[4].split()[0])>5:
        raise RuntimeError('selected_GPU_unavailable')
    if any(gpu_uuid in row for row in evidence['compute'].splitlines()[1:]):
        raise RuntimeError('other_GPU_compute_process_present')
    return evidence


def render_prepared(output,gpu_uuid,budget_seconds,video=False):
    """Only this explicit call starts Blender and a persistent CPU consumer."""
    from human_data_generator.runtime.roi_run_utils import disk_snapshot
    output=Path(output).resolve()
    config=json.loads((output/'run_config.json').read_text())
    recipe=json.loads((output/'recipe.json').read_text())
    if config['schema']!='human_run_config_v1' or recipe['schema']!='human_run_v1':
        raise ValueError('current_prepared_task_required')
    if any((output/name).exists() for name in ('started.json','render_calls.jsonl','failure.json','observations.json')):
        raise FileExistsError('task_already_started')
    if type(budget_seconds) is not int or budget_seconds<1:
        raise ValueError('positive_budget_required')
    evidence=_resources(output,gpu_uuid)
    start=time.monotonic();deadline=start+budget_seconds
    config.update(gpu_uuid=gpu_uuid,deadline_monotonic=deadline)
    _write(output/'run_config.json',config)
    _write(output/'resources_before.json',evidence)
    _write(output/'started.json',dict(gpu_uuid=gpu_uuid,budget_seconds=budget_seconds))
    env=dict(os.environ,PYTHONPATH=str(PROJECT_ROOT/'src'),CUDA_VISIBLE_DEVICES=gpu_uuid,
             OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
             MPLCONFIGDIR='/tmp/human-data-generator-mpl')
    blender=resource_path('blender_executable')
    command=[str(blender),'-b','-t','4','--python-exit-code','1','--python',
             str(PROJECT_ROOT/'src/human_data_generator/render/multirig.py'),'--','--root',str(output)]
    with (output/'blender.log').open('w') as log:
        process=subprocess.Popen(command,cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        timed_out=False
        while process.poll() is None:
            if time.monotonic()>=deadline:
                process.kill();timed_out=True;break
            time.sleep(1)
        code=process.wait()
    _write(output/'blender_process.json',dict(returncode=code,timed_out=timed_out,
           wall_seconds=time.monotonic()-start))
    if code or timed_out:
        _write(output/'failure.json',dict(stage='render_or_consume',returncode=code,
              timed_out=timed_out,failed_site_preserved=True))
        raise RuntimeError('render_or_consume_failed')
    if config['preview_only']:
        _write(output/'preview_result.json',dict(status='completed',frames=config['selected_frames']))
        return output
    spec=json.loads((output/'multirig.json').read_text())
    for rig in spec['rigs']:
        child=Path(rig['root'])
        with (child/'finish.log').open('w') as log:
            subprocess.run([sys.executable,'-m','human_data_generator.consumer','finish','--root',str(child)],
                           cwd=PROJECT_ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,
                           stderr=subprocess.STDOUT,check=True,timeout=max(.1,deadline-time.monotonic()))
    from human_data_generator.data.mamma_export import export_existing
    manifest=export_existing(output)
    _write(output/'observations.json',manifest)
    _write(output/'data_complete.json',dict(observations=len(manifest['observations']),
           observed=sum(r['status']=='observed' for r in manifest['observations']),
           wall_seconds=time.monotonic()-start,disk=disk_snapshot(output)))
    if video:
        for rig in spec['rigs']:
            child=Path(rig['root'])
            try:
                subprocess.run([sys.executable,'-m','human_data_generator.consumer','video','--root',str(child)],
                               cwd=PROJECT_ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True,
                               timeout=max(.1,deadline-time.monotonic()))
            except Exception as error:
                _write(child/'video_failure.json',dict(error=f'{type(error).__name__}: {error}'))
    return output


def main(argv=None):
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    for action in ('plan','prepare','render'):
        p=sub.add_parser(action)
        p.add_argument('--scene',type=Path,required=True)
        if action!='plan':p.add_argument('--name',required=True)
        p.add_argument('--frame',type=int,action='append')
        p.add_argument('--rig',action='append')
        p.add_argument('--preview',action='store_true')
        if action=='render':
            p.add_argument('--gpu-uuid',required=True)
            p.add_argument('--budget-seconds',type=int,default=1800)
            p.add_argument('--video',action='store_true')
    p=sub.add_parser('run-prepared')
    p.add_argument('--name',required=True);p.add_argument('--preview',action='store_true')
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--budget-seconds',type=int,default=1800)
    p.add_argument('--video',action='store_true')
    p=sub.add_parser('export')
    p.add_argument('--name',required=True)
    args=parser.parse_args(argv)
    if args.action=='plan':
        print(json.dumps(plan(args.scene,args.frame,args.rig,args.preview),indent=2))
    elif args.action=='prepare':
        output,config=prepare(args.scene,args.name,args.frame,args.rig,args.preview)
        print(json.dumps(dict(output=str(output),camera_calls=config['max_camera_calls'])))
    elif args.action=='render':
        output,_=prepare(args.scene,args.name,args.frame,args.rig,args.preview)
        render_prepared(output,args.gpu_uuid,args.budget_seconds,args.video)
    elif args.action=='run-prepared':
        render_prepared(_task_path(args.name,args.preview),args.gpu_uuid,args.budget_seconds,args.video)
    else:
        from human_data_generator.data.mamma_export import export_existing
        output=_task_path(args.name)
        manifest=export_existing(output)
        destination=output/'observations.json'
        if destination.exists():raise FileExistsError(destination)
        _write(destination,manifest)
        print(json.dumps(dict(output=str(destination),observations=len(manifest['observations']))))


if __name__=='__main__':main()
