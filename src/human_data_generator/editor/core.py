"""Current scene editor data and checks; render uses the same run entry."""
import json
from pathlib import Path

import numpy as np

from human_data_generator.config import PROJECT_ROOT,resources
from human_data_generator.runtime.scene import validate_scene,_curve

ROOT=PROJECT_ROOT
OUTPUT=ROOT/'outputs'


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def map_data(environment):
    if environment=='city':
        return dict(bounds=[-90,-90,90,95],roads=[[-3.5,-80,3.5,80],[-80,80,80,88]],
                    water=[],obstacles=[])
    if environment=='riverfront':
        return dict(bounds=[-22,-22,22,22],roads=[[-2.5,-20,2.5,20]],
                    water=[[-22,-22,-3,22]],obstacles=[])
    raise ValueError('unknown_environment')


def default_scene(environment='city'):
    if environment=='city':
        people=[dict(person_id=1,asset='city_motion_1',motion_start=0,
                     route_xy_m=[[0,77.96],[0,92]],speed_m_s=1.2,start_progress_m=0)]
        rigs=[dict(rig_id='rig_side',origin_m=[6.035128805859624,75.97759765371869,0],yaw_deg=298)]
    elif environment=='riverfront':
        people=[dict(person_id=1,asset='riverfront_motion_1',motion_start=0,
                     route_xy_m=[[0,1.8],[0,14]],speed_m_s=1.2,start_progress_m=0)]
        rigs=[dict(rig_id='rig_edge',origin_m=[0,-.0676685317844754,0],yaw_deg=90.70578757417066)]
    else:raise ValueError('unknown_environment')
    return dict(schema='human_scene_v1',scene_id=environment+'_example',environment=environment,
                seed=20261023,fps=8,frame_count=16,people=people,rigs=rigs)


def additional_person(scene,person_id):
    if type(person_id) is not int or person_id<1:raise ValueError('invalid_person_id')
    catalog=list(resources()['assets'])
    if not catalog:raise ValueError('no_available_assets')
    base=default_scene(scene['environment'])['people'][0]
    item=dict(base,person_id=person_id,asset=catalog[(person_id-1)%len(catalog)])
    shift=(person_id-1)*1.2
    item['route_xy_m']=[[x+shift,y] for x,y in base['route_xy_m']]
    return item


def validate(scene,frames=None,rig_ids=None):
    errors=[]
    try:validate_scene(scene)
    except Exception as exc:return dict(status='rejected',errors=[str(exc)],checked_frames=[])
    chosen=list(range(scene['frame_count'])) if frames is None else frames
    if not chosen or chosen!=sorted(set(chosen)) or any(type(k) is not int or not 0<=k<scene['frame_count'] for k in chosen):
        return dict(status='rejected',errors=['invalid_selected_frames'],checked_frames=[])
    rigs=[r for r in scene['rigs'] if rig_ids is None or r['rig_id'] in rig_ids]
    if not rigs or (rig_ids is not None and {r['rig_id'] for r in rigs}!=set(rig_ids)):
        return dict(status='rejected',errors=['invalid_selected_rigs'],checked_frames=chosen)
    roads=map_data(scene['environment'])['roads']
    for k in chosen:
        roots=[]
        for p in scene['people']:
            try:
                xy,_=_curve(p['route_xy_m'],p['start_progress_m']+p['speed_m_s']*k/scene['fps'])
                roots.append((p['person_id'],xy))
                if not any(a<=xy[0]<=c and b<=xy[1]<=d for a,b,c,d in roads):
                    errors.append(f"p{p['person_id']}:frame{k}:outside_verified_road")
            except ValueError as exc:errors.append(f"p{p['person_id']}:frame{k}:{exc}")
        for a in range(len(roots)):
            for b in range(a+1,len(roots)):
                if np.linalg.norm(roots[a][1]-roots[b][1])<.9:
                    errors.append(f"p{roots[a][0]}_p{roots[b][0]}:frame{k}:root_clearance")
        for rig in rigs:
            for pid,xy in roots:
                if np.linalg.norm(xy-np.asarray(rig['origin_m'][:2]))<.5:
                    errors.append(f"{rig['rig_id']}:p{pid}:frame{k}:camera_clearance")
    return dict(status='passed' if not errors else 'rejected',errors=errors[:100],checked_frames=chosen)
