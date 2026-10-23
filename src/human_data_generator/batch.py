"""Deterministic small-scene sampler; writes scenes, never starts rendering."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from human_data_generator.config import PROJECT_ROOT
from human_data_generator.components import describe
from human_data_generator.editor.core import default_scene,validate,write


def _rng(seed,index,purpose):
    material=f'{seed}:{index}:{purpose}'.encode()
    return random.Random(int.from_bytes(hashlib.blake2b(material,digest_size=8).digest(),'big'))


def sample_scene(rule,seed,index):
    if rule not in ('city_single_ride','city_single_walk','city_crossing_pair','riverfront_single_ride'):
        raise ValueError('unknown_batch_rule')
    scene=default_scene('riverfront' if rule.startswith('riverfront') else 'city')
    scene['scene_id']=f'batch_{index:03d}_{rule}'
    scene['seed']=_rng(seed,index,'render').randrange(1,2**31)
    inventory=describe()
    def choose(kind,purpose,behavior=None):
        items=sorted(row['id'] for row in inventory[kind] if row['status']=='ready' and
                     (kind!='bodies' or row['default_version']) and
                     (kind!='clothing' or row['kind']=='simple_barycentric') and
                     (behavior is None or row['behavior']==behavior))
        if not items:raise ValueError('no_usable_batch_component:'+kind)
        return _rng(seed,index,purpose).choice(items)
    rng=_rng(seed,index,'trajectory')
    if rule=='city_single_walk':
        scene['people'][0]['route_xy_m']=[[rng.uniform(-1.0,1.0),77.96],[rng.uniform(-1.0,1.0),92]]
        scene['people'][0]['speed_m_s']=rng.uniform(.8,1.3)
    elif rule=='city_crossing_pair':
        first=scene['people'][0]
        first['route_xy_m']=[[-1.4,77.96],[-1.4,91]]
        second=dict(person_id=2,body='M02',motion='p1_comfortable',clothing='M02_simple',hair='M02_groom',motion_start=0,
                    route_xy_m=[[1.4,82],[1.4,72]],speed_m_s=rng.uniform(.9,1.3),
                    start_progress_m=rng.uniform(0,.4))
        scene['people'].append(second)
    else:
        person=scene['people'][0]
        person['speed_m_s']=rng.uniform(.9,1.4)
        person['start_progress_m']=rng.uniform(0,.4)
    motion_rng=_rng(seed,index,'motion_start')
    for person_index,person in enumerate(scene['people']):
        purpose=f'person_{person_index}'
        person['body']=choose('bodies',purpose+'_body')
        person['motion']=choose('motions',purpose+'_motion','walk' if rule=='city_single_walk' else 'ride')
        person['clothing']=choose('clothing',purpose+'_clothing')
        person['hair']=choose('hair',purpose+'_hair')
        person['motion_start']=motion_rng.randint(0,4)
    camera_rng=_rng(seed,index,'camera')
    for rig in scene['rigs']:
        rig['origin_m'][0]+=camera_rng.uniform(-.25,.25)
        rig['origin_m'][1]+=camera_rng.uniform(-.25,.25)
        rig['yaw_deg']+=camera_rng.uniform(-4,4)
    check=validate(scene)
    if check['status']!='passed':raise ValueError(check['errors'])
    return scene


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--rules',type=Path,required=True)
    parser.add_argument('--name',required=True)
    args=parser.parse_args(argv)
    rules=json.loads(args.rules.read_text())
    if set(rules)!={'schema','seed','scenes'} or rules['schema']!='human_batch_v1':
        raise ValueError('invalid_batch_rules')
    output=PROJECT_ROOT/'outputs/scenes'/args.name
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    paths=[]
    for index,rule in enumerate(rules['scenes']):
        scene=sample_scene(rule,rules['seed'],index)
        path=output/f"{scene['scene_id']}.json"
        write(path,scene)
        paths.append(str(path))
    write(output/'batch.json',dict(schema='human_batch_plan_v1',seed=rules['seed'],scenes=paths))
    print(json.dumps(dict(output=str(output),scenes=paths),ensure_ascii=False))


if __name__=='__main__':main()
