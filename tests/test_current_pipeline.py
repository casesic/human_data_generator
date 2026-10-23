"""Bounded CPU checks for the single current generation/data path."""
import json
import importlib
import os
import shutil
import sys
import time
from types import ModuleType, SimpleNamespace
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from human_data_generator import batch
from human_data_generator.batch import sample_scene
from human_data_generator.components import describe,select
from human_data_generator.composition import build,_cached,_key
from human_data_generator.config import PROJECT_ROOT
from human_data_generator.editor.core import default_scene
from human_data_generator.editor import server
from human_data_generator.consumer import consume_camera_arrays
from human_data_generator.data.mamma_export import export_existing
from human_data_generator.data.mamma_crop import mask_native_support
from human_data_generator.run import render_prepared
from human_data_generator import run
from human_data_generator.runtime.multirig_prepare import prepare
from human_data_generator.runtime.scene import compile_scene,validate_scene
from human_data_generator.runtime.scene import _curve
from human_data_generator.runtime.task_paths import run_config, task_path
from human_data_generator.imaging.output_modes import prepare_source_workspace, cleanup_owned_source, record_failure


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def synthetic_bundle(root):
    root=Path(root)
    put(root/'scene.json',dict(schema='human_scene_v2'))
    people=[dict(person_id=7,asset_id='fixture_asset',source_manifest_id='fixture_source'),
            dict(person_id=9,asset_id='fixture_asset',source_manifest_id='fixture_source')]
    put(root/'recipe.json',dict(schema='human_run_v1',scene_episode_id='fixture_episode',
        frame_count=1,fps=8,people=people))
    np.save(root/'scene_matrices.npy',np.broadcast_to(np.eye(4),(1,2,4,4)))
    put(root/'run_config.json',dict(schema='human_run_config_v2',task_id='a'*32,
        rig_ids=['rig_alpha','rig_beta'],selected_frames=[0],full_image_size_wh=[256,256],
        generation_contract='mamma_512_v1',
        mamma_512_label=dict(template_id='MAMMA_verts_512_ordered_v1')))
    valid=np.ones((256,256),bool)
    rgb=np.zeros((256,256,3),np.uint8)
    rng=np.full((256,256),2.0,np.float32)
    pose=np.eye(4)
    timing=dict(episode_time_tick=0,time_base=[1,8],person_records=[
        dict(asset_frame=0,source_frame=3,source_time_tick=3,source_time_base=[1,30]),
        dict(asset_frame=0,source_frame=3,source_time_tick=3,source_time_base=[1,30])])
    triangle=np.array([[-.2,-.2,2],[.2,-.2,2],[0,.2,2]],np.float64)
    geometry=dict(person_ids=np.array([7,9],np.int32),asset_frames=np.array([0,0]),
                  source_frames=np.array([3,3]),timing=timing,
                  world=np.tile(np.array([[[0.,0.,2.]]]),(2,512,1)),
                  full_body=[triangle,triangle],faces=[np.array([[0,1,2]],np.int32)]*2,
                  vertex_ids=np.arange(512,dtype=np.int32))
    for rid,second_visible in [('rig_alpha',True),('rig_beta',False)]:
        child=root/'rigs'/rid
        child.mkdir(parents=True)
        mask=child/'physical.png'
        Image.fromarray(valid.astype(np.uint8)*255).save(mask)
        put(child/'rig.json',dict(rig_id=rid,cameras=[dict(id='cam2'),dict(id='cam4')],
                                  valid_masks={'cam2':str(mask),'cam4':str(mask)}))
        put(child/'planned_regions.json',{'0':{'cam2':{},'cam4':{}}})
        put(child/'label_policy.json',dict(visibility_policy='fixture'))
    for rid,second_visible in [('rig_alpha',True),('rig_beta',False)]:
        child=root/'rigs'/rid
        instance=np.zeros((256,256),np.float32)
        instance[80:170,80:170]=7
        if second_visible:instance[80:170,175:235]=9
        (child/'frame_000').mkdir()
        for camera in ('cam2','cam4'):
            consume_camera_arrays(child,0,camera,rgb,rng,instance,[0,0,256,256],
                                  pose,valid,geometry)
    manifest=export_existing(root)
    put(root/'observations.json',manifest)
    return manifest


class CurrentPipelineTest(unittest.TestCase):
    def test_components_and_short_real_build(self):
        inventory=describe()
        self.assertEqual({k:v['ready'] for k,v in inventory['counts'].items()},
                         {'bodies':15,'motions':7,'clothing':11,'hair':10})
        self.assertEqual(next(x for x in inventory['motions'] if x['id']=='KIT314_walk_raw')['status'],'raw_only')
        person=dict(body='M03',motion='KIT11_walk',clothing='M02_simple',hair='M01_groom')
        made=build(person,[0,1])
        self.assertEqual(made['component_ids'],person)
        self.assertTrue(all(build(person,[0,1])['cache_hits'].values()))
        asset=json.loads(Path(made['asset_manifest']).read_text())
        self.assertEqual(np.load(asset['body_vertices'],mmap_mode='r').shape,(2,10475,3))
        self.assertEqual(np.load(asset['head_transforms'],mmap_mode='r').shape,(2,4,4))
        self.assertTrue(np.isfinite(np.load(asset['cloth_vertices'],mmap_mode='r')).all())
        self.assertTrue((Path(asset['groom_reference'])/'groom_manifest.json').is_file())
        native=build(dict(person,clothing='M03_KIT11_BEDLAM_transfer'),[0,1])
        native_asset=json.loads(Path(native['asset_manifest']).read_text())
        self.assertTrue((Path(native_asset['clothing_reference'])/'native_material.json').is_file())
        with self.assertRaisesRegex(ValueError,'native_clothing_locked_body_motion'):
            select(dict(person,body='M01',clothing='M03_KIT11_BEDLAM_transfer'))
        with self.assertRaisesRegex(ValueError,'unknown_body'):
            select(dict(person,body='missing'))
        retargeted=build(dict(person,body='M01',motion='p1_comfortable',
                              clothing='M01_simple',hair='M01_groom'),[0,1])
        target=json.loads(Path(select(dict(person,body='M01',motion='p1_comfortable',
                                clothing='M01_simple',hair='M01_groom'))['body']['manifest']).read_text())
        with np.load(json.loads(Path(retargeted['asset_manifest']).read_text())['parameters']) as params:
            np.testing.assert_array_equal(params['betas'],np.asarray(target['betas'],np.float32))
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';source.write_text('a')
            first,first_id=_key('fixture',{'version':1},[source])
            source.write_text('changed')
            second,second_id=_key('fixture',{'version':1},[source])
            self.assertNotEqual(first_id,second_id)
            with patch('human_data_generator.composition.CACHE',Path(temp)):
                incomplete=Path(temp)/'fixture'/second_id
                incomplete.mkdir(parents=True)
                with self.assertRaises(FileNotFoundError):
                    _cached('fixture',second,second_id,('result.npy',),lambda p:None)

    def test_walking_root_and_shared_rigs(self):
        scene=default_scene('city');scene['frame_count']=2
        scene['people'][0].update(body='M03',motion='KIT11_walk',clothing='M02_simple',hair='M01_groom')
        other=json.loads(json.dumps(scene['rigs'][0]));other['rig_id']='rig_other';other['origin_m'][0]+=.3
        scene['rigs'].append(other)
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'scene.json';put(source,scene)
            output=Path(temp)/'prepared'
            config=prepare(output,scene_path=source)
            self.assertEqual(config['max_camera_calls'],8)
            self.assertEqual(config['rig_ids'],['rig_side','rig_other'])
            self.assertEqual(len([p for p in output.rglob('*') if p.is_file()]),8)
            self.assertTrue((output/'scene_matrices.npy').is_file())
            self.assertFalse((output/'rigs/rig_other/scene_matrices.npy').exists())
            matrix=np.load(output/'scene_matrices.npy')[:,0]
            recipe=json.loads((output/'recipe.json').read_text())
            asset=json.loads(Path(recipe['people'][0]['asset_manifest']).read_text())
            with np.load(asset['parameters']) as parameters:
                root=parameters['source_root_relative']
            xy0,_=_curve(scene['people'][0]['route_xy_m'],0)
            xy1,_=_curve(scene['people'][0]['route_xy_m'],scene['people'][0]['speed_m_s']/scene['fps'])
            local=[row['asset_frame'] for row in recipe['person_timelines']['1']]
            delta=np.array([root[local[1],0]-root[local[0],0],0,
                            root[local[1],2]-root[local[0],2]])
            expected=xy1-xy0-(matrix[1,:3,:3]@delta)[:2]
            np.testing.assert_allclose(matrix[1,:2,3]-matrix[0,:2,3],expected,atol=1e-6)

    def test_editor_component_api_and_save(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(server,'RECIPES',Path(temp)):
                def request(path,body=None):
                    handler=server.Handler.__new__(server.Handler)
                    handler.path=path;handler.wfile=BytesIO()
                    handler.send_response=lambda code:None
                    handler.send_header=lambda *args:None
                    handler.end_headers=lambda:None
                    if body is None:handler.do_GET()
                    else:
                        handler._body=lambda:body
                        handler.do_POST()
                    return json.loads(handler.wfile.getvalue())
                items=request('/api/components')
                self.assertIn('M01', {x['id'] for x in items['bodies']})
                scene=default_scene('city')
                scene['people'][0].update(body='M03',motion='KIT11_walk',
                                           clothing='M02_simple',hair='M01_groom')
                saved=request('/api/save',{'scene':scene})
                self.assertTrue(Path(saved['path']).is_file())
                self.assertEqual(json.loads(Path(saved['path']).read_text())['people'][0]['hair'],'M01_groom')

    def test_scene_sampling_and_clock(self):
        rules=json.loads((PROJECT_ROOT/'config/batch_rules.json').read_text())
        self.assertEqual(len(rules['scenes']),4)
        for i,item in enumerate(rules['scenes']):
            self.assertEqual(sample_scene(item,rules['seed'],i)['schema'],'human_scene_v2')
        rule=rules['scenes'][2]
        a=sample_scene(rule,rules['seed'],2)
        b=sample_scene(rule,rules['seed'],2)
        self.assertEqual(a,b)
        self.assertTrue(all(k in a['people'][0] for k in ('body','motion','clothing','hair')))
        custom=deepcopy(rule)
        custom['scene_id']='custom_pair'
        custom['people'][0]['speed_m_s']=[1.05,1.05]
        custom['people'][0]['route_start_xy_range_m'][0]=[-1.2,-1.2]
        changed=sample_scene(custom,rules['seed'],2)
        self.assertEqual(changed['scene_id'],'batch_002_custom_pair')
        self.assertEqual(changed['people'][0]['speed_m_s'],1.05)
        self.assertEqual(changed['people'][0]['route_xy_m'][0][0],-1.2)
        with self.assertRaises(ValueError):sample_scene('city_crossing_pair',rules['seed'],2)
        inventory=describe()
        self.assertGreaterEqual(inventory['counts']['bodies']['ready'],10)
        a=default_scene('city');a['frame_count']=2
        second=dict(a['people'][0],person_id=17,route_xy_m=[[1.5,77.96],[1.5,92]])
        a['people'].append(second)
        validate_scene(a)
        people,times,matrices=compile_scene(a,[0,1])
        self.assertEqual([p['person_id'] for p in people],[1,17])
        self.assertEqual(matrices.shape,(2,2,4,4))
        self.assertEqual(len(times['17']),2)
        self.assertEqual(people[0]['asset_id'],people[1]['asset_id'])
        a['people'][1]['body']='M03'
        shared_motion=compile_scene(a,[0])[0]
        self.assertEqual(shared_motion[0]['component_ids']['motion'],shared_motion[1]['component_ids']['motion'])
        self.assertNotEqual(shared_motion[0]['asset_id'],shared_motion[1]['asset_id'])
        a['people'].reverse()
        self.assertEqual([p['person_id'] for p in compile_scene(a,[0])[0]],[17,1])
        a['frame_count']=999
        with self.assertRaises(ValueError):validate_scene(a)
        a['frame_count']=2
        a['people'][0]['asset']='unsupported'
        with self.assertRaisesRegex(ValueError,'person_fields'):validate_scene(a)
        a['people'][0].pop('asset')
        a['qualification_registry']='unsupported'
        with self.assertRaisesRegex(ValueError,'scene_fields'):validate_scene(a)

    def test_batch_component_candidate_pools(self):
        rules=json.loads((PROJECT_ROOT/'config/batch_rules.json').read_text())
        rule=deepcopy(rules['scenes'][0]);pool=rule['people'][0]['components']
        pool['body']['ids']=['M03','M01']
        pool['motion']['ids']=['KIT425_walk','KIT11_walk']
        pool['clothing']['ids']=['M03_simple','M01_simple']
        pool['hair']['ids']=['M03_groom','M01_groom']
        chosen=sample_scene(rule,rules['seed'],0)
        self.assertIn(chosen['people'][0]['body'],pool['body']['ids'])
        self.assertIn(chosen['people'][0]['motion'],pool['motion']['ids'])
        self.assertIn(chosen['people'][0]['clothing'],pool['clothing']['ids'])
        self.assertIn(chosen['people'][0]['hair'],pool['hair']['ids'])
        reordered=deepcopy(rule)
        for item in reordered['people'][0]['components'].values():item['ids'].reverse()
        self.assertEqual(chosen,sample_scene(reordered,rules['seed'],0))

        fixed=deepcopy(rule);pool=fixed['people'][0]['components']
        pool['body']={'default_versions_only':False,'ids':['F04_v1']}
        pool['motion']['ids']=['KIT11_walk']
        pool['clothing']['ids']=['M03_simple']
        pool['hair']['ids']=['M01_groom']
        person=sample_scene(fixed,rules['seed'],0)['people'][0]
        self.assertEqual((person['body'],person['motion'],person['clothing'],person['hair']),
                         ('F04_v1','KIT11_walk','M03_simple','M01_groom'))
        pool['body']['default_versions_only']=True
        with self.assertRaisesRegex(ValueError,'batch_component_filter_conflict'):
            sample_scene(fixed,rules['seed'],0)

        pair=deepcopy(rules['scenes'][2])
        for spec,ident in zip(pair['people'],('M01','M03')):
            spec['components']['body']['ids']=[ident]
        self.assertEqual([p['body'] for p in sample_scene(pair,rules['seed'],2)['people']],
                         ['M01','M03'])

    def test_batch_locked_clothing_and_rejections(self):
        rules=json.loads((PROJECT_ROOT/'config/batch_rules.json').read_text())
        rule=deepcopy(rules['scenes'][0]);rule['frame_count']=2
        spec=rule['people'][0];pool=spec['components']
        pool['body']['ids']=['M03']
        pool['motion']['ids']=['KIT11_walk']
        pool['clothing']={'kinds':['native_locked'],'ids':['M03_KIT11_BEDLAM_transfer']}
        spec['motion_start']=[0,0]
        self.assertEqual(sample_scene(rule,rules['seed'],0)['people'][0]['clothing'],
                         'M03_KIT11_BEDLAM_transfer')
        for kind,ident in (('body','M01'),('motion','KIT425_walk')):
            bad=deepcopy(rule);bad['people'][0]['components'][kind]['ids']=[ident]
            with self.assertRaisesRegex(ValueError,'native_clothing_body_motion_or_window_conflict'):
                sample_scene(bad,rules['seed'],0)
        inventory=deepcopy(describe())
        locked=next(row for row in inventory['clothing'] if row['id']=='M03_KIT11_BEDLAM_transfer')
        locked['locked_window']=[0,1]
        with patch('human_data_generator.batch.describe',return_value=inventory):
            with self.assertRaisesRegex(ValueError,'native_clothing_body_motion_or_window_conflict'):
                sample_scene(rule,rules['seed'],0)
        bad=deepcopy(rule);bad['people'][0]['components']['clothing']['ids']=[]
        bad['people'][0]['components']['body']['ids']=['M01']
        with self.assertRaisesRegex(ValueError,'no_usable_batch_component:clothing'):
            sample_scene(bad,rules['seed'],0)

        base=rules['scenes'][0]
        unavailable=next(row['id'] for row in describe()['motions'] if row['status']!='ready')
        failures=(('body',['missing_body'],'unknown_batch_component'),
                  ('motion',['KIT11_walk','KIT11_walk'],'invalid_component_ids'),
                  ('motion',[unavailable],'unavailable_batch_component'),
                  ('motion',['APD6526'],'batch_component_filter_conflict'),
                  ('clothing',['M03_KIT11_BEDLAM_transfer'],'batch_component_filter_conflict'))
        for kind,ids,reason in failures:
            bad=deepcopy(base)
            bad['people'][0]['components'][kind]['ids']=ids
            with self.assertRaisesRegex(ValueError,reason):sample_scene(bad,rules['seed'],0)
        for kind,key,value,reason in (('body','default_versions_only','true','invalid_default_versions_only'),
                                      ('clothing','kinds',[],'invalid_clothing_kinds'),
                                      ('hair','ids','M01_groom','invalid_component_ids')):
            bad=deepcopy(base)
            bad['people'][0]['components'][kind][key]=value
            with self.assertRaisesRegex(ValueError,reason):sample_scene(bad,rules['seed'],0)
        old=deepcopy(base)
        old['people'][0]['motion_behavior']='walk'
        with self.assertRaisesRegex(ValueError,'invalid_batch_rule:person'):
            sample_scene(old,rules['seed'],0)

    def test_batch_cli_writes_only_scene_files(self):
        rules=PROJECT_ROOT/'config/batch_rules.json'
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(batch,'PROJECT_ROOT',Path(tmp)), \
                 patch('human_data_generator.composition.build',side_effect=AssertionError('built geometry')), \
                 patch('subprocess.Popen',side_effect=AssertionError('started process')):
                batch.main(['--rules',str(rules),'--name','sample'])
            folder=Path(tmp)/'outputs/scenes/sample'
            plan=json.loads((folder/'batch.json').read_text())
            self.assertEqual(len(plan['scenes']),4)
            self.assertTrue(all(Path(path).is_file() for path in plan['scenes']))
            self.assertEqual(len(list(folder.glob('*.json'))),5)
            old=Path(tmp)/'old_rules.json'
            old.write_text(json.dumps({'schema':'human_batch_v1','seed':1,
                                       'scenes':['city_single_walk']}))
            with patch.object(batch,'PROJECT_ROOT',Path(tmp)):
                with self.assertRaisesRegex(ValueError,'invalid_batch_rules'):
                    batch.main(['--rules',str(old),'--name','old'])
            self.assertFalse((Path(tmp)/'outputs/scenes/old').exists())

    def test_prepare_budget_and_mock_launch(self):
        scene=PROJECT_ROOT/'config/scene_example.json'
        with tempfile.TemporaryDirectory(prefix='hdg_current_prepare_') as temp:
            output=Path(temp)/'run'
            config=prepare(output,scene_path=scene,frame_indices=[5],
                           selected_rig_ids=['rig_side'],preview=True)
            self.assertEqual(config['selected_frames'],[5])
            person=json.loads((output/'recipe.json').read_text())['people'][0]
            asset=json.loads(Path(person['asset_manifest']).read_text())
            self.assertEqual(np.load(asset['body_vertices'],mmap_mode='r').shape[0],1)
            self.assertNotIn('deadline_monotonic',config)
            self.assertEqual(config['rig_ids'],['rig_side'])
            self.assertEqual(sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()),
                ['recipe.json','rigs/rig_side/planned_regions.json','rigs/rig_side/rig.json',
                 'run_config.json','scene.json','scene_matrices.npy'])
            self.assertFalse((output/'frame_005').exists())
            with patch('human_data_generator.run._resources',return_value={}),\
                 patch('human_data_generator.run.subprocess.Popen',side_effect=RuntimeError('mock_launch')) as popen:
                with self.assertRaisesRegex(RuntimeError,'mock_launch'):
                    render_prepared(output,'MOCK_GPU',60)
            command=popen.call_args.args[0]
            self.assertEqual(command[command.index('--python')+1],
                str(PROJECT_ROOT/'src/human_data_generator/render/multirig.py'))
            self.assertEqual(json.loads((output/'run_config.json').read_text())['gpu_uuid'],'MOCK_GPU')
            self.assertEqual(json.loads((output/'run_config.json').read_text())['budget_seconds'],60)
            self.assertNotIn('remaining_budget_seconds',json.loads((output/'run_config.json').read_text()))

    def test_render_command_prepares_before_start(self):
        events=[]
        def fake_prepare(scene_path,name,frames,rigs,preview):
            events.append(('prepare',name))
            return Path('/tmp/mock_current_run'),{}
        def fake_start(output,gpu,budget,video):
            events.append(('start',budget))
        with patch.object(run,'prepare',side_effect=fake_prepare),patch.object(run,'render_prepared',side_effect=fake_start):
            run.main(['render','--scene','config/scene_example.json','--name','mock_current',
                      '--gpu-uuid','MOCK','--budget-seconds','70'])
        self.assertEqual(events,[('prepare','mock_current'),('start',70)])

    def test_relocated_prepared_task_and_current_layout(self):
        scene=PROJECT_ROOT/'config/scene_example.json'
        with tempfile.TemporaryDirectory(prefix='hdg_relocate_') as temp:
            old=Path(temp)/'old';new=Path(temp)/'relocated'
            prepare(old,scene_path=scene,frame_indices=[1,5],selected_rig_ids=['rig_side'])
            original=run_config(old)
            shutil.move(str(old),str(new))
            self.assertFalse(old.exists())
            elsewhere=Path(temp)/'elsewhere';elsewhere.mkdir()
            cwd=Path.cwd()
            try:
                os.chdir(elsewhere)
                self.assertEqual(run_config(new)['selected_frames'],[1,5])
                self.assertEqual(run_config(new)['rig_ids'],['rig_side'])
                self.assertEqual(json.loads(task_path(new,'recipe.json').read_text())['people'][0]['person_id'],1)
                self.assertIn('1',json.loads(task_path(new,'rigs/rig_side/planned_regions.json').read_text()))
                obsolete_camera=new/'rigs/rig_side/camera_configs';obsolete_camera.mkdir()
                with self.assertRaisesRegex(ValueError,'obsolete_prepared_layout'):
                    run_config(new)
                obsolete_camera.rmdir()
                with_old_field=dict(run_config(new),reprocess_units_before_render=[])
                put(new/'run_config.json',with_old_field)
                with self.assertRaisesRegex(ValueError,'obsolete_run_config_fields'):
                    run_config(new)
                put(new/'run_config.json',original)
                with patch('human_data_generator.run._resources',return_value={}),\
                     patch('human_data_generator.run.subprocess.Popen',side_effect=RuntimeError('launch_boundary')) as popen:
                    with self.assertRaisesRegex(RuntimeError,'launch_boundary'):
                        render_prepared(new,'MOCK_GPU',47)
                self.assertEqual(popen.call_args.args[0][-1],str(new))
                started=run_config(new)
                self.assertEqual(started['task_id'],original['task_id'])
                self.assertEqual(started['gpu_uuid'],'MOCK_GPU')
                self.assertEqual(started['budget_seconds'],47)
                self.assertGreater(started['deadline_monotonic'],0)
            finally:os.chdir(cwd)
            with self.assertRaisesRegex(ValueError,'current_prepared_layout_required'):
                run.main(['run-prepared','--root',str(old),'--gpu-uuid','MOCK'])
            obsolete=Path(temp)/'obsolete';obsolete.mkdir()
            put(obsolete/'run_config.json',dict(schema='human_run_config_v1',selected_frames=[0]))
            with self.assertRaisesRegex(ValueError,'current_prepared_layout_required'):
                run_config(obsolete)

    def test_owned_buffer_cleanup_rejections(self):
        with tempfile.TemporaryDirectory(prefix='hdg_owned_') as temp:
            root=Path(temp)/'task';other=Path(temp)/'other';outside=Path(temp)/'external'
            task_id='a'*32
            output=root/'rigs/rig_side/frame_005'
            source=prepare_source_workspace(output,'compact',task_root=root,task_id=task_id)
            (source/'evidence.txt').write_text('retained on failure')
            record_failure(output,'compact',source,'consume',RuntimeError('fixture'))
            self.assertTrue((source/'evidence.txt').is_file())
            for changed in (outside,other/'rigs/rig_side/frame_005'):
                with self.assertRaises(ValueError):
                    cleanup_owned_source(changed,source,task_root=root,task_id=task_id)
            with self.assertRaises(RuntimeError):
                cleanup_owned_source(output,source,task_root=root,task_id='b'*32)
            marker=output/'_temporary_ds_source/.human_data_temporary_source.json'
            saved=marker.read_text();marker.write_text(saved.replace(task_id,'c'*32))
            with self.assertRaisesRegex(RuntimeError,'ownership mismatch'):
                cleanup_owned_source(output,source,task_root=root,task_id=task_id)
            self.assertTrue((source/'evidence.txt').is_file())
            marker.write_text(saved)
            cleanup_owned_source(output,source,task_root=root,task_id=task_id)
            self.assertFalse(source.exists())

    def test_camera_object_and_multirig_callbacks_without_blender(self):
        fake_bpy=ModuleType('bpy')
        fake_mathutils=ModuleType('mathutils')
        fake_mathutils.Matrix=lambda value:np.asarray(value)
        fake_mathutils.Vector=lambda value:np.asarray(value)
        with patch.dict(sys.modules,{'bpy':fake_bpy,'mathutils':fake_mathutils}):
            camera_module=importlib.import_module('human_data_generator.rendering.blender.assembly')
            multi=importlib.import_module('human_data_generator.render.multirig')
            roi_module=importlib.import_module('human_data_generator.render.roi')
        with tempfile.TemporaryDirectory(prefix='hdg_callbacks_') as temp:
            root=Path(temp)
            image=np.eye(4).tolist()
            camera=dict(id='cam2',camera_to_world=image)
            render_calls=[]
            def fake_render(*,write_still):
                render_calls.append(write_still)
                (root/'cam2_0001.exr').write_bytes(b'fixture')
            fake_bpy.ops=SimpleNamespace(render=SimpleNamespace(render=fake_render))
            fake_scene=SimpleNamespace(render=SimpleNamespace(filepath=''))
            cam=SimpleNamespace(matrix_world=None)
            node=SimpleNamespace(file_name='')
            record=camera_module.render_cameras(fake_scene,cam,node,camera,root,root)
            self.assertEqual(record[0]['source'],'cam2.exr')
            self.assertEqual(record[0]['camera_to_world'],image)
            self.assertEqual(render_calls,[True])

            seen_cameras=[]
            source=root/'roi_fixture';source.mkdir()
            scene=SimpleNamespace(render=SimpleNamespace(),cycles=SimpleNamespace())
            def capture_camera(scene,cam,node,camera,dest,task_root):
                seen_cameras.append((camera['id'],task_root))
                return [dict(id=camera['id'],source='fixture.exr')]
            with patch.object(roi_module,'render_cameras',side_effect=capture_camera),\
                 patch.object(roi_module,'project_erp',return_value=(np.array([[1.,1.]]),None)),\
                 patch.object(roi_module,'plan_person_crop_union',return_value=dict(roi_xyxy=[0,0,4,4])),\
                 patch.object(roi_module,'border_coordinates',return_value=(0.,1.,0.,1.)):
                result=roi_module.render_rig_roi(scene,None,SimpleNamespace(directory=''),source,root,
                    dict(rig_id='rig_alpha',people=[dict(person_id=7)]),
                    [dict(id=cid,camera_to_world=image) for cid in ('cam2','cam4')],
                    {'body':np.array([[0.,0.,1.]])},{'body':7},dict(storage_mode='roi_only'),1,
                    lambda *args,**kwargs:1)
            self.assertEqual(seen_cameras,[('cam2',root),('cam4',root)])
            self.assertEqual([row['id'] for row in result['records']],['cam2','cam4'])

            put(root/'scene.json',dict(schema='human_scene_v2'))
            put(root/'recipe.json',dict(schema='human_run_v1',scene_episode_id='fixture',
                people=[dict(person_id=7),dict(person_id=9)]))
            np.save(root/'scene_matrices.npy',np.broadcast_to(np.eye(4),(2,2,4,4)))
            put(root/'run_config.json',dict(schema='human_run_config_v2',task_id='d'*32,
                rig_ids=['rig_alpha','rig_beta'],selected_frames=[1,3],
                scene_settings={},deadline_monotonic=time.monotonic()+60,
                consumer_python=sys.executable,max_camera_calls=8))
            for rid in ('rig_alpha','rig_beta'):
                put(root/f'rigs/{rid}/rig.json',dict(rig_id=rid,
                    cameras=[dict(id=cid,camera_to_world=image) for cid in ('cam2','cam4')]))
                put(root/f'rigs/{rid}/planned_regions.json',
                    {str(frame):{cid:dict(roi_xyxy=[0,0,4,4]) for cid in ('cam2','cam4')}
                     for frame in (1,3)})
            calls=[]
            class Worker:
                def __init__(self,*args,**kwargs):
                    self.process=SimpleNamespace(pid=1234);self.startup_seconds=0
                    self.responses=[]
                def consume(self,frame,rig_id):
                    self.responses.append(dict(frame=frame,rig_id=rig_id))
                    return dict(status='ok',frame=frame,rig_id=rig_id)
                def close(self):return dict(status='closed')
                def abort(self):raise AssertionError('worker aborted')
            def fake_roi(scene,cam,node,folder,task_root,recipe,cameras,parts,owners,config,frame,started):
                calls.append((frame,recipe['rig_id'],[c['id'] for c in cameras],task_root))
                put(folder/'roi_plan.json',dict(cameras={c['id']:dict(roi_xyxy=[0,0,4,4]) for c in cameras}))
                return dict(records=[dict(id=c['id'],source=f'rigs/{recipe["rig_id"]}/frame_{frame:03d}/source.exr')
                                     for c in cameras])
            def fake_assembly(*,root,render_cameras_fn,frame_complete_fn,frame_source_dir_fn,reuse_session):
                scene=SimpleNamespace(objects=[],render=SimpleNamespace(use_persistent_data=True))
                for frame in (1,3):
                    folder=frame_source_dir_fn(frame)
                    result=render_cameras_fn(scene,None,None,folder,frame_index=frame)
                    frame_complete_fn(frame,folder,dict(sources=result,frame_index=frame))
                put(root/'assembly_result.json',dict(scene_reused=False))
            with patch.object(multi,'FrameConsumer',Worker),\
                 patch.object(multi,'render_rig_roi',side_effect=fake_roi),\
                 patch.object(multi,'scene_settings',return_value={}),\
                 patch.object(multi.assembly,'main',side_effect=fake_assembly):
                multi.main(root)
            self.assertEqual([(frame,rid,cids) for frame,rid,cids,_ in calls],
                [(frame,rid,['cam2','cam4']) for frame in (1,3) for rid in ('rig_alpha','rig_beta')])
            self.assertTrue(all(path==root for *_,path in calls))
            self.assertFalse((root/'frame_001').exists())
            self.assertEqual(json.loads((root/'multirig_render_result.json').read_text())['normal_path_geometry_updates'],2)
            for rid in ('rig_alpha','rig_beta'):
                self.assertTrue((root/f'rigs/{rid}/frame_001/_temporary_ds_source/source/source_manifest.json').is_file())

    def test_assembly_updates_shared_geometry_once_per_frame(self):
        fake_bpy=ModuleType('bpy');fake_mathutils=ModuleType('mathutils')
        fake_mathutils.Matrix=lambda value:np.asarray(value)
        fake_mathutils.Vector=lambda value:np.asarray(value)
        with patch.dict(sys.modules,{'bpy':fake_bpy,'mathutils':fake_mathutils}):
            assembly=importlib.import_module('human_data_generator.render.assembly')
        class Vertices:
            def __init__(self,points):self.points=np.asarray(points,np.float32).copy()
            def foreach_set(self,attribute,values):self.points=np.asarray(values,np.float32).reshape(-1,3).copy()
            def foreach_get(self,attribute,target):target[:]=self.points.ravel()
        def mesh_object(points):
            return SimpleNamespace(data=SimpleNamespace(vertices=Vertices(points),update=lambda:None),
                matrix_world=np.eye(4),hide_get=lambda:False,hide_set=lambda value:None)
        with tempfile.TemporaryDirectory(prefix='hdg_assembly_') as temp:
            root=Path(temp);blend=root/'scene.blend';blend.write_bytes(b'fixture')
            environment=root/'environment.json'
            put(environment,dict(schema='human_environment_v1',blend_path=str(blend),hide_objects=[]))
            vertices=np.array([[[0,0,1],[.1,0,1],[0,.1,1]],
                               [[0,0,1.1],[.1,0,1.1],[0,.1,1.1]]],np.float32)
            body_path=root/'body.npy';np.save(body_path,vertices)
            asset_path=root/'asset.json';put(asset_path,dict(body_vertices=str(body_path)))
            person=dict(person_id=7,asset_manifest=str(asset_path),root_mode='fixture')
            matrices=np.broadcast_to(np.eye(4),(2,1,4,4)).copy()
            matrices[1,0,0,3]=.2
            put(root/'scene.json',dict(schema='human_scene_v2'))
            put(root/'recipe.json',dict(schema='human_run_v1',scene=str(environment),
                scene_episode_id='fixture',frame_count=2,fps=8,geometry_acceptance='passed',
                people=[person],person_timelines={'7':[dict(asset_frame=k,source_frame=k) for k in (0,1)]},
                render=dict(source_resolution=[4,4],samples=1,seed=2)))
            np.save(root/'scene_matrices.npy',matrices)
            put(root/'run_config.json',dict(schema='human_run_config_v2',task_id='e'*32,
                rig_ids=['rig_side'],selected_frames=[0,1],gpu_uuid='MOCK',
                deadline_monotonic=time.monotonic()+60))
            put(root/'rigs/rig_side/rig.json',dict(rig_id='rig_side',cameras=[]))
            put(root/'rigs/rig_side/planned_regions.json',{})
            body=mesh_object(vertices[0]);cloth=mesh_object(vertices[0])
            road=SimpleNamespace(name='Road')
            scene=SimpleNamespace(objects=[body,cloth],render=SimpleNamespace(),cycles=SimpleNamespace(max_bounces=2),
                view_layers=[SimpleNamespace()],ray_cast=lambda *args,**kwargs:
                    (True,SimpleNamespace(z=0),SimpleNamespace(z=1),0,road,None))
            device=SimpleNamespace(name='mock',id='mock',type='CUDA',use=False)
            preferences=SimpleNamespace(compute_device_type=None,get_devices=lambda:None,devices=[device])
            fake_bpy.context=SimpleNamespace(scene=scene,view_layer=SimpleNamespace(update=lambda:None),
                evaluated_depsgraph_get=lambda:None,
                preferences=SimpleNamespace(addons={'cycles':SimpleNamespace(preferences=preferences)}))
            fake_bpy.app=SimpleNamespace(version_string='mock')
            key=(str(blend),blend.stat().st_size,blend.stat().st_mtime_ns,
                ((7,str(asset_path),asset_path.stat().st_size,asset_path.stat().st_mtime_ns),),
                (4,4),1,'MOCK')
            prior=(person,{},vertices,vertices,np.broadcast_to(np.eye(4),(2,4,4)),matrices,
                   body,cloth,[],[],[0,1])
            assembly._SESSION=dict(key=key,scene=scene,objects=[prior],camera='mock-camera',
                                   file_node=SimpleNamespace(directory=''))
            seen=[]
            def render_callback(scene,camera,node,folder,*,frame_index):
                seen.append((frame_index,body.data.vertices.points.copy(),cloth.data.vertices.points.copy(),
                             np.asarray(body.matrix_world).copy()))
                return [dict(id='cam2',source='fixture')]
            def complete(frame,folder,manifest):
                self.assertEqual(manifest['frame_index'],frame)
                self.assertEqual(manifest['episode_recipe'],'recipe.json')
            def source_folder(frame):
                folder=root/f'scratch_{frame}';folder.mkdir()
                return folder
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'MOCK'}),\
                 patch.object(assembly,'configure_render'),\
                 patch.object(assembly,'geometry_paths',return_value={'body_vertices':body_path}),\
                 patch.object(assembly,'support_limits',return_value=(.9,1.2)):
                assembly.main(root=root,render_cameras_fn=render_callback,
                    frame_complete_fn=complete,frame_source_dir_fn=source_folder,reuse_session=True)
            self.assertEqual([row[0] for row in seen],[0,1])
            np.testing.assert_array_equal(seen[1][1],vertices[1])
            np.testing.assert_array_equal(seen[1][2],vertices[1])
            self.assertEqual(seen[1][3][0,3],.2)
            self.assertEqual(json.loads((root/'assembly_result.json').read_text())['scene_reused'],True)
            assembly._SESSION=None

    def test_consumer_export_and_rejections(self):
        with tempfile.TemporaryDirectory(prefix='hdg_current_bundle_') as temp:
            root=Path(temp)
            manifest=synthetic_bundle(root)
            self.assertEqual((len(manifest['observations']),
                              sum(r['status']=='observed' for r in manifest['observations'])),(8,6))
            self.assertEqual({r['person_id'] for r in manifest['observations']},{7,9})
            self.assertEqual({r['rig'] for r in manifest['observations']},{'rig_alpha','rig_beta'})
            with np.load(root/manifest['observations'][0]['labels']) as gt:
                self.assertNotIn('visibility_known',gt)
                self.assertNotIn('contact_value',gt)
                self.assertTrue(np.isin(gt['visibility_value'],[-1,0,1]).all())
            path=root/'rigs/rig_alpha/frame_000/cam2/mamma_512.npz'
            with np.load(path,allow_pickle=False) as data:
                original={key:data[key] for key in data.files}
            for key,value in [('person_ids',np.array([9,7])),('frame_index',np.array(1)),
                              ('time_tick',np.array(1)),
                              ('camera_id',np.array('cam4')),
                              ('ordered_vertex_ids',original['ordered_vertex_ids'][::-1])]:
                edited=dict(original,**{key:value})
                np.savez(path,**edited)
                with self.assertRaisesRegex(ValueError,'label_identity_time_conflict|invalid_mamma_512_label|template_point_order_conflict'):
                    export_existing(root)
            np.savez(path,**original)
            path.rename(root/'missing.npz')
            with self.assertRaisesRegex(ValueError,'missing_required_camera_product'):
                export_existing(root)
            near_edge=np.zeros((256,256),np.uint8)
            near_edge[:40,:40]=255
            with self.assertRaisesRegex(ValueError,'crop_filter_support_outside_roi'):
                mask_native_support(near_edge,[0,0,256,256],[256,256])


if __name__=='__main__':unittest.main()
