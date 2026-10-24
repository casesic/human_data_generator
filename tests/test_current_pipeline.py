"""Bounded CPU checks for the single current generation/data path."""
import json
import importlib
import os
import shutil
import sys
import time
from types import ModuleType, SimpleNamespace
from copy import deepcopy
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from human_data_generator import batch
from human_data_generator.batch import sample_scene, sample_batch
from human_data_generator.components import describe,select
from human_data_generator.composition import build,_cached,_key
from human_data_generator.config import PROJECT_ROOT
from human_data_generator.editor.core import default_scene
from human_data_generator.editor import server
from human_data_generator.camera.rig import rig_poses
from human_data_generator.consumer import consume_camera_arrays, load_rig_context
from human_data_generator.data.mamma_export import export_existing
from human_data_generator.data.mamma_crop import mask_native_support
from human_data_generator.run import render_prepared
from human_data_generator import run
from human_data_generator.runtime.multirig_prepare import prepare, _surface_may_enter_valid_mask
from human_data_generator.runtime.scene import compile_scene, route_positions, validate, validate_scene
from human_data_generator.runtime.scene import _curve
from human_data_generator.runtime.task_paths import run_config, task_path
from human_data_generator.runtime.frame_consumer import serve_frames
from human_data_generator.imaging.output_modes import prepare_source_workspace, cleanup_owned_source
from human_data_generator.imaging.roi_planning import plan_native_crop_region
from human_data_generator.data.occlusion_events import point_occlusion_evidence


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def synthetic_bundle(root, *, frames=(0,), rigs=(('rig_alpha',True),('rig_beta',False)),
                     single_pixel_second=False):
    root=Path(root)
    put(root/'scene.json',dict(schema='human_scene_v2'))
    people=[dict(person_id=7,asset_id='fixture_asset',source_manifest_id='fixture_source'),
            dict(person_id=9,asset_id='fixture_asset',source_manifest_id='fixture_source')]
    put(root/'recipe.json',dict(schema='human_run_v1',scene_episode_id='fixture_episode',
        frame_count=max(frames)+1,fps=8,people=people))
    np.save(root/'scene_matrices.npy',np.broadcast_to(np.eye(4),(max(frames)+1,2,4,4)))
    put(root/'run_config.json',dict(schema='human_run_config_v2',task_id='a'*32,
        rig_ids=[rid for rid,_ in rigs],selected_frames=list(frames),full_image_size_wh=[256,256],
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
    for rid,second_visible in rigs:
        child=root/'rigs'/rid
        child.mkdir(parents=True)
        mask=child/'physical.png'
        Image.fromarray(valid.astype(np.uint8)*255).save(mask)
        put(child/'rig.json',dict(rig_id=rid,cameras=[dict(id='cam2'),dict(id='cam4')],
                                  valid_masks={'cam2':str(mask),'cam4':str(mask)}))
        put(child/'planned_regions.json',
            {str(frame):{'cam2':{},'cam4':{}} for frame in frames})
        put(child/'label_policy.json',dict(visibility_policy='fixture'))
    for rid,second_visible in rigs:
        child=root/'rigs'/rid
        recipe, config = load_rig_context(child)
        instance=np.zeros((256,256),np.float32)
        instance[80:170,80:170]=7
        if second_visible:
            if single_pixel_second:instance[80,175]=9
            else:instance[80:170,175:235]=9
        for frame in frames:
            (child/f'frame_{frame:03d}').mkdir()
            state=dict(geometry,timing=dict(timing,episode_time_tick=frame))
            for camera in ('cam2','cam4'):
                consume_camera_arrays(child,frame,camera,rgb,rng,instance,[0,0,256,256],
                                      pose,valid,state,recipe,config)
    manifest=export_existing(root)
    put(root/'observations.json',manifest)
    return manifest


class CurrentPipelineTest(unittest.TestCase):
    def test_degenerate_target_mask_is_absent_without_losing_other_targets(self):
        with tempfile.TemporaryDirectory(prefix='hdg_degenerate_mask_') as temp:
            manifest = synthetic_bundle(
                Path(temp), rigs=(('rig_alpha', True),), single_pixel_second=True
            )
            rows = manifest['observations']
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                {(row['person_id'], row['status'], row['absent_reason']) for row in rows},
                {(7, 'observed', None), (9, 'absent', 'degenerate_target_mask_box')},
            )
            self.assertTrue(all(row['crop_support'] is None for row in rows if row['person_id'] == 9))

    def test_periodic_seam_roi_crop_and_visibility_neighbors(self):
        normal = plan_native_crop_region(
            {"body": np.array([[100., 110.], [120., 130.]])}, (256, 256)
        )
        self.assertEqual(normal["roi_xyxy"], [54, 64, 166, 176])
        xy = np.array([[250., 110.], [3., 112.], [6., 134.], [247., 131.]])
        region = plan_native_crop_region({"body": xy}, (256, 256))
        self.assertEqual(region["roi_xyxy"][0::2], [0, 256])
        mask = np.zeros((256, 256), np.uint8)
        mask[110:135, 247:] = 255
        mask[110:135, :7] = 255
        support = mask_native_support(mask, [0, 0, 256, 256], [256, 256])
        self.assertEqual(support["crop_policy_version"], "mask_bbox_periodic_v2")
        self.assertEqual(support["box_xyxy"], [247, 110, 262, 134])
        self.assertLess(support["sampling_rect_erp"][2] - support["sampling_rect_erp"][0], 256)
        depth = np.ones((64, 64), np.float32)
        instance = np.ones((64, 64), np.float32)
        evidence = point_occlusion_evidence(
            np.array([[0.2, 20.], [63.2, 20.]]), np.array([1.1, 1.1]),
            np.array([False, False]), np.array([True, True]),
            np.array([2, 2]), [1, 2], depth, instance, strict_coverage=False
        )
        self.assertTrue(evidence["evidence_neighborhood_complete"].all())

        valid = np.zeros((64, 64), bool)
        valid[29:32, 29:32] = True
        vertices = np.array([[10., 10.], [50., 10.], [30., 50.]])
        self.assertTrue(_surface_may_enter_valid_mask(
            vertices, np.array([[0, 1, 2]]), valid
        ))
        self.assertFalse(_surface_may_enter_valid_mask(
            vertices + 70, np.array([[0, 1, 2]]), valid
        ))

    def test_shared_gpu_check_keeps_uuid_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            blender = Path(temp) / 'blender'
            blender.touch()
            snapshots = [
                'index, uuid, memory.used [MiB], memory.free [MiB], utilization.gpu [%]\n'
                '1, GPU-BUSY, 14000 MiB, 10000 MiB, 75 %\n',
                'gpu_uuid, pid, process_name, used_gpu_memory [MiB]\n'
                'GPU-BUSY, 123, python, 4000 MiB\n',
            ]
            with patch('human_data_generator.run.resource_path', return_value=blender), \
                 patch('human_data_generator.run.subprocess.check_output', side_effect=snapshots):
                self.assertEqual(run._check_gpu('GPU-BUSY')['selected_index'], 1)
            with patch('human_data_generator.run.resource_path', return_value=blender), \
                 patch('human_data_generator.run.subprocess.check_output', side_effect=snapshots):
                with self.assertRaisesRegex(RuntimeError, 'selected_GPU_unavailable'):
                    run._check_gpu('GPU-MISSING')

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

    def test_editor_timeline_uses_generation_route_and_clock(self):
        scene = default_scene('city')
        scene['fps'] = 8
        scene['frame_count'] = 6
        scene['people'][0].update(person_id=7, route_xy_m=[[0, 77.96], [2, 84], [0, 92]],
                                  start_progress_m=1.0, speed_m_s=1.5)
        second = deepcopy(scene['people'][0])
        second['person_id'] = 13
        second['route_xy_m'] = [[x + 2, y] for x, y in second['route_xy_m']]
        scene['people'].append(second)
        handler = server.Handler.__new__(server.Handler)
        handler.path = '/api/trajectory'
        handler.wfile = BytesIO()
        handler.send_response = lambda code: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler._body = lambda: {'scene': scene}
        handler.do_POST()
        positions = json.loads(handler.wfile.getvalue())
        self.assertEqual(set(positions['people']), {'7', '13'})
        self.assertEqual(positions['fps'], 8)
        self.assertEqual(len(positions['people']['7']), 6)
        for frame in (0, 3, 5):
            expected, direction = _curve(scene['people'][0]['route_xy_m'], 1.0 + 1.5 * frame / 8)
            np.testing.assert_allclose(positions['people']['7'][frame]['xy'], expected, atol=1e-10)
            np.testing.assert_allclose(positions['people']['7'][frame]['direction'], direction,
                                       atol=1e-10)
            np.testing.assert_allclose(positions['people']['13'][frame]['xy'], expected + [2, 0],
                                       atol=1e-10)
        self.assertGreater(np.linalg.norm(np.array(positions['people']['7'][5]['xy']) -
                                           positions['people']['7'][0]['xy']), 0.5)
        third = deepcopy(scene['people'][0])
        third['person_id'] = 21
        third['start_progress_m'] = 999
        scene['people'].append(third)
        handler.wfile = BytesIO()
        handler.do_POST()
        partial = json.loads(handler.wfile.getvalue())
        self.assertEqual(set(partial['people']), {'7', '13'})
        self.assertIn('person_21:frame_0:route_time_out_of_bounds', partial['errors']['21'])
        third['start_progress_m'] = 0
        third['route_xy_m'] = [[0, 78], [0, 78.4]]
        handler.wfile = BytesIO()
        handler.do_POST()
        partial = json.loads(handler.wfile.getvalue())
        self.assertEqual(set(partial['people']), {'7', '13', '21'})
        self.assertEqual(len(partial['people']['21']), 3)
        self.assertIn('person_21:frame_3:route_time_out_of_bounds', partial['errors']['21'])
        scene['people'].pop()
        scene['people'][0]['start_progress_m'] = 999
        handler.wfile = BytesIO()
        handler.do_POST()
        partial = json.loads(handler.wfile.getvalue())
        self.assertEqual(set(partial['people']), {'13'})
        self.assertIn('person_7:frame_0:route_time_out_of_bounds', partial['errors']['7'])
        with self.assertRaisesRegex(ValueError, 'person_7:frame_0:route_time_out_of_bounds'):
            route_positions(scene)

    def test_editor_camera_arrow_convention_matches_physical_lenses(self):
        for yaw_deg in (0, 90, 298):
            _, lenses, _ = rig_poses([0, 0, 0], yaw_deg, 0.18)
            angle = np.deg2rad(yaw_deg)
            cam2_axis = np.array([-np.sin(angle), np.cos(angle), 0])
            np.testing.assert_allclose(lenses['cam2'][:3, 2], cam2_axis, atol=1e-12)
            np.testing.assert_allclose(lenses['cam4'][:3, 2], -cam2_axis, atol=1e-12)

    def test_overlapping_people_and_camera_colocation_allowed_but_road_checked(self):
        scene = default_scene('city')
        second = deepcopy(scene['people'][0])
        second['person_id'] = 2
        scene['people'].append(second)
        scene['rigs'][0]['origin_m'][:2] = scene['people'][0]['route_xy_m'][0]
        self.assertEqual(validate(scene)['status'], 'passed')
        template = {'template_id': 'overlap', 'base_scene': scene}
        self.assertEqual([p['person_id'] for p in sample_scene(template, scene['seed'], 0)['people']],
                         [1, 2])

        off_road = deepcopy(scene)
        off_road['people'][1]['route_xy_m'] = [[x + 10, y] for x, y in second['route_xy_m']]
        self.assertTrue(any('outside_verified_road' in error for error in
                            validate(off_road, frames=[0])['errors']))

    def test_editor_rules_export_batch_and_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
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
            scene=default_scene('city');scene['frame_count']=2
            scene['people'][0]['route_xy_m'].insert(1,[0,82])
            other=deepcopy(scene['people'][0]);other['person_id']=7
            other['route_xy_m']=[[x+1.3,y] for x,y in other['route_xy_m']]
            scene['people'].append(other)
            rig=deepcopy(scene['rigs'][0]);rig['rig_id']='rig_other';rig['origin_m'][0]+=.4
            scene['rigs'].append(rig)
            rules=dict(schema='human_batch_v4',seed=47,templates=[
                dict(template_id='pair',base_scene=scene,
                     people=[dict(person_id=7,speed_m_s=[1.1,1.2])],
                     rigs=[dict(rig_id='rig_other',offset_yaw_deg=[0,1])])])
            with patch.object(server,'RULES',root/'rules'), \
                 patch.object(server,'RECIPES',root/'recipes'), \
                 patch.object(batch,'PROJECT_ROOT',root), \
                 patch('subprocess.Popen',side_effect=AssertionError('started process')):
                saved=request('/api/save',{'scene':scene})
                self.assertEqual(request('/api/recipes')['names'],[saved['id']])
                preview=request('/api/try-rules',{'rules':rules})['scenes']
                self.assertEqual(len(preview),2)
                self.assertEqual(sample_batch(rules,3)[:2],preview)
                exported=request('/api/export-rules',{'name':'pair-rules','rules':rules})
                self.assertTrue(Path(exported['path']).is_file())
                loaded=request('/api/rule?name=pair-rules')
                self.assertEqual(loaded,rules)
                self.assertNotIn('count',loaded['templates'][0])
                self.assertIn('--count 10',exported['command'])
                self.assertIn('--name pair-rules',exported['command'])
                self.assertEqual(request('/api/rules')['names'],['pair-rules'])
                put(root/'rules/old.json',dict(schema='human_batch_v3',seed=47,templates=[]))
                self.assertEqual(request('/api/rules')['names'],['pair-rules'])
                self.assertEqual(request('/api/try-rules',{'rules':loaded})['scenes'],preview)
                duplicate=request('/api/export-rules',{'name':'pair-rules','rules':rules})
                self.assertIn('File exists',duplicate['error'])
                batch.main(['--rules',exported['path'],'--name','pair-sample','--count','2'])
                paths=sorted((root/'outputs/scenes/pair-sample').glob('*.json'))
                self.assertEqual(len(paths),2)
                self.assertEqual(json.loads(paths[0].read_text()),preview[0])
                with patch('sys.stdout',new_callable=StringIO) as stdout:
                    run.main(['plan','--scene',str(paths[0])])
                self.assertEqual(json.loads(stdout.getvalue())['camera_calls'],8)
                multi=deepcopy(rules)
                second=deepcopy(multi['templates'][0]);second['template_id']='second'
                second['base_scene']['scene_id']='second_base'
                multi['templates'].append(second)
                self.assertEqual(len(request('/api/try-rules',{'rules':multi})['scenes']),4)
                self.assertEqual(len(request('/api/export-rules',{'name':'multi','rules':multi}).get('error','')),0)
                self.assertEqual(len(request('/api/rule?name=multi')['templates']),2)

    def test_scene_sampling_and_clock(self):
        too_long=default_scene('city');too_long['frame_count']=26
        self.assertIn('person_1:APD6526:motion_exceeds_cache:required_s=3.125',
                      validate(too_long)['errors'][0])
        rules=json.loads((PROJECT_ROOT/'config/batch_rules.json').read_text())
        self.assertEqual(rules['schema'],'human_batch_v4')
        self.assertEqual(len(rules['templates']),4)
        self.assertEqual(len(sample_batch(rules,1)),4)
        self.assertEqual(len(sample_batch(rules,2)),8)
        template=deepcopy(rules['templates'][2])
        template['base_scene']['people'][0]['route_xy_m'].insert(1,[-1.4,84])
        second=deepcopy(template['base_scene']['rigs'][0]);second['rig_id']='rig_other'
        second['origin_m'][0]+=.3
        template['base_scene']['rigs'].append(second)
        template.pop('route_offset_xy_m')
        template.pop('people')
        template.pop('rigs')
        frozen=deepcopy(template)
        scene=sample_scene(template,rules['seed'],0)
        self.assertEqual(template,frozen)
        self.assertEqual(len(scene['people'][0]['route_xy_m']),3)
        self.assertEqual(scene['people'][0]['route_xy_m'],frozen['base_scene']['people'][0]['route_xy_m'])
        self.assertEqual(scene['rigs'],frozen['base_scene']['rigs'])
        self.assertEqual([p['person_id'] for p in scene['people']],[1,2])
        expected=deepcopy(frozen['base_scene'])
        expected['scene_id']=scene['scene_id'];expected['seed']=scene['seed']
        self.assertEqual(scene,expected)
        self.assertEqual(scene,sample_scene(template,rules['seed'],0))
        with self.assertRaises(ValueError):sample_batch({'schema':'human_batch_v2','seed':1,'scenes':[]},1)
        with self.assertRaisesRegex(ValueError,'invalid_batch_count'):sample_batch(rules,0)
        with self.assertRaisesRegex(ValueError,'invalid_template_fields'):
            sample_batch(dict(rules,templates=[dict(rules['templates'][0],count=2)]),1)
        validate_scene(scene)
        people,times,matrices=compile_scene(scene,[0,1])
        self.assertEqual([p['person_id'] for p in people],[1,2])
        self.assertEqual(matrices.shape,(16,2,4,4))
        self.assertEqual(len(times['2']),16)

    def test_batch_offsets_and_candidates(self):
        rules=json.loads((PROJECT_ROOT/'config/batch_rules.json').read_text())
        template=deepcopy(rules['templates'][2])
        template['base_scene']['people'][0]['route_xy_m'].insert(1,[-1.4,84])
        template['route_offset_xy_m']=[[.1,.1],[.2,.2]]
        template['people'][0]['route_offset_xy_m']=[[.3,.3],[0,0]]
        template['rigs'][0]['offset_xy_m']=[[.5,.5],[0,0]]
        template['rigs'][0]['offset_yaw_deg']=[2,2]
        untouched_rig=deepcopy(template['base_scene']['rigs'][0]);untouched_rig['rig_id']='rig_other'
        untouched_rig['origin_m'][0]+=.2
        template['base_scene']['rigs'].append(untouched_rig)
        base=deepcopy(template['base_scene'])
        scene=sample_scene(template,rules['seed'],0)
        for original,shifted in zip(base['people'][0]['route_xy_m'],scene['people'][0]['route_xy_m']):
            np.testing.assert_allclose(np.array(shifted)-original,[.4,.2])
        for original,shifted in zip(base['people'][1]['route_xy_m'],scene['people'][1]['route_xy_m']):
            np.testing.assert_allclose(np.array(shifted)-original,[.1,.2])
        self.assertAlmostEqual(scene['rigs'][0]['origin_m'][0]-base['rigs'][0]['origin_m'][0],.5)
        self.assertAlmostEqual(scene['rigs'][0]['yaw_deg']-base['rigs'][0]['yaw_deg'],2)
        self.assertEqual(scene['rigs'][1],base['rigs'][1])
        policy=template['people'][0]['components']
        policy['body']={'default_versions_only':False,'ids':['F04_v1']}
        policy['motion']={'behavior':'ride','ids':['APD6526']}
        policy['clothing']={'kinds':['simple_barycentric'],'ids':['M03_simple']}
        policy['hair']={'ids':['M01_groom']}
        selected=sample_scene(template,rules['seed'],0)['people'][0]
        self.assertEqual((selected['body'],selected['motion'],selected['clothing'],selected['hair']),
                         ('F04_v1','APD6526','M03_simple','M01_groom'))
        reordered=deepcopy(template)
        reordered['people'][0]['components']['hair']['ids']=['M03_groom','M01_groom']
        normal=sample_scene(reordered,rules['seed'],0)
        reordered['people'][0]['components']['hair']['ids'].reverse()
        self.assertEqual(normal,sample_scene(reordered,rules['seed'],0))
        by_id={item['scene_id']:item for item in sample_batch(rules,1)}
        rules['templates'].reverse()
        self.assertEqual({item['scene_id']:item for item in sample_batch(rules,1)},by_id)
        policy['body']['default_versions_only']=True
        with self.assertRaisesRegex(ValueError,'person_1:body_candidate_conflict'):
            sample_scene(template,rules['seed'],0)

    def test_batch_locked_clothing_and_rejections(self):
        rules=json.loads((PROJECT_ROOT/'config/batch_rules.json').read_text())
        template=deepcopy(rules['templates'][0]);template['base_scene']['frame_count']=2
        template.pop('route_offset_xy_m');template.pop('rigs')
        person=template['people'][0]
        person['components']={
            'body':{'default_versions_only':True,'ids':['M03']},
            'motion':{'behavior':'walk','ids':['KIT11_walk']},
            'clothing':{'kinds':['native_locked'],'ids':['M03_KIT11_BEDLAM_transfer']},
        }
        person['motion_start']=[0,0]
        self.assertEqual(sample_scene(template,rules['seed'],0)['people'][0]['clothing'],
                         'M03_KIT11_BEDLAM_transfer')
        bad=deepcopy(template)
        bad['people'][0]['components']['body']['ids']=['M01']
        with self.assertRaisesRegex(ValueError,'person_1:clothing_candidate_conflict'):
            sample_scene(bad,rules['seed'],0)
        bad=deepcopy(template)
        bad['people'][0]['components']['motion']['ids']=['missing_motion']
        with self.assertRaisesRegex(ValueError,'person_1:unknown_motion'):
            sample_scene(bad,rules['seed'],0)
        inventory=deepcopy(describe())
        locked=next(row for row in inventory['clothing'] if row['id']=='M03_KIT11_BEDLAM_transfer')
        locked['locked_window']=[0,1]
        with self.assertRaisesRegex(ValueError,'person_1:clothing_candidate_conflict'):
            sample_scene(template,rules['seed'],0,inventory)
        for key,value in [('motion_start',[9,1]),('speed_m_s',[0,float('nan')])]:
            bad=deepcopy(template);bad['people'][0][key]=value
            with self.assertRaises(ValueError):sample_scene(bad,rules['seed'],0)
        bad=deepcopy(template);bad['people'][0]['person_id']=42
        with self.assertRaisesRegex(ValueError,'invalid_person_rule'):
            sample_scene(bad,rules['seed'],0)
        bad=deepcopy(template);bad['rigs']=[{'rig_id':'rig_missing','offset_yaw_deg':[0,1]}]
        with self.assertRaisesRegex(ValueError,'invalid_rig_rule'):
            sample_scene(bad,rules['seed'],0)
        bad=deepcopy(template);bad['people'][0]['motion_behavior']='walk'
        with self.assertRaisesRegex(ValueError,'invalid_person_rule_fields'):
            sample_scene(bad,rules['seed'],0)

    def test_batch_cli_writes_only_scene_files(self):
        rules=PROJECT_ROOT/'config/batch_rules.json'
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(batch,'PROJECT_ROOT',Path(tmp)), \
                 patch('human_data_generator.composition.build',side_effect=AssertionError('built geometry')), \
                 patch('subprocess.Popen',side_effect=AssertionError('started process')):
                batch.main(['--rules',str(rules),'--name','sample','--count','1'])
            folder=Path(tmp)/'outputs/scenes/sample'
            files=sorted(folder.glob('*.json'))
            self.assertEqual(len(files),4)
            self.assertTrue(all(json.loads(path.read_text())['schema']=='human_scene_v2' for path in files))
            with patch('sys.stdout',new_callable=StringIO) as stdout:
                run.main(['plan','--scene',str(files[0])])
            self.assertGreater(json.loads(stdout.getvalue())['camera_calls'],0)
            old=Path(tmp)/'old_rules.json'
            old.write_text(json.dumps({'schema':'human_batch_v2','seed':1,'scenes':[]}))
            with patch.object(batch,'PROJECT_ROOT',Path(tmp)):
                with self.assertRaisesRegex(ValueError,'invalid_batch_rules'):
                    batch.main(['--rules',str(old),'--name','old','--count','1'])
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
            with patch('human_data_generator.run._preflight_consumer'),\
                 patch('human_data_generator.run._check_gpu',return_value={}),\
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

    def test_task_status_has_one_parent_writer_and_no_false_completion(self):
        scene=PROJECT_ROOT/'config/scene_example.json'
        with tempfile.TemporaryDirectory(prefix='hdg_status_') as temp:
            prepared=Path(temp)/'prepared'
            prepare(prepared,scene_path=scene,frame_indices=[5],selected_rig_ids=['rig_side'])
            for name in ('success','render_failed','timed_out','export_failed','video_failed'):
                shutil.copytree(prepared,Path(temp)/name)

            def execute(name, *, blender=None, export_error=None, video_error=None, video=False):
                output=Path(temp)/name
                render_result=blender or dict(returncode=0,timed_out=False,wall_seconds=.1)
                def consumer_stage(stage,*args,**kwargs):
                    if stage=='video' and video_error:raise RuntimeError(video_error)
                def exporter(root):
                    if export_error:raise RuntimeError(export_error)
                    return dict(observations=[dict(status='observed')])
                with patch('human_data_generator.run._preflight_consumer'),\
                     patch('human_data_generator.run._check_gpu',return_value={}),\
                     patch('human_data_generator.run._run_blender',return_value=render_result),\
                     patch('human_data_generator.run._run_consumer_stage',side_effect=consumer_stage),\
                     patch('human_data_generator.data.mamma_export.export_existing',side_effect=exporter):
                    if name in ('success','video_failed'):
                        render_prepared(output,'MOCK_GPU',60,video=video)
                    else:
                        with self.assertRaises(RuntimeError):
                            render_prepared(output,'MOCK_GPU',60)
                return output,json.loads((output/'status.json').read_text())

            success,status=execute('success')
            self.assertEqual((status['state'],status['data_complete']),('data_complete',True))
            self.assertTrue((success/'observations.json').is_file())
            self.assertFalse(any((success/name).exists() for name in
                                 ('started.json','failure.json','data_complete.json','blender_process.json')))
            (Path(temp)/'render_failed'/'partial.exr').write_bytes(b'failure site')
            failure,status=execute('render_failed',blender=dict(returncode=1,timed_out=False,wall_seconds=.1))
            self.assertEqual((status['state'],status['data_complete']),('failed',False))
            self.assertTrue((failure/'partial.exr').is_file())
            self.assertFalse((failure/'observations.json').exists())
            _,status=execute('timed_out',blender=dict(returncode=-9,timed_out=True,wall_seconds=60))
            self.assertTrue(status['blender']['timed_out'])
            self.assertEqual(status['state'],'failed')
            output,status=execute('export_failed',export_error='fixture_export_rejected')
            self.assertEqual((status['stage'],status['data_complete']),('export',False))
            self.assertFalse((output/'observations.json').exists())
            output,status=execute('video_failed',video_error='fixture_encoder',video=True)
            self.assertEqual((status['state'],status['video_status']),('data_complete','failed'))
            self.assertTrue((output/'observations.json').is_file())

    def test_consumer_protocol_keeps_stdout_as_json_lines(self):
        with tempfile.TemporaryDirectory(prefix='hdg_consumer_protocol_') as temp:
            root=Path(temp)
            put(root/'run_config.json',dict(schema='human_run_config_v2',
                rig_ids=['rig_side'],selected_frames=[0],task_id='a'*32))
            rig=root/'rigs/rig_side'
            (rig/'frame_000').mkdir(parents=True)
            put(root/'scene.json',{})
            put(root/'recipe.json',{})
            np.save(root/'scene_matrices.npy',np.empty((0,)))
            put(rig/'rig.json',{})
            put(rig/'planned_regions.json',{})
            seen=[]
            def consume(path,frame):
                print('consumer diagnostic')
                seen.append((path,frame))
            incoming=StringIO(json.dumps(dict(command='frame',frame=0,rig_id='rig_side'))+'\n'
                +json.dumps(dict(command='stop'))+'\n')
            outgoing=StringIO()
            self.assertEqual(serve_frames(root,consume,incoming=incoming,outgoing=outgoing),0)
            replies=[json.loads(line) for line in outgoing.getvalue().splitlines()]
            self.assertEqual([row['status'] for row in replies],['ready','ok','stopped'])
            self.assertEqual(seen,[(rig,0)])
            self.assertIn('consumer diagnostic',(rig/'frame_000/consume.log').read_text())
            bad=StringIO(json.dumps(dict(command='frame',frame=0,rig_id='rig_other'))+'\n')
            result=StringIO()
            self.assertEqual(serve_frames(root,consume,incoming=bad,outgoing=result),1)
            self.assertEqual(json.loads(result.getvalue().splitlines()[-1])['status'],'error')

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
                with patch('human_data_generator.run._preflight_consumer'),\
                     patch('human_data_generator.run._check_gpu',return_value={}),\
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
            (root/'rigs/rig_alpha').mkdir(parents=True)
            put(root/'rigs/rig_alpha/planned_regions.json',
                {'1': {cid: {'roi_xyxy': [0,0,4,4]} for cid in ('cam2','cam4')}})
            with patch.object(roi_module,'render_cameras',side_effect=capture_camera),\
                 patch.object(roi_module,'border_coordinates',return_value=(0.,1.,0.,1.)):
                result=roi_module.render_rig_roi(scene,None,SimpleNamespace(directory=''),source,root,
                    dict(rig_id='rig_alpha',people=[dict(person_id=7)]),
                    [dict(id=cid,camera_to_world=image) for cid in ('cam2','cam4')],
                    {'body':np.array([[0.,0.,1.]])},{'body':7},1,
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
                    return dict(status='ok',frame=frame,rig_id=rig_id,seconds=0.,pid=1234)
                def close(self):return dict(pid=1234,frames=self.responses,returncode=0)
                def abort(self):raise AssertionError('worker aborted')
            def fake_roi(scene,cam,node,folder,task_root,recipe,cameras,parts,owners,frame,started):
                calls.append((frame,recipe['rig_id'],[c['id'] for c in cameras],task_root))
                put(folder/'roi_plan.json',dict(cameras={c['id']:dict(roi_xyxy=[0,0,4,4]) for c in cameras}))
                return dict(records=[dict(id=c['id'],source=f'rigs/{recipe["rig_id"]}/frame_{frame:03d}/source.exr')
                                     for c in cameras])
            def fake_assembly(*,root,render_cameras_fn,frame_complete_fn,frame_source_dir_fn,
                              reuse_session,event_fn):
                scene=SimpleNamespace(objects=[],render=SimpleNamespace(use_persistent_data=True))
                for frame in (1,3):
                    folder=frame_source_dir_fn(frame)
                    result=render_cameras_fn(scene,None,None,folder,frame_index=frame)
                    frame_complete_fn(frame,folder,dict(sources=result,frame_index=frame))
                return dict(scene_reused=False)
            with patch.object(multi,'FrameConsumer',Worker),\
                 patch.object(multi,'render_rig_roi',side_effect=fake_roi),\
                 patch.object(multi,'scene_settings',return_value={}),\
                 patch.object(multi.assembly,'main',side_effect=fake_assembly):
                multi.main(root)
            self.assertEqual([(frame,rid,cids) for frame,rid,cids,_ in calls],
                [(frame,rid,['cam2','cam4']) for frame in (1,3) for rid in ('rig_alpha','rig_beta')])
            self.assertTrue(all(path==root for *_,path in calls))
            self.assertFalse((root/'frame_001').exists())
            events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
            self.assertEqual([event['frame_index'] for event in events
                              if event['event']=='geometry_updated'],[1,3])
            self.assertEqual(events[-1]['geometry_updates'],2)
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
            prior=assembly.AssembledPerson(
                person,{},vertices,vertices,np.broadcast_to(np.eye(4),(2,4,4)),matrices,
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
                assembly_result=assembly.main(root=root,render_cameras_fn=render_callback,
                    frame_complete_fn=complete,frame_source_dir_fn=source_folder,reuse_session=True)
            self.assertEqual([row[0] for row in seen],[0,1])
            np.testing.assert_array_equal(seen[1][1],vertices[1])
            np.testing.assert_array_equal(seen[1][2],vertices[1])
            self.assertEqual(seen[1][3][0,3],.2)
            self.assertTrue(assembly_result['scene_reused'])
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

    def test_two_frame_one_rig_json_inventory(self):
        with tempfile.TemporaryDirectory(prefix='hdg_record_inventory_') as temp:
            root=Path(temp)
            manifest=synthetic_bundle(root,frames=(0,5),rigs=(('rig_side',True),))
            put(root/'status.json',dict(schema='human_task_status_v1',state='data_complete',
                                        data_complete=True))
            (root/'events.jsonl').write_text(json.dumps(dict(event='render_completed'))+'\n')
            files=sorted(str(path.relative_to(root)) for path in root.rglob('*.json'))
            self.assertEqual(files,[
                'observations.json','recipe.json',
                'rigs/rig_side/frame_000/cam2/raster_storage.json',
                'rigs/rig_side/frame_000/cam4/raster_storage.json',
                'rigs/rig_side/frame_005/cam2/raster_storage.json',
                'rigs/rig_side/frame_005/cam4/raster_storage.json',
                'rigs/rig_side/label_policy.json','rigs/rig_side/planned_regions.json',
                'rigs/rig_side/rig.json','run_config.json','scene.json','status.json'])
            self.assertEqual(len(manifest['observations']),8)
            self.assertFalse(any(root.rglob('generation_status.json')))


if __name__=='__main__':unittest.main()
