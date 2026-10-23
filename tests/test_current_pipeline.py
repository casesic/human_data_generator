"""Bounded CPU checks for the single current generation/data path."""
import json
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


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def synthetic_bundle(root):
    root=Path(root)
    people=[dict(person_id=7,asset_id='fixture_asset',source_manifest_id='fixture_source'),
            dict(person_id=9,asset_id='fixture_asset',source_manifest_id='fixture_source')]
    put(root/'recipe.json',dict(schema='human_run_v1',scene_episode_id='fixture_episode',
        frame_count=1,fps=8,people=people))
    put(root/'run_config.json',dict(selected_frames=[0],full_image_size_wh=[256,256],
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
    rigs=[]
    for rid,second_visible in [('rig_alpha',True),('rig_beta',False)]:
        child=root/'rigs'/rid
        child.mkdir(parents=True)
        mask=child/'physical.png'
        Image.fromarray(valid.astype(np.uint8)*255).save(mask)
        put(child/'rig.json',dict(rig_id=rid,cameras=[dict(id='cam2'),dict(id='cam4')],
                                  valid_masks={'cam2':str(mask),'cam4':str(mask)}))
        put(child/'label_policy.json',dict(visibility_policy='fixture'))
        instance=np.zeros((256,256),np.float32)
        instance[80:170,80:170]=7
        if second_visible:instance[80:170,175:235]=9
        (child/'frame_000').mkdir()
        for camera in ('cam2','cam4'):
            consume_camera_arrays(child,0,camera,rgb,rng,instance,[0,0,256,256],
                                  pose,valid,geometry)
        rigs.append(dict(rig_id=rid,root=str(child)))
    put(root/'multirig.json',dict(schema='human_multirig_v1',rigs=rigs))
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
            self.assertEqual(json.loads((output/'job.json').read_text())['frames'],[5])
            self.assertFalse((output/'frame_005').exists())
            with patch('human_data_generator.run._resources',return_value={}),\
                 patch('human_data_generator.run.subprocess.Popen',side_effect=RuntimeError('mock_launch')) as popen:
                with self.assertRaisesRegex(RuntimeError,'mock_launch'):
                    render_prepared(output,'MOCK_GPU',60)
            command=popen.call_args.args[0]
            self.assertEqual(command[command.index('--python')+1],
                str(PROJECT_ROOT/'src/human_data_generator/render/multirig.py'))
            self.assertEqual(json.loads((output/'run_config.json').read_text())['gpu_uuid'],'MOCK_GPU')
            self.assertNotIn('remaining_budget_seconds',json.loads((output/'job.json').read_text()))

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
