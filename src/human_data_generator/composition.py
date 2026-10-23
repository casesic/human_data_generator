"""Build only the selected SMPL-X frames and appearance for a current scene.

The source parameters and character references stay read-only. Cache keys use
component versions and file metadata; an incomplete cache is always an error.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .components import select
from .config import PROJECT_ROOT

POSE = ('global_orient','body_pose','jaw_pose','leye_pose','reye_pose',
        'left_hand_pose','right_hand_pose','expression')
DIMS = (3,63,3,3,3,45,45,10)
WALK = 'source_root_relative_to_start_plus_fixed_scene_matrix'
FIXED = 'fixed_pelvis_scene_owns_progression'
CONV = np.array([[1,0,0],[0,0,-1],[0,1,0]],np.float32)
CACHE = PROJECT_ROOT/'outputs/cache/components'


def _read(path):return json.loads(Path(path).read_text())
def _write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def _stamp(path):
    path=Path(path).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    stat=path.stat()
    if stat.st_size==0:raise ValueError('empty_component_dependency:'+str(path))
    return [str(path),stat.st_size,stat.st_mtime_ns]


def _key(kind,fields,files):
    frozen=dict(schema='human_component_cache_v1',kind=kind,fields=fields,
                dependencies=[_stamp(p) for p in files])
    digest=hashlib.blake2b(json.dumps(frozen,sort_keys=True).encode(),digest_size=16).hexdigest()
    return frozen,digest


def _cached(kind,frozen,digest,required,make):
    folder=CACHE/kind/digest
    if folder.exists():
        if _read(folder/'complete.json')!=frozen or any(not (folder/name).is_file() for name in required):
            raise ValueError('incomplete_or_changed_component_cache:'+str(folder))
        return folder,True
    folder.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.building_',dir=folder.parent))
    try:
        make(temporary)
        if any(not (temporary/name).is_file() for name in required):
            raise ValueError('component_build_missing_output')
        _write(temporary/'complete.json',frozen)
        if folder.exists():raise FileExistsError('concurrent_component_build:'+str(folder))
        os.replace(temporary,folder)
    except Exception:
        # Preserve a failed build for inspection; it is never accepted as cache.
        raise
    return folder,False


def _vertex_normals(vertices,faces):
    tri=vertices[faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    total=np.zeros_like(vertices,dtype=np.float64)
    for corner in range(3):np.add.at(total,faces[:,corner],normal)
    return total/np.maximum(np.linalg.norm(total,axis=1,keepdims=True),1e-12)


def _bound(vertices,faces,ids,weights,offset):
    surface=(vertices[ids]*weights[...,None]).sum(1)
    normal=(_vertex_normals(vertices,faces)[ids]*weights[...,None]).sum(1)
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    result=(surface+offset*normal).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('nonfinite_bound_clothing')
    return result


def _head_transform(local,parents,posed,rest):
    rotations=np.asarray(local).copy()
    for k in range(1,16):rotations[k]=rotations[int(parents[k])]@rotations[k]
    matrix=np.eye(4)
    matrix[:3,:3]=rotations[15]
    matrix[:3,3]=posed-matrix[:3,:3]@rest
    if not np.isfinite(matrix).all():raise ValueError('nonfinite_head_transform')
    return matrix


def _triangle_frame(vertices,tri):
    face=vertices[tri]
    tangent=face[:,1]-face[:,0]
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-10)
    normal=np.cross(tangent,face[:,2]-face[:,0])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-10)
    return np.stack((tangent,np.cross(normal,tangent),normal),axis=2)


def _transfer_hair(source,target_rest,folder):
    """Frozen scalp-triangle frames; one static groom, not a per-frame copy."""
    gm=_read(source/'groom_manifest.json')
    with np.load(Path(gm['binding'])/'binding.npz',allow_pickle=False) as binding:
        tri=np.asarray(binding['body_faces'])
        weights=np.asarray(binding['weights'])
        offsets=np.asarray(binding['curve_offsets'])
        local=np.asarray(binding['local_points'])
        frames=np.asarray(binding['target_rest_frames'])
    source_rest=np.load(Path(gm['body_geometry'])/'vertices.npy',mmap_mode='r')[0]@CONV.T
    target=target_rest@CONV.T
    rotation=np.einsum('nij,nkj->nik',_triangle_frame(target,tri),_triangle_frame(source_rest,tri))
    posed_frames=np.einsum('nij,njk->nik',rotation,frames)
    roots=(target[tri]*weights[:,:,None]).sum(1)+posed_frames[:,:,2]*.0004
    correction=np.load(source/'collision_correction.npy',mmap_mode='r')
    points=np.lib.format.open_memmap(folder/'points.npy',mode='w+',dtype=np.float32,shape=local.shape)
    for first in range(0,len(offsets)-1,4096):
        end=min(first+4096,len(offsets)-1)
        start_point,stop_point=int(offsets[first]),int(offsets[end])
        counts=np.diff(offsets[first:end+1])
        curve=np.repeat(np.arange(first,end),counts)
        base=roots[curve]+np.einsum('nij,nj->ni',posed_frames[curve],local[start_point:stop_point])
        adjust=np.einsum('nij,nj->ni',rotation[curve],correction[start_point:stop_point]@CONV.T)
        points[start_point:stop_point]=((base+adjust)@CONV).astype(np.float32)
    points.flush();del points
    for name in ('radius.npy','curve_offsets.npy','active_curve_mask.npy'):
        shutil.copyfile(source/name,folder/name)
    gm['transfer']='target_rest_scalp_triangle_frame_v1'
    gm['source_groom_reference']=str(source)
    _write(folder/'groom_manifest.json',gm)
    if not np.isfinite(np.load(folder/'points.npy',mmap_mode='r')).all():
        raise ValueError('nonfinite_transferred_hair')


def _model_rest(body):
    import torch
    import smplx
    character=_read(body['manifest'])
    model=smplx.SMPLX(character['model_dir'],gender=character['gender'],ext='npz',
                      num_betas=30,use_pca=False,flat_hand_mean=True,batch_size=1).eval()
    model.requires_grad_(False)
    if model.use_pca or not model.flat_hand_mean or tuple(model.shapedirs.shape)!=(10475,3,30):
        raise ValueError('unexpected_smplx_basis')
    beta=np.asarray(character['betas'],np.float32)
    with torch.no_grad():
        zero=model(betas=torch.from_numpy(beta[None]))
        translation=-zero.joints[0,0].numpy().astype(np.float32)
        rest=model(betas=torch.from_numpy(beta[None]),transl=torch.from_numpy(translation[None]))
    faces=np.asarray(model.faces)
    reference=Path(character['reference']['geometry'])
    np.testing.assert_array_equal(faces,np.load(reference/'faces.npy'))
    if np.max(np.abs(rest.vertices[0].numpy()-np.load(reference/'vertices.npy')[0]))>2e-6:
        raise ValueError('target_rest_reference_conflict')
    return model,beta,translation,rest.vertices[0].numpy(),rest.joints[0].numpy(),faces


def build(person,source_indices):
    """Return a render-consumable descriptor and cache reuse report."""
    chosen=select(person)
    body,motion,cloth,hair=(chosen[k] for k in ('body','motion','clothing','hair'))
    rows=_read(motion['timeline'])
    if not source_indices or min(source_indices)<0 or max(source_indices)>=len(rows):
        raise ValueError('motion_window_out_of_bounds')
    first,end=min(source_indices),max(source_indices)+1
    window=rows[first:end]
    if any(row['asset_frame']!=i for i,row in enumerate(rows)):
        raise ValueError('source_timeline_not_dense')
    character=_read(body['manifest'])
    model_file=Path(character['model_file'])
    body_files=[body['manifest'],model_file,motion['parameters'],motion['timeline'],
                Path(character['reference']['geometry'])/'faces.npy',
                Path(character['reference']['geometry'])/'vertices.npy']
    body_frozen,body_digest=_key('body',[body['id'],body['version'],motion['id'],motion['version'],first,end],body_files)
    def make_body(out):
        import torch
        from smplx.lbs import batch_rodrigues
        torch.set_num_threads(2)
        model,beta,translation,rest,joints_rest,faces=_model_rest(body)
        with np.load(motion['parameters'],allow_pickle=False) as source:
            n=len(source['global_orient'])
            if n!=len(rows) or source['betas'].shape not in ((10,),(30,)) or source['transl'].shape!=(n,3):
                raise ValueError('motion_shape_or_timeline_conflict')
            for key,dim in zip(POSE,DIMS):
                if key not in source and key=='expression' and motion.get('expression_policy')=='neutral_zero_missing':continue
                if key not in source or source[key].shape!=(n,dim):raise ValueError('motion_pose_basis:'+key)
            root_mode=motion['root_mode']
            if root_mode==WALK:
                if 'source_root_relative' not in source or source['source_root_relative'].shape!=(n,3):
                    raise ValueError('walking_source_root_missing')
                relative=np.asarray(source['source_root_relative'][first:end],np.float32)
                if not np.allclose(source['transl']-source['transl'][0],source['source_root_relative'],atol=2e-6,rtol=0):
                    raise ValueError('walking_source_root_conflict')
            elif root_mode==FIXED:
                if not np.allclose(source['transl'],source['transl'][0],atol=1e-7) or 'source_root_relative' in source:
                    raise ValueError('fixed_motion_root_conflict')
                relative=np.zeros((end-first,3),np.float32)
            else:raise ValueError('unsupported_motion_root_mode')
            params={key:(np.asarray(source[key][first:end],np.float32).copy() if key in source else np.zeros((end-first,10),np.float32)) for key in POSE}
        params.update(betas=beta,transl=relative+translation[None])
        if motion['root_mode']==WALK:params['source_root_relative']=relative
        vertices=np.empty((end-first,10475,3),np.float32)
        heads=np.empty((end-first,4,4),np.float64)
        with torch.no_grad():
            for i in range(end-first):
                kwargs={key:torch.from_numpy(params[key][i:i+1]) for key in POSE}
                result=model(**kwargs,betas=torch.from_numpy(beta[None]),
                             transl=torch.from_numpy(params['transl'][i:i+1]))
                vertices[i]=result.vertices[0].numpy()
                local=batch_rodrigues(torch.from_numpy(np.concatenate((params['global_orient'][i],params['body_pose'][i])).reshape(-1,3))).numpy()
                heads[i]=_head_transform(local,model.parents.numpy(),result.joints[0,15].numpy(),joints_rest[15])
        if not np.isfinite(vertices).all() or not np.isfinite(heads).all():raise ValueError('nonfinite_body')
        np.save(out/'vertices.npy',vertices);np.save(out/'faces.npy',faces)
        np.save(out/'head_transforms.npy',heads);np.savez(out/'parameters.npz',**params)
    body_dir,body_hit=_cached('body',body_frozen,body_digest,
                              ('vertices.npy','faces.npy','head_transforms.npy','parameters.npz'),make_body)
    vertices=np.load(body_dir/'vertices.npy',mmap_mode='r')
    faces=np.load(body_dir/'faces.npy',mmap_mode='r')
    if cloth['kind']=='simple_barycentric':
        cloth_source=Path(cloth['reference']);ids=np.load(cloth_source/'body_barycentric_indices.npy')
        weights=np.load(cloth_source/'body_barycentric_weights.npy')
        cloth_faces=np.load(cloth_source/'clothing_faces.npy')
        if (ids.shape!=weights.shape or ids.ndim!=2 or ids.shape[1]!=3 or ids.min()<0 or ids.max()>=10475 or
            not np.allclose(weights.sum(1),1,atol=1e-6) or cloth_faces.min()<0 or cloth_faces.max()>=len(ids)):
            raise ValueError('invalid_simple_clothing_binding')
        cloth_files=[cloth_source/x for x in ('body_barycentric_indices.npy','body_barycentric_weights.npy',
                                               'clothing_faces.npy','materials.json','garment_material.json')]
        offset=float(cloth['offset_m'])
        cloth_frozen,cloth_digest=_key('cloth',[body_digest,cloth['id'],cloth['version'],offset],cloth_files)
        def make_cloth(out):
            array=np.lib.format.open_memmap(out/'clothing_vertices.npy',mode='w+',dtype=np.float32,
                                            shape=(len(vertices),len(ids),3))
            for i in range(len(vertices)):array[i]=_bound(vertices[i],faces,ids,weights,offset)
            array.flush();del array
            np.save(out/'clothing_faces.npy',cloth_faces)
            model,beta,translation,rest,_,_=_model_rest(body)
            np.save(out/'clothing_rest_vertices.npy',_bound(rest,faces,ids,weights,offset))
            for name in ('materials.json','garment_material.json'):shutil.copyfile(cloth_source/name,out/name)
        cloth_dir,cloth_hit=_cached('cloth',cloth_frozen,cloth_digest,
                                    ('clothing_vertices.npy','clothing_faces.npy','clothing_rest_vertices.npy',
                                     'materials.json','garment_material.json'),make_cloth)
    else:
        native=_read(cloth['asset_manifest'])
        low,high=cloth['locked_window']
        if body['id']!=cloth['locked_body'] or motion['id']!=cloth['locked_motion'] or first<low or end>high:
            raise ValueError('native_clothing_locked_body_motion_time')
        if native['asset_id']!=cloth['locked_asset_id'] or native['motion_source_id']!=motion['id']:
            raise ValueError('native_clothing_authority_conflict')
        native_body=np.load(native['target_body_vertices'],mmap_mode='r')
        if (not np.array_equal(np.load(native['target_body_faces']),faces) or
            np.max(np.abs(native_body[first:end]-vertices))>3e-6):
            raise ValueError('native_clothing_target_body_mismatch')
        native_faces=np.load(native['cloth_faces'],mmap_mode='r')
        native_uv=np.load(native['cloth_uv_vertices'],mmap_mode='r')
        native_uv_faces=np.load(native['cloth_uv_faces'],mmap_mode='r')
        if (native_faces.ndim!=2 or native_faces.shape[1]!=3 or
            native_uv.ndim!=2 or native_uv.shape[1]!=2 or not np.isfinite(native_uv).all() or
            native_uv_faces.shape!=native_faces.shape or native_uv_faces.min()<0 or
            native_uv_faces.max()>=len(native_uv)):
            raise ValueError('native_clothing_uv_topology_conflict')
        cloth_files=[cloth['asset_manifest'],native['target_body_vertices'],native['cloth_vertices'],
                     native['cloth_faces'],native['cloth_uv_vertices'],native['cloth_uv_faces'],
                     native['garment_texture_01']['diffuse'],native['garment_texture_01']['normal']]
        cloth_frozen,cloth_digest=_key('native_cloth',[body_digest,cloth['id'],cloth['version'],first,end],cloth_files)
        def make_native(out):
            source=np.load(native['cloth_vertices'],mmap_mode='r')
            if source.shape[0]<end or source.ndim!=3 or source.shape[2]!=3:
                raise ValueError('native_clothing_frame_conflict')
            np.save(out/'clothing_vertices.npy',np.asarray(source[first:end]))
            np.save(out/'clothing_faces.npy',np.load(native['cloth_faces']))
            _write(out/'native_material.json',dict(diffuse=native['garment_texture_01']['diffuse'],
                normal=native['garment_texture_01']['normal'],
                uv_vertices=native['cloth_uv_vertices'],uv_faces=native['cloth_uv_faces']))
        cloth_dir,cloth_hit=_cached('native_cloth',cloth_frozen,cloth_digest,
                                    ('clothing_vertices.npy','clothing_faces.npy','native_material.json'),make_native)
    hair_source=Path(hair['reference']);gm=_read(hair_source/'groom_manifest.json')
    hair_files=[hair_source/x for x in ('points.npy','radius.npy','curve_offsets.npy',
                                        'active_curve_mask.npy','groom_manifest.json','collision_correction.npy')]
    hair_files += [Path(gm['binding'])/'binding.npz',
                   Path(gm['body_geometry'])/'vertices.npy',Path(gm['source_abc'])]
    hair_frozen,hair_digest=_key('hair',[body['id'],body['version'],hair['id'],hair['version'],'render_groom_v3'],
                                [body['manifest'],model_file,Path(character['reference']['geometry'])/'vertices.npy']+hair_files)
    owner_manifest=_read(hair_source/'groom_manifest.json')
    if Path(owner_manifest['body_geometry']).resolve()==Path(character['reference']['geometry']).resolve():
        groom_dir=hair_source;hair_hit=True
    else:
        def make_hair(out):
            model,beta,translation,rest,_,_=_model_rest(body)
            _transfer_hair(hair_source,rest,out)
        groom_dir,hair_hit=_cached('hair',hair_frozen,hair_digest,
                                    ('points.npy','radius.npy','curve_offsets.npy','active_curve_mask.npy','groom_manifest.json'),make_hair)
    selected=[i-first for i in source_indices]
    timeline=[dict(rows[i],asset_frame=i-first) for i in range(first,end)]
    bundle_frozen,bundle_digest=_key('asset',[body_digest,cloth_digest,hair_digest,first,end],[])
    def make_asset(out):
        _write(out/'timeline.json',timeline)
        _write(out/'manifest.json',dict(schema='human_component_asset_v1',
             asset_id='component_'+bundle_digest,components={k:v['id'] for k,v in chosen.items()},
             character_manifest=body['manifest'],parameters=str(body_dir/'parameters.npz'),
             body_vertices=str(body_dir/'vertices.npy'),body_faces=str(body_dir/'faces.npy'),
             cloth_vertices=str(cloth_dir/'clothing_vertices.npy'),
             cloth_faces=str(cloth_dir/'clothing_faces.npy'),clothing_reference=str(cloth_dir),
             groom_reference=str(groom_dir),head_transforms=str(body_dir/'head_transforms.npy'),
             timeline_ref=str(CACHE/'asset'/bundle_digest/'timeline.json'),
             frame_count=end-first,source_group=motion['source_group'],
             root_mode=motion['root_mode'],time_base=timeline[0]['time_base']))
    asset_dir,asset_hit=_cached('asset',bundle_frozen,bundle_digest,('timeline.json','manifest.json'),make_asset)
    low=float(np.min(vertices[selected,:,1]));high=float(np.max(np.min(vertices[selected,:,1],axis=1)))
    gap=.02 if motion['behavior']=='walk' else .25
    placement=-low+gap
    if motion['behavior']=='walk' and high-low>.10:raise ValueError('walking_ground_motion_range')
    support=dict(body_min_clearance_m=[-.02,.12] if motion['behavior']=='walk' else [.05,.7],
                 pelvis_height_above_road_m=[placement-.15,placement+.15])
    return dict(asset_manifest=str(asset_dir/'manifest.json'),asset_id='component_'+bundle_digest,
                source_manifest_id=motion['source_group'],root_mode=motion['root_mode'],
                support_policy='pedestrian_source_root_constant_lift' if motion['behavior']=='walk' else 'fixed_scene_placement_height_no_pedestrian_foot_contact',
                support_checks=support,placement_height_m=placement,
                component_ids={k:v['id'] for k,v in chosen.items()},
                source_indices=selected,
                cache_hits=dict(body=body_hit,clothing=cloth_hit,hair=hair_hit,asset=asset_hit))
