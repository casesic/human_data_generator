"""Blender consumer of generic dynamic asset manifests and explicit scene matrices.

The recipe owns root/ground/rig placement. No automatic per-frame grounding,
motion-source selection or clothing rebinding occurs in this consumer.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import time
import bpy
import numpy as np
from mathutils import Matrix, Vector
from human_data_generator.rendering.blender.assembly import (
    configure_render,
    configure_compositor,
    create_erp_camera,
    create_mesh_object,
)
from human_data_generator.rendering.blender.materials import (
    apply_character_materials,
    add_clothing_materials,
)
from human_data_generator.rendering.blender.groom import add_groom
from human_data_generator.data.dynamic_scene import (
    geometry_paths,
    frame_record,
    support_limits,
    ROOT_MODE,
)
from human_data_generator.runtime.task_paths import run_config as load_run_config, task_path


_SESSION = None


@dataclass
class LoadedPerson:
    person: dict
    asset: dict
    geometry: dict
    body_vertices: np.ndarray
    matrices: np.ndarray
    asset_frames: list[int]


@dataclass
class AssembledPerson:
    person: dict
    asset: dict
    body_vertices: np.ndarray
    cloth_vertices: np.ndarray
    head_transforms: np.ndarray
    matrices: np.ndarray
    body_object: object
    garment_object: object
    hair_objects: list
    hair_topology: list
    asset_frames: list[int]


def _update_person_geometry(item, frame):
    """Apply one scene matrix and source frame, preserving exact Blender readback checks."""
    p = item.person
    v = item.body_vertices
    cv = item.cloth_vertices
    H = item.head_transforms
    matrices = item.matrices
    body = item.body_object
    garment = item.garment_object
    hair = item.hair_objects
    counts = item.hair_topology
    asset_frames = item.asset_frames
    M = matrices[frame]
    body.matrix_world = Matrix(M.tolist())
    garment.matrix_world = body.matrix_world
    ix = asset_frames[frame]
    body.data.vertices.foreach_set("co", v[ix].ravel())
    body.data.update()
    garment.data.vertices.foreach_set("co", cv[ix].ravel())
    garment.data.update()
    for obj, count in zip(hair, counts):
        obj.matrix_world = Matrix((M @ H[ix]).tolist())
        assert (len(obj.data.curves), len(obj.data.points)) == count
        first = np.asarray(obj.data.position_data[0].vector)
        planned_head_world = M @ H[ix]
        actual_head_world = np.asarray(obj.matrix_world)
        np.testing.assert_array_equal(
            actual_head_world.astype(np.float32), planned_head_world.astype(np.float32)
        )
        actual = np.asarray(obj.matrix_world @ Vector(first))
        expected = planned_head_world[:3, :3] @ first + planned_head_world[:3, 3]
        # The groom is also placed through a float32 Blender Matrix.
        # Compare the exact matrix first, then admit one float32 ULP
        # for its world-space point (not a fixed metric tolerance at
        # an ~80 m scene offset).
        head_ulp = np.spacing(np.float32(np.abs(expected)))
        head_delta = np.abs(actual - expected)
        if not (
            np.isfinite(actual).all() and np.all(head_delta <= np.maximum(2e-6, head_ulp))
        ):
            raise AssertionError(
                f'groom_world_changed:frame={frame}:person={p["person_id"]}:'
                f"max_abs_m={np.nanmax(head_delta)}"
            )
    check = np.empty(v[ix].size, np.float32)
    body.data.vertices.foreach_get("co", check)
    np.testing.assert_array_equal(check.reshape(-1, 3), v[ix])
    cloth_check = np.empty(cv[ix].size, np.float32)
    garment.data.vertices.foreach_get("co", cloth_check)
    np.testing.assert_array_equal(cloth_check.reshape(-1, 3), cv[ix])
    np.testing.assert_array_equal(
        np.asarray(garment.matrix_world), np.asarray(body.matrix_world)
    )
    actual_matrix = np.asarray(body.matrix_world)
    np.testing.assert_array_equal(actual_matrix.astype(np.float32), M.astype(np.float32))


def main(*, root, render_cameras_fn, frame_complete_fn, frame_source_dir_fn,
         reuse_session=False, event_fn=None):
    root = Path(root).resolve()
    config = load_run_config(root)
    ep = json.loads(task_path(root, "recipe.json").read_text())
    frames = config["selected_frames"]
    render = dict(ep["render"], gpu_uuid=config["gpu_uuid"])
    began = time.monotonic()
    assert (
        ep["geometry_acceptance"] == "passed"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == render["gpu_uuid"]
    )
    assert 1 <= len(ep["people"]) <= 5 and frames
    assert all(k < ep["frame_count"] for k in frames)
    folders = [Path(frame_source_dir_fn(frame)) for frame in frames]
    if len(folders) != len(frames):
        raise ValueError("frame_source_count_mismatch")
    cfg = json.loads(Path(ep["scene"]).read_text())
    if cfg.get("schema") != "human_environment_v1":
        raise ValueError("current_environment_required")
    blend = Path(cfg["blend_path"]).resolve()
    global _SESSION
    asset_key = tuple(
        (
            p["person_id"],
            p["asset_manifest"],
            Path(p["asset_manifest"]).stat().st_size,
            Path(p["asset_manifest"]).stat().st_mtime_ns,
        )
        for p in ep["people"]
    )
    key = (
        str(blend),
        blend.stat().st_size,
        blend.stat().st_mtime_ns,
        asset_key,
        tuple(render["source_resolution"]),
        render["samples"],
        render["gpu_uuid"],
    )
    warm = bool(
        reuse_session
        and _SESSION is not None
        and _SESSION["key"] == key
        and bpy.context.scene == _SESSION["scene"]
    )
    if warm:
        s = _SESSION["scene"]
    else:
        _SESSION = None
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        s = bpy.context.scene
    if cfg.get("lod_camera_anchor_object"):
        # The forest scatter LOD references Camera1. Keep one fixed anchor for
        # both calibrated cameras and every frame, never switch it per view.
        anchor = s.objects[cfg["lod_camera_anchor_object"]]
        rig = json.loads(task_path(root, f'rigs/{config["rig_ids"][0]}/rig.json').read_text())[
            "cameras"
        ]
        anchor.location = tuple(
            sum(float(c["camera_to_world"][j][3]) for c in rig) / len(rig) for j in range(3)
        )
        bpy.context.view_layer.update()
    if not warm:
        for o in s.objects:
            o.pass_index = 0
    for name in cfg["hide_objects"]:
        if name in s.objects:
            s.objects[name].hide_render = True
    configure_render(s, *render["source_resolution"], render["samples"])
    s.render.use_border = False
    s.render.use_crop_to_border = False
    s.cycles.seed = render["seed"]
    s.cycles.use_animated_seed = False
    s.cycles.use_adaptive_sampling = False
    s.render.use_persistent_data = True
    s.render.threads_mode = "FIXED"
    s.render.threads = 4
    s.view_layers[0].use_pass_z = True
    s.view_layers[0].use_pass_object_index = True
    s.render.use_compositing = True
    pref = bpy.context.preferences.addons["cycles"].preferences
    pref.compute_device_type = "CUDA"
    pref.get_devices()
    for d in pref.devices:
        d.use = d.type == "CUDA"
    devices = [dict(name=d.name, id=d.id, type=d.type) for d in pref.devices if d.use]
    assert len(devices) == 1
    s.cycles.device = "GPU"
    # A resident scene still probes native road, with the cached people hidden
    # from ray casting while their geometry is not yet updated for this job.
    hidden = []
    if warm:
        for item in _SESSION["objects"]:
            for obj in (item.body_object, item.garment_object, *item.hair_objects):
                hidden.append((obj, obj.hide_get()))
                obj.hide_set(True)
        bpy.context.view_layer.update()
    # Probe native road before importing any people. Constant transform throughout.
    ground = []
    loaded = []
    shared_matrices = np.load(task_path(root, "scene_matrices.npy"), mmap_mode="r")
    if shared_matrices.shape != (ep["frame_count"], len(ep["people"]), 4, 4):
        raise ValueError("scene_matrices_shape_conflict")
    for person_index, p in enumerate(ep["people"]):
        asset = json.loads(Path(p["asset_manifest"]).read_text())
        g = geometry_paths(asset)
        v = np.load(g["body_vertices"], mmap_mode="r")
        matrices = shared_matrices[:, person_index]
        asset_frames = [row["asset_frame"] for row in ep["person_timelines"][str(p["person_id"])]]
        if any(
            type(asset_frames[k]) is not int or not 0 <= asset_frames[k] < len(v) for k in frames
        ):
            raise ValueError("asset_frame_out_of_bounds")
        bounds = support_limits(p)
        cycling = p["root_mode"] == ROOT_MODE
        for k in frames:
            M = matrices[k]
            world = v[asset_frames[k]] @ M[:3, :3].T + M[:3, 3]
            low = world[world[:, 2] < world[:, 2].min() + 0.02][::10]
            gaps = []
            names = []
            for x in low:
                hit, point, normal, face, obj, _ = s.ray_cast(
                    bpy.context.evaluated_depsgraph_get(),
                    Vector((float(x[0]), float(x[1]), float(x[2] + 0.5))),
                    Vector((0, 0, -1)),
                    distance=1,
                )
                if not hit:
                    raise ValueError("No native road under planned support")
                if cycling:
                    gate = p["support_checks"]
                    if obj.name not in gate["road_object_names"] and not any(
                        obj.name.startswith(prefix) for prefix in gate["road_object_prefixes"]
                    ):
                        raise ValueError("cycling_path_hits_nonroad_object")
                    if (
                        abs(point.z - ep["road_height_m"]) > gate["road_z_tolerance_m"]
                        or normal.z < 0.9
                    ):
                        raise ValueError("cycling_road_height_or_normal_conflict")
                    lo, hi = gate["pelvis_height_above_road_m"]
                    if not lo <= M[2, 3] - point.z <= hi:
                        raise ValueError("cycling_pelvis_height_out_of_range")
                gaps.append(float(x[2] - point.z))
                names.append(obj.name)
            ground.append(
                dict(
                    person_id=p["person_id"],
                    frame_index=k,
                    min_gap_m=min(gaps),
                    max_gap_m=max(gaps),
                    hit_objects=sorted(set(names)),
                )
            )
            if min(gaps) < bounds[0] or max(gaps) > bounds[1]:
                raise ValueError(f"Road support outside explicit {bounds} gate: {ground[-1]}")
        loaded.append(LoadedPerson(p, asset, g, v, matrices, asset_frames))
    if event_fn is not None:
        event_fn(dict(event="ground_checked", checks=len(ground)))
    for obj, was_hidden in hidden:
        obj.hide_set(was_hidden)
    if hidden:
        bpy.context.view_layer.update()
    objects = []
    if warm:
        for item, prior in zip(loaded, _SESSION["objects"]):
            objects.append(AssembledPerson(
                item.person, item.asset, item.body_vertices, prior.cloth_vertices,
                prior.head_transforms, item.matrices, prior.body_object,
                prior.garment_object, prior.hair_objects, prior.hair_topology,
                item.asset_frames,
            ))
        camera = _SESSION["camera"]
        file_node = _SESSION["file_node"]
    else:
        for item in loaded:
            p = item.person
            asset = item.asset
            g = item.geometry
            v = item.body_vertices
            matrices = item.matrices
            asset_frames = item.asset_frames
            M = matrices[frames[0]]
            m = json.loads(Path(asset["character_manifest"]).read_text())
            ix = asset_frames[frames[0]]
            faces = np.load(g["body_faces"])
            cloth = Path(asset["clothing_reference"])
            body, mesh = create_mesh_object(
                s, f'Body{p["person_id"]}', v[ix], faces, p["person_id"]
            )
            body.matrix_world = Matrix(M.tolist())
            apply_character_materials(mesh, faces, m["appearance"])
            cv = np.load(g["cloth_vertices"], mmap_mode="r")
            garment, cm = create_mesh_object(
                s, f'Clothes{p["person_id"]}', cv[ix], np.load(g["cloth_faces"]), p["person_id"]
            )
            garment.matrix_world = body.matrix_world
            add_clothing_materials(cm, cloth)
            for mesh in [mesh, cm]:
                for poly in mesh.polygons:
                    poly.use_smooth = True
            hair = add_groom(asset["groom_reference"], body.matrix_world, p["person_id"])
            H = np.load(g["head_transforms"])
            counts = [(len(o.data.curves), len(o.data.points)) for o in hair]
            objects.append(AssembledPerson(
                p, asset, v, cv, H, matrices, body, garment, hair, counts, asset_frames,
            ))
        camera = create_erp_camera(s)
        file_node = configure_compositor(s, folders[0], "SequenceSourceCompositor")
        if reuse_session:
            _SESSION = dict(key=key, scene=s, objects=objects, camera=camera, file_node=file_node)
    setup = time.monotonic() - began
    timings = []
    deadline = config["deadline_monotonic"]
    for frame, folder in zip(frames, folders):
        if time.monotonic() > deadline:
            raise RuntimeError("Render budget exhausted; failed scene retained")
        start = time.monotonic()
        folder = Path(folder)
        if hasattr(file_node, "directory"):
            file_node.directory = str(folder)
        else:
            file_node.base_path = str(folder)
        for item in objects:
            _update_person_geometry(item, frame)
        bpy.context.view_layer.update()
        geometry_seconds = time.monotonic() - start
        sources = render_cameras_fn(s, camera, file_node, folder, frame_index=frame)
        elapsed = time.monotonic() - start
        timing = frame_record(ep, frame)
        manifest = dict(
            sources=sources,
            frame_index=frame,
            time_tick=timing["episode_time_tick"],
            time_base=timing["time_base"],
            selected_time_record=timing,
            episode_recipe="recipe.json",
            source_resolution=render["source_resolution"],
            blender_version=bpy.app.version_string,
            samples=render["samples"],
            gpu_uuid=render["gpu_uuid"],
            devices=devices,
            denoising=False,
            adaptive_sampling=False,
            render_seconds=elapsed,
            body_readback_exact=True,
            head_transform_once=True,
            groom_topology_fixed=True,
            native_scene_max_bounces=s.cycles.max_bounces,
            world_transform_owner="shared scene matrices",
            root_owner="scene trajectory",
            people_state=[
                dict(
                    person_id=item.person["person_id"],
                    asset_frame=item.asset_frames[frame],
                    scene_from_model=item.matrices[frame].tolist(),
                )
                for item in objects
            ],
        )
        consume_start = time.monotonic()
        frame_complete_fn(frame, folder, manifest)
        timings.append(
            dict(
                frame_index=frame,
                seconds=time.monotonic() - start,
                geometry_seconds=geometry_seconds,
                render_callback_seconds=elapsed - geometry_seconds,
                consumer_callback_seconds=time.monotonic() - consume_start,
            )
        )
        if event_fn is not None:
            event_fn(dict(event="frame_completed", **timings[-1]))
        print("ROAD_PASS_FRAME_DONE", frame, elapsed, flush=True)
    return dict(
        setup_seconds=setup,
        elapsed_seconds=time.monotonic() - began,
        frames=timings,
        gpu_uuid=render["gpu_uuid"],
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        devices=devices,
        scene_reused=warm,
    )
