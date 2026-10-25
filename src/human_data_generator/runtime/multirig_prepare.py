"""CPU-only preparation of one scene and selected frames for paired ROI rigs."""

from pathlib import Path
import sys
import uuid

import numpy as np
from PIL import Image

from human_data_generator.camera.rig import rig_poses
from human_data_generator.camera.erp import project_erp
from human_data_generator.camera.domain import point_in_view
from human_data_generator.config import resources
from human_data_generator.data.mamma_labels import ordered_vertex_ids
from human_data_generator.data.scene_geometry import person_geometry, transform
from human_data_generator.data.dynamic_scene import geometry_paths
from human_data_generator.imaging.roi_planning import plan_person_crop_union
from human_data_generator.io.json_file import write_json
from human_data_generator.runtime.scene import read_scene, compile_scene, validate, map_data
from human_data_generator.runtime.task_paths import RUN_SCHEMA


def _surface_may_enter_valid_mask(projected, faces, valid):
    """Conservative raster-box test for a triangle clipped by the lens edge.

    A valid pixel inside a triangle's projected bounding box keeps the person.
    False positives only enlarge ROI; a vertex-only visibility check could
    silently discard a thin visible part of body or clothing.
    """
    height, width = valid.shape
    tri = projected[faces]
    x = tri[:, :, 0]
    x = x[:, :1] + (x - x[:, :1] + width / 2) % width - width / 2
    x0 = np.floor(x.min(1) - 2).astype(int)
    x1 = np.ceil(x.max(1) + 2).astype(int)
    y0 = np.clip(np.floor(tri[:, :, 1].min(1) - 2).astype(int), 0, height)
    y1 = np.clip(np.ceil(tri[:, :, 1].max(1) + 2).astype(int), 0, height)
    integral = np.pad(valid.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for shift in (-width, 0, width):
        left = np.clip(x0 + shift, 0, width)
        right = np.clip(x1 + shift, 0, width)
        candidate = (left < right) & (y0 < y1)
        overlap = integral[y1, right] - integral[y0, right] - integral[y1, left] + integral[y0, left]
        if np.any(candidate & (overlap > 0)):
            return True
    return False


def prepare(output, *, scene_path, frame_indices=None, selected_rig_ids=None, preview=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("run_output_not_empty")
    scene = read_scene(scene_path)
    frames = list(range(scene["frame_count"])) if frame_indices is None else list(frame_indices)
    if (
        not frames
        or frames != sorted(set(frames))
        or any(type(k) is not int or not 0 <= k < scene["frame_count"] for k in frames)
    ):
        raise ValueError("invalid_selected_frames")
    selected = (
        scene["rigs"]
        if selected_rig_ids is None
        else [r for r in scene["rigs"] if r["rig_id"] in selected_rig_ids]
    )
    if not selected or (
        selected_rig_ids is not None and {r["rig_id"] for r in selected} != {*selected_rig_ids}
    ):
        raise ValueError("unknown_selected_rig")
    if preview and (len(frames) != 1 or len(selected) != 1):
        raise ValueError("preview_requires_one_frame_and_rig")
    check = validate(scene, frames, [r["rig_id"] for r in selected])
    if check["status"] != "passed":
        raise ValueError(check["errors"])
    resource = resources()
    environment = resource["environments"][scene["environment"]]
    people, timelines, matrices = compile_scene(scene, frames)
    template = Path(resource["mamma_template"])
    ordered_vertex_ids(template)
    recipe = dict(
        schema="human_run_v1",
        scene_episode_id=scene["scene_id"],
        frame_count=scene["frame_count"],
        fps=scene["fps"],
        people=people,
        person_timelines=timelines,
        scene=environment["scene_config"],
        geometry_acceptance="passed",
        road_height_m=0,
        render=dict(source_resolution=[5760, 2880], samples=64, seed=scene["seed"]),
    )
    bodies = person_geometry(recipe, frames)
    surface_faces = []
    for body in bodies:
        paths = geometry_paths(body["asset"])
        surface_faces.append(
            {name: np.load(paths[name + "_faces"], mmap_mode="r") for name in ("body", "cloth")}
        )
    roads = map_data(scene["environment"])["roads"]
    world_parts = {}
    for frame in frames:
        for j, body in enumerate(bodies):
            M = matrices[frame, j]
            parts = dict(
                body=transform(body["body"][frame], M),
                cloth=transform(body["cloth"][frame], M),
                groom=transform(transform(body["hair"], body["head"][frame]), M),
            )
            xy = np.concatenate((parts["body"][:, :2], parts["cloth"][:, :2]))
            supported = np.zeros(len(xy), bool)
            for x0, y0, x1, y1 in roads:
                supported |= (
                    (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
                )
            if not supported.all():
                raise ValueError(
                    f'body_or_cloth_outside_road:p{people[j]["person_id"]}:frame{frame}'
                )
            world_parts[frame, j] = parts
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "scene.json", scene)
    write_json(output / "recipe.json", recipe)
    np.save(output / "scene_matrices.npy", matrices)
    for rig in selected:
        rid = rig["rig_id"]
        child = output / "rigs" / rid
        child.mkdir(parents=True, exist_ok=False)
        poses, lenses, _ = rig_poses(rig["origin_m"], rig["yaw_deg"], 0.18)
        masks = {
            camera: Path(environment["calibration_dir"]) / f"{camera}_valid_mask.png"
            for camera in poses
        }
        for path in masks.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        cameras = [
            dict(
                id=camera,
                camera_to_world=poses[camera].tolist(),
                physical_lens_to_world=lenses[camera].tolist(),
            )
            for camera in ("cam2", "cam4")
        ]
        rig_json = dict(
            schema="paired_rig_v1",
            rig_id=rid,
            calibration_target="20260320-riding-dataset3",
            origin_m=rig["origin_m"],
            yaw_deg=rig["yaw_deg"],
            baseline_m=0.18,
            cameras=cameras,
            valid_masks={c: str(path) for c, path in masks.items()},
        )
        write_json(child / "rig.json", rig_json)
        valid = {c: np.asarray(Image.open(path).convert("L")) > 0 for c, path in masks.items()}
        planned = {}
        for frame in frames:
            frame_plans = {}
            for camera in cameras:
                cid = camera["id"]
                pose = np.asarray(camera["camera_to_world"])
                grouped = {}
                for j, body in enumerate(bodies):
                    parts = world_parts[frame, j]
                    projected = {}
                    eligible = False
                    for name, points in parts.items():
                        xy, _, _, _, in_view = point_in_view(points, pose, cid, valid[cid])
                        projected[name] = xy
                        eligible |= bool(in_view.any())
                    if not eligible:
                        # A clipped triangle can enter the lens even if all
                        # vertices are outside its physical valid pixels.
                        for name in ("body", "cloth"):
                            if _surface_may_enter_valid_mask(
                                projected[name], surface_faces[j][name], valid[cid]
                            ):
                                eligible = True
                                break
                    if eligible:
                        grouped[str(people[j]["person_id"])] = projected
                if grouped:
                    region = plan_person_crop_union(grouped, (5760, 2880))
                else:
                    # Keep a real (but small) camera product so all people
                    # still receive absent observation slots.
                    y, x = np.unravel_index(int(np.argmax(valid[cid])), valid[cid].shape)
                    x0 = int(np.clip(x - 32, 0, 5760 - 64))
                    y0 = int(np.clip(y - 32, 0, 2880 - 64))
                    region = dict(
                        roi_xyxy=[x0, y0, x0 + 64, y0 + 64],
                        person_regions={},
                        planning_method="empty_physical_camera",
                    )
                frame_plans[cid] = region
            planned[frame] = frame_plans
        write_json(child / "planned_regions.json", planned)
    config = dict(
        schema=RUN_SCHEMA,
        task_id=uuid.uuid4().hex,
        selected_frames=frames,
        rig_ids=[r["rig_id"] for r in selected],
        preview_only=preview,
        consumer_python=sys.executable,
        generation_contract="preview_rgb_v1" if preview else "mamma_512_v1",
        max_camera_calls=len(frames) * 2 * len(selected),
        full_image_size_wh=[5760, 2880],
        scene_settings=environment["scene_settings"],
        mamma_512_label=dict(
            template_id="MAMMA_verts_512_ordered_v1",
            template_path=str(template),
            visibility_policy="mamma512_conservative_visibility_v1_compatible_definition_pending",
        ),
    )
    write_json(output / "run_config.json", config)
    return config
