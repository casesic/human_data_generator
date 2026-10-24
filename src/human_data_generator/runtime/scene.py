"""One explicit scene format shared by the editor, sampler and run planner."""

import json
import math
import re
from pathlib import Path

import numpy as np

from human_data_generator.config import resources
from human_data_generator.components import catalog, select
from human_data_generator.composition import build
from human_data_generator.data.dynamic_scene import geometry_paths, checked_matrices

SCHEMA = "human_scene_v2"
PERSON_KEYS = {
    "person_id",
    "body",
    "motion",
    "clothing",
    "hair",
    "motion_start",
    "route_xy_m",
    "speed_m_s",
    "start_progress_m",
}
RIG_KEYS = {"rig_id", "origin_m", "yaw_deg"}
SCENE_KEYS = {"schema", "scene_id", "environment", "seed", "fps", "frame_count", "people", "rigs"}


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name}_fields_must_be_{sorted(expected)}")


def read_scene(path):
    scene = json.loads(Path(path).read_text())
    validate_scene(scene)
    return scene


def _curve(points, distance):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("route_requires_two_finite_points")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 0.01):
        raise ValueError("route_points_too_close")
    samples = []
    for i in range(len(points) - 1):
        a, b, c, d = (
            points[max(0, i - 1)],
            points[i],
            points[i + 1],
            points[min(len(points) - 1, i + 2)],
        )
        for t in np.linspace(0, 1, 41, endpoint=False):
            samples.append(
                0.5
                * (
                    (2 * b)
                    + (-a + c) * t
                    + (2 * a - 5 * b + 4 * c - d) * t * t
                    + (-a + 3 * b - 3 * c + d) * t * t * t
                )
            )
    samples.append(points[-1])
    sample = np.asarray(samples)
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sample, axis=0), axis=1))]
    if not 0 <= distance <= lengths[-1]:
        raise ValueError("route_time_out_of_bounds")
    xy = np.array([np.interp(distance, lengths, sample[:, axis]) for axis in range(2)])
    lo = max(0, distance - 0.025)
    hi = min(lengths[-1], distance + 0.025)
    tangent = np.array(
        [
            np.interp(hi, lengths, sample[:, axis]) - np.interp(lo, lengths, sample[:, axis])
            for axis in range(2)
        ]
    )
    tangent /= np.linalg.norm(tangent)
    return xy, tangent


def _asset_timeline(motion, start, count, fps):
    rows = json.loads(Path(motion["timeline"]).read_text())
    if type(start) is not int or start < 0 or start >= len(rows):
        raise ValueError("motion_start_out_of_bounds")
    ticks = np.array([int(row["source_time_tick"]) for row in rows], dtype=np.int64)
    base = rows[0]["time_base"]
    if any(row["time_base"] != base for row in rows) or np.any(np.diff(ticks) <= 0):
        raise ValueError("invalid_asset_timeline")
    seconds = (ticks - ticks[start]) * base[0] / base[1]
    selected = []
    for k in range(count):
        target = k / fps
        if target > seconds[-1] + 1e-9:
            raise ValueError("motion_exceeds_cache")
        j = int(np.argmin(np.abs(seconds[start:] - target))) + start
        row = rows[j]
        if row["asset_frame"] >= len(rows):
            raise ValueError("asset_frame_out_of_bounds")
        selected.append(
            dict(
                asset_frame=int(row["asset_frame"]),
                source_frame=int(row["source_frame"]),
                source_time_tick=int(row["source_time_tick"]),
                source_time_base=base,
            )
        )
    return selected


def validate_scene(scene):
    _exact_keys(scene, SCENE_KEYS, "scene")
    if (
        scene["schema"] != SCHEMA
        or re.fullmatch(r"[a-z0-9_]{1,64}", str(scene["scene_id"])) is None
    ):
        raise ValueError("invalid_scene_id_or_schema")
    if scene["environment"] not in resources()["environments"]:
        raise ValueError("unknown_environment")
    if any(type(scene[key]) is not int or scene[key] < 1 for key in ("seed", "fps", "frame_count")):
        raise ValueError("invalid_seed_or_clock")
    people = scene["people"]
    rigs = scene["rigs"]
    if not 1 <= len(people) <= 5 or not 1 <= len(rigs) <= 8:
        raise ValueError("invalid_person_or_rig_count")
    ids = []
    rig_ids = []
    components = catalog()
    for p in people:
        _exact_keys(p, PERSON_KEYS, "person")
        pid = p["person_id"]
        ids.append(pid)
        if type(pid) is not int or pid <= 0:
            raise ValueError("invalid_person_id")
        if not 0.1 <= float(p["speed_m_s"]) <= 4 or float(p["start_progress_m"]) < 0:
            raise ValueError("invalid_person_speed_or_start")
        chosen = select(p, components)
        records = _asset_timeline(
            chosen["motion"], p["motion_start"], scene["frame_count"], scene["fps"]
        )
        if chosen["clothing"]["kind"] == "native_locked":
            low, high = chosen["clothing"]["locked_window"]
            if (
                min(x["asset_frame"] for x in records) < low
                or max(x["asset_frame"] for x in records) >= high
            ):
                raise ValueError("native_clothing_source_window_out_of_bounds")
        for k in range(scene["frame_count"]):
            _curve(p["route_xy_m"], p["start_progress_m"] + p["speed_m_s"] * k / scene["fps"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_person_id")
    for rig in rigs:
        _exact_keys(rig, RIG_KEYS, "rig")
        rig_ids.append(rig["rig_id"])
        if re.fullmatch(r"rig_[a-z0-9_]+", str(rig["rig_id"])) is None:
            raise ValueError("invalid_rig_id")
        origin = np.asarray(rig["origin_m"], float)
        if origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(rig["yaw_deg"]):
            raise ValueError("invalid_rig_pose")
    if len(rig_ids) != len(set(rig_ids)):
        raise ValueError("duplicate_rig_id")
    return scene


def compile_scene(scene, selected_frames=None):
    """Resolve assets once and calculate one shared body state for all rigs."""
    validate_scene(scene)
    environment = json.loads(
        Path(resources()["environments"][scene["environment"]]["scene_config"]).read_text()
    )
    count = scene["frame_count"]
    fps = scene["fps"]
    selected = list(range(count)) if selected_frames is None else list(selected_frames)
    if (
        not selected
        or selected != sorted(set(selected))
        or any(type(k) is not int or not 0 <= k < count for k in selected)
    ):
        raise ValueError("invalid_selected_frames")
    matrices = np.empty((count, len(scene["people"]), 4, 4), dtype=np.float64)
    timelines = {}
    people = []
    faces_ref = None
    components = catalog()
    for j, p in enumerate(scene["people"]):
        chosen = select(p, components)
        source_records = _asset_timeline(chosen["motion"], p["motion_start"], count, fps)
        needed = [source_records[k]["asset_frame"] for k in selected]
        descriptor = build(p, needed)
        source_first = min(needed)
        source_last = max(needed)
        asset_path = Path(descriptor["asset_manifest"])
        asset = json.loads(asset_path.read_text())
        paths = geometry_paths(asset)
        vertices = np.load(paths["body_vertices"], mmap_mode="r")
        faces = np.load(paths["body_faces"], mmap_mode="r")
        if (
            vertices.ndim != 3
            or vertices.shape[1:] != (10475, 3)
            or faces.ndim != 2
            or faces.shape[1] != 3
            or faces.min() < 0
            or faces.max() >= 10475
        ):
            raise ValueError("invalid_smplx_geometry")
        if faces_ref is None:
            faces_ref = np.array(faces)
        elif not np.array_equal(faces_ref, faces):
            raise ValueError("asset_face_order_conflict")
        records = [
            dict(
                row,
                asset_frame=(
                    row["asset_frame"] - source_first
                    if source_first <= row["asset_frame"] <= source_last
                    else None
                ),
            )
            for row in source_records
        ]
        last = source_last - source_first
        cloth = np.load(paths["cloth_vertices"], mmap_mode="r")
        cloth_faces = np.load(paths["cloth_faces"], mmap_mode="r")
        head = np.load(paths["head_transforms"], mmap_mode="r")
        if (
            last >= len(vertices)
            or cloth.ndim != 3
            or cloth.shape[2] != 3
            or last >= len(cloth)
            or cloth_faces.ndim != 2
            or cloth_faces.shape[1] != 3
            or cloth_faces.min() < 0
            or cloth_faces.max() >= cloth.shape[1]
            or head.ndim != 3
            or head.shape[1:] != (4, 4)
            or last >= len(head)
        ):
            raise ValueError("asset_body_cloth_head_incompatible")
        if (
            not Path(asset["character_manifest"]).is_file()
            or not Path(asset["parameters"]).is_file()
        ):
            raise ValueError("missing_character_or_motion_parameters")
        groom = Path(asset["groom_reference"])
        if any(
            not (groom / name).is_file()
            for name in ("points.npy", "active_curve_mask.npy", "curve_offsets.npy")
        ):
            raise ValueError("missing_groom_geometry")
        timelines[str(p["person_id"])] = records
        person = {
            key: descriptor[key]
            for key in (
                "asset_id",
                "asset_manifest",
                "placement_height_m",
                "root_mode",
                "support_policy",
                "component_ids",
            )
        }
        person["support_checks"] = dict(
            descriptor["support_checks"],
            road_object_names=environment["road_object_names"],
            road_object_prefixes=environment["road_object_prefixes"],
            road_z_tolerance_m=environment["road_z_tolerance_m"],
        )
        person["person_id"] = p["person_id"]
        person["source_manifest_id"] = descriptor["source_manifest_id"]
        people.append(person)
        source_roots = None
        if descriptor["root_mode"] == "source_root_relative_to_start_plus_fixed_scene_matrix":
            with np.load(chosen["motion"]["parameters"], allow_pickle=False) as parameters:
                source_roots = np.asarray(parameters["source_root_relative"])
        for k in range(count):
            xy, tangent = _curve(p["route_xy_m"], p["start_progress_m"] + p["speed_m_s"] * k / fps)
            tx, ty = tangent
            M = np.eye(4)
            M[:3, :3] = [[-ty, 0, tx], [tx, 0, ty], [0, 1, 0]]
            M[:3, 3] = [xy[0], xy[1], descriptor["placement_height_m"]]
            if source_roots is not None:
                delta = (
                    source_roots[source_records[k]["asset_frame"]]
                    - source_roots[source_records[0]["asset_frame"]]
                )
                horizontal = np.array([delta[0], 0, delta[2]])
                M[:2, 3] -= (M[:3, :3] @ horizontal)[:2]
            matrices[k, j] = M
        checked_matrices(matrices[:, j], count)
    return people, timelines, matrices


def map_data(environment):
    if environment == "city":
        return dict(
            bounds=[-90, -90, 90, 95],
            roads=[[-3.5, -80, 3.5, 80], [-80, 80, 80, 88]],
            water=[],
            obstacles=[],
        )
    if environment == "riverfront":
        return dict(
            bounds=[-22, -22, 22, 22],
            roads=[[-2.5, -20, 2.5, 20]],
            water=[[-22, -22, -3, 22]],
            obstacles=[],
        )
    raise ValueError("unknown_environment")


def validate(scene, frames=None, rig_ids=None):
    errors = []
    try:
        validate_scene(scene)
    except Exception as exc:
        return dict(status="rejected", errors=[str(exc)], checked_frames=[])
    chosen = list(range(scene["frame_count"])) if frames is None else frames
    if (
        not chosen
        or chosen != sorted(set(chosen))
        or any(type(k) is not int or not 0 <= k < scene["frame_count"] for k in chosen)
    ):
        return dict(status="rejected", errors=["invalid_selected_frames"], checked_frames=[])
    rigs = [r for r in scene["rigs"] if rig_ids is None or r["rig_id"] in rig_ids]
    if not rigs or (rig_ids is not None and {r["rig_id"] for r in rigs} != set(rig_ids)):
        return dict(status="rejected", errors=["invalid_selected_rigs"], checked_frames=chosen)
    roads = map_data(scene["environment"])["roads"]
    for k in chosen:
        roots = []
        for p in scene["people"]:
            try:
                xy, _ = _curve(
                    p["route_xy_m"], p["start_progress_m"] + p["speed_m_s"] * k / scene["fps"]
                )
                roots.append((p["person_id"], xy))
                if not any(a <= xy[0] <= c and b <= xy[1] <= d for a, b, c, d in roads):
                    errors.append(f"p{p['person_id']}:frame{k}:outside_verified_road")
            except ValueError as exc:
                errors.append(f"p{p['person_id']}:frame{k}:{exc}")
        for a in range(len(roots)):
            for b in range(a + 1, len(roots)):
                if np.linalg.norm(roots[a][1] - roots[b][1]) < 0.9:
                    errors.append(f"p{roots[a][0]}_p{roots[b][0]}:frame{k}:root_clearance")
        for rig in rigs:
            for pid, xy in roots:
                if np.linalg.norm(xy - np.asarray(rig["origin_m"][:2])) < 0.5:
                    errors.append(f"{rig['rig_id']}:p{pid}:frame{k}:camera_clearance")
    return dict(
        status="passed" if not errors else "rejected", errors=errors[:100], checked_frames=chosen
    )
