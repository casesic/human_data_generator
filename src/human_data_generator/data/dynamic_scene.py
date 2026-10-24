"""Asset geometry, shared scene matrices and integer person-frame records."""

from pathlib import Path

import numpy as np

ROOT_MODE = "fixed_pelvis_scene_owns_progression"
WALK_ROOT = "source_root_relative_to_start_plus_fixed_scene_matrix"


def geometry_paths(asset):
    required = ("body_vertices", "body_faces", "cloth_vertices", "cloth_faces", "head_transforms")
    if any(key not in asset for key in required):
        raise ValueError("current_asset_geometry_paths_required")
    paths = {key: Path(asset[key]) for key in required}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("missing_asset_geometry")
    return paths


def checked_matrices(value, count):
    array = np.asarray(value, float)
    if array.shape != (count, 4, 4) or not np.isfinite(array).all():
        raise ValueError("invalid_scene_matrices")
    if (
        not np.allclose(array[:, 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(
            array[:, :3, :3] @ array[:, :3, :3].transpose(0, 2, 1), np.eye(3), atol=1e-6, rtol=0
        )
        or not np.allclose(np.linalg.det(array[:, :3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("nonrigid_scene_matrix")
    return array


def frame_record(recipe, frame):
    if (
        recipe.get("schema") != "human_run_v1"
        or type(frame) is not int
        or not 0 <= frame < recipe["frame_count"]
    ):
        raise ValueError("invalid_current_frame_record")
    records = []
    for person in recipe["people"]:
        rows = recipe["person_timelines"][str(person["person_id"])]
        if len(rows) != recipe["frame_count"]:
            raise ValueError("person_time_count_conflict")
        records.append(dict(rows[frame], person_id=person["person_id"]))
    return dict(
        frame_index=frame,
        episode_time_tick=frame,
        time_base=[1, recipe["fps"]],
        person_records=records,
    )


def support_limits(person):
    mode = person["root_mode"]
    if mode == WALK_ROOT and person["support_policy"] == "pedestrian_source_root_constant_lift":
        return -0.02, 0.12
    if (
        mode == ROOT_MODE
        and person["support_policy"] == "fixed_scene_placement_height_no_pedestrian_foot_contact"
    ):
        lo, hi = person["support_checks"]["body_min_clearance_m"]
        if not 0 <= lo < hi <= 1:
            raise ValueError("invalid_body_clearance_limits")
        return lo, hi
    raise ValueError("unsupported_root_or_support_policy")
