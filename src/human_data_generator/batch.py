"""Sample explicit small-scene rules into scene files; never start rendering."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from human_data_generator.config import PROJECT_ROOT
from human_data_generator.components import catalog, describe
from human_data_generator.io.json_file import write_json
from human_data_generator.runtime.scene import _asset_timeline, validate


def _rng(seed, index, purpose):
    material = f"{seed}:{index}:{purpose}".encode()
    return random.Random(int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big"))


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("invalid_batch_rule:" + label)


def _range(value, label, *, integer=False):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("invalid_batch_range:" + label)
    if integer:
        if any(type(v) is not int for v in value):
            raise ValueError("invalid_batch_range:" + label)
    elif any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError("invalid_batch_range:" + label)
    if value[0] > value[1]:
        raise ValueError("invalid_batch_range:" + label)
    return value


def _sample(rng, value, label, *, integer=False):
    low, high = _range(value, label, integer=integer)
    return rng.randint(low, high) if integer else rng.uniform(low, high)


def _point(rng, value, label):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("invalid_batch_point:" + label)
    return [_sample(rng, value[axis], label) for axis in range(2)]


def _component_policy(value):
    _keys(value, ("body", "motion", "clothing", "hair"), "components")
    for kind, keys in (
        ("body", ("default_versions_only", "ids")),
        ("motion", ("behavior", "ids")),
        ("clothing", ("kinds", "ids")),
        ("hair", ("ids",)),
    ):
        _keys(value[kind], keys, "components." + kind)
        ids = value[kind]["ids"]
        if (
            not isinstance(ids, list)
            or any(not isinstance(ident, str) or not ident for ident in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("invalid_component_ids:" + kind)
    if type(value["body"]["default_versions_only"]) is not bool:
        raise ValueError("invalid_default_versions_only")
    if value["motion"]["behavior"] not in ("walk", "ride"):
        raise ValueError("invalid_motion_behavior")
    kinds = value["clothing"]["kinds"]
    if (
        not isinstance(kinds, list)
        or not kinds
        or any(kind not in ("simple_barycentric", "native_locked") for kind in kinds)
        or len(kinds) != len(set(kinds))
    ):
        raise ValueError("invalid_clothing_kinds")
    return value


def _choose(
    inventory, kind, policy, seed, index, person_id, *, body=None, motion=None, records=None
):
    catalog_kind = {"body": "bodies", "motion": "motions", "clothing": "clothing", "hair": "hair"}[
        kind
    ]
    rows = {row["id"]: row for row in inventory[catalog_kind]}
    ids = policy["ids"]
    for ident in ids:
        if ident not in rows:
            raise ValueError(f"unknown_batch_component:{kind}:{ident}")
        if rows[ident]["status"] != "ready":
            raise ValueError(f'unavailable_batch_component:{kind}:{ident}:{rows[ident]["status"]}')

    def allowed(row):
        if kind == "body":
            return not policy["default_versions_only"] or row["default_version"]
        if kind == "motion":
            return row["behavior"] == policy["behavior"]
        if kind == "clothing":
            if row["kind"] not in policy["kinds"]:
                return False
            if row["kind"] == "native_locked":
                low, high = row["locked_window"]
                return (
                    row["locked_body"] == body
                    and row["locked_motion"] == motion
                    and records is not None
                    and all(low <= record["asset_frame"] < high for record in records)
                )
        return True

    candidates = []
    for ident in (ids if ids else rows):
        row = rows[ident]
        if row["status"] != "ready":
            continue
        if not allowed(row):
            if ids:
                reason = (
                    "native_clothing_body_motion_or_window_conflict"
                    if kind == "clothing"
                    and row["kind"] == "native_locked"
                    and row["kind"] in policy["kinds"]
                    else "batch_component_filter_conflict"
                )
                raise ValueError(f"{reason}:{kind}:{ident}")
            continue
        candidates.append(ident)
    if not candidates:
        raise ValueError("no_usable_batch_component:" + kind)
    return _rng(seed, index, f"person_{person_id}_{kind}").choice(sorted(candidates))


def sample_scene(rule, seed, index):
    """Return one current-format scene from one fully specified sampling rule."""
    if type(seed) is not int or seed < 1 or type(index) is not int or index < 0:
        raise ValueError("invalid_batch_seed_or_index")
    _keys(
        rule,
        ("scene_id", "environment", "fps", "frame_count", "people", "rigs", "camera_jitter"),
        "scene",
    )
    _keys(rule["camera_jitter"], ("xy_m", "yaw_deg"), "camera_jitter")
    xy_jitter = rule["camera_jitter"]["xy_m"]
    yaw_jitter = rule["camera_jitter"]["yaw_deg"]
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in (xy_jitter, yaw_jitter)
    ):
        raise ValueError("invalid_camera_jitter")
    found = catalog()
    inventory = describe(found)
    motions = found["motions"]

    people = []
    for spec in rule["people"]:
        _keys(
            spec,
            (
                "person_id",
                "components",
                "route_start_xy_range_m",
                "route_end_xy_range_m",
                "speed_m_s",
                "start_progress_m",
                "motion_start",
            ),
            "person",
        )
        pid = spec["person_id"]
        if type(pid) is not int or pid < 1:
            raise ValueError("invalid_batch_person")
        policy = _component_policy(spec["components"])
        body = _choose(inventory, "body", policy["body"], seed, index, pid)
        motion = _choose(inventory, "motion", policy["motion"], seed, index, pid)
        motion_start = _sample(
            _rng(seed, index, f"person_{pid}_motion_start"),
            spec["motion_start"],
            "motion_start",
            integer=True,
        )
        records = _asset_timeline(motions[motion], motion_start, rule["frame_count"], rule["fps"])
        clothing = _choose(
            inventory,
            "clothing",
            policy["clothing"],
            seed,
            index,
            pid,
            body=body,
            motion=motion,
            records=records,
        )
        hair = _choose(inventory, "hair", policy["hair"], seed, index, pid)
        route_rng = _rng(seed, index, f"person_{pid}_route")
        people.append(
            dict(
                person_id=pid,
                body=body,
                motion=motion,
                clothing=clothing,
                hair=hair,
                motion_start=motion_start,
                route_xy_m=[
                    _point(route_rng, spec["route_start_xy_range_m"], "route_start"),
                    _point(route_rng, spec["route_end_xy_range_m"], "route_end"),
                ],
                speed_m_s=_sample(
                    _rng(seed, index, f"person_{pid}_speed"), spec["speed_m_s"], "speed_m_s"
                ),
                start_progress_m=_sample(
                    _rng(seed, index, f"person_{pid}_progress"),
                    spec["start_progress_m"],
                    "start_progress_m",
                ),
            )
        )

    rigs = []
    for rig in rule["rigs"]:
        _keys(rig, ("rig_id", "origin_m", "yaw_deg"), "rig")
        if (
            not isinstance(rig["origin_m"], list)
            or len(rig["origin_m"]) != 3
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (*rig["origin_m"], rig["yaw_deg"])
            )
        ):
            raise ValueError("invalid_batch_rig")
        camera_rng = _rng(seed, index, "camera_" + str(rig["rig_id"]))
        rigs.append(
            dict(
                rig_id=rig["rig_id"],
                origin_m=[
                    rig["origin_m"][0] + camera_rng.uniform(-xy_jitter, xy_jitter),
                    rig["origin_m"][1] + camera_rng.uniform(-xy_jitter, xy_jitter),
                    rig["origin_m"][2],
                ],
                yaw_deg=rig["yaw_deg"] + camera_rng.uniform(-yaw_jitter, yaw_jitter),
            )
        )

    scene = dict(
        schema="human_scene_v2",
        scene_id=f"batch_{index:03d}_{rule['scene_id']}",
        environment=rule["environment"],
        seed=_rng(seed, index, "render").randrange(1, 2**31),
        fps=rule["fps"],
        frame_count=rule["frame_count"],
        people=people,
        rigs=rigs,
    )
    check = validate(scene)
    if check["status"] != "passed":
        raise ValueError(check["errors"])
    return scene


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample scene JSON files; no geometry build or rendering."
    )
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    rules = json.loads(args.rules.read_text())
    _keys(rules, ("schema", "seed", "scenes"), "root")
    if (
        rules["schema"] != "human_batch_v2"
        or not isinstance(rules["scenes"], list)
        or not rules["scenes"]
    ):
        raise ValueError("invalid_batch_rules")
    output = PROJECT_ROOT / "outputs/scenes" / args.name
    if output.exists():
        raise FileExistsError(output)
    # Validate and sample every rule before creating an output directory.
    scenes = [
        sample_scene(rule, rules["seed"], index) for index, rule in enumerate(rules["scenes"])
    ]
    output.mkdir(parents=True)
    paths = []
    for scene in scenes:
        path = output / f"{scene['scene_id']}.json"
        write_json(path, scene)
        paths.append(str(path))
    write_json(
        output / "batch.json", dict(schema="human_batch_plan_v1", seed=rules["seed"], scenes=paths)
    )
    print(json.dumps(dict(output=str(output), scenes=paths), ensure_ascii=False))


if __name__ == "__main__":
    main()
