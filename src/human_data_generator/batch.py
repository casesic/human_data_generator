"""Sample variants of complete editor scenes; this module never builds or renders assets."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re

from human_data_generator.config import PROJECT_ROOT
from human_data_generator.components import catalog, describe
from human_data_generator.io.json_file import write_json
from human_data_generator.runtime.scene import _asset_timeline, validate, validate_scene

SCHEMA = "human_batch_v4"
COMPONENTS = {"body": "bodies", "motion": "motions", "clothing": "clothing", "hair": "hair"}


def _rng(seed, template_id, index, purpose):
    material = f"{seed}:{template_id}:{index}:{purpose}".encode()
    return random.Random(int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big"))


def _keys(value, required, optional=(), label="rule"):
    if (
        not isinstance(value, dict)
        or not set(required) <= set(value)
        or set(value) - set(required) - set(optional)
    ):
        raise ValueError(f"invalid_{label}_fields")


def _number(value, label, *, integer=False):
    if integer:
        if type(value) is not int:
            raise ValueError(f"invalid_{label}")
    elif type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"invalid_{label}")
    return value


def _range(value, label, *, integer=False):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"invalid_{label}_range")
    low, high = (_number(v, label, integer=integer) for v in value)
    if low > high:
        raise ValueError(f"invalid_{label}_range")
    return low, high


def _xy_ranges(value, label):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"invalid_{label}_xy")
    return [_range(axis, label) for axis in value]


def _sample(rng, value, label, *, integer=False):
    low, high = _range(value, label, integer=integer)
    return rng.randint(low, high) if integer else rng.uniform(low, high)


def _sample_xy(rng, value, label):
    return [rng.uniform(low, high) for low, high in _xy_ranges(value, label)]


def _component_policy(kind, value):
    fields = {
        "body": ("default_versions_only", "ids"),
        "motion": ("behavior", "ids"),
        "clothing": ("kinds", "ids"),
        "hair": ("ids",),
    }[kind]
    _keys(value, fields, label=f"{kind}_candidate")
    ids = value["ids"]
    if (
        not isinstance(ids, list)
        or any(not isinstance(i, str) or not i for i in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError(f"invalid_{kind}_ids")
    if kind == "body" and type(value["default_versions_only"]) is not bool:
        raise ValueError("invalid_default_versions_only")
    if kind == "motion" and value["behavior"] not in ("walk", "ride"):
        raise ValueError("invalid_motion_behavior")
    if kind == "clothing":
        kinds = value["kinds"]
        if (
            not isinstance(kinds, list)
            or not kinds
            or any(k not in ("simple_barycentric", "native_locked") for k in kinds)
            or len(kinds) != len(set(kinds))
        ):
            raise ValueError("invalid_clothing_kinds")


def _choose(inventory, kind, policy, rng, *, body=None, motion=None, records=None):
    _component_policy(kind, policy)
    rows = {row["id"]: row for row in inventory[COMPONENTS[kind]]}
    ids = policy["ids"]
    for ident in ids:
        if ident not in rows:
            raise ValueError(f"unknown_{kind}:{ident}")
        if rows[ident]["status"] != "ready":
            raise ValueError(f"unavailable_{kind}:{ident}:{rows[ident]['status']}")

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
                raise ValueError(f"{kind}_candidate_conflict:{ident}")
            continue
        candidates.append(ident)
    if not candidates:
        raise ValueError(f"no_usable_{kind}_candidate")
    return rng.choice(sorted(candidates))


def _template(template):
    _keys(
        template,
        ("template_id", "base_scene"),
        ("route_offset_xy_m", "people", "rigs"),
        "template",
    )
    ident = template["template_id"]
    if not isinstance(ident, str) or re.fullmatch(r"[a-z0-9_]{1,48}", ident) is None:
        raise ValueError("invalid_template_id")
    validate_scene(template["base_scene"])
    people = template.get("people", [])
    rigs = template.get("rigs", [])
    if not isinstance(people, list) or not isinstance(rigs, list):
        raise ValueError(f"{ident}:invalid_targets")
    base_people = {p["person_id"] for p in template["base_scene"]["people"]}
    base_rigs = {r["rig_id"] for r in template["base_scene"]["rigs"]}
    seen_people, seen_rigs = set(), set()
    if "route_offset_xy_m" in template:
        try:
            _xy_ranges(template["route_offset_xy_m"], "shared_route_offset")
        except ValueError as error:
            raise ValueError(f"{ident}:{error}") from error
    for item in people:
        _keys(
            item,
            ("person_id",),
            ("components", "speed_m_s", "motion_start", "start_progress_m", "route_offset_xy_m"),
            "person_rule",
        )
        pid = item["person_id"]
        if type(pid) is not int or pid not in base_people or pid in seen_people:
            raise ValueError(f"{ident}:invalid_person_rule:{pid}")
        seen_people.add(pid)
        if "components" in item:
            policies = item["components"]
            if not isinstance(policies, dict) or not policies or set(policies) - set(COMPONENTS):
                raise ValueError(f"{ident}:person_{pid}:invalid_components")
            for kind, policy in policies.items():
                try:
                    _component_policy(kind, policy)
                except ValueError as error:
                    raise ValueError(f"{ident}:person_{pid}:{error}") from error
        for field in ("speed_m_s", "motion_start", "start_progress_m"):
            if field in item:
                try:
                    _range(item[field], field, integer=field == "motion_start")
                except ValueError as error:
                    raise ValueError(f"{ident}:person_{pid}:{error}") from error
        if "route_offset_xy_m" in item:
            try:
                _xy_ranges(item["route_offset_xy_m"], "person_route_offset")
            except ValueError as error:
                raise ValueError(f"{ident}:person_{pid}:{error}") from error
    for item in rigs:
        _keys(item, ("rig_id",), ("offset_xy_m", "offset_yaw_deg"), "rig_rule")
        rid = item["rig_id"]
        if not isinstance(rid, str) or rid not in base_rigs or rid in seen_rigs:
            raise ValueError(f"{ident}:invalid_rig_rule:{rid}")
        seen_rigs.add(rid)
        if "offset_xy_m" in item:
            try:
                _xy_ranges(item["offset_xy_m"], "rig_offset")
            except ValueError as error:
                raise ValueError(f"{ident}:{rid}:{error}") from error
        if "offset_yaw_deg" in item:
            try:
                _range(item["offset_yaw_deg"], "rig_yaw_offset")
            except ValueError as error:
                raise ValueError(f"{ident}:{rid}:{error}") from error


def sample_scene(template, seed, index, inventory=None, motions=None):
    """Return one deterministic scene; never mutate the embedded base scene."""
    _number(seed, "seed", integer=True)
    if seed < 1 or type(index) is not int or index < 0:
        raise ValueError("invalid_seed_or_index")
    _template(template)
    ident = template["template_id"]
    inventory = describe() if inventory is None else inventory
    motions = catalog()["motions"] if motions is None else motions
    scene = deepcopy(template["base_scene"])
    scene["scene_id"] = f"batch_{ident}_v{index + 1:04d}"
    scene["seed"] = _rng(seed, ident, index, "render").randrange(1, 2**31)
    shared = (
        _sample_xy(
            _rng(seed, ident, index, "shared_route"),
            template["route_offset_xy_m"],
            "shared_route_offset",
        )
        if "route_offset_xy_m" in template
        else [0, 0]
    )
    person_rules = {p["person_id"]: p for p in template.get("people", [])}
    for person in scene["people"]:
        pid = person["person_id"]
        spec = person_rules.get(pid, {})
        policies = spec.get("components", {})
        try:
            for kind in ("body", "motion"):
                if kind in policies:
                    person[kind] = _choose(
                        inventory,
                        kind,
                        policies[kind],
                        _rng(seed, ident, index, f"person_{pid}_{kind}"),
                    )
            for field in ("motion_start", "speed_m_s", "start_progress_m"):
                if field in spec:
                    person[field] = _sample(
                        _rng(seed, ident, index, f"person_{pid}_{field}"),
                        spec[field],
                        field,
                        integer=field == "motion_start",
                    )
            if "clothing" in policies:
                records = _asset_timeline(
                    motions[person["motion"]],
                    person["motion_start"],
                    scene["frame_count"],
                    scene["fps"],
                )
                person["clothing"] = _choose(
                    inventory,
                    "clothing",
                    policies["clothing"],
                    _rng(seed, ident, index, f"person_{pid}_clothing"),
                    body=person["body"],
                    motion=person["motion"],
                    records=records,
                )
            if "hair" in policies:
                person["hair"] = _choose(
                    inventory,
                    "hair",
                    policies["hair"],
                    _rng(seed, ident, index, f"person_{pid}_hair"),
                )
            personal = (
                _sample_xy(
                    _rng(seed, ident, index, f"person_{pid}_route"),
                    spec["route_offset_xy_m"],
                    "person_route_offset",
                )
                if "route_offset_xy_m" in spec
                else [0, 0]
            )
            person["route_xy_m"] = [
                [x + shared[0] + personal[0], y + shared[1] + personal[1]]
                for x, y in person["route_xy_m"]
            ]
        except (ValueError, KeyError) as error:
            raise ValueError(f"{ident}:person_{pid}:{error}") from error
    rig_rules = {r["rig_id"]: r for r in template.get("rigs", [])}
    for rig in scene["rigs"]:
        rid = rig["rig_id"]
        spec = rig_rules.get(rid, {})
        if "offset_xy_m" in spec:
            dx, dy = _sample_xy(
                _rng(seed, ident, index, f"rig_{rid}_xy"), spec["offset_xy_m"], "rig_offset"
            )
            rig["origin_m"][0] += dx
            rig["origin_m"][1] += dy
        if "offset_yaw_deg" in spec:
            rig["yaw_deg"] += _sample(
                _rng(seed, ident, index, f"rig_{rid}_yaw"), spec["offset_yaw_deg"], "rig_yaw_offset"
            )
    check = validate(scene)
    if check["status"] != "passed":
        raise ValueError(f"{ident}:variant_{index + 1}:{check['errors']}")
    return scene


def sample_batch(rules, count):
    """Draw count scenes from each template; count belongs to this invocation."""
    _keys(rules, ("schema", "seed", "templates"), label="batch_rules")
    if (
        rules["schema"] != SCHEMA
        or type(rules["seed"]) is not int
        or rules["seed"] < 1
        or not isinstance(rules["templates"], list)
        or not rules["templates"]
    ):
        raise ValueError("invalid_batch_rules")
    if type(count) is not int or count < 1:
        raise ValueError("invalid_batch_count")
    templates = rules["templates"]
    for template in templates:
        _template(template)
    ids = [t["template_id"] for t in templates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_template_id")
    found = catalog()
    inventory = describe(found)
    scenes = []
    for template in templates:
        for index in range(count):
            scenes.append(sample_scene(template, rules["seed"], index, inventory, found["motions"]))
    return scenes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample editor scene variants; no geometry or rendering."
    )
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--count", type=int, required=True, help="number of scenes per template")
    args = parser.parse_args(argv)
    if re.fullmatch(r"[a-z0-9_-]{1,64}", args.name) is None:
        parser.error("invalid_batch_name")
    rules = json.loads(args.rules.read_text())
    output = PROJECT_ROOT / "outputs/scenes" / args.name
    if output.exists():
        raise FileExistsError(output)
    scenes = sample_batch(rules, args.count)
    output.mkdir(parents=True)
    paths = []
    for scene in scenes:
        path = output / f"{scene['scene_id']}.json"
        write_json(path, scene)
        paths.append(str(path))
    print(json.dumps(dict(output=str(output), scenes=paths), ensure_ascii=False))


if __name__ == "__main__":
    main()
