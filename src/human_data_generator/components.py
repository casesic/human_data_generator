"""Read-only catalogue and technical compatibility checks for four components."""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import resources

KINDS = ("bodies", "motions", "clothing", "hair")


def _check_motion(entry):
    rows = json.loads(Path(entry["timeline"]).read_text())
    if not rows or any(row["asset_frame"] != i for i, row in enumerate(rows)):
        raise ValueError("motion_timeline_order")
    ticks = [row["source_time_tick"] for row in rows]
    if len({tuple(row["time_base"]) for row in rows}) != 1 or any(
        a >= b for a, b in zip(ticks, ticks[1:])
    ):
        raise ValueError("motion_integer_time")
    with np.load(entry["parameters"], allow_pickle=False) as params:
        n = len(rows)
        dims = {
            "global_orient": 3,
            "body_pose": 63,
            "jaw_pose": 3,
            "leye_pose": 3,
            "reye_pose": 3,
            "left_hand_pose": 45,
            "right_hand_pose": 45,
            "expression": 10,
        }
        for key, dim in dims.items():
            if (
                key == "expression"
                and key not in params
                and entry.get("expression_policy") == "neutral_zero_missing"
            ):
                continue
            if key not in params or params[key].shape != (n, dim):
                raise ValueError("motion_pose_basis:" + key)
        if params["transl"].shape != (n, 3) or params["betas"].shape not in ((10,), (30,)):
            raise ValueError("motion_translation_or_shape")
        if entry["root_mode"] == "source_root_relative_to_start_plus_fixed_scene_matrix":
            if "source_root_relative" not in params or params["source_root_relative"].shape != (
                n,
                3,
            ):
                raise ValueError("walking_root_missing")
        elif entry["root_mode"] != "fixed_pelvis_scene_owns_progression":
            raise ValueError("unsupported_motion_root_mode")
    return rows


def catalog():
    paths = resources()["component_catalog"]
    if set(paths) != set(KINDS):
        raise ValueError("component_catalog_paths_required")
    result = {}
    for kind in KINDS:
        data = json.loads(Path(paths[kind]).read_text())
        if data.get("schema") != "human_components_v1" or not isinstance(data.get("items"), dict):
            raise ValueError("invalid_component_catalog:" + kind)
        result[kind] = data["items"]
    return result


def _needed(entry, keys):
    return [Path(entry[key]) for key in keys]


def describe(found=None):
    """A compact inventory. Status comes from real required-file checks."""
    if found is None:
        found = catalog()
    result = {}
    for kind, items in found.items():
        rows = []
        for ident, entry in sorted(items.items()):
            status = entry["status"]
            reason = entry.get("reason")
            timeline = None
            required = {
                "bodies": ("manifest",),
                "motions": ("parameters", "timeline"),
                "clothing": (
                    ("asset_manifest",) if entry.get("kind") == "native_locked" else ("reference",)
                ),
                "hair": ("reference",),
            }[kind]
            if status == "ready":
                missing = [str(p) for p in _needed(entry, required) if not p.exists()]
                if missing:
                    status, reason = "missing_files", "missing: " + ", ".join(missing)
                elif kind == "bodies":
                    body = json.loads(Path(entry["manifest"]).read_text())
                    paths = [
                        Path(body["model_file"]),
                        Path(body["reference"]["geometry"]) / "faces.npy",
                        Path(body["reference"]["geometry"]) / "vertices.npy",
                    ]
                    if any(not p.is_file() for p in paths):
                        status, reason = "missing_files", "body model or reference geometry missing"
                    elif len(body.get("betas", [])) != 30 or body.get("num_betas") != 30:
                        status, reason = (
                            "needs_conversion",
                            "current composition requires 30 target betas",
                        )
                elif kind == "motions":
                    try:
                        timeline = _check_motion(entry)
                    except (ValueError, KeyError) as error:
                        status, reason = "invalid_format", str(error)
                elif kind == "clothing" and entry.get("kind") == "simple_barycentric":
                    if any(
                        not (Path(entry["reference"]) / f).is_file()
                        for f in (
                            "body_barycentric_indices.npy",
                            "body_barycentric_weights.npy",
                            "clothing_faces.npy",
                            "materials.json",
                            "garment_material.json",
                        )
                    ):
                        status, reason = (
                            "missing_files",
                            "simple garment binding or material missing",
                        )
                elif kind == "hair":
                    if any(
                        not (Path(entry["reference"]) / f).is_file()
                        for f in (
                            "points.npy",
                            "radius.npy",
                            "curve_offsets.npy",
                            "active_curve_mask.npy",
                            "groom_manifest.json",
                        )
                    ):
                        status, reason = "missing_files", "groom arrays or binding metadata missing"
                    else:
                        gm = json.loads(
                            (Path(entry["reference"]) / "groom_manifest.json").read_text()
                        )
                        if (
                            not (Path(gm["binding"]) / "binding.npz").is_file()
                            or not Path(gm["source_abc"]).is_file()
                        ):
                            status, reason = (
                                "missing_files",
                                "groom binding or Alembic source missing",
                            )
            motion_info = {}
            if kind == "motions" and status == "ready":
                motion_info = dict(
                    available_frames=len(timeline),
                    first_source_frame=timeline[0]["source_frame"],
                    last_source_frame=timeline[-1]["source_frame"],
                    time_base=timeline[0]["time_base"],
                )
            rows.append(
                dict(
                    id=ident,
                    status=status,
                    version=entry.get("version"),
                    default_version=entry.get("default_version", True),
                    behavior=entry.get("behavior"),
                    reason=reason,
                    resource=entry.get(
                        "manifest",
                        entry.get(
                            "parameters",
                            entry.get(
                                "reference", entry.get("asset_manifest", entry.get("source"))
                            ),
                        ),
                    ),
                    family=entry.get("family"),
                    source_group=entry.get("source_group"),
                    kind=entry.get("kind"),
                    locked_body=entry.get("locked_body"),
                    locked_motion=entry.get("locked_motion"),
                    locked_window=entry.get("locked_window"),
                    reference_body=entry.get("reference_body"),
                    fitted_body=entry.get("fitted_body"),
                    root_mode=entry.get("root_mode"),
                    license=entry.get("license"),
                    restriction=reason if entry.get("kind") == "native_locked" else None,
                    **motion_info,
                )
            )
        result[kind] = rows
    return dict(
        schema="human_component_inventory_v1",
        counts={
            k: dict(total=len(v), ready=sum(x["status"] == "ready" for x in v))
            for k, v in result.items()
        },
        **result,
    )


def select(person, found=None):
    if found is None:
        found = catalog()
    selection = {}
    for kind, field in (
        ("bodies", "body"),
        ("motions", "motion"),
        ("clothing", "clothing"),
        ("hair", "hair"),
    ):
        ident = person[field]
        if ident not in found[kind]:
            raise ValueError("unknown_" + field + ":" + str(ident))
        entry = found[kind][ident]
        if entry["status"] != "ready":
            raise ValueError("component_not_ready:" + ident + ":" + entry.get("reason", ""))
        selection[field] = dict(id=ident, **entry)
    body = json.loads(Path(selection["body"]["manifest"]).read_text())
    if len(body.get("betas", [])) != 30 or body.get("gender") not in ("male", "female"):
        raise ValueError("invalid_target_body_shape")
    if not Path(body["model_file"]).is_file():
        raise FileNotFoundError("body_model_missing")
    target_faces = np.load(Path(body["reference"]["geometry"]) / "faces.npy", mmap_mode="r")
    for field in ("motion", "clothing", "hair"):
        owner = selection[field]["fitted_body" if field == "motion" else "reference_body"]
        other = found["bodies"].get(owner)
        if other is None or other["status"] != "ready":
            raise ValueError("unavailable_component_reference_body:" + owner)
        source = json.loads(Path(other["manifest"]).read_text())
        if Path(source["model_dir"]).resolve() != Path(body["model_dir"]).resolve():
            raise ValueError("different_smplx_model_release")
        faces = np.load(Path(source["reference"]["geometry"]) / "faces.npy", mmap_mode="r")
        if not np.array_equal(faces, target_faces):
            raise ValueError("semantic_topology_face_order_conflict")
    motion = selection["motion"]
    if any(not Path(motion[key]).is_file() for key in ("parameters", "timeline")):
        raise FileNotFoundError("motion_conversion_missing")
    _check_motion(motion)
    if (
        selection["clothing"]["kind"] not in ("simple_barycentric", "native_locked")
        or selection["hair"]["kind"] != "static_groom"
    ):
        raise ValueError("unsupported_component_kind")
    cloth = (
        Path(selection["clothing"]["reference"])
        if selection["clothing"]["kind"] == "simple_barycentric"
        else None
    )
    groom = Path(selection["hair"]["reference"])
    if cloth is not None:
        if not 0 < float(selection["clothing"]["offset_m"]) <= 0.01:
            raise ValueError("invalid_simple_clothing_offset")
        if any(
            not (cloth / name).is_file()
            for name in (
                "body_barycentric_indices.npy",
                "body_barycentric_weights.npy",
                "clothing_faces.npy",
                "materials.json",
                "garment_material.json",
                "clothing_manifest.json",
            )
        ):
            raise FileNotFoundError("simple_clothing_files_missing")
        if (
            Path(
                json.loads((cloth / "clothing_manifest.json").read_text())["body_geometry"]
            ).resolve()
            != Path(
                json.loads(
                    Path(
                        found["bodies"][selection["clothing"]["reference_body"]]["manifest"]
                    ).read_text()
                )["reference"]["geometry"]
            ).resolve()
        ):
            raise ValueError("clothing_binding_owner_conflict")
    else:
        locked = selection["clothing"]
        if person["body"] != locked["locked_body"] or person["motion"] != locked["locked_motion"]:
            raise ValueError("native_clothing_locked_body_motion")
    gm = json.loads((groom / "groom_manifest.json").read_text())
    if any(
        not (groom / name).is_file()
        for name in (
            "points.npy",
            "radius.npy",
            "curve_offsets.npy",
            "active_curve_mask.npy",
            "collision_correction.npy",
        )
    ):
        raise FileNotFoundError("groom_files_missing")
    hair_owner = json.loads(
        Path(found["bodies"][selection["hair"]["reference_body"]]["manifest"]).read_text()
    )
    if Path(gm["body_geometry"]).resolve() != Path(hair_owner["reference"]["geometry"]).resolve():
        raise ValueError("hair_binding_owner_conflict")
    if not (Path(gm["binding"]) / "binding.npz").is_file() or not Path(gm["source_abc"]).is_file():
        raise FileNotFoundError("hair_binding_or_alembic_missing")
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect existing body, motion, clothing and hair components without building geometry."
    )
    parser.add_argument("--kind", choices=KINDS)
    args = parser.parse_args(argv)
    data = describe()
    print(
        json.dumps(
            (
                data
                if args.kind is None
                else {args.kind: data[args.kind], "counts": data["counts"][args.kind]}
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
