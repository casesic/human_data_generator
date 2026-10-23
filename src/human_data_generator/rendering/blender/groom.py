"""Frozen NURBS groom assets share the body's coordinate transform and instance ID."""

import json
from pathlib import Path
import bpy
import numpy as np


def add_groom(folder, transform, person_id=1):
    folder = Path(folder)
    info = json.loads((folder / "groom_manifest.json").read_text())
    points = np.load(folder / "points.npy")
    radius = np.load(folder / "radius.npy")
    active = (
        np.load(folder / "active_curve_mask.npy")
        if (folder / "active_curve_mask.npy").exists()
        else None
    )
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.alembic_import(
        filepath=info["source_abc"], as_background_job=False, set_frame_range=False
    )
    objects = [o for o in bpy.context.scene.objects if o not in before]
    groups = {g["name"]: g for g in info["source_objects"]}
    objects.sort(key=lambda obj: groups[obj.name]["point_start"])
    material = bpy.data.materials.new("GroomDarkBrown")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfHairPrincipled")
    shader.parametrization = "COLOR"
    shader.inputs["Color"].default_value = (0.022, 0.009, 0.004, 1)
    shader.inputs["Roughness"].default_value = 0.4
    material.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
    for obj in objects:
        group = groups[obj.name]
        if obj.type != "CURVES":
            raise ValueError("Unsupported groom object")
        for modifier in list(obj.modifiers):
            obj.modifiers.remove(modifier)
        start = group["point_start"]
        stop = start + group["point_count"]
        if len(obj.data.points) != stop - start:
            raise ValueError("Groom point ordering/count differs from import audit")
        obj.data.position_data.foreach_set("vector", points[start:stop].ravel())
        obj.data.attributes["radius"].data.foreach_set("value", radius[start:stop])
        if active is not None:
            first = group["curve_start"]
            last = first + group["curve_count"]
            remove = np.flatnonzero(~active[first:last]).tolist()
            if remove:
                obj.data.remove_curves(indices=remove)
            if len(obj.data.curves) != int(active[first:last].sum()):
                raise ValueError("Groom pruning did not reach rendered geometry")
        obj.data.materials.clear()
        obj.data.materials.append(material)
        obj.matrix_world = transform
        obj.pass_index = person_id
        obj.data.update_tag()
    # Avoid Alembic name collisions when another identity uses the same groom.
    for obj in objects:
        obj.name = f"Person{person_id}_{obj.name}"
    return objects
