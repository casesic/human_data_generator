"""Per-character skin and native-eye shaders, shared by review and DS renders."""

from pathlib import Path
import bpy
import numpy as np


def apply_character_materials(mesh, faces, appearance, *, clay=False):
    with np.load(appearance["body_uv"]) as uv:
        np.testing.assert_array_equal(uv["faces"], faces)
        layer = mesh.uv_layers.new(name="SMPLX_UV")
        for polygon, indices in zip(mesh.polygons, uv["faces_uv"]):
            for loop, index in zip(polygon.loop_indices, indices):
                layer.data[loop].uv = uv["uv"][index]
    skin = bpy.data.materials.new(f"Skin_{appearance['character_id']}")
    skin.use_nodes = True
    nodes, links = skin.node_tree.nodes, skin.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.58
    mesh.materials.append(skin)
    if clay:
        bsdf.inputs["Base Color"].default_value = (0.4, 0.4, 0.4, 1)
        return
    original = nodes.new("ShaderNodeTexImage")
    original.image = bpy.data.images.load(
        str(Path(appearance["body_texture"]).resolve()), check_existing=True
    )
    original.image.colorspace_settings.name = "sRGB"
    config = appearance["eye_patch"]
    uv = nodes.new("ShaderNodeTexCoord")
    # Optional local SMPL-albedo -> SMPL-X eyelid alignment. This affects
    # shader sampling only, never geometry, UV topology or semantic labels.
    sample_uv = uv.outputs["UV"]
    alignment = appearance.get("face_uv_alignment")
    if alignment:
        xy = nodes.new("ShaderNodeSeparateXYZ")
        links.new(sample_uv, xy.inputs[0])

        def ramp(socket, low, high, reverse=False):
            node = nodes.new("ShaderNodeMapRange")
            node.clamp = True
            links.new(socket, node.inputs["Value"])
            node.inputs["From Min"].default_value = low
            node.inputs["From Max"].default_value = high
            node.inputs["To Min"].default_value = 1 if reverse else 0
            node.inputs["To Max"].default_value = 0 if reverse else 1
            return node.outputs[0]

        weight = None
        for axis, knots in [("X", alignment["x_knots"]), ("Y", alignment["y_knots"])]:
            for value in [
                ramp(xy.outputs[axis], knots[0], knots[1]),
                ramp(xy.outputs[axis], knots[2], knots[3], True),
            ]:
                if weight is None:
                    weight = value
                else:
                    mul = nodes.new("ShaderNodeMath")
                    mul.operation = "MULTIPLY"
                    links.new(weight, mul.inputs[0])
                    links.new(value, mul.inputs[1])
                    weight = mul.outputs[0]
        delta = nodes.new("ShaderNodeVectorMath")
        delta.operation = "MULTIPLY"
        links.new(sample_uv, delta.inputs[0])
        delta.inputs[1].default_value = (alignment["u_scale"] - 1, 0, 0)
        shift = nodes.new("ShaderNodeVectorMath")
        shift.operation = "ADD"
        links.new(delta.outputs["Vector"], shift.inputs[0])
        shift.inputs[1].default_value = (*alignment["offset"], 0)
        scaled = nodes.new("ShaderNodeVectorMath")
        scaled.operation = "SCALE"
        links.new(shift.outputs["Vector"], scaled.inputs[0])
        links.new(weight, scaled.inputs["Scale"])
        added = nodes.new("ShaderNodeVectorMath")
        added.operation = "ADD"
        links.new(sample_uv, added.inputs[0])
        links.new(scaled.outputs["Vector"], added.inputs[1])
        sample_uv = added.outputs["Vector"]
    for warp in appearance.get("eye_uv_warps", []):
        delta = nodes.new("ShaderNodeVectorMath")
        delta.operation = "SUBTRACT"
        links.new(uv.outputs["UV"], delta.inputs[0])
        delta.inputs[1].default_value = (*warp["center"], 0)
        scaled = nodes.new("ShaderNodeVectorMath")
        scaled.operation = "MULTIPLY"
        links.new(delta.outputs["Vector"], scaled.inputs[0])
        scaled.inputs[1].default_value = (1 / warp["radii"][0], 1 / warp["radii"][1], 0)
        length = nodes.new("ShaderNodeVectorMath")
        length.operation = "LENGTH"
        links.new(scaled.outputs["Vector"], length.inputs[0])
        weight = nodes.new("ShaderNodeMapRange")
        weight.clamp = True
        weight.interpolation_type = "SMOOTHSTEP"
        links.new(length.outputs["Value"], weight.inputs["Value"])
        weight.inputs["From Min"].default_value = 0.55
        weight.inputs["From Max"].default_value = 1
        weight.inputs["To Min"].default_value = 1
        weight.inputs["To Max"].default_value = 0
        shift = nodes.new("ShaderNodeVectorMath")
        shift.operation = "SCALE"
        shift.inputs[0].default_value = (*warp["offset"], 0)
        links.new(weight.outputs[0], shift.inputs["Scale"])
        added = nodes.new("ShaderNodeVectorMath")
        added.operation = "ADD"
        links.new(sample_uv, added.inputs[0])
        links.new(shift.outputs["Vector"], added.inputs[1])
        sample_uv = added.outputs["Vector"]
    links.new(sample_uv, original.inputs["Vector"])
    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "ADD"
    links.new(uv.outputs["UV"], offset.inputs[0])
    offset.inputs[1].default_value = (*config["sample_offset"], 0)
    adjacent = nodes.new("ShaderNodeTexImage")
    adjacent.image = original.image
    links.new(offset.outputs["Vector"], adjacent.inputs["Vector"])
    color = original.outputs["Color"]
    for center in config["centers"]:
        delta = nodes.new("ShaderNodeVectorMath")
        delta.operation = "SUBTRACT"
        links.new(uv.outputs["UV"], delta.inputs[0])
        delta.inputs[1].default_value = (*center, 0)
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "MULTIPLY"
        links.new(delta.outputs["Vector"], scale.inputs[0])
        scale.inputs[1].default_value = (1 / config["radii"][0], 1 / config["radii"][1], 0)
        radius = nodes.new("ShaderNodeVectorMath")
        radius.operation = "LENGTH"
        links.new(scale.outputs["Vector"], radius.inputs[0])
        feather = nodes.new("ShaderNodeMapRange")
        feather.clamp = True
        feather.interpolation_type = "SMOOTHSTEP"
        links.new(radius.outputs["Value"], feather.inputs["Value"])
        feather.inputs["From Min"].default_value = config["feather_inner"]
        feather.inputs["From Max"].default_value = 1
        feather.inputs["To Min"].default_value = 1
        feather.inputs["To Max"].default_value = 0
        mix = nodes.new("ShaderNodeMixRGB")
        links.new(feather.outputs[0], mix.inputs[0])
        links.new(color, mix.inputs[1])
        links.new(adjacent.outputs["Color"], mix.inputs[2])
        color = mix.outputs[0]
    scalp = appearance.get("scalp_replacement")
    if scalp:
        values = np.load(scalp["vertex_mask"]).astype(np.float32)
        if values.shape != (len(mesh.vertices),) or not np.isfinite(values).all():
            raise ValueError("Invalid scalp mask")
        attribute = mesh.attributes.new("ScalpReplacement", "FLOAT", "POINT")
        attribute.data.foreach_set("value", values)
        source = nodes.new("ShaderNodeAttribute")
        source.attribute_name = attribute.name
        luminance = nodes.new("ShaderNodeRGBToBW")
        links.new(color, luminance.inputs[0])
        darkness = nodes.new("ShaderNodeMapRange")
        darkness.clamp = True
        links.new(luminance.outputs[0], darkness.inputs["Value"])
        darkness.inputs["From Min"].default_value = 0.015
        darkness.inputs["From Max"].default_value = 0.055
        darkness.inputs["To Min"].default_value = 1
        darkness.inputs["To Max"].default_value = 0
        weight = nodes.new("ShaderNodeMath")
        weight.operation = "MULTIPLY"
        links.new(source.outputs["Fac"], weight.inputs[0])
        if scalp.get("darkness_gate", True):
            links.new(darkness.outputs[0], weight.inputs[1])
        else:
            weight.inputs[1].default_value = 1
        replacement = nodes.new("ShaderNodeMixRGB")
        links.new(weight.outputs[0], replacement.inputs[0])
        links.new(color, replacement.inputs[1])
        replacement.inputs[2].default_value = (*scalp["color_linear"], 1)
        color = replacement.outputs[0]
    links.new(color, bsdf.inputs["Base Color"])
    with np.load(appearance["eye_mapping"]) as mapping:
        np.testing.assert_array_equal(mapping["faces"], faces)
        coords = mapping["coordinates"]
        labels = mapping["component_labels"]
    if coords.shape != (len(mesh.vertices), 3) or not np.isfinite(coords).all():
        raise ValueError("Invalid rest-eye coordinates")
    attribute = mesh.attributes.new("EyeRestCoordinates", "FLOAT_VECTOR", "POINT")
    attribute.data.foreach_set("vector", coords.astype(np.float32).ravel())
    eye = bpy.data.materials.new(f"Eyes_{appearance['character_id']}")
    eye.use_nodes = True
    nodes, links = eye.node_tree.nodes, eye.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.36
    bsdf.inputs["Specular IOR Level"].default_value = 0.2
    source = nodes.new("ShaderNodeAttribute")
    source.attribute_name = attribute.name
    split = nodes.new("ShaderNodeSeparateXYZ")
    links.new(source.outputs["Vector"], split.inputs[0])
    planar = nodes.new("ShaderNodeVectorMath")
    planar.operation = "MULTIPLY"
    planar.inputs[1].default_value = (1, 1, 0)
    links.new(source.outputs["Vector"], planar.inputs[0])
    radius = nodes.new("ShaderNodeVectorMath")
    radius.operation = "LENGTH"
    links.new(planar.outputs["Vector"], radius.inputs[0])
    front = nodes.new("ShaderNodeMath")
    front.operation = "GREATER_THAN"
    links.new(split.outputs["Z"], front.inputs[0])
    front.inputs[1].default_value = 0
    color = None
    for size, fill in [
        (0.0057, (0.015, 0.025, 0.028, 1)),
        (0.0051, appearance["iris_color"]),
        (0.0022, (0.002, 0.003, 0.004, 1)),
    ]:
        less = nodes.new("ShaderNodeMath")
        less.operation = "LESS_THAN"
        links.new(radius.outputs["Value"], less.inputs[0])
        less.inputs[1].default_value = size
        mask = nodes.new("ShaderNodeMath")
        mask.operation = "MULTIPLY"
        links.new(less.outputs[0], mask.inputs[0])
        links.new(front.outputs[0], mask.inputs[1])
        mix = nodes.new("ShaderNodeMixRGB")
        if color is None:
            mix.inputs[1].default_value = (0.62, 0.64, 0.65, 1)
        else:
            links.new(color, mix.inputs[1])
        mix.inputs[2].default_value = fill
        links.new(mask.outputs[0], mix.inputs[0])
        color = mix.outputs[0]
    links.new(color, bsdf.inputs["Base Color"])
    mesh.materials.append(eye)
    for polygon in mesh.polygons:
        if np.all(labels[faces[polygon.index]] != 0):
            polygon.material_index = len(mesh.materials) - 1


def add_clothing_materials(mesh, folder):
    import json

    native = Path(folder) / "native_material.json"
    if native.is_file():
        config = json.loads(native.read_text())
        uv = np.load(config["uv_vertices"])
        faces_uv = np.load(config["uv_faces"])
        if len(faces_uv) != len(mesh.polygons) or faces_uv.shape[1] != 3 or uv.shape[1] != 2:
            raise ValueError("Native clothing UV topology mismatch")
        layer = mesh.uv_layers.new(name="GarmentUV")
        for polygon, indices in zip(mesh.polygons, faces_uv):
            for loop, index in zip(polygon.loop_indices, indices):
                layer.data[loop].uv = uv[index]
        material = bpy.data.materials.new("NativeClothing")
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        color = nodes.new("ShaderNodeTexImage")
        color.image = bpy.data.images.load(config["diffuse"], check_existing=True)
        color.image.colorspace_settings.name = "sRGB"
        links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
        normal = nodes.new("ShaderNodeTexImage")
        normal.image = bpy.data.images.load(config["normal"], check_existing=True)
        normal.image.colorspace_settings.name = "Non-Color"
        mapping = nodes.new("ShaderNodeNormalMap")
        links.new(normal.outputs["Color"], mapping.inputs["Color"])
        links.new(mapping.outputs["Normal"], bsdf.inputs["Normal"])
        mesh.materials.append(material)
        return
    materials = json.loads((Path(folder) / "materials.json").read_text())
    continuous = Path(folder) / "garment_material.json"
    if continuous.exists():
        config = json.loads(continuous.read_text())
        rest = np.load(Path(folder) / "clothing_rest_vertices.npy")
        if rest.shape != (len(mesh.vertices), 3):
            raise ValueError("Garment rest-coordinate mismatch")
        attr = mesh.attributes.new("GarmentRest", "FLOAT_VECTOR", "POINT")
        attr.data.foreach_set("vector", rest.astype(np.float32).ravel())
        material = bpy.data.materials.new("ContinuousJerseyShorts")
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        bsdf.inputs["Roughness"].default_value = 0.75
        source = nodes.new("ShaderNodeAttribute")
        source.attribute_name = attr.name
        split = nodes.new("ShaderNodeSeparateXYZ")
        links.new(source.outputs["Vector"], split.inputs[0])

        def above(value):
            node = nodes.new("ShaderNodeMath")
            node.operation = "GREATER_THAN"
            links.new(split.outputs["Y"], node.inputs[0])
            node.inputs[1].default_value = value
            return node.outputs[0]

        base = nodes.new("ShaderNodeMixRGB")
        base.inputs[1].default_value = materials[1]["color"]
        base.inputs[2].default_value = materials[0]["color"]
        links.new(above(config["waist_y"]), base.inputs[0])
        band = nodes.new("ShaderNodeMath")
        band.operation = "SUBTRACT"
        links.new(above(config["stripe_min_y"]), band.inputs[0])
        links.new(above(config["stripe_max_y"]), band.inputs[1])
        final = nodes.new("ShaderNodeMixRGB")
        links.new(base.outputs[0], final.inputs[1])
        final.inputs[2].default_value = materials[2]["color"]
        links.new(band.outputs[0], final.inputs[0])
        links.new(final.outputs[0], bsdf.inputs["Base Color"])
        mesh.materials.append(material)
        return
    ids = np.load(Path(folder) / "face_material_ids.npy")
    if len(ids) != len(mesh.polygons) or ids.min() < 0 or ids.max() >= len(materials):
        raise ValueError("Invalid clothing material assignments")
    for info in materials:
        material = bpy.data.materials.new(info["name"])
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = info["color"]
        bsdf.inputs["Roughness"].default_value = 0.75
        mesh.materials.append(material)
    for polygon, index in zip(mesh.polygons, ids):
        polygon.material_index = int(index)
