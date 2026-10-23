"""Shared bpy construction primitives; callers retain lighting/ID/default policy.

Import only inside Blender. Creation order, node names and numeric expressions
are retained from the two historical ERP renderers. No scene loading occurs here.
"""

import json
import math
import bpy
import numpy as np
from mathutils import Matrix


def configure_render(s, width, height, samples):
    """Set shared raster/sample policy; caller retains seed, device, look and pass flags."""
    s.render.resolution_x = width
    s.render.resolution_y = height
    s.render.resolution_percentage = 100
    # Eevee cannot render panoramic cameras, so use Cycles with the same static geometry.
    s.render.engine = "CYCLES"
    s.cycles.samples = samples
    s.cycles.use_denoising = False
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGBA"
    s.render.film_transparent = False


def configure_compositor(s, output_dir, name):
    """Create the named 32-bit Combined/Depth/IndexOB output using the existing API branch."""
    if hasattr(s, "compositing_node_group"):
        nt = bpy.data.node_groups.new(name, "CompositorNodeTree")
        s.compositing_node_group = nt
    else:
        s.use_nodes = True
        nt = s.node_tree
        nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    rl.scene = s
    file_node = nt.nodes.new("CompositorNodeOutputFile")
    if hasattr(file_node, "directory"):
        file_node.directory = str(output_dir)
        file_node.file_name = "source"
        file_node.format.file_format = "OPEN_EXR_MULTILAYER"
        file_node.format.color_depth = "32"
        file_node.file_output_items.new("RGBA", "Combined")
        file_node.file_output_items.new("FLOAT", "Depth")
        file_node.file_output_items.new("FLOAT", "IndexOB")
    else:
        file_node.base_path = str(output_dir)
        file_node.format.file_format = "OPEN_EXR_MULTILAYER"
        file_node.format.color_depth = "32"
        file_node.file_slots.clear()
        file_node.file_slots.new("Combined")
        file_node.file_slots.new("Depth")
        file_node.file_slots.new("IndexOB")
    index_output = next((socket for socket in rl.outputs if "Index" in socket.name), None)
    if index_output is None:
        raise RuntimeError(
            f"object index render pass unavailable: {[x.name for x in rl.outputs]}"
        )
    nt.links.new(rl.outputs["Image"], file_node.inputs["Combined"])
    nt.links.new(rl.outputs["Depth"], file_node.inputs["Depth"])
    nt.links.new(index_output, file_node.inputs["IndexOB"])
    return file_node


def create_environment(s, name, hdri):
    """Create the historical HDRI/constant world only when the calling path requests it."""
    world = bpy.data.worlds.new(name)
    s.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    wo = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1
    if hdri:
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(hdri.resolve()))
        env.image.colorspace_settings.name = "Linear Rec.709"
        links.new(env.outputs["Color"], bg.inputs["Color"])
    else:
        bg.inputs["Color"].default_value = (0.08, 0.1, 0.14, 1)
    links.new(bg.outputs["Background"], wo.inputs["Surface"])


def create_erp_camera(s):
    """Create full-sphere ERP camera: longitude ±pi, latitude ±pi/2 in radians."""
    cd = bpy.data.cameras.new("OmniSourceCamera")
    cd.type = "PANO"
    pano = cd.cycles if hasattr(cd, "cycles") else cd
    pano.panorama_type = "EQUIRECTANGULAR"
    pano.longitude_min = -math.pi
    pano.longitude_max = math.pi
    pano.latitude_min = -math.pi / 2
    pano.latitude_max = math.pi / 2
    cam = bpy.data.objects.new("OmniSourceCamera", cd)
    s.collection.objects.link(cam)
    s.camera = cam
    return cam


def render_cameras(s, cam, file_node, cameras_path, output_dir):
    """CV camera-to-world -> Blender right/up/back axes; render and resolve legacy filenames."""
    records = []
    for item in json.loads(cameras_path.read_text())["cameras"]:
        pose = np.asarray(item["camera_to_world"], float)
        r = pose[:3, :3]
        # Blender local axes: X right, Y up, -Z forward. CV: X right, Y down, Z forward.
        matrix = np.eye(4)
        matrix[:3, 0] = r[:, 0]
        matrix[:3, 1] = -r[:, 1]
        matrix[:3, 2] = -r[:, 2]
        matrix[:3, 3] = pose[:3, 3]
        cam.matrix_world = Matrix(matrix.tolist())
        if hasattr(file_node, "file_name"):
            file_node.file_name = item["id"]
        else:
            for slot in file_node.file_slots:
                slot.path = item["id"]
        s.render.filepath = str(output_dir / f"{item['id']}_preview.png")
        bpy.ops.render.render(write_still=True)
        candidates = sorted(output_dir.glob(item["id"] + "*.exr"))
        if len(candidates) != 1:
            raise RuntimeError(
                f'cannot locate compositor output for {item["id"]}: {candidates}'
            )
        target = output_dir / f"{item['id']}.exr"
        if candidates[0] != target:
            candidates[0].replace(target)
        records.append(
            {
                "id": item["id"],
                "source": str(target.resolve()),
                "camera_to_world": item["camera_to_world"],
            }
        )
    return records


def create_mesh_object(scene, name, vertices, faces, pass_index):
    """Create local-metre [V,3] mesh and return (object, mesh); caller sets its world transform."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.pass_index = pass_index
    return obj, mesh


def add_body_material(mesh, name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bs = mat.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Roughness"].default_value = 0.58
    bs.inputs["Base Color"].default_value = (0.55, 0.25, 0.16, 1)
    mesh.materials.append(mat)
    return mat, bs


def add_surface_material(mesh, name, color, roughness):
    """Append caller-selected RGBA/roughness; keep historical per-path material values."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    mesh.materials.append(material)
    return material
