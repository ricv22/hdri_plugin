bl_info = {
    "name": "Photo → HDRI World (API)",
    "author": "Cursor AI",
    "version": (0, 1, 7),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > HDRI",
    "description": "Upload a photo to an API, get a 2:1 HDRI (.hdr/.exr), apply to World lighting (Cycles).",
    "category": "Lighting",
}

import base64
import json
import math
import os
import tempfile
import time
import urllib.request
import urllib.error
import webbrowser

import bpy
import bpy.utils.previews
import blf
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup


_LIB_PREVIEW_COLLECTION = None
_LIB_PREVIEW_FOLDER = ""
_LIB_PREVIEW_SIGNATURE: tuple[tuple[str, float], ...] = ()
_LIB_PREVIEW_ITEMS: list[tuple[str, str, str, int, int]] = []


def _addon_prefs():
    return bpy.context.preferences.addons[__name__].preferences


def _set_env_image_colorspace(img: bpy.types.Image):
    """Pick a valid scene/linear colorspace (Blender 4.x removed the old name 'Linear')."""
    for name in (
        "Linear Rec.709",
        "scene_linear",
        "Non-Color",
        "Linear CIE-XYZ D65",
    ):
        try:
            img.colorspace_settings.name = name
            return
        except Exception:
            continue


def _ensure_world_nodes(world: bpy.types.World):
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    out = next((n for n in nodes if n.type == "OUTPUT_WORLD"), None)
    if out is None:
        out = nodes.new("ShaderNodeOutputWorld")
        out.location = (400, 0)

    bg = next((n for n in nodes if n.type == "BACKGROUND"), None)
    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
        bg.location = (150, 0)

    if not bg.outputs["Background"].is_linked:
        links.new(bg.outputs["Background"], out.inputs["Surface"])

    env = next(
        (
            n
            for n in nodes
            if n.bl_idname == "ShaderNodeTexEnvironment"
            and n.label not in ("HDRI Blur Source", "HDRI Ground Projected")
        ),
        None,
    )
    if env is None:
        env = nodes.new("ShaderNodeTexEnvironment")
        env.location = (-350, 0)
    env.label = "HDRI Environment"

    env_blur = next(
        (
            n
            for n in nodes
            if n.bl_idname == "ShaderNodeTexEnvironment"
            and n.label == "HDRI Blur Source"
        ),
        None,
    )
    if env_blur is None:
        env_blur = nodes.new("ShaderNodeTexEnvironment")
        env_blur.label = "HDRI Blur Source"
        env_blur.location = (-350, -220)

    env_projected = next(
        (
            n
            for n in nodes
            if n.bl_idname == "ShaderNodeTexEnvironment"
            and n.label == "HDRI Ground Projected"
        ),
        None,
    )
    if env_projected is None:
        env_projected = nodes.new("ShaderNodeTexEnvironment")
        env_projected.label = "HDRI Ground Projected"
        env_projected.location = (-350, -440)

    mix = next((n for n in nodes if n.bl_idname == "ShaderNodeMixRGB" and n.label == "HDRI Blur Mix"), None)
    if mix is None:
        mix = next((n for n in nodes if n.bl_idname == "ShaderNodeMixRGB"), None)
    if mix is None:
        mix = nodes.new("ShaderNodeMixRGB")
        mix.location = (-120, -80)
        mix.blend_type = "MIX"
        mix.inputs["Fac"].default_value = 0.0
    mix.label = "HDRI Blur Mix"

    ground_mix = next(
        (n for n in nodes if n.bl_idname == "ShaderNodeMixRGB" and n.label == "HDRI Ground Mix"),
        None,
    )
    if ground_mix is None:
        ground_mix = nodes.new("ShaderNodeMixRGB")
        ground_mix.location = (-120, -280)
        ground_mix.blend_type = "MIX"
        ground_mix.inputs["Fac"].default_value = 0.0
    ground_mix.label = "HDRI Ground Mix"

    hue_sat = next((n for n in nodes if n.bl_idname == "ShaderNodeHueSaturation"), None)
    if hue_sat is None:
        hue_sat = nodes.new("ShaderNodeHueSaturation")
        hue_sat.location = (30, -80)

    tint_mix = next((n for n in nodes if n.bl_idname == "ShaderNodeMixRGB" and n.label == "HDRI Tint Mix"), None)
    if tint_mix is None:
        tint_mix = nodes.new("ShaderNodeMixRGB")
        tint_mix.location = (110, -80)
        tint_mix.blend_type = "MIX"
        tint_mix.inputs["Fac"].default_value = 0.0
        tint_mix.inputs["Color2"].default_value = (1.0, 1.0, 1.0, 1.0)
    tint_mix.label = "HDRI Tint Mix"

    mapping = next((n for n in nodes if n.bl_idname == "ShaderNodeMapping"), None)
    if mapping is None:
        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (-650, 0)
    mapping.vector_type = "TEXTURE"

    texcoord = next((n for n in nodes if n.bl_idname == "ShaderNodeTexCoord"), None)
    if texcoord is None:
        texcoord = nodes.new("ShaderNodeTexCoord")
        texcoord.location = (-850, 0)

    if not mapping.inputs["Vector"].is_linked:
        links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    mapping_vector = mapping.outputs["Vector"]
    for env_node in (env, env_blur):
        for link in list(env_node.inputs["Vector"].links):
            links.remove(link)
        links.new(mapping_vector, env_node.inputs["Vector"])
    if not mix.inputs["Color1"].is_linked:
        links.new(env.outputs["Color"], mix.inputs["Color1"])
    if not ground_mix.inputs["Color1"].is_linked:
        links.new(env_blur.outputs["Color"], ground_mix.inputs["Color1"])
    if not ground_mix.inputs["Color2"].is_linked:
        links.new(env_projected.outputs["Color"], ground_mix.inputs["Color2"])
    for link in list(mix.inputs["Color2"].links):
        links.remove(link)
    links.new(ground_mix.outputs["Color"], mix.inputs["Color2"])
    if not hue_sat.inputs["Color"].is_linked:
        links.new(mix.outputs["Color"], hue_sat.inputs["Color"])
    if not tint_mix.inputs["Color1"].is_linked:
        links.new(hue_sat.outputs["Color"], tint_mix.inputs["Color1"])

    for link in list(bg.inputs["Color"].links):
        links.remove(link)
    links.new(tint_mix.outputs["Color"], bg.inputs["Color"])

    return {
        "nt": nt,
        "env": env,
        "env_blur": env_blur,
        "env_projected": env_projected,
        "ground_mix": ground_mix,
        "mix": mix,
        "tint_mix": tint_mix,
        "hue_sat": hue_sat,
        "bg": bg,
        "mapping": mapping,
        "texcoord": texcoord,
    }


def _ensure_cycles():
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        scene.render.engine = "CYCLES"


def _ensure_preview_sphere(name="HDRI_PreviewSphere"):
    obj = bpy.data.objects.get(name)
    if obj and obj.type == "MESH":
        return obj

    mesh = bpy.data.meshes.new(name + "_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = None
    try:
        import bmesh

        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
        bm.to_mesh(mesh)
    finally:
        if bm:
            bm.free()

    obj.location = (0, 0, 1)

    mat = bpy.data.materials.get(name + "_Mat")
    if mat is None:
        mat = bpy.data.materials.new(name + "_Mat")
        mat.use_nodes = True
        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links

        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.0

        out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out is None:
            out = nodes.new("ShaderNodeOutputMaterial")
        if not bsdf.outputs["BSDF"].is_linked:
            links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return obj


_FAKE_GROUND_OBJ = "HDRI_FakeGround"
_FAKE_GROUND_MAT = "HDRI_FakeGround_Mat"


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 1e-4:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(round(float(sigma) * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return (k / np.sum(k)).astype(np.float32)


def _convolve_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    out = np.empty_like(arr)
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        for i in range(arr.shape[0]):
            out[i] = np.tensordot(kernel, padded[i : i + len(kernel)], axes=([0], [0]))
        return out
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
    for i in range(arr.shape[1]):
        out[:, i] = np.tensordot(kernel, padded[:, i : i + len(kernel)], axes=([0], [0]))
    return out


def _blur_rgba_array(rgba: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 1e-4:
        return rgba
    kernel = _gaussian_kernel1d(float(sigma))
    out = _convolve_axis(rgba.astype(np.float32, copy=True), kernel, axis=0)
    out = _convolve_axis(out, kernel, axis=1)
    return np.clip(out, 0.0, None)


def _blur_image_copy(source_img: bpy.types.Image, blur_amount: float) -> bpy.types.Image:
    if source_img is None:
        return source_img
    amount = float(blur_amount)
    if amount <= 1e-4:
        return source_img
    sigma = max(0.5, amount * 8.0)
    blur_name = f"{source_img.name}__hdri_blur_{amount:.3f}"
    blur_img = bpy.data.images.get(blur_name)
    if blur_img is None:
        blur_img = source_img.copy()
        blur_img.name = blur_name
    width = int(source_img.size[0])
    height = int(source_img.size[1])
    if width <= 0 or height <= 0:
        return source_img
    px = np.array(source_img.pixels[:], dtype=np.float32)
    rgba = px.reshape(height, width, 4)
    blurred = _blur_rgba_array(rgba, sigma)
    if tuple(blur_img.size) != (width, height):
        blur_img.scale(width, height)
    blur_img.pixels[:] = blurred.reshape(-1).tolist()
    blur_img.colorspace_settings.name = source_img.colorspace_settings.name
    return blur_img


def _assign_hdri_images(env_node, env_blur_node, env_projected_node, source_img, blur_amount: float):
    env_node.image = source_img
    blur_img = _blur_image_copy(source_img, blur_amount)
    env_blur_node.image = blur_img
    if env_projected_node is not None:
        env_projected_node.image = source_img


_HDRI_GP = "HDRI GP"
_CUSTOM_GROUND_GROUP_NAME = "HDRI Ground Projection"


def _addon_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _default_ground_projection_blend_path() -> str:
    return os.path.join(_addon_dir(), "templates", "ground_projection.blend")


def _set_fake_ground_visible(visible: bool):
    """Hide legacy mesh plane if an old file still has HDRI_FakeGround."""
    obj = bpy.data.objects.get(_FAKE_GROUND_OBJ)
    if obj is None:
        return
    obj.hide_viewport = not visible
    obj.hide_render = not visible


def _find_node_by_label(nodes, label: str):
    for node in nodes:
        if node.label == label:
            return node
    return None


def _gp_get(nodes, label: str, node_type: str):
    full_label = f"{_HDRI_GP} {label}"
    node = _find_node_by_label(nodes, full_label)
    if node is None:
        node = nodes.new(node_type)
        node.label = full_label
    return node


def _gp_link(links, src, dst):
    for link in list(dst.links):
        links.remove(link)
    links.new(src, dst)


def _remove_broken_ground_group_nodes(nt: bpy.types.NodeTree) -> None:
    for node in list(nt.nodes):
        if node.label.startswith("HDRI Ground Projection "):
            nt.nodes.remove(node)


def _clear_inline_ground_nodes(nt: bpy.types.NodeTree) -> None:
    prefix = f"{_HDRI_GP} "
    for node in list(nt.nodes):
        if node.label.startswith(prefix):
            nt.nodes.remove(node)


def _clear_world_ground_group_node(nt: bpy.types.NodeTree) -> None:
    for node in list(nt.nodes):
        if node.bl_idname == "ShaderNodeGroup" and node.label == _CUSTOM_GROUND_GROUP_NAME:
            nt.nodes.remove(node)


def _ngroup_socket_names(ng: bpy.types.NodeTree, *, in_out: str) -> set[str]:
    names: set[str] = set()
    if hasattr(ng, "interface"):
        for item in ng.interface.items_tree:
            socket_in_out = getattr(item, "in_out", None)
            if socket_in_out == in_out and getattr(item, "name", None):
                names.add(item.name)
        return names
    coll = ng.inputs if in_out == "INPUT" else ng.outputs
    for item in coll:
        names.add(item.name)
    return names


def _ground_group_is_valid(ng: bpy.types.NodeTree | None) -> bool:
    if ng is None:
        return False
    inputs = _ngroup_socket_names(ng, in_out="INPUT")
    outputs = _ngroup_socket_names(ng, in_out="OUTPUT")
    return (
        "Vector" in inputs
        and "Size" in inputs
        and "Horizon" in inputs
        and "Rotation" in inputs
        and "Vector" in outputs
    )


def _load_ground_group_from_blend(filepath: str, group_name: str) -> bpy.types.NodeTree | None:
    if not filepath or not os.path.isfile(filepath):
        return None
    existing = bpy.data.node_groups.get(group_name)
    if existing is not None and _ground_group_is_valid(existing):
        return existing
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        if group_name not in data_from.node_groups:
            return None
        data_to.node_groups = [group_name]
    ng = bpy.data.node_groups.get(group_name)
    if ng is None:
        for candidate in bpy.data.node_groups:
            if candidate.name.startswith(group_name):
                ng = candidate
                break
    return ng if _ground_group_is_valid(ng) else None


def _resolve_custom_ground_group() -> bpy.types.NodeTree | None:
    prefs = _addon_prefs()
    if prefs is None or not prefs.use_custom_ground_projection:
        return None
    path = (prefs.ground_projection_blend or "").strip() or _default_ground_projection_blend_path()
    ng = bpy.data.node_groups.get(_CUSTOM_GROUND_GROUP_NAME)
    if _ground_group_is_valid(ng):
        return ng
    ng = _load_ground_group_from_blend(path, _CUSTOM_GROUND_GROUP_NAME)
    if ng is not None:
        return ng
    return _ensure_default_ground_projection_template()


def _build_ground_projection_node_group_asset() -> bpy.types.NodeTree:
    """Create the bundled HDRI Ground Projection node group (Easy HDRI topology)."""
    name = _CUSTOM_GROUND_GROUP_NAME
    existing = bpy.data.node_groups.get(name)
    if existing is not None and _ground_group_is_valid(existing):
        return existing
    if existing is not None:
        bpy.data.node_groups.remove(existing)

    ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
    _node_group_add_socket(ng, "Vector", in_out="INPUT", socket_type="NodeSocketVector")
    _node_group_add_socket(ng, "Size", in_out="INPUT", socket_type="NodeSocketVector")
    _node_group_add_socket(ng, "Horizon", in_out="INPUT", socket_type="NodeSocketVector")
    _node_group_add_socket(ng, "Rotation", in_out="INPUT", socket_type="NodeSocketVectorEuler")
    _node_group_add_socket(ng, "Vector", in_out="OUTPUT", socket_type="NodeSocketVector")

    nodes = ng.nodes
    links = ng.links
    gi, go = _group_io_nodes(ng)

    vec_in = gi.outputs.get("Vector") or gi.outputs[0]
    size_in = gi.outputs.get("Size") or gi.outputs[1]
    horizon_in = gi.outputs.get("Horizon") or gi.outputs[2]
    rot_in = gi.outputs.get("Rotation") or gi.outputs[3]
    vec_out = go.inputs.get("Vector") or go.inputs[0]

    vtransform = nodes.new("ShaderNodeVectorTransform")
    vtransform.vector_type = "POINT"
    vtransform.convert_from = "CAMERA"
    vtransform.convert_to = "WORLD"
    vtransform.location = (-1166, 60)

    vrotate = nodes.new("ShaderNodeVectorRotate")
    vrotate.rotation_type = "EULER_XYZ"
    vrotate.invert = True
    vrotate.inputs["Center"].default_value = (0.0, 0.0, 0.0)
    vrotate.location = (-936, 60)

    sep_rot = nodes.new("ShaderNodeSeparateXYZ")
    sep_rot.location = (-690, 60)

    dot_down = nodes.new("ShaderNodeVectorMath")
    dot_down.operation = "DOT_PRODUCT"
    dot_down.inputs[1].default_value = (0.0, 0.0, -1.0)
    dot_down.location = (-872, -140)

    ratio = nodes.new("ShaderNodeMath")
    ratio.operation = "DIVIDE"
    ratio.use_clamp = False
    ratio.location = (-446, 60)

    scale_vec = nodes.new("ShaderNodeVectorMath")
    scale_vec.operation = "SCALE"
    scale_vec.location = (-202, 60)

    add_vec = nodes.new("ShaderNodeVectorMath")
    add_vec.operation = "ADD"
    add_vec.location = (42, 60)

    sub_size = nodes.new("ShaderNodeVectorMath")
    sub_size.operation = "SUBTRACT"
    sub_size.location = (287, 60)

    normalize = nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    normalize.location = (531, 60)

    sep_orig = nodes.new("ShaderNodeSeparateXYZ")
    sep_orig.location = (286, -140)

    sky_test = nodes.new("ShaderNodeMath")
    sky_test.operation = "GREATER_THAN"
    sky_test.inputs[1].default_value = 0.0
    sky_test.location = (531, -140)

    try:
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "VECTOR"
        mix.clamp_factor = False
        mix.location = (776, 60)
        mix_a = mix.inputs.get("A") or mix.inputs[6]
        mix_b = mix.inputs.get("B") or mix.inputs[7]
        mix_fac = mix.inputs.get("Factor") or mix.inputs[0]
        mix_out = mix.outputs.get("Vector") or mix.outputs[0]
    except Exception:
        mix = nodes.new("ShaderNodeMixRGB")
        mix.location = (776, 60)
        mix_a = mix.inputs["Color1"]
        mix_b = mix.inputs["Color2"]
        mix_fac = mix.inputs["Fac"]
        mix_out = mix.outputs["Color"]

    sub_horizon = nodes.new("ShaderNodeVectorMath")
    sub_horizon.operation = "SUBTRACT"
    sub_horizon.location = (1020, 60)

    _gp_link(links, vec_in, vtransform.inputs[0])
    _gp_link(links, vec_in, dot_down.inputs[0])
    _gp_link(links, vec_in, scale_vec.inputs[0])
    _gp_link(links, vec_in, sep_orig.inputs["Vector"])
    _gp_link(links, vec_in, mix_b)
    _gp_link(links, rot_in, vrotate.inputs["Rotation"])
    _gp_link(links, vtransform.outputs["Vector"], vrotate.inputs["Vector"])
    _gp_link(links, vrotate.outputs["Vector"], sep_rot.inputs["Vector"])
    _gp_link(links, sep_rot.outputs["Z"], ratio.inputs[0])
    _gp_link(links, dot_down.outputs["Value"], ratio.inputs[1])
    _gp_link(links, ratio.outputs["Value"], scale_vec.inputs["Scale"])
    _gp_link(links, scale_vec.outputs["Vector"], add_vec.inputs[0])
    _gp_link(links, vrotate.outputs["Vector"], add_vec.inputs[1])
    _gp_link(links, add_vec.outputs["Vector"], sub_size.inputs[0])
    _gp_link(links, size_in, sub_size.inputs[1])
    _gp_link(links, sub_size.outputs["Vector"], normalize.inputs[0])
    _gp_link(links, normalize.outputs["Vector"], mix_a)
    _gp_link(links, sep_orig.outputs["Z"], sky_test.inputs[0])
    _gp_link(links, sky_test.outputs["Value"], mix_fac)
    _gp_link(links, mix_out, sub_horizon.inputs[0])
    _gp_link(links, horizon_in, sub_horizon.inputs[1])
    _gp_link(links, sub_horizon.outputs["Vector"], vec_out)

    return ng


def _ensure_default_ground_projection_template() -> bpy.types.NodeTree | None:
    """Load templates/ground_projection.blend or create the default group and save it."""
    path = _default_ground_projection_blend_path()
    ng = _load_ground_group_from_blend(path, _CUSTOM_GROUND_GROUP_NAME)
    if ng is not None:
        return ng
    ng = _build_ground_projection_node_group_asset()
    if not _ground_group_is_valid(ng):
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bpy.data.libraries.write(path, {ng}, fake_user=True)
    except Exception:
        pass
    return ng


def _ensure_custom_ground_group_node(nt: bpy.types.NodeTree, ng: bpy.types.NodeTree) -> bpy.types.Node:
    node = _find_node_by_label(nt.nodes, _CUSTOM_GROUND_GROUP_NAME)
    if node is None:
        node = nt.nodes.new("ShaderNodeGroup")
        node.label = _CUSTOM_GROUND_GROUP_NAME
        node.location = (-710, -560)
    node.node_tree = ng
    node.name = _CUSTOM_GROUND_GROUP_NAME
    return node


def _wire_custom_ground_group(
    nt: bpy.types.NodeTree,
    gp_node: bpy.types.Node,
    *,
    mapping_node: bpy.types.Node,
    settings,
):
    links = nt.links
    vec = mapping_node.outputs["Vector"]
    _gp_link(links, vec, gp_node.inputs["Vector"])
    gp_node.inputs["Size"].default_value = _ground_projection_size(settings)
    gp_node.inputs["Horizon"].default_value = _ground_projection_horizon(settings)
    gp_node.inputs["Rotation"].default_value = _ground_projection_rotation(mapping_node)
    return gp_node.outputs.get("Vector") or gp_node.outputs[0]


def _ensure_gp_mix_node(nodes):
    label = f"{_HDRI_GP} Mix"
    existing = _find_node_by_label(nodes, label)
    if existing is not None and existing.bl_idname not in ("ShaderNodeMix", "ShaderNodeMixRGB"):
        nodes.remove(existing)
        existing = None
    if existing is not None and existing.bl_idname == "ShaderNodeMixRGB":
        return existing, existing.inputs["Color1"], existing.inputs["Color2"], existing.inputs["Fac"], existing.outputs["Color"]
    if existing is not None:
        try:
            existing.data_type = "VECTOR"
            existing.clamp_factor = False
            return (
                existing,
                existing.inputs.get("A") or existing.inputs[6],
                existing.inputs.get("B") or existing.inputs[7],
                existing.inputs.get("Factor") or existing.inputs[0],
                existing.outputs.get("Result") or existing.outputs.get("Vector") or existing.outputs[0],
            )
        except Exception:
            nodes.remove(existing)
    try:
        mix = nodes.new("ShaderNodeMix")
        mix.label = label
        mix.data_type = "VECTOR"
        mix.clamp_factor = False
        return (
            mix,
            mix.inputs.get("A") or mix.inputs[6],
            mix.inputs.get("B") or mix.inputs[7],
            mix.inputs.get("Factor") or mix.inputs[0],
            mix.outputs.get("Vector") or mix.outputs[0],
        )
    except Exception:
        mix = nodes.new("ShaderNodeMixRGB")
        mix.label = label
        return mix, mix.inputs["Color1"], mix.inputs["Color2"], mix.inputs["Fac"], mix.outputs["Color"]


def _ensure_ground_projection_output(nt: bpy.types.NodeTree, *, mapping_node, settings):
    """Return ground projection vector output — custom saved group or built-in inline chain."""
    custom_ng = _resolve_custom_ground_group()
    if custom_ng is not None:
        _clear_inline_ground_nodes(nt)
        gp_node = _ensure_custom_ground_group_node(nt, custom_ng)
        output = _wire_custom_ground_group(nt, gp_node, mapping_node=mapping_node, settings=settings)
        return output
    _clear_world_ground_group_node(nt)
    block = _ensure_ground_projection_chain(nt)
    _wire_ground_projection_externals(nt, block, mapping_node=mapping_node, settings=settings)
    return block["output"]


def _ensure_ground_projection_chain(nt: bpy.types.NodeTree) -> dict:
    """Full Easy HDRI ground projection node chain, inline in the world tree."""
    _remove_broken_ground_group_nodes(nt)
    nodes = nt.nodes
    links = nt.links

    vtransform = _gp_get(nodes, "Transform", "ShaderNodeVectorTransform")
    vtransform.vector_type = "POINT"
    vtransform.convert_from = "CAMERA"
    vtransform.convert_to = "WORLD"
    vtransform.location = (-1166, -560)

    vrotate = _gp_get(nodes, "Rotate", "ShaderNodeVectorRotate")
    vrotate.rotation_type = "EULER_XYZ"
    vrotate.invert = True
    vrotate.inputs["Center"].default_value = (0.0, 0.0, 0.0)
    vrotate.location = (-936, -560)

    sep_rot = _gp_get(nodes, "Sep Rot", "ShaderNodeSeparateXYZ")
    sep_rot.location = (-690, -560)

    dot_down = _gp_get(nodes, "Dot Down", "ShaderNodeVectorMath")
    dot_down.operation = "DOT_PRODUCT"
    dot_down.inputs[1].default_value = (0.0, 0.0, -1.0)
    dot_down.location = (-872, -760)

    ratio = _gp_get(nodes, "Ratio", "ShaderNodeMath")
    ratio.operation = "DIVIDE"
    ratio.use_clamp = False
    ratio.location = (-446, -560)

    scale_vec = _gp_get(nodes, "Scale", "ShaderNodeVectorMath")
    scale_vec.operation = "SCALE"
    scale_vec.location = (-202, -560)

    add_vec = _gp_get(nodes, "Add", "ShaderNodeVectorMath")
    add_vec.operation = "ADD"
    add_vec.location = (42, -560)

    sub_size = _gp_get(nodes, "Sub Size", "ShaderNodeVectorMath")
    sub_size.operation = "SUBTRACT"
    sub_size.location = (287, -560)

    normalize = _gp_get(nodes, "Normalize", "ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    normalize.location = (531, -560)

    sep_orig = _gp_get(nodes, "Sep Orig", "ShaderNodeSeparateXYZ")
    sep_orig.location = (286, -760)

    sky_test = _gp_get(nodes, "Sky Test", "ShaderNodeMath")
    sky_test.operation = "GREATER_THAN"
    sky_test.inputs[1].default_value = 0.0
    sky_test.location = (531, -760)

    mix, mix_a, mix_b, mix_fac, mix_out = _ensure_gp_mix_node(nodes)
    mix.location = (776, -560)

    sub_horizon = _gp_get(nodes, "Sub Horizon", "ShaderNodeVectorMath")
    sub_horizon.operation = "SUBTRACT"
    sub_horizon.location = (1020, -560)

    # Internal wiring (fixed topology — matches Easy HDRI Ground Projection group)
    _gp_link(links, vtransform.outputs["Vector"], vrotate.inputs["Vector"])
    _gp_link(links, vrotate.outputs["Vector"], sep_rot.inputs["Vector"])
    _gp_link(links, sep_rot.outputs["Z"], ratio.inputs[0])
    _gp_link(links, dot_down.outputs["Value"], ratio.inputs[1])
    _gp_link(links, ratio.outputs["Value"], scale_vec.inputs["Scale"])
    _gp_link(links, scale_vec.outputs["Vector"], add_vec.inputs[0])
    _gp_link(links, vrotate.outputs["Vector"], add_vec.inputs[1])
    _gp_link(links, add_vec.outputs["Vector"], sub_size.inputs[0])
    _gp_link(links, sub_size.outputs["Vector"], normalize.inputs[0])
    _gp_link(links, normalize.outputs["Vector"], mix_a)
    _gp_link(links, sep_orig.outputs["Z"], sky_test.inputs[0])
    _gp_link(links, sky_test.outputs["Value"], mix_fac)
    _gp_link(links, mix_out, sub_horizon.inputs[0])

    return {
        "vtransform": vtransform,
        "vrotate": vrotate,
        "dot_down": dot_down,
        "scale_vec": scale_vec,
        "sep_orig": sep_orig,
        "mix_b": mix_b,
        "sub_size": sub_size,
        "sub_horizon": sub_horizon,
        "output": sub_horizon.outputs["Vector"],
    }


def _ground_projection_size(settings) -> tuple[float, float, float]:
    size_z = max(0.5, float(settings.fake_ground_lift) * 20.0)
    return (0.0, 0.0, size_z)


def _ground_projection_horizon(settings) -> tuple[float, float, float]:
    return (0.0, 0.0, float(settings.fake_ground_z_offset))


def _ground_projection_rotation(mapping_node: bpy.types.Node) -> tuple[float, float, float]:
    rot = mapping_node.inputs["Rotation"].default_value
    return (float(rot[0]), float(rot[1]), float(rot[2]))


def _wire_ground_projection_externals(
    nt: bpy.types.NodeTree,
    block: dict,
    *,
    mapping_node: bpy.types.Node,
    settings,
) -> None:
    """Connect mapping vector + Size / Horizon / Rotation inputs to the chain."""
    links = nt.links
    vec = mapping_node.outputs["Vector"]
    _gp_link(links, vec, block["vtransform"].inputs[0])
    _gp_link(links, vec, block["dot_down"].inputs[0])
    _gp_link(links, vec, block["scale_vec"].inputs[0])
    _gp_link(links, vec, block["sep_orig"].inputs["Vector"])
    _gp_link(links, vec, block["mix_b"])
    block["sub_size"].inputs[1].default_value = _ground_projection_size(settings)
    block["sub_horizon"].inputs[1].default_value = _ground_projection_horizon(settings)
    block["vrotate"].inputs["Rotation"].default_value = _ground_projection_rotation(mapping_node)


def _route_ground_projected_vector(
    nt: bpy.types.NodeTree,
    *,
    env_projected_node: bpy.types.Node,
    mapping_node: bpy.types.Node,
    ground_out,
    enabled: bool,
) -> None:
    links = nt.links
    for link in list(env_projected_node.inputs["Vector"].links):
        links.remove(link)

    if enabled and ground_out is not None:
        links.new(ground_out, env_projected_node.inputs["Vector"])
    else:
        links.new(mapping_node.outputs["Vector"], env_projected_node.inputs["Vector"])


def _apply_world_ground_projection(world: bpy.types.World, settings, *, enabled: bool) -> None:
    nodes = _ensure_world_nodes(world)
    nt = nodes["nt"]
    mapping = nodes["mapping"]
    ground_mix = nodes["ground_mix"]
    _set_fake_ground_visible(False)

    ground_mix.inputs["Fac"].default_value = 1.0 if enabled else 0.0

    ground_out = _ensure_ground_projection_output(nt, mapping_node=mapping, settings=settings)
    _route_ground_projected_vector(
        nt,
        env_projected_node=nodes["env_projected"],
        mapping_node=mapping,
        ground_out=ground_out if enabled else None,
        enabled=enabled,
    )


def _apply_look_controls_to_nodes(
    settings,
    mapping_node: bpy.types.Node,
    mix_node: bpy.types.Node,
    tint_mix_node: bpy.types.Node,
    hue_sat_node: bpy.types.Node,
    bg_node: bpy.types.Node,
):
    mapping_node.inputs["Rotation"].default_value[0] = settings.pitch_degrees * (3.141592653589793 / 180.0)
    mapping_node.inputs["Rotation"].default_value[1] = settings.roll_degrees * (3.141592653589793 / 180.0)
    mapping_node.inputs["Rotation"].default_value[2] = settings.yaw_degrees * (3.141592653589793 / 180.0)
    hue_sat_node.inputs["Hue"].default_value = 0.5 + settings.hue_shift
    hue_sat_node.inputs["Saturation"].default_value = settings.saturation
    mix_node.inputs["Fac"].default_value = settings.blur_amount
    bg_node.inputs["Strength"].default_value = settings.exposure * settings.post_exposure

    tint_mix_node.inputs["Fac"].default_value = settings.tint_strength
    tint_mix_node.inputs["Color2"].default_value = (
        settings.tint_color[0],
        settings.tint_color[1],
        settings.tint_color[2],
        1.0,
    )


def _sync_world_and_ground_look(context, settings):
    scene = getattr(context, "scene", None)
    if scene is None or scene.world is None:
        return
    nodes = _ensure_world_nodes(scene.world)
    _apply_look_controls_to_nodes(
        settings,
        nodes["mapping"],
        nodes["mix"],
        nodes["tint_mix"],
        nodes["hue_sat"],
        nodes["bg"],
    )
    _refresh_hdri_blur_images(settings, nodes["env"], nodes["env_blur"], nodes["env_projected"])

    img = nodes["env"].image
    _apply_world_ground_projection(scene.world, settings, enabled=bool(settings.fake_ground and img is not None))


def _refresh_hdri_blur_images(settings, env_node, env_blur_node, env_projected_node=None):
    src = env_node.image
    if src is None:
        env_blur_node.image = None
        if env_projected_node is not None:
            env_projected_node.image = None
        return
    env_blur_node.image = _blur_image_copy(src, settings.blur_amount)
    if env_projected_node is not None:
        env_projected_node.image = src


def _update_look_controls(self, context):
    if context is None:
        return
    try:
        _sync_world_and_ground_look(context, self)
    except Exception:
        # Property updates should not break UI interaction if nodes are not ready yet.
        pass


def _http_post_json(url: str, payload: dict, headers: dict, timeout_s: int):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        if v:
            req.add_header(k, v)

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8"))


def _http_get_json(url: str, headers: dict, timeout_s: int):
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        if v:
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_bytes(url: str, headers: dict, timeout_s: int):
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        if v:
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _safe_get_account(base_url: str, headers: dict, timeout_s: int) -> dict | None:
    try:
        data = _http_get_json(f"{base_url.rstrip('/')}/v1/account", headers=headers, timeout_s=timeout_s)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _running_ascii_spinner() -> str:
    frames = ("|", "/", "-", "\\")
    idx = int(time.monotonic() * 8.0) % len(frames)
    return f"[{frames[idx]}]"


def _format_duration_compact(seconds: float) -> str:
    if seconds <= 0 or math.isnan(seconds) or math.isinf(seconds):
        return "0s"
    secs = max(1, int(round(seconds)))
    if secs < 60:
        return f"{secs}s"
    m = secs // 60
    sec = secs % 60
    return f"{m}m {sec}s"


def _default_expected_remote_job_seconds(quality_mode: str) -> float:
    """Heuristic ETA when no prior completion time exists (remote panorama + HDR)."""
    q = str(quality_mode or "balanced").strip().lower()
    if q == "fast":
        return 90.0
    if q == "high":
        return 540.0
    return 240.0


def _expected_remote_job_seconds(settings) -> float:
    prev = float(getattr(settings, "last_completed_job_wall_s", 0.0) or 0.0)
    if prev >= 15.0:
        return prev
    return _default_expected_remote_job_seconds(getattr(settings, "quality_mode", "balanced"))


def _list_hdri_files(folder_path: str) -> list[str]:
    if not folder_path or not os.path.isdir(folder_path):
        return []
    try:
        names = [
            name
            for name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, name))
            and name.lower().endswith((".hdr", ".exr"))
        ]
    except Exception:
        return []
    names.sort(key=lambda n: os.path.getmtime(os.path.join(folder_path, n)), reverse=True)
    return names


def _clear_library_previews() -> None:
    global _LIB_PREVIEW_COLLECTION, _LIB_PREVIEW_FOLDER, _LIB_PREVIEW_SIGNATURE, _LIB_PREVIEW_ITEMS
    if _LIB_PREVIEW_COLLECTION is not None:
        try:
            bpy.utils.previews.remove(_LIB_PREVIEW_COLLECTION)
        except Exception:
            pass
    _LIB_PREVIEW_COLLECTION = None
    _LIB_PREVIEW_FOLDER = ""
    _LIB_PREVIEW_SIGNATURE = ()
    _LIB_PREVIEW_ITEMS = []


def _library_signature(folder_path: str, names: list[str]) -> tuple[tuple[str, float], ...]:
    out: list[tuple[str, float]] = []
    for name in names:
        abs_path = os.path.join(folder_path, name)
        try:
            mtime = float(os.path.getmtime(abs_path))
        except Exception:
            mtime = 0.0
        out.append((name, mtime))
    return tuple(out)


def _refresh_library_previews(folder_path: str) -> None:
    global _LIB_PREVIEW_COLLECTION, _LIB_PREVIEW_FOLDER, _LIB_PREVIEW_SIGNATURE, _LIB_PREVIEW_ITEMS
    names = _list_hdri_files(folder_path)
    signature = _library_signature(folder_path, names)
    if (
        _LIB_PREVIEW_COLLECTION is not None
        and _LIB_PREVIEW_FOLDER == folder_path
        and _LIB_PREVIEW_SIGNATURE == signature
    ):
        return

    _clear_library_previews()
    _LIB_PREVIEW_FOLDER = folder_path
    _LIB_PREVIEW_SIGNATURE = signature
    if not names:
        _LIB_PREVIEW_ITEMS = [("__none__", "No panoramas", "No .hdr/.exr files in library folder", 0, 0)]
        return

    _LIB_PREVIEW_COLLECTION = bpy.utils.previews.new()
    items: list[tuple[str, str, str, int, int]] = []
    for idx, name in enumerate(names):
        abs_path = os.path.join(folder_path, name)
        icon_id = 0
        try:
            thumb = _LIB_PREVIEW_COLLECTION.load(name, abs_path, "IMAGE")
            icon_id = int(thumb.icon_id)
        except Exception:
            icon_id = 0
        items.append((name, name, abs_path, icon_id, idx))
    _LIB_PREVIEW_ITEMS = items


def _library_gallery_items(self, context):
    folder_raw = (getattr(self, "library_folder", "") or "").strip()
    folder = bpy.path.abspath(folder_raw) if folder_raw else ""
    if not folder or not os.path.isdir(folder):
        _clear_library_previews()
        return [("__none__", "No gallery", "Choose a valid library folder", 0, 0)]
    _refresh_library_previews(folder)
    return _LIB_PREVIEW_ITEMS or [("__none__", "No panoramas", "No .hdr/.exr files in library folder", 0, 0)]


def _update_library_gallery_item(self, context):
    if context is None:
        return
    selected = str(getattr(self, "library_gallery_item", "") or "").strip()
    if not selected or selected == "__none__":
        return
    folder_raw = (getattr(self, "library_folder", "") or "").strip()
    folder = bpy.path.abspath(folder_raw) if folder_raw else ""
    if not folder or not os.path.isdir(folder):
        return
    path = os.path.join(folder, selected)
    if not os.path.isfile(path):
        return
    ok, _message = _apply_hdri_image_path(context, self, path)
    if ok:
        self.last_library_path = path


def _build_library_output_path(settings, suffix: str) -> str | None:
    folder_raw = (getattr(settings, "library_folder", "") or "").strip()
    if not folder_raw:
        return None
    folder = os.path.normpath(bpy.path.abspath(folder_raw))
    if not folder:
        return None
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception:
        return None

    src_path = bpy.path.abspath(getattr(settings, "input_image_path", "") or "")
    src_stem = os.path.splitext(os.path.basename(src_path))[0].strip() if src_path else ""
    if not src_stem:
        src_stem = "hdri"
    src_stem = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in src_stem).strip("_") or "hdri"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"{src_stem}_{timestamp}"
    file_name = f"{base_name}{suffix}"
    candidate = os.path.join(folder, file_name)
    if not os.path.exists(candidate):
        return candidate
    for idx in range(1, 1000):
        candidate = os.path.join(folder, f"{base_name}_{idx:03d}{suffix}")
        if not os.path.exists(candidate):
            return candidate
    return None


def _write_hdri_output_file(settings, file_bytes: bytes, suffix: str) -> tuple[str | None, str | None, bool]:
    """
    Writes HDRI bytes. If Library Folder is set, saves there when possible; on write errors
    falls back to a temp file so Apply can still succeed. Returns (path, warning_or_error, saved_to_library).
    """
    folder_raw = (getattr(settings, "library_folder", "") or "").strip()
    folder_set = bool(folder_raw)

    if folder_set:
        lib_path = _build_library_output_path(settings, suffix)
        if lib_path is None:
            try:
                fd, tmp_path = tempfile.mkstemp(prefix="hdri_api_", suffix=suffix)
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    f.write(file_bytes)
                return (
                    tmp_path,
                    (
                        "Library Folder path is invalid or could not be created; "
                        f"saved HDRI to temp only. Path was: {bpy.path.abspath(folder_raw)!r}"
                    ),
                    False,
                )
            except Exception as e2:
                return None, f"Library Folder unusable and temp write failed: {e2}", False
        try:
            with open(lib_path, "wb") as f:
                f.write(file_bytes)
            return lib_path, None, True
        except Exception as e:
            try:
                fd, tmp_path = tempfile.mkstemp(prefix="hdri_api_", suffix=suffix)
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    f.write(file_bytes)
                return tmp_path, f"Library Folder save failed ({e}); saved to temp instead.", False
            except Exception as e2:
                return None, f"Library save failed ({e}) and temp save failed ({e2})", False

    try:
        fd, tmp_path = tempfile.mkstemp(prefix="hdri_api_", suffix=suffix)
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        return tmp_path, None, False
    except Exception as e:
        return None, f"Failed to write temp HDRI file: {e}", False


def _apply_hdri_image_path(context, settings, image_path: str) -> tuple[bool, str]:
    try:
        _ensure_cycles()
        world = context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            context.scene.world = world

        nodes = _ensure_world_nodes(world)
        env_node = nodes["env"]
        env_blur_node = nodes["env_blur"]
        env_projected_node = nodes["env_projected"]
        mix_node = nodes["mix"]
        tint_mix_node = nodes["tint_mix"]
        hue_sat_node = nodes["hue_sat"]
        bg_node = nodes["bg"]
        mapping_node = nodes["mapping"]

        img = bpy.data.images.load(image_path, check_existing=True)
        _set_env_image_colorspace(img)
        _assign_hdri_images(env_node, env_blur_node, env_projected_node, img, settings.blur_amount)

        _apply_look_controls_to_nodes(
            settings,
            mapping_node,
            mix_node,
            tint_mix_node,
            hue_sat_node,
            bg_node,
        )

        if settings.add_preview_sphere:
            _ensure_preview_sphere()

        _apply_world_ground_projection(
            world,
            settings,
            enabled=bool(settings.fake_ground and img is not None),
        )
    except Exception as e:
        return False, f"Failed to apply HDRI: {e}"
    return True, "HDRI applied to World."


_PLACEMENT_PREVIEW_IMAGE_NAME = "HDRI_Placement_Preview"


def _clampf(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


# Equirectangular u positions for world-facing labels (matches _render_placement_preview_rgb).
_PLACEMENT_COMPASS_LABELS = (
    ("FRONT", 0.50),
    ("LEFT", 0.25),
    ("RIGHT", 0.75),
    ("BACK", 0.04),
    ("BACK", 0.96),
)


def _placement_label_font_size(canvas_w: int) -> int:
    return max(12, min(20, int(canvas_w / 36)))


def _draw_text_label(
    text: str,
    cx: float,
    cy: float,
    *,
    font_size: int,
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.95),
    shadow: bool = True,
) -> None:
    font_id = 0
    blf.size(font_id, font_size)
    width, _height = blf.dimensions(font_id, text)
    blf.position(font_id, cx - width * 0.5, cy, 0)
    if shadow:
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 3, 0, 0, 0, 0.9)
    blf.color(font_id, *color)
    blf.draw(font_id, text)
    if shadow:
        blf.disable(font_id, blf.SHADOW)


def _draw_placement_compass_labels(x: int, y: int, w: int, h: int) -> None:
    """Draw FRONT/LEFT/RIGHT/BACK and SKY/GROUND on the 2:1 ERP preview."""
    font_size = _placement_label_font_size(w)
    cx = x + w * 0.5
    horizon_y = y + h * 0.5

    for text, u in _PLACEMENT_COMPASS_LABELS:
        label_x = x + w * u
        accent = (1.0, 0.92, 0.45, 0.98) if text == "FRONT" else (1.0, 1.0, 1.0, 0.9)
        _draw_text_label(text, label_x, horizon_y - font_size * 0.35, font_size=font_size, color=accent)

    _draw_text_label(
        "SKY",
        cx,
        y + h * 0.88,
        font_size=font_size,
        color=(0.72, 0.88, 1.0, 0.95),
    )
    _draw_text_label(
        "GROUND",
        cx,
        y + h * 0.06,
        font_size=font_size,
        color=(0.82, 0.72, 0.55, 0.95),
    )

    hint = "Drag photo · wheel = scale · R = rotate drag"
    hint_size = max(10, font_size - 3)
    _draw_text_label(hint, cx, y + h + 18, font_size=hint_size, color=(0.85, 0.85, 0.85, 0.85))


def _coverage_to_hfov_deg(coverage: float) -> float:
    # Map UI coverage to a more camera-like rectilinear hFOV:
    # 0.15 -> 35deg, 0.85 -> 95deg.
    cov = _clampf(float(coverage), 0.15, 0.85)
    t = (cov - 0.15) / 0.70
    return 35.0 + t * 60.0


def _sample_rgb_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]

    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]

    c0 = c00 * (1.0 - fx) + c10 * fx
    c1 = c01 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fy) + c1 * fy


def _placement_source_rgb_from_path(path: str, max_edge: int = 384) -> np.ndarray | None:
    try:
        img = bpy.data.images.load(path, check_existing=True)
        w = int(img.size[0])
        h = int(img.size[1])
        if w <= 0 or h <= 0:
            return None
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        rgba = px.reshape((h, w, 4))
        rgb = np.clip(rgba[:, :, :3], 0.0, 1.0)
        long_edge = max(w, h)
        if long_edge > max_edge:
            scale = max_edge / float(long_edge)
            out_w = max(1, int(round(w * scale)))
            out_h = max(1, int(round(h * scale)))
            x_idx = np.linspace(0, w - 1, out_w).astype(np.int32)
            y_idx = np.linspace(0, h - 1, out_h).astype(np.int32)
            rgb = rgb[np.ix_(y_idx, x_idx)]
        return np.ascontiguousarray(rgb.astype(np.float32))
    except Exception:
        return None


def _render_placement_preview_rgb(
    source_rgb: np.ndarray,
    *,
    yaw_deg: float,
    pitch_deg: float,
    rotation_deg: float,
    coverage: float,
    h_fov_deg: float | None = None,
    out_w: int = 512,
    out_h: int = 256,
) -> np.ndarray:
    out_w = max(64, int(out_w))
    out_h = max(32, int(out_h))
    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        return np.zeros((out_h, out_w, 3), dtype=np.float32)

    src_h = max(1, int(source_rgb.shape[0]))
    src_w = max(1, int(source_rgb.shape[1]))
    source_aspect = float(src_w) / float(src_h)

    hfov_deg = (
        float(h_fov_deg)
        if (h_fov_deg is not None and float(h_fov_deg) > 0.0)
        else _coverage_to_hfov_deg(coverage)
    )
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) / max(source_aspect, 1e-6))
    rot = math.radians(float(rotation_deg))

    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    lon = (xs * 2.0 - 1.0) * math.pi
    lat = (0.5 - ys) * math.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    dirs = np.stack(
        [
            np.cos(lat_grid) * np.sin(lon_grid),
            np.sin(lat_grid),
            np.cos(lat_grid) * np.cos(lon_grid),
        ],
        axis=-1,
    ).astype(np.float32)

    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    cp = math.cos(pitch)
    forward = np.array([cp * math.sin(yaw), math.sin(pitch), cp * math.cos(yaw)], dtype=np.float32)
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-8)

    x_cam = np.tensordot(dirs, right, axes=([-1], [0]))
    y_cam = np.tensordot(dirs, up, axes=([-1], [0]))
    z_cam = np.tensordot(dirs, forward, axes=([-1], [0]))

    if abs(rot) > 1e-8:
        cr = math.cos(rot)
        sr = math.sin(rot)
        x_rot = x_cam * cr + y_cam * sr
        y_rot = -x_cam * sr + y_cam * cr
        x_cam = x_rot
        y_cam = y_rot

    tan_h = math.tan(hfov * 0.5)
    tan_v = math.tan(vfov * 0.5)
    eps = 1e-6
    z_safe = np.where(z_cam > eps, z_cam, 1.0)
    u = x_cam / (z_safe * tan_h)
    v = y_cam / (z_safe * tan_v)

    visible = (z_cam > eps) & (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
    sx = (u + 1.0) * 0.5 * (src_w - 1)
    sy = (1.0 - (v + 1.0) * 0.5) * (src_h - 1)

    out = np.empty((out_h, out_w, 3), dtype=np.float32)
    out[:, :, :] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if np.any(visible):
        sampled = _sample_rgb_bilinear(source_rgb, sx, sy)
        out[visible] = sampled[visible]
    return np.clip(out, 0.0, 1.0)


def _ensure_placement_preview_image(width: int, height: int) -> bpy.types.Image:
    img = bpy.data.images.get(_PLACEMENT_PREVIEW_IMAGE_NAME)
    if img is None:
        img = bpy.data.images.new(_PLACEMENT_PREVIEW_IMAGE_NAME, width=width, height=height, alpha=True, float_buffer=False)
    elif int(img.size[0]) != width or int(img.size[1]) != height:
        img.scale(width, height)
    return img


def _update_preview_image_pixels(img: bpy.types.Image, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    img.pixels.foreach_set(rgba.reshape(-1))
    img.update()


def _refresh_placement_preview(settings, context=None, source_rgb: np.ndarray | None = None) -> None:
    if context is None:
        context = bpy.context
    if context is None:
        return

    if source_rgb is None:
        src_path = bpy.path.abspath(getattr(settings, "input_image_path", "") or "")
        if not src_path or not os.path.exists(src_path):
            return
        source_rgb = _placement_source_rgb_from_path(src_path)
    if source_rgb is None:
        return

    rgb = _render_placement_preview_rgb(
        source_rgb,
        yaw_deg=float(getattr(settings, "placement_yaw_deg", 0.0)),
        pitch_deg=float(getattr(settings, "placement_pitch_deg", 0.0)),
        rotation_deg=float(getattr(settings, "placement_rotation_deg", 0.0)),
        coverage=float(getattr(settings, "placement_coverage", 0.6)),
        h_fov_deg=float(getattr(settings, "placement_hfov_deg", 0.0)),
    )
    img = _ensure_placement_preview_image(int(rgb.shape[1]), int(rgb.shape[0]))
    _update_preview_image_pixels(img, rgb)


def _update_placement_controls(self, context):
    try:
        ref_cov = float(getattr(self, "reference_coverage", 0.6))
        place_cov = float(getattr(self, "placement_coverage", ref_cov))
        if abs(ref_cov - place_cov) > 1e-6:
            self.reference_coverage = place_cov
    except Exception:
        pass
    _refresh_placement_preview(self, context)


class HDRI_OT_save_ground_projection(Operator):
    bl_idname = "hdri.save_ground_projection"
    bl_label = "Save Ground Template"
    bl_description = (
        "Save the node group 'HDRI Ground Projection' from this file to the addon templates folder"
    )

    def execute(self, context):
        ng = bpy.data.node_groups.get(_CUSTOM_GROUND_GROUP_NAME)
        if ng is None:
            self.report(
                {"ERROR"},
                f'Create a node group named "{_CUSTOM_GROUND_GROUP_NAME}" first (see addon help).',
            )
            return {"CANCELLED"}
        if not _ground_group_is_valid(ng):
            self.report(
                {"ERROR"},
                "Group needs inputs Vector, Size, Horizon, Rotation and output Vector.",
            )
            return {"CANCELLED"}
        path = _default_ground_projection_blend_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bpy.data.libraries.write(path, {ng}, fake_user=True)
        prefs = _addon_prefs()
        prefs.ground_projection_blend = path
        prefs.use_custom_ground_projection = True
        self.report({"INFO"}, f"Saved ground template to {path}")
        return {"FINISHED"}


class HDRI_OT_reload_ground_projection(Operator):
    bl_idname = "hdri.reload_ground_projection"
    bl_label = "Reload Ground Template"
    bl_description = "Reload the ground projection node group from the template blend file"

    def execute(self, context):
        prefs = _addon_prefs()
        path = (prefs.ground_projection_blend or "").strip() or _default_ground_projection_blend_path()
        if bpy.data.node_groups.get(_CUSTOM_GROUND_GROUP_NAME) is not None:
            bpy.data.node_groups.remove(bpy.data.node_groups[_CUSTOM_GROUND_GROUP_NAME])
        ng = _load_ground_group_from_blend(path, _CUSTOM_GROUND_GROUP_NAME)
        if ng is None:
            self.report({"ERROR"}, f"Could not load a valid group from {path}")
            return {"CANCELLED"}
        prefs.use_custom_ground_projection = True
        if context.scene.world is not None:
            _apply_world_ground_projection(
                context.scene.world,
                context.scene.hdri_api_settings,
                enabled=bool(context.scene.hdri_api_settings.fake_ground),
            )
        self.report({"INFO"}, f"Loaded {_CUSTOM_GROUND_GROUP_NAME}")
        return {"FINISHED"}


class HDRI_OT_reset_ground_projection(Operator):
    bl_idname = "hdri.reset_ground_projection"
    bl_label = "Use Built-in Ground"
    bl_description = "Switch back to the addon's built-in inline ground projection nodes"

    def execute(self, context):
        prefs = _addon_prefs()
        prefs.use_custom_ground_projection = False
        if context.scene.world is not None:
            _apply_world_ground_projection(
                context.scene.world,
                context.scene.hdri_api_settings,
                enabled=bool(context.scene.hdri_api_settings.fake_ground),
            )
        self.report({"INFO"}, "Using built-in ground projection nodes")
        return {"FINISHED"}


class HDRI_API_Preferences(AddonPreferences):
    bl_idname = __name__

    api_base_url: StringProperty(
        name="API Base URL",
        description="HDRI API root (no trailing slash), e.g. http://127.0.0.1:8000 — must match where uvicorn runs",
        default="http://127.0.0.1:8000",
    )
    register_email: StringProperty(
        name="Email",
        description="Email for account registration and login",
        default="",
    )
    account_password: StringProperty(
        name="Password",
        description="Account password (registration and login)",
        default="",
        subtype="PASSWORD",
    )
    api_key: StringProperty(
        name="API Key (optional)",
        description="Stored automatically after login/register; sent as Authorization: Bearer",
        default="",
        subtype="PASSWORD",
    )
    checkout_package_id: StringProperty(
        name="Token package",
        description="Package id from GET /v1/billing/packages (e.g. tokens_50)",
        default="tokens_50",
    )
    timeout_s: FloatProperty(
        name="Timeout (seconds)",
        default=60.0,
        min=5.0,
        max=600.0,
    )
    use_custom_ground_projection: BoolProperty(
        name="Use custom ground projection",
        description="Use the HDRI Ground Projection node group (templates/ground_projection.blend)",
        default=True,
    )
    ground_projection_blend: StringProperty(
        name="Ground projection template",
        description=(
            "Blend file with node group 'HDRI Ground Projection'. "
            "Leave empty to use the addon default under templates/"
        ),
        default="",
        subtype="FILE_PATH",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "api_base_url")
        layout.label(text="Account")
        layout.prop(self, "register_email")
        layout.prop(self, "account_password")
        row = layout.row(align=True)
        row.operator("hdri.login_account", icon="KEYINGSET")
        row.operator("hdri.register_account", icon="USER")
        layout.prop(self, "api_key")
        layout.prop(self, "checkout_package_id")
        layout.prop(self, "timeout_s")
        layout.operator("hdri.buy_tokens", icon="FUND")
        box = layout.box()
        box.label(text="Ground projection template", icon="NODETREE")
        box.prop(self, "use_custom_ground_projection")
        box.prop(self, "ground_projection_blend")
        row = box.row(align=True)
        row.operator("hdri.save_ground_projection", icon="EXPORT")
        row.operator("hdri.reload_ground_projection", icon="FILE_REFRESH")
        row.operator("hdri.reset_ground_projection", icon="LOOP_BACK")


class HDRI_API_Settings(PropertyGroup):
    input_image_path: StringProperty(
        name="Input Image",
        description="Path to a photo (jpg/png/webp)",
        default="",
        subtype="FILE_PATH",
    )
    library_folder: StringProperty(
        name="Library Folder",
        description="Optional folder for generated panoramas (.hdr/.exr). When empty, files stay temporary/local only",
        default="",
        subtype="DIR_PATH",
    )
    last_library_path: StringProperty(
        name="Last Saved HDRI",
        description="Most recent generated HDRI path saved by this addon",
        default="",
    )
    library_gallery_item: EnumProperty(
        name="Gallery",
        description="Choose a panorama from library gallery",
        items=_library_gallery_items,
        update=_update_library_gallery_item,
    )

    scene_mode: EnumProperty(
        name="Scene",
        items=[
            ("auto", "Auto", "Let server decide"),
            ("outdoor", "Outdoor", "Outdoor-biased lighting"),
            ("indoor", "Indoor", "Indoor-biased lighting"),
            ("studio", "Studio", "Studio-like lighting"),
        ],
        default="auto",
    )

    quality_mode: EnumProperty(
        name="Quality",
        items=[
            ("fast", "Fast (1–3s)", "Lighting-only / fastest"),
            ("balanced", "Balanced (5–10s)", "Basic HDRI (recommended)"),
            ("high", "High (15–30s)", "Diffusion refinement"),
        ],
        default="balanced",
    )
    output_resolution: EnumProperty(
        name="Output Resolution",
        items=[
            ("1024x512", "1024x512", "Fast preview size"),
            ("2048x1024", "2048x1024", "Default (recommended)"),
        ],
        default="2048x1024",
    )

    preset: EnumProperty(
        name="Style",
        items=[
            ("none", "None", "No creative edit"),
            ("sunset", "Sunset", "Warm golden-hour look"),
            ("overcast", "Overcast", "Soft diffuse sky"),
            ("dramatic", "Dramatic Sky", "High-contrast clouds"),
            ("studio_soft", "Studio Softbox", "Soft even studio"),
            ("cyberpunk", "Cyberpunk", "Neon-magenta/cyan vibe"),
        ],
        default="none",
    )

    yaw_degrees: FloatProperty(
        name="Yaw",
        description="Rotate HDRI around Z (degrees). User-tweakable.",
        default=0.0,
        min=-180.0,
        max=180.0,
        update=_update_look_controls,
    )
    pitch_degrees: FloatProperty(
        name="Pitch",
        description="Rotate HDRI around X (advanced)",
        default=0.0,
        min=-90.0,
        max=90.0,
        update=_update_look_controls,
    )
    roll_degrees: FloatProperty(
        name="Roll",
        description="Rotate HDRI around Y (advanced)",
        default=0.0,
        min=-180.0,
        max=180.0,
        update=_update_look_controls,
    )

    exposure: FloatProperty(
        name="Exposure",
        description="Multiply World background strength (artistic)",
        default=1.0,
        min=0.0,
        soft_max=10.0,
        update=_update_look_controls,
    )
    post_exposure: FloatProperty(
        name="Post Exposure",
        description="Extra world strength multiplier applied in Blender nodes",
        default=1.0,
        min=0.0,
        soft_max=10.0,
        update=_update_look_controls,
    )
    blur_amount: FloatProperty(
        name="Blur",
        description="Mixes in a Gaussian-blurred copy of the HDRI for softer lighting",
        default=0.0,
        min=0.0,
        max=1.0,
        update=_update_look_controls,
    )
    hue_shift: FloatProperty(
        name="Hue Shift",
        description="Shift hue in Blender (non-destructive)",
        default=0.0,
        min=-0.5,
        max=0.5,
        update=_update_look_controls,
    )
    saturation: FloatProperty(
        name="Saturation",
        description="Saturation multiplier in Blender (1.0 = unchanged)",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_update_look_controls,
    )
    tint_strength: FloatProperty(
        name="Tint Amount",
        description="Blend a tint color over the HDRI output (0 = off)",
        default=0.0,
        min=0.0,
        max=1.0,
        update=_update_look_controls,
    )
    tint_color: FloatVectorProperty(
        name="Tint Color",
        description="Color picker tint for world and fake ground",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0),
        update=_update_look_controls,
    )
    bake_adjustments_on_server: BoolProperty(
        name="Bake controls on server",
        description="If enabled, blur/hue/sat/post exposure are baked into the returned HDR file",
        default=False,
    )

    add_preview_sphere: BoolProperty(
        name="Add preview sphere",
        description="Adds a reflective sphere to preview the HDRI (optional)",
        default=False,
    )

    fake_ground: BoolProperty(
        name="Fake ground plane",
        description=(
            "Project the HDRI ground below the horizon using a circular shader falloff (no geometry). "
            "Mixes projected ground samples into the blurred environment branch"
        ),
        default=False,
        update=_update_look_controls,
    )
    fake_ground_z_offset: FloatProperty(
        name="Horizon offset",
        description="Horizon vector Z subtracted after ground projection (Easy HDRI Horizon input)",
        default=0.0,
        min=-1000.0,
        max=1000.0,
        update=_update_look_controls,
    )
    fake_ground_lift: FloatProperty(
        name="Ground size",
        description=(
            "Ground projection Size Z — distance of the virtual floor below the camera "
            "(Easy HDRI default is 20; this slider multiplies that value)"
        ),
        default=1.0,
        min=0.05,
        soft_max=10.0,
        update=_update_look_controls,
    )

    # Option D: "Panorama diffusion endpoint + server HDR lift"
    provider: EnumProperty(
        name="Provider",
        items=[
            ("D", "D (External panorama→HDRI)", "API returns 2:1 EXR HDRI"),
        ],
        default="D",
    )

    # Sent to POST /v1/hdri — forwarded to PANORAMA_MODE=http_json worker (img2img / outpainting)
    panorama_prompt: StringProperty(
        name="Extra prompt",
        description=(
            "Optional style/scene hints prepended before the required ERP outpaint prompt "
            "(seamless 360°, horizon level, edge match). Leave empty for defaults only"
        ),
        default="",
    )
    panorama_negative_prompt: StringProperty(
        name="Negative prompt",
        description="Deprecated: workflow defaults are used; not shown in UI",
        default="",
        options={"HIDDEN"},
    )
    panorama_seed: IntProperty(
        name="Seed",
        description="Random seed for the worker. −1 = omit (worker decides)",
        default=-1,
        min=-1,
        max=2_147_483_647,
    )
    panorama_strength: FloatProperty(
        name="Img2img strength",
        description="0–1 if your worker supports strength. −1 = omit",
        default=-1.0,
        min=-1.0,
        max=1.0,
    )
    panorama_extra_json: StringProperty(
        name="Extra JSON",
        description='Optional JSON object merged into the worker request, e.g. {"foo": 1}',
        default="",
    )
    erp_layout_mode: EnumProperty(
        name="ERP Layout",
        items=[
            ("single_front", "Single Front", "Place source image at front-center on ERP canvas"),
        ],
        default="single_front",
    )
    reference_coverage: FloatProperty(
        name="Reference Coverage",
        description="Legacy alias for placement coverage",
        default=0.60,
        min=0.15,
        max=0.85,
    )
    placement_coverage: FloatProperty(
        name="Placement Scale",
        description="Relative on-sphere size; converted to camera hFOV (~35-95 deg) when Placement hFOV is 0",
        default=0.60,
        min=0.15,
        max=0.85,
        update=_update_placement_controls,
    )
    placement_yaw_deg: FloatProperty(
        name="Placement Yaw",
        description="Horizontal panorama placement in degrees",
        default=0.0,
        min=-180.0,
        max=180.0,
        update=_update_placement_controls,
    )
    placement_pitch_deg: FloatProperty(
        name="Placement Pitch",
        description="Vertical panorama placement in degrees",
        default=0.0,
        min=-85.0,
        max=85.0,
        update=_update_placement_controls,
    )
    placement_rotation_deg: FloatProperty(
        name="Placement Rotation",
        description="Sticker in-plane rotation in degrees",
        default=0.0,
        min=-180.0,
        max=180.0,
        update=_update_placement_controls,
    )
    placement_hfov_deg: FloatProperty(
        name="Placement hFOV",
        description="Explicit camera horizontal FOV override in degrees (0 = use Placement Scale mapping)",
        default=0.0,
        min=0.0,
        max=179.0,
        update=_update_placement_controls,
    )
    seam_fix: BoolProperty(
        name="Seam Fix",
        description="API post-process: blur-blend the ERP left/right wrap after generation (works on all backends)",
        default=True,
    )
    erp_canvas_width: IntProperty(
        name="ERP Canvas Width",
        description="Optional worker control canvas width (-1 = use output width)",
        default=-1,
        min=-1,
        max=16384,
    )
    erp_canvas_height: IntProperty(
        name="ERP Canvas Height",
        description="Optional worker control canvas height (-1 = use output height)",
        default=-1,
        min=-1,
        max=8192,
    )

    hdr_reconstruction_mode: EnumProperty(
        name="HDR Reconstruction",
        items=[
            ("comfyui_hdr", "ComfyUI HDR", "GMNet via local HDR worker (:8001) — needs HDR_HTTP_URL on API host"),
            ("heuristic", "Heuristic", "Legacy heuristic HDR lift (no ComfyUI)"),
            ("off", "Off", "Flat linear export (least boosted)"),
        ],
        default="comfyui_hdr",
    )
    hdr_exposure_bias: FloatProperty(
        name="HDR Exposure Bias (EV)",
        description="Post-HDR exposure bias applied by server (comfyui_hdr and heuristic)",
        default=0.0,
        min=-4.0,
        max=4.0,
    )
    heuristic_hdr_lift: BoolProperty(
        name="Legacy HDR boost toggle",
        description="Backward compatibility only; use HDR Reconstruction mode instead",
        default=True,
    )

    # Filled by GET /v1/config (Query API mode) and by last successful Apply
    server_config_panorama_mode: StringProperty(
        name="Server PANORAMA_MODE",
        description="From API GET /v1/config — what the server process was started with",
        default="",
    )
    last_panorama_mode: StringProperty(
        name="Last job panorama_mode",
        description="panorama_mode returned by the last successful Generate & Apply",
        default="",
    )
    current_job_id: StringProperty(
        name="Current Job ID",
        description="Latest async job id returned by /v1/jobs/hdri",
        default="",
    )
    current_job_status: StringProperty(
        name="Current Job Status",
        description="Current async job status (queued/running/succeeded/failed)",
        default="",
    )
    last_job_error: StringProperty(
        name="Last Job Error",
        description="Last async job error message",
        default="",
    )
    tokens_remaining: IntProperty(
        name="Tokens Remaining",
        description="Latest token balance returned by /v1/account (-1 means unknown)",
        default=-1,
        min=-1,
    )

    job_started_monotonic: FloatProperty(
        name="Job Start (mono)",
        default=-1.0,
        description="Internal: time.monotonic() when polling started (-1 means idle)",
        options={"HIDDEN"},
    )
    last_completed_job_wall_s: FloatProperty(
        name="Last Job Duration",
        default=0.0,
        min=0.0,
        description="Internal: wall duration of last successful job for ETA refinement",
        options={"HIDDEN"},
    )


class HDRI_OT_login_account(Operator):
    bl_idname = "hdri.login_account"
    bl_label = "Log in"
    bl_description = "Log in with email and password; stores a fresh API key in preferences"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = _addon_prefs()
        email = (prefs.register_email or "").strip()
        password = prefs.account_password or ""
        if not email:
            self.report({"ERROR"}, "Enter your email in add-on preferences.")
            return {"CANCELLED"}
        if not password:
            self.report({"ERROR"}, "Enter your password in add-on preferences.")
            return {"CANCELLED"}
        base = prefs.api_base_url.rstrip("/")
        try:
            resp = _http_post_json(
                f"{base}/v1/login",
                {"email": email, "password": password},
                headers={},
                timeout_s=int(prefs.timeout_s),
            )
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", str(e))
            except Exception:
                detail = str(e)
            self.report({"ERROR"}, f"Login failed: {detail}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Login failed: {e}")
            return {"CANCELLED"}

        api_key = str(resp.get("api_key", "")).strip()
        if not api_key:
            self.report({"ERROR"}, "Login succeeded but no API key was returned.")
            return {"CANCELLED"}
        prefs.api_key = api_key
        settings = context.scene.hdri_api_settings
        try:
            settings.tokens_remaining = int(resp.get("tokens_remaining", -1))
        except Exception:
            settings.tokens_remaining = -1
        self.report({"INFO"}, f"Logged in as {resp.get('account_id', 'account')}.")
        return {"FINISHED"}


class HDRI_OT_register_account(Operator):
    bl_idname = "hdri.register_account"
    bl_label = "Register"
    bl_description = "Create a new account with email and password; stores the API key"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = _addon_prefs()
        email = (prefs.register_email or "").strip()
        password = prefs.account_password or ""
        if not email:
            self.report({"ERROR"}, "Enter an email in add-on preferences first.")
            return {"CANCELLED"}
        if len(password) < 8:
            self.report({"ERROR"}, "Password must be at least 8 characters.")
            return {"CANCELLED"}
        base = prefs.api_base_url.rstrip("/")
        try:
            resp = _http_post_json(
                f"{base}/v1/register",
                {"email": email, "password": password},
                headers={},
                timeout_s=int(prefs.timeout_s),
            )
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", str(e))
            except Exception:
                detail = str(e)
            self.report({"ERROR"}, f"Registration failed: {detail}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Registration failed: {e}")
            return {"CANCELLED"}

        api_key = str(resp.get("api_key", "")).strip()
        if not api_key:
            self.report({"ERROR"}, "Registration succeeded but no API key was returned.")
            return {"CANCELLED"}
        prefs.api_key = api_key
        settings = context.scene.hdri_api_settings
        try:
            settings.tokens_remaining = int(resp.get("tokens_remaining", -1))
        except Exception:
            settings.tokens_remaining = -1
        self.report({"INFO"}, f"Registered {resp.get('account_id', 'account')}. API key saved to preferences.")
        return {"FINISHED"}


class HDRI_OT_buy_tokens(Operator):
    bl_idname = "hdri.buy_tokens"
    bl_label = "Buy tokens"
    bl_description = "Open Stripe checkout in your browser (requires API key and server billing config)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = _addon_prefs()
        if not (prefs.api_key or "").strip():
            self.report({"ERROR"}, "Log in or register in add-on preferences first.")
            return {"CANCELLED"}
        base = prefs.api_base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {prefs.api_key.strip()}"}
        package_id = (prefs.checkout_package_id or "tokens_50").strip()
        try:
            resp = _http_post_json(
                f"{base}/v1/billing/checkout",
                {"package_id": package_id},
                headers=headers,
                timeout_s=int(prefs.timeout_s),
            )
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", str(e))
            except Exception:
                detail = str(e)
            self.report({"ERROR"}, f"Checkout failed: {detail}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Checkout failed: {e}")
            return {"CANCELLED"}

        url = str(resp.get("checkout_url", "")).strip()
        if not url:
            self.report({"ERROR"}, "Server did not return a checkout URL.")
            return {"CANCELLED"}
        try:
            webbrowser.open(url, new=2)
        except Exception as e:
            self.report({"WARNING"}, f"Open this URL manually: {url} ({e})")
            return {"FINISHED"}
        self.report({"INFO"}, "Opened token checkout in your browser.")
        return {"FINISHED"}


class HDRI_OT_refresh_server_config(Operator):
    bl_idname = "hdri_api.refresh_server_config"
    bl_label = "Query API mode"
    bl_description = "GET /v1/config — shows PANORAMA_MODE on the API server (resize vs http_json, etc.)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = _addon_prefs()
        s = context.scene.hdri_api_settings
        base = prefs.api_base_url.rstrip("/")
        url = f"{base}/v1/config"
        headers = {}
        if prefs.api_key:
            headers["Authorization"] = f"Bearer {prefs.api_key}"
        try:
            data = _http_get_json(url, headers=headers, timeout_s=min(30.0, float(prefs.timeout_s)))
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            self.report({"ERROR"}, f"Config error {e.code}: {body[:200]}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Config request failed: {e}")
            return {"CANCELLED"}
        mode = str(data.get("panorama_mode", "?"))
        s.server_config_panorama_mode = mode
        self.report({"INFO"}, f"Server PANORAMA_MODE={mode}")
        return {"FINISHED"}


class HDRI_OT_open_placement_editor(Operator):
    bl_idname = "hdri.open_placement_editor"
    bl_label = "Open placement editor"
    bl_description = (
        "Drag the source image on a labeled 2:1 panorama "
        "(FRONT/LEFT/RIGHT/BACK/SKY/GROUND) and scale/rotate it"
    )
    bl_options = {"REGISTER", "UNDO"}

    _draw_handle = None
    _timer = None
    _area = None
    _source_rgb = None
    _dragging = False
    _rotation_drag = False
    _last_mouse = (0, 0)
    _canvas_rect = (0, 0, 0, 0)
    _start_values = None
    _request_close = False
    _active_instance = None

    @classmethod
    def request_close(cls) -> bool:
        inst = cls._active_instance
        if inst is None:
            return False
        inst._request_close = True
        if inst._area is not None:
            try:
                inst._area.tag_redraw()
            except Exception:
                pass
        return True

    def _cleanup(self, context):
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        if self._draw_handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, "WINDOW")
            except Exception:
                pass
            self._draw_handle = None
        if self._area is not None:
            try:
                self._area.tag_redraw()
            except Exception:
                pass
        if HDRI_OT_open_placement_editor._active_instance is self:
            HDRI_OT_open_placement_editor._active_instance = None

    @staticmethod
    def _canvas_for_region(region) -> tuple[int, int, int, int]:
        margin_x = 40
        margin_y = 60
        cw = max(320, min(760, int(region.width - margin_x * 2)))
        ch = cw // 2
        max_h = max(180, int(region.height - margin_y * 2))
        if ch > max_h:
            ch = max_h
            cw = ch * 2
        x = int((region.width - cw) * 0.5)
        y = int((region.height - ch) * 0.5)
        return x, y, cw, ch

    def _inside_canvas(self, mx: int, my: int) -> bool:
        x, y, w, h = self._canvas_rect
        return (x <= mx <= x + w) and (y <= my <= y + h)

    def _render_preview(self, context):
        settings = context.scene.hdri_api_settings
        _refresh_placement_preview(settings, context=context, source_rgb=self._source_rgb)
        if self._area is not None:
            self._area.tag_redraw()

    def _draw_callback(self):
        context = bpy.context
        settings = context.scene.hdri_api_settings
        region = context.region
        if region is None:
            return

        x, y, w, h = self._canvas_for_region(region)
        self._canvas_rect = (x, y, w, h)
        img = bpy.data.images.get(_PLACEMENT_PREVIEW_IMAGE_NAME)

        if img is not None:
            try:
                tex = gpu.texture.from_image(img)
                shader = gpu.shader.from_builtin("IMAGE")
                batch = batch_for_shader(
                    shader,
                    "TRI_FAN",
                    {
                        "pos": ((x, y), (x + w, y), (x + w, y + h), (x, y + h)),
                        "texCoord": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                    },
                )
                gpu.state.blend_set("ALPHA")
                shader.bind()
                shader.uniform_sampler("image", tex)
                batch.draw(shader)
            except Exception:
                pass

        outline = gpu.shader.from_builtin("UNIFORM_COLOR")
        outline.bind()
        outline.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
        frame = batch_for_shader(
            outline,
            "LINE_LOOP",
            {"pos": ((x, y), (x + w, y), (x + w, y + h), (x, y + h))},
        )
        frame.draw(outline)

        cx = x + w * 0.5
        cy = y + h * 0.5
        cross = batch_for_shader(
            outline,
            "LINES",
            {
                "pos": (
                    (cx, y),
                    (cx, y + h),
                    (x + w * 0.25, y),
                    (x + w * 0.25, y + h),
                    (x + w * 0.75, y),
                    (x + w * 0.75, y + h),
                    (x, cy),
                    (x + w, cy),
                )
            },
        )
        outline.uniform_float("color", (1.0, 1.0, 1.0, 0.22))
        cross.draw(outline)

        # Ground-level guide (horizon / pitch=0) to help manual alignment.
        horizon = batch_for_shader(
            outline,
            "LINES",
            {"pos": ((x, cy), (x + w, cy))},
        )
        outline.uniform_float("color", (1.0, 0.85, 0.25, 0.95))
        horizon.draw(outline)
        gpu.state.blend_set("NONE")

        _draw_placement_compass_labels(x, y, w, h)

    def invoke(self, context, event):
        if HDRI_OT_open_placement_editor._active_instance is not None:
            self.report({"INFO"}, "Placement editor is already open. Use Close Editor to exit it.")
            return {"CANCELLED"}

        settings = context.scene.hdri_api_settings
        src_path = bpy.path.abspath(settings.input_image_path)
        if not src_path or not os.path.exists(src_path):
            self.report({"ERROR"}, "Pick an input image first.")
            return {"CANCELLED"}

        source_rgb = _placement_source_rgb_from_path(src_path)
        if source_rgb is None:
            self.report({"ERROR"}, "Unable to decode the input image for placement preview.")
            return {"CANCELLED"}

        self._source_rgb = source_rgb
        self._request_close = False
        self._start_values = (
            float(settings.placement_yaw_deg),
            float(settings.placement_pitch_deg),
            float(settings.placement_rotation_deg),
            float(settings.placement_coverage),
            float(settings.reference_coverage),
            float(settings.placement_hfov_deg),
        )
        self._area = context.area
        HDRI_OT_open_placement_editor._active_instance = self
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(self._draw_callback, (), "WINDOW", "POST_PIXEL")
        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        self._render_preview(context)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        settings = context.scene.hdri_api_settings

        if self._request_close:
            self._cleanup(context)
            return {"FINISHED"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value in {"PRESS", "RELEASE", "CLICK"}:
            (
                settings.placement_yaw_deg,
                settings.placement_pitch_deg,
                settings.placement_rotation_deg,
                settings.placement_coverage,
                settings.reference_coverage,
                settings.placement_hfov_deg,
            ) = self._start_values
            self._render_preview(context)
            self._cleanup(context)
            return {"CANCELLED"}

        if event.type in {"SPACE", "Q"} and event.value == "PRESS":
            self._cleanup(context)
            return {"FINISHED"}

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._cleanup(context)
            return {"FINISHED"}

        if event.type == "R" and event.value == "PRESS":
            self._rotation_drag = not self._rotation_drag
            return {"RUNNING_MODAL"}

        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS":
            step = 0.02 if event.type == "WHEELUPMOUSE" else -0.02
            settings.placement_coverage = _clampf(float(settings.placement_coverage) + step, 0.15, 0.85)
            settings.reference_coverage = float(settings.placement_coverage)
            self._render_preview(context)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS" and self._inside_canvas(event.mouse_region_x, event.mouse_region_y):
                self._dragging = True
                self._last_mouse = (event.mouse_region_x, event.mouse_region_y)
                return {"RUNNING_MODAL"}
            if event.value == "RELEASE" and self._dragging:
                self._dragging = False
                self._cleanup(context)
                return {"FINISHED"}

        if event.type == "MOUSEMOVE" and self._dragging:
            mx = int(event.mouse_region_x)
            my = int(event.mouse_region_y)
            x, y, w, h = self._canvas_rect
            if self._rotation_drag:
                dx = float(mx - self._last_mouse[0])
                settings.placement_rotation_deg = _clampf(float(settings.placement_rotation_deg) + dx * 0.35, -180.0, 180.0)
            else:
                nx = _clampf((mx - x) / max(1, w), 0.0, 1.0)
                ny = _clampf((my - y) / max(1, h), 0.0, 1.0)
                settings.placement_yaw_deg = (nx - 0.5) * 360.0
                settings.placement_pitch_deg = (0.5 - ny) * 170.0
            self._last_mouse = (mx, my)
            self._render_preview(context)
            return {"RUNNING_MODAL"}

        if event.type == "TIMER":
            if self._area is not None:
                self._area.tag_redraw()
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


class HDRI_OT_close_placement_editor(Operator):
    bl_idname = "hdri.close_placement_editor"
    bl_label = "Close placement editor"
    bl_description = "Close the active placement editor overlay"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if HDRI_OT_open_placement_editor.request_close():
            self.report({"INFO"}, "Closing placement editor.")
            return {"FINISHED"}
        self.report({"INFO"}, "Placement editor is not open.")
        return {"CANCELLED"}


class HDRI_OT_apply_from_api(Operator):
    bl_idname = "hdri.apply_from_api"
    bl_label = "Generate & Apply HDRI"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _job_id = ""
    _base_url = ""
    _headers = {}
    _deadline = 0.0

    @staticmethod
    def _resolution_pair(value: str) -> tuple[int, int]:
        try:
            w_s, h_s = value.lower().split("x", 1)
            return int(w_s), int(h_s)
        except Exception:
            return 2048, 1024

    def _clear_modal_timer(self, context):
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

    def _refresh_tokens(self, settings):
        acct = _safe_get_account(self._base_url, self._headers, timeout_s=10)
        if acct and "tokens_remaining" in acct:
            try:
                settings.tokens_remaining = int(acct["tokens_remaining"])
            except Exception:
                pass

    def _cancel_remote_job(self):
        if not self._job_id:
            return
        try:
            _http_post_json(
                f"{self._base_url}/v1/jobs/{self._job_id}/cancel",
                {},
                headers=self._headers,
                timeout_s=8,
            )
        except Exception:
            pass

    def _apply_hdri_response(self, context, settings, prefs, resp: dict) -> tuple[bool, str]:
        # Safety net: block accidental resize responses from misconfigured servers.
        if str(resp.get("panorama_mode", "")).strip().lower() == "resize":
            settings.last_panorama_mode = "resize"
            return (
                False,
                "API returned panorama_mode=resize (stretched source image). Fix server mode to http_json and retry.",
            )

        # Prefer signed URL: hdri_url (Radiance .hdr) or exr_url (same URL or .exr)
        download_url = resp.get("hdri_url") or resp.get("exr_url")
        file_bytes = None
        if download_url:
            try:
                file_bytes = _download_bytes(download_url, headers=self._headers, timeout_s=int(prefs.timeout_s))
            except Exception as e:
                return False, f"Failed to download HDRI: {e}"
        elif resp.get("exr_base64"):
            try:
                file_bytes = base64.b64decode(resp["exr_base64"])
            except Exception as e:
                return False, f"Bad exr_base64: {e}"
        else:
            return False, "API response missing hdri_url, exr_url, or exr_base64."

        suffix = ".hdr"
        if download_url:
            path_part = download_url.split("?", 1)[0].lower()
            if path_part.endswith(".exr"):
                suffix = ".exr"
            elif path_part.endswith(".hdr"):
                suffix = ".hdr"
        elif resp.get("exr_base64"):
            suffix = ".exr"

        image_path, write_warn, saved_to_library = _write_hdri_output_file(settings, file_bytes, suffix)
        if image_path is None:
            return False, write_warn or "Failed to write HDRI file."

        settings.last_library_path = image_path

        ok_apply, apply_message = _apply_hdri_image_path(context, settings, image_path)
        if not ok_apply:
            return False, apply_message

        mode = str(resp.get("panorama_mode", "")).strip()
        if mode:
            settings.last_panorama_mode = mode
        settings.current_job_status = "succeeded"
        self._refresh_tokens(settings)

        hints: list[str] = []
        if saved_to_library:
            hints.append(f"Saved to library: {os.path.basename(image_path)}.")
        else:
            if write_warn:
                hints.append(str(write_warn))
            elif (settings.library_folder or "").strip():
                hints.append("Library Folder is set but the file was not saved there.")

        suffix_msg = (" " + " ".join(hints)) if hints else ""
        if mode:
            if mode == "resize":
                return (
                    True,
                    "HDRI applied (panorama_mode=resize — photo stretched to 2:1; prompts unused until API uses PANORAMA_MODE=http_json)."
                    + suffix_msg,
                )
            return True, f"HDRI applied (panorama_mode={mode}).{suffix_msg}"
        return True, "HDRI applied to World." + suffix_msg

    def execute(self, context):
        prefs = _addon_prefs()
        s = context.scene.hdri_api_settings

        if not s.input_image_path:
            self.report({"ERROR"}, "Pick an input image first.")
            return {"CANCELLED"}

        img_path = bpy.path.abspath(s.input_image_path)
        if not os.path.exists(img_path):
            self.report({"ERROR"}, f"File not found: {img_path}")
            return {"CANCELLED"}

        try:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read image: {e}")
            return {"CANCELLED"}

        base = prefs.api_base_url.rstrip("/")
        submit_url = f"{base}/v1/jobs/hdri"

        headers = {}
        if prefs.api_key:
            headers["Authorization"] = f"Bearer {prefs.api_key}"

        # Hard-stop resize fallback. If this triggers, the wrong API mode/process is running.
        try:
            cfg = _http_get_json(f"{base}/v1/config", headers=headers, timeout_s=min(15.0, float(prefs.timeout_s)))
            cfg_mode = str(cfg.get("panorama_mode", "")).strip().lower()
            if cfg_mode == "resize":
                s.server_config_panorama_mode = "resize"
                self.report(
                    {"ERROR"},
                    "Server is in PANORAMA_MODE=resize. Start API with PANORAMA_MODE=http_json (and worker) to generate real panoramas.",
                )
                return {"CANCELLED"}
            if cfg_mode:
                s.server_config_panorama_mode = cfg_mode
        except Exception:
            # Do not block if /v1/config is unavailable; request path below may still succeed.
            pass

        out_w, out_h = self._resolution_pair(s.output_resolution)
        payload = {
            "provider": s.provider,
            "image_b64": img_b64,
            "scene_mode": s.scene_mode,
            "quality_mode": s.quality_mode,
            "preset": s.preset,
            "output_width": out_w,
            "output_height": out_h,
            "assume_upright": True,
        }
        # Match hdri_api_server/app.py HdriRequest — only add keys when set
        if s.panorama_prompt.strip():
            payload["panorama_prompt"] = s.panorama_prompt.strip()
        if s.panorama_seed >= 0:
            payload["panorama_seed"] = int(s.panorama_seed)
        if s.panorama_strength >= 0.0:
            payload["panorama_strength"] = float(s.panorama_strength)
        payload["erp_layout_mode"] = s.erp_layout_mode
        # Clamp + round coverage to avoid float32 spillover (e.g. 0.8500000238)
        # tripping server-side <= 0.85 validation.
        coverage_send = round(_clampf(float(s.placement_coverage), 0.15, 0.85), 6)
        payload["reference_coverage"] = coverage_send
        payload["placement_coverage"] = coverage_send
        payload["placement_yaw_deg"] = float(s.placement_yaw_deg)
        payload["placement_pitch_deg"] = float(s.placement_pitch_deg)
        payload["placement_rotation_deg"] = float(s.placement_rotation_deg)
        if s.placement_hfov_deg > 0.0:
            payload["placement_hfov_deg"] = float(s.placement_hfov_deg)
        payload["seam_fix"] = bool(s.seam_fix)
        if s.erp_canvas_width > 0:
            payload["erp_canvas_width"] = int(s.erp_canvas_width)
        if s.erp_canvas_height > 0:
            payload["erp_canvas_height"] = int(s.erp_canvas_height)
        if s.panorama_extra_json.strip():
            try:
                payload["panorama_extra"] = json.loads(s.panorama_extra_json.strip())
            except json.JSONDecodeError as e:
                self.report({"ERROR"}, f"Extra JSON invalid: {e}")
                return {"CANCELLED"}
        payload["hdr_reconstruction_mode"] = s.hdr_reconstruction_mode
        payload["hdr_exposure_bias"] = float(s.hdr_exposure_bias)
        # Keep legacy field for older API versions.
        payload["heuristic_hdr_lift"] = bool(s.hdr_reconstruction_mode == "heuristic")
        if s.bake_adjustments_on_server:
            payload["blur_sigma"] = float(s.blur_amount * 6.0)
            payload["hue_shift"] = float(s.hue_shift)
            payload["sat_scale"] = float(s.saturation)
            payload["color_gain"] = float(s.post_exposure)

        try:
            create_resp = _http_post_json(submit_url, payload, headers=headers, timeout_s=int(prefs.timeout_s))
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            self.report({"ERROR"}, f"API error {e.code}: {body[:300]}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"API request failed: {e}")
            return {"CANCELLED"}

        job_id = str(create_resp.get("job_id", "") if isinstance(create_resp, dict) else "").strip()
        if not job_id:
            self.report({"ERROR"}, "API response missing job_id.")
            return {"CANCELLED"}

        s.current_job_id = job_id
        s.current_job_status = "queued"
        s.last_job_error = ""
        s.job_started_monotonic = float(time.monotonic())
        self._job_id = job_id
        self._base_url = base
        self._headers = headers
        self._deadline = time.monotonic() + float(prefs.timeout_s)
        self._timer = context.window_manager.event_timer_add(2.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        s = context.scene.hdri_api_settings
        if event.type == "ESC":
            self._clear_modal_timer(context)
            self._cancel_remote_job()
            s.current_job_status = ""
            s.current_job_id = ""
            s.last_job_error = "cancelled by user"
            s.job_started_monotonic = -1.0
            self._refresh_tokens(s)
            self.report({"INFO"}, "HDRI generation cancelled.")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if time.monotonic() >= self._deadline:
            self._clear_modal_timer(context)
            self._cancel_remote_job()
            s.current_job_status = "failed"
            s.last_job_error = "Job polling timed out."
            s.job_started_monotonic = -1.0
            self._refresh_tokens(s)
            self.report({"ERROR"}, "Job polling timed out.")
            return {"CANCELLED"}

        try:
            status_resp = _http_get_json(
                f"{self._base_url}/v1/jobs/{self._job_id}",
                headers=self._headers,
                timeout_s=20,
            )
        except Exception as e:
            self._clear_modal_timer(context)
            s.current_job_status = "failed"
            s.last_job_error = f"Polling failed: {e}"
            s.job_started_monotonic = -1.0
            self.report({"ERROR"}, s.last_job_error[:300])
            return {"CANCELLED"}

        if not isinstance(status_resp, dict):
            self._clear_modal_timer(context)
            s.current_job_status = "failed"
            s.last_job_error = "Invalid job status response."
            s.job_started_monotonic = -1.0
            self.report({"ERROR"}, s.last_job_error)
            return {"CANCELLED"}

        status = str(status_resp.get("status", "")).strip().lower()
        if status:
            s.current_job_status = status
        if status in {"queued", "running"}:
            # Force panel redraw so spinner/status hints animate while modal polling runs.
            wm = getattr(context, "window_manager", None)
            windows = getattr(wm, "windows", []) if wm is not None else []
            for win in windows:
                screen = getattr(win, "screen", None)
                if screen is None:
                    continue
                for area in screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
            return {"RUNNING_MODAL"}

        self._clear_modal_timer(context)
        if status == "failed":
            err = str(status_resp.get("error", "Job failed without details."))
            s.last_job_error = err
            s.job_started_monotonic = -1.0
            self._refresh_tokens(s)
            self.report({"ERROR"}, f"Job failed: {err[:300]}")
            return {"CANCELLED"}
        if status == "succeeded":
            start_mono = float(getattr(s, "job_started_monotonic", -1.0))
            ok, message = self._apply_hdri_response(context, s, _addon_prefs(), status_resp)
            if ok:
                if start_mono >= 0.0:
                    elapsed_job = float(time.monotonic()) - start_mono
                    # Skip very short durations (misconfig/timeouts/noise).
                    if elapsed_job >= 45.0:
                        s.last_completed_job_wall_s = elapsed_job
                s.job_started_monotonic = -1.0
                s.current_job_id = ""
                s.last_job_error = ""
                self.report({"INFO"}, message)
                return {"FINISHED"}
            s.current_job_status = "failed"
            s.last_job_error = message
            s.job_started_monotonic = -1.0
            self.report({"ERROR"}, message[:300])
            return {"CANCELLED"}

        s.current_job_status = "failed"
        s.last_job_error = f"Unexpected job status: {status or '(missing)'}"
        s.job_started_monotonic = -1.0
        self.report({"ERROR"}, s.last_job_error)
        return {"CANCELLED"}


class HDRI_OT_apply_library_hdri(Operator):
    bl_idname = "hdri.apply_library_hdri"
    bl_label = "Apply Library HDRI"
    bl_options = {"REGISTER", "UNDO"}

    file_name: StringProperty(default="")

    def execute(self, context):
        s = context.scene.hdri_api_settings
        folder = bpy.path.abspath((s.library_folder or "").strip())
        if not folder:
            self.report({"ERROR"}, "Set Library Folder first.")
            return {"CANCELLED"}
        if not os.path.isdir(folder):
            self.report({"ERROR"}, f"Library folder not found: {folder}")
            return {"CANCELLED"}

        target_name = (self.file_name or "").strip()
        if not target_name:
            selected = str(getattr(s, "library_gallery_item", "") or "").strip()
            if selected and selected != "__none__":
                target_name = selected
        if not target_name:
            files = _list_hdri_files(folder)
            if not files:
                self.report({"ERROR"}, "No .hdr/.exr files found in Library Folder.")
                return {"CANCELLED"}
            target_name = files[0]

        path = os.path.join(folder, target_name)
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        ok, message = _apply_hdri_image_path(context, s, path)
        if not ok:
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        s.last_library_path = path
        self.report({"INFO"}, f"Applied library HDRI: {target_name}")
        return {"FINISHED"}


class HDRI_OT_cancel_job(Operator):
    bl_idname = "hdri.cancel_job"
    bl_label = "Cancel Job"

    def execute(self, context):
        prefs = _addon_prefs()
        s = context.scene.hdri_api_settings
        job_id = (s.current_job_id or "").strip()
        if not job_id:
            self.report({"INFO"}, "No active job.")
            return {"CANCELLED"}

        base = prefs.api_base_url.rstrip("/")
        headers = {}
        if prefs.api_key:
            headers["Authorization"] = f"Bearer {prefs.api_key}"
        try:
            _http_post_json(f"{base}/v1/jobs/{job_id}/cancel", {}, headers=headers, timeout_s=min(20, int(prefs.timeout_s)))
        except Exception as e:
            self.report({"WARNING"}, f"Server cancel failed, clearing local state: {e}")

        s.current_job_status = ""
        s.current_job_id = ""
        s.last_job_error = "cancelled by user"
        s.job_started_monotonic = -1.0
        acct = _safe_get_account(base, headers, timeout_s=10)
        if acct and "tokens_remaining" in acct:
            try:
                s.tokens_remaining = int(acct["tokens_remaining"])
            except Exception:
                pass
        self.report({"INFO"}, "Cancelled current job.")
        return {"FINISHED"}


class HDRI_PT_panel(Panel):
    bl_label = "Photo → HDRI (API)"
    bl_idname = "HDRI_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "HDRI"

    def draw(self, context):
        layout = self.layout
        s = context.scene.hdri_api_settings

        col = layout.column(align=True)
        col.prop(s, "input_image_path")
        col.label(text="Placement")
        col.prop(s, "placement_coverage")
        row_place = col.row(align=True)
        row_place.prop(s, "placement_yaw_deg")
        row_place.prop(s, "placement_pitch_deg")
        col.prop(s, "placement_rotation_deg")
        col.prop(s, "placement_hfov_deg")
        row_editor = col.row(align=True)
        row_editor.operator(HDRI_OT_open_placement_editor.bl_idname, text="Open Editor", icon="ORIENTATION_VIEW")
        row_editor.operator(HDRI_OT_close_placement_editor.bl_idname, text="Close Editor", icon="X")
        preview_img = bpy.data.images.get(_PLACEMENT_PREVIEW_IMAGE_NAME)
        if preview_img is not None:
            col.template_preview(preview_img, show_buttons=False)
        col.separator()
        col.prop(s, "library_folder")
        library_folder = bpy.path.abspath((s.library_folder or "").strip()) if s.library_folder else ""
        if library_folder:
            lib_box = col.box()
            lib_box.label(text="Library gallery (.hdr/.exr)")
            if os.path.isdir(library_folder):
                files = _list_hdri_files(library_folder)
                if files:
                    row = lib_box.row(align=True)
                    row.operator(HDRI_OT_apply_library_hdri.bl_idname, text="Apply Latest", icon="WORLD")
                    selected = str(getattr(s, "library_gallery_item", "") or "")
                    apply_row = lib_box.row(align=True)
                    apply_sel = apply_row.operator(HDRI_OT_apply_library_hdri.bl_idname, text="Apply Selected", icon="CHECKMARK")
                    if selected and selected != "__none__":
                        apply_sel.file_name = selected
                    else:
                        apply_row.enabled = False
                    lib_box.template_icon_view(s, "library_gallery_item", show_labels=True, scale=6.0, scale_popup=4.0)
                    if s.last_library_path:
                        lib_box.label(text=f"Last saved: {os.path.basename(s.last_library_path)}", icon="FILE_TICK")
                else:
                    lib_box.label(text="No panoramas found yet.", icon="INFO")
            else:
                lib_box.label(text="Folder does not exist yet (will be created on save).", icon="INFO")
        col.prop(s, "provider")

        col.separator()
        col.prop(s, "scene_mode")
        col.prop(s, "quality_mode")
        col.prop(s, "output_resolution")
        col.prop(s, "preset")

        col.separator()
        col.prop(s, "yaw_degrees")
        col.prop(s, "pitch_degrees")
        col.prop(s, "roll_degrees")
        col.prop(s, "exposure")
        col.prop(s, "post_exposure")
        col.prop(s, "blur_amount")
        col.prop(s, "hue_shift")
        col.prop(s, "saturation")
        col.prop(s, "tint_strength")
        col.prop(s, "tint_color")
        col.template_color_picker(s, "tint_color", value_slider=True)
        col.prop(s, "bake_adjustments_on_server")
        col.prop(s, "add_preview_sphere")
        col.prop(s, "fake_ground")
        fg = col.column(align=True)
        fg.enabled = s.fake_ground
        fg.prop(s, "fake_ground_z_offset")
        fg.prop(s, "fake_ground_lift")
        gp_box = col.box()
        gp_box.label(text="Custom ground node group", icon="NODETREE")
        prefs = _addon_prefs()
        if prefs.use_custom_ground_projection:
            gp_box.label(text="Mode: saved template", icon="CHECKMARK")
        else:
            gp_box.label(text="Mode: built-in inline nodes", icon="INFO")
        row = gp_box.row(align=True)
        row.operator(HDRI_OT_save_ground_projection.bl_idname, text="Save template")
        row.operator(HDRI_OT_reload_ground_projection.bl_idname, text="", icon="FILE_REFRESH")
        gp_box.operator(HDRI_OT_reset_ground_projection.bl_idname, text="Use built-in nodes")
        gp_box.label(text="1. Clean up nodes → Ctrl+G → name group", icon="BLANK1")
        gp_box.label(text='2. "HDRI Ground Projection" + Vector/Size/Horizon/Rotation', icon="BLANK1")
        gp_box.label(text="3. Save template → commit templates/ground_projection.blend", icon="BLANK1")

        box = layout.box()
        row = box.row(align=True)
        row.label(text="Panorama backend")
        row.operator(HDRI_OT_refresh_server_config.bl_idname, text="", icon="FILE_REFRESH")
        cfg = (s.server_config_panorama_mode or "").strip()
        last = (s.last_panorama_mode or "").strip()
        if cfg:
            box.label(text=f"Server env: {cfg}", icon="SETTINGS")
        else:
            box.label(text="Server env: (click refresh)", icon="QUESTION")
        if last:
            box.label(text=f"Last job: {last}", icon="CHECKMARK")
        if cfg == "resize" or last == "resize":
            box.label(
                text="resize = only stretch photo to 2:1. Prompts/seed/strength are ignored.",
                icon="ERROR",
            )
            box.label(
                text="On the API host set PANORAMA_MODE=http_json and PANORAMA_HTTP_URL=…",
                icon="INFO",
            )
        active_job = (s.current_job_status or "").strip().lower() in {"queued", "running"}
        if s.current_job_id:
            box.label(text=f"Current job: {s.current_job_id}", icon="TIME")
        if active_job:
            prefs_eta = _addon_prefs()
            t0 = float(getattr(s, "job_started_monotonic", -1.0))
            if t0 >= 0.0:
                elapsed = float(time.monotonic()) - t0
                expected = float(_expected_remote_job_seconds(s))
                remaining = max(0.0, expected - elapsed)
                budget = max(1.0, float(prefs_eta.timeout_s))
                learns = ""
                if float(getattr(s, "last_completed_job_wall_s", 0.0) or 0.0) >= 15.0:
                    learns = " (from last job)"
                elif expected > 0.0:
                    learns = " (typical heuristic)"
                box.label(
                    text=(
                        f"Elapsed {_format_duration_compact(elapsed)}  ·  est. left ~{_format_duration_compact(remaining)}"
                        f"  ·  timeout {_format_duration_compact(budget)}{learns}"
                    ),
                    icon="TIME",
                )
            status_now = (s.current_job_status or "").strip().lower()
            if status_now == "running":
                box.label(text=f"Job status: 🟢 running", icon="INFO")
                box.label(text=f"  {_running_ascii_spinner()} generating", icon="BLANK1")
            elif status_now == "queued":
                box.label(text=f"Job status: 🔵 queued", icon="INFO")
            else:
                box.label(text=f"Job status: {s.current_job_status}", icon="INFO")
        elif s.current_job_status:
            box.label(text=f"Job status: {s.current_job_status}", icon="INFO")
        if s.last_job_error:
            box.label(text=f"Last error: {s.last_job_error[:100]}", icon="ERROR")
        if s.tokens_remaining >= 0:
            box.label(text=f"Tokens remaining: {s.tokens_remaining}", icon="SOLO_ON")
        acct = box.column(align=True)
        acct.prop(_addon_prefs(), "register_email", text="Email")
        acct.prop(_addon_prefs(), "account_password", text="Password")
        row_acct = acct.row(align=True)
        row_acct.operator("hdri.login_account", icon="KEYINGSET")
        row_acct.operator("hdri.register_account", icon="USER")
        acct.operator("hdri.buy_tokens", icon="FUND")

        box.label(text="Panorama options")
        box.label(text="Extra prompt is prepended before required ERP outpaint instructions.", icon="INFO")
        col2 = box.column(align=True)
        col2.prop(s, "panorama_prompt")
        row = col2.row(align=True)
        row.prop(s, "panorama_seed")
        row.prop(s, "panorama_strength")
        col2.prop(s, "erp_layout_mode")
        col2.prop(s, "seam_fix")
        row2 = col2.row(align=True)
        row2.prop(s, "erp_canvas_width")
        row2.prop(s, "erp_canvas_height")
        col2.prop(s, "panorama_extra_json")
        col2.prop(s, "hdr_reconstruction_mode")
        col2.prop(s, "hdr_exposure_bias")

        col.separator()
        row = col.row(align=True)
        row.enabled = not active_job
        row.operator(HDRI_OT_apply_from_api.bl_idname, icon="WORLD")
        if active_job:
            col.operator(HDRI_OT_cancel_job.bl_idname, icon="CANCEL")


classes = (
    HDRI_API_Preferences,
    HDRI_API_Settings,
    HDRI_OT_save_ground_projection,
    HDRI_OT_reload_ground_projection,
    HDRI_OT_reset_ground_projection,
    HDRI_OT_login_account,
    HDRI_OT_register_account,
    HDRI_OT_buy_tokens,
    HDRI_OT_refresh_server_config,
    HDRI_OT_open_placement_editor,
    HDRI_OT_close_placement_editor,
    HDRI_OT_apply_from_api,
    HDRI_OT_apply_library_hdri,
    HDRI_OT_cancel_job,
    HDRI_PT_panel,
)


def register():
    _clear_library_previews()
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.hdri_api_settings = PointerProperty(type=HDRI_API_Settings)
    _ensure_default_ground_projection_template()


def unregister():
    _clear_library_previews()
    if hasattr(bpy.types.Scene, "hdri_api_settings"):
        del bpy.types.Scene.hdri_api_settings
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

