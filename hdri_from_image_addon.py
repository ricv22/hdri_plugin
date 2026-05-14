bl_info = {
    "name": "Photo → HDRI World (API)",
    "author": "Cursor AI",
    "version": (0, 1, 5),
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

import bpy
import bpy.utils.previews
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

    env = next((n for n in nodes if n.bl_idname == "ShaderNodeTexEnvironment"), None)
    if env is None:
        env = nodes.new("ShaderNodeTexEnvironment")
        env.location = (-350, 0)

    env_blur = next((n for n in nodes if n.bl_idname == "ShaderNodeTexEnvironment" and n != env), None)
    if env_blur is None:
        env_blur = nodes.new("ShaderNodeTexEnvironment")
        env_blur.label = "HDRI Blur Source"
        env_blur.location = (-350, -220)

    mix = next((n for n in nodes if n.bl_idname == "ShaderNodeMixRGB" and n.label == "HDRI Blur Mix"), None)
    if mix is None:
        mix = next((n for n in nodes if n.bl_idname == "ShaderNodeMixRGB"), None)
    if mix is None:
        mix = nodes.new("ShaderNodeMixRGB")
        mix.location = (-120, -80)
        mix.blend_type = "MIX"
        mix.inputs["Fac"].default_value = 0.0
    mix.label = "HDRI Blur Mix"

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

    texcoord = next((n for n in nodes if n.bl_idname == "ShaderNodeTexCoord"), None)
    if texcoord is None:
        texcoord = nodes.new("ShaderNodeTexCoord")
        texcoord.location = (-850, 0)

    if not mapping.inputs["Vector"].is_linked:
        links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    if not env.inputs["Vector"].is_linked:
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    if not env_blur.inputs["Vector"].is_linked:
        links.new(mapping.outputs["Vector"], env_blur.inputs["Vector"])
    if not mix.inputs["Color1"].is_linked:
        links.new(env.outputs["Color"], mix.inputs["Color1"])
    if not mix.inputs["Color2"].is_linked:
        links.new(env_blur.outputs["Color"], mix.inputs["Color2"])
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
        "mix": mix,
        "tint_mix": tint_mix,
        "hue_sat": hue_sat,
        "bg": bg,
        "mapping": mapping,
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


def _set_fake_ground_visible(visible: bool):
    obj = bpy.data.objects.get(_FAKE_GROUND_OBJ)
    if obj is None:
        return
    obj.hide_viewport = not visible
    obj.hide_render = not visible


def _build_fake_ground_mesh(mesh: bpy.types.Mesh):
    """Fill mesh with a 2×2 XY quad (−1..1)."""
    bm = None
    try:
        import bmesh

        bm = bmesh.new()
        v0 = bm.verts.new((-1.0, -1.0, 0.0))
        v1 = bm.verts.new((1.0, -1.0, 0.0))
        v2 = bm.verts.new((1.0, 1.0, 0.0))
        v3 = bm.verts.new((-1.0, 1.0, 0.0))
        bm.faces.new((v0, v1, v2, v3))
        try:
            bm.normal_update()
        except AttributeError:
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        if bm:
            bm.free()


def _link_fake_ground_to_scene(context, obj: bpy.types.Object):
    """Orphaned objects exist in bpy.data but are invisible — ensure scene membership."""
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    if scene.objects.get(obj.name) is None:
        try:
            scene.collection.objects.link(obj)
        except RuntimeError:
            pass


def _ensure_fake_ground_object(context):
    obj = bpy.data.objects.get(_FAKE_GROUND_OBJ)
    if obj is not None and obj.type == "MESH":
        _link_fake_ground_to_scene(context, obj)
        if len(obj.data.polygons) == 0:
            _build_fake_ground_mesh(obj.data)
        return obj

    mesh = bpy.data.meshes.new(_FAKE_GROUND_OBJ + "_Mesh")
    _build_fake_ground_mesh(mesh)
    obj = bpy.data.objects.new(_FAKE_GROUND_OBJ, mesh)
    col = getattr(context, "collection", None)
    if col is None:
        sc = getattr(context, "scene", None)
        col = sc.collection if sc is not None else None
    if col is not None:
        try:
            col.objects.link(obj)
        except RuntimeError:
            pass
    _link_fake_ground_to_scene(context, obj)
    return obj


def _rebuild_fake_ground_material(
    img: bpy.types.Image,
    mapping_src: bpy.types.Node,
    mix_src: bpy.types.Node,
    tint_mix_src: bpy.types.Node,
    hue_sat_src: bpy.types.Node,
    bg_src: bpy.types.Node,
    lift: float,
):
    """Emissive ground using the same HDRI sampling as the world (blur + hue/sat + strength)."""
    mat = bpy.data.materials.get(_FAKE_GROUND_MAT)
    if mat is None:
        mat = bpy.data.materials.new(_FAKE_GROUND_MAT)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    try:
        geom = nodes.new("ShaderNodeNewGeometry")
    except (RuntimeError, TypeError):
        geom = nodes.new("ShaderNodeGeometry")
    geom.location = (-1400, 0)

    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-1200, 0)

    comb = nodes.new("ShaderNodeCombineXYZ")
    comb.location = (-1000, 0)
    comb.inputs["Z"].default_value = -float(lift)

    norm = nodes.new("ShaderNodeVectorMath")
    norm.location = (-800, 0)
    norm.operation = "NORMALIZE"

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-600, 0)
    rot = mapping_src.inputs["Rotation"].default_value
    loc = mapping_src.inputs["Location"].default_value
    scl = mapping_src.inputs["Scale"].default_value
    mapping.inputs["Location"].default_value[0] = loc[0]
    mapping.inputs["Location"].default_value[1] = loc[1]
    mapping.inputs["Location"].default_value[2] = loc[2]
    mapping.inputs["Rotation"].default_value[0] = rot[0]
    mapping.inputs["Rotation"].default_value[1] = rot[1]
    mapping.inputs["Rotation"].default_value[2] = rot[2]
    mapping.inputs["Scale"].default_value[0] = scl[0]
    mapping.inputs["Scale"].default_value[1] = scl[1]
    mapping.inputs["Scale"].default_value[2] = scl[2]

    env = nodes.new("ShaderNodeTexEnvironment")
    env.location = (-400, 80)
    env.image = img

    env_blur = nodes.new("ShaderNodeTexEnvironment")
    env_blur.location = (-400, -120)
    env_blur.image = img
    env_blur.label = "HDRI Blur Source"

    mix = nodes.new("ShaderNodeMixRGB")
    mix.location = (0, -40)
    mix.blend_type = "MIX"
    mix.inputs["Fac"].default_value = mix_src.inputs["Fac"].default_value

    hue_sat = nodes.new("ShaderNodeHueSaturation")
    hue_sat.location = (200, -40)
    hue_sat.inputs["Hue"].default_value = hue_sat_src.inputs["Hue"].default_value
    hue_sat.inputs["Saturation"].default_value = hue_sat_src.inputs["Saturation"].default_value

    tint_mix = nodes.new("ShaderNodeMixRGB")
    tint_mix.location = (260, -40)
    tint_mix.blend_type = "MIX"
    tint_mix.inputs["Fac"].default_value = tint_mix_src.inputs["Fac"].default_value
    tint_color = tint_mix_src.inputs["Color2"].default_value
    tint_mix.inputs["Color2"].default_value = (tint_color[0], tint_color[1], tint_color[2], 1.0)

    emit = nodes.new("ShaderNodeEmission")
    emit.location = (420, 0)
    emit.inputs["Strength"].default_value = bg_src.inputs["Strength"].default_value

    links.new(geom.outputs["Position"], sep.inputs["Vector"])
    links.new(sep.outputs["X"], comb.inputs["X"])
    links.new(sep.outputs["Y"], comb.inputs["Y"])
    links.new(comb.outputs["Vector"], norm.inputs["Vector"])
    links.new(norm.outputs["Vector"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env_blur.inputs["Vector"])
    links.new(env.outputs["Color"], mix.inputs["Color1"])
    links.new(env_blur.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], hue_sat.inputs["Color"])
    links.new(hue_sat.outputs["Color"], tint_mix.inputs["Color1"])
    links.new(tint_mix.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    return mat


def _apply_fake_ground(
    context,
    settings,
    img: bpy.types.Image,
    mapping_node: bpy.types.Node,
    mix_node: bpy.types.Node,
    tint_mix_node: bpy.types.Node,
    hue_sat_node: bpy.types.Node,
    bg_node: bpy.types.Node,
):
    obj = _ensure_fake_ground_object(context)
    mat = _rebuild_fake_ground_material(
        img,
        mapping_src=mapping_node,
        mix_src=mix_node,
        tint_mix_src=tint_mix_node,
        hue_sat_src=hue_sat_node,
        bg_src=bg_node,
        lift=settings.fake_ground_lift,
    )
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    obj.location = (0.0, 0.0, float(settings.fake_ground_z_offset))
    s = float(settings.fake_ground_size) / 2.0
    obj.scale = (s, s, 1.0)

    obj.hide_viewport = False
    obj.hide_render = False


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

    img = nodes["env"].image
    if settings.fake_ground and img is not None:
        _apply_fake_ground(
            context,
            settings,
            img,
            nodes["mapping"],
            nodes["mix"],
            nodes["tint_mix"],
            nodes["hue_sat"],
            nodes["bg"],
        )
    else:
        _set_fake_ground_visible(False)


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
        mix_node = nodes["mix"]
        tint_mix_node = nodes["tint_mix"]
        hue_sat_node = nodes["hue_sat"]
        bg_node = nodes["bg"]
        mapping_node = nodes["mapping"]

        img = bpy.data.images.load(image_path, check_existing=True)
        _set_env_image_colorspace(img)
        env_node.image = img
        env_blur_node.image = img

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

        if settings.fake_ground:
            _apply_fake_ground(
                context,
                settings,
                img,
                mapping_node,
                mix_node,
                tint_mix_node,
                hue_sat_node,
                bg_node,
            )
        else:
            _set_fake_ground_visible(False)
    except Exception as e:
        return False, f"Failed to apply HDRI: {e}"
    return True, "HDRI applied to World."


_PLACEMENT_PREVIEW_IMAGE_NAME = "HDRI_Placement_Preview"


def _clampf(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _coverage_to_hfov_deg(coverage: float) -> float:
    fov = float(coverage) * 212.5
    return _clampf(fov, 35.0, 140.0)


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

    hfov_deg = _coverage_to_hfov_deg(coverage)
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


class HDRI_API_Preferences(AddonPreferences):
    bl_idname = __name__

    api_base_url: StringProperty(
        name="API Base URL",
        description="HDRI API root (no trailing slash), e.g. http://127.0.0.1:8000 — must match where uvicorn runs",
        default="http://127.0.0.1:8000",
    )
    api_key: StringProperty(
        name="API Key (optional)",
        description="Sent as Authorization: Bearer <key>",
        default="",
        subtype="PASSWORD",
    )
    timeout_s: FloatProperty(
        name="Timeout (seconds)",
        default=60.0,
        min=5.0,
        max=600.0,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "api_base_url")
        layout.prop(self, "api_key")
        layout.prop(self, "timeout_s")


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
            ("2048x1024", "2048x1024", "Default local ComfyUI target"),
            ("4096x2048", "4096x2048", "High resolution (heavy GPU load)"),
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
        description="Mixes in a blurred copy of the HDRI for softer lighting",
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
            "Adds a large emissive floor that projects the lower part of the HDRI so the "
            "environment reads as 3D ground + sky instead of a floating bubble"
        ),
        default=False,
        update=_update_look_controls,
    )
    fake_ground_size: FloatProperty(
        name="Ground size",
        description="Edge length of the ground plane (scene units)",
        default=100.0,
        min=1.0,
        soft_max=1000.0,
        update=_update_look_controls,
    )
    fake_ground_z_offset: FloatProperty(
        name="Ground Z",
        description="Height of the plane (Z-up). Slightly below 0 avoids z-fighting with the grid",
        default=-0.01,
        min=-1000.0,
        max=1000.0,
        update=_update_look_controls,
    )
    fake_ground_lift: FloatProperty(
        name="Projection lift",
        description=(
            "Virtual Z used when sampling the panorama from the plane (higher = horizon "
            "reaches the edges sooner; tweak if the floor looks too sky-like or too dark)"
        ),
        default=1.0,
        min=0.01,
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
        name="Panorama prompt",
        description="Prompt for your panorama worker (http_json). Empty = server/worker defaults",
        default="",
    )
    panorama_negative_prompt: StringProperty(
        name="Negative prompt",
        description="Negative prompt for the panorama worker (optional)",
        default="",
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
        description="How much panorama width the source image should occupy",
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
        description="Optional explicit horizontal FOV override (0 = use Placement Scale)",
        default=0.0,
        min=0.0,
        max=179.0,
        update=_update_placement_controls,
    )
    seam_fix: BoolProperty(
        name="Seam Fix",
        description="Worker post-blend at ERP left/right wrap (can soften real detail; leave off if panorama is already good)",
        default=False,
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
            ("ai_fast", "AI Fast", "Use server-side AI HDR reconstruction (recommended)"),
            ("comfyui_hdr", "ComfyUI HDR", "Run HDR restoration inside the ComfyUI worker workflow"),
            ("heuristic", "Heuristic", "Legacy heuristic HDR lift"),
            ("off", "Off", "Flat linear export (least boosted)"),
        ],
        default="ai_fast",
    )
    hdr_exposure_bias: FloatProperty(
        name="HDR Exposure Bias (EV)",
        description="Post-HDR exposure bias applied by server AI/heuristic stage",
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
    bl_description = "Drag the source image on a 2:1 panorama preview and scale/rotate it"
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
                    (x, cy),
                    (x + w, cy),
                )
            },
        )
        outline.uniform_float("color", (1.0, 1.0, 1.0, 0.35))
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
        if s.panorama_negative_prompt.strip():
            payload["panorama_negative_prompt"] = s.panorama_negative_prompt.strip()
        if s.panorama_seed >= 0:
            payload["panorama_seed"] = int(s.panorama_seed)
        if s.panorama_strength >= 0.0:
            payload["panorama_strength"] = float(s.panorama_strength)
        payload["erp_layout_mode"] = s.erp_layout_mode
        payload["reference_coverage"] = float(s.placement_coverage)
        payload["placement_coverage"] = float(s.placement_coverage)
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
        fg.prop(s, "fake_ground_size")
        fg.prop(s, "fake_ground_z_offset")
        fg.prop(s, "fake_ground_lift")

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

        box.label(text="Panorama worker fields (http_json only)")
        box.label(text="Prompts go to your worker; server may still HDR-tonemap after.", icon="INFO")
        box.label(text="Local mode: run API server + worker + ComfyUI before Generate.", icon="INFO")
        col2 = box.column(align=True)
        col2.prop(s, "panorama_prompt")
        col2.prop(s, "panorama_negative_prompt")
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


def unregister():
    _clear_library_previews()
    if hasattr(bpy.types.Scene, "hdri_api_settings"):
        del bpy.types.Scene.hdri_api_settings
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

