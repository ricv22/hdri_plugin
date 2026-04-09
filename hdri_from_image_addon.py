bl_info = {
    "name": "Photo → HDRI World (API)",
    "author": "Cursor AI",
    "version": (0, 1, 4),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > HDRI",
    "description": "Upload a photo to an API, get a 2:1 HDRI (.hdr/.exr), apply to World lighting (Cycles).",
    "category": "Lighting",
}

import base64
import json
import os
import tempfile
import time
import urllib.request
import urllib.error

import bpy
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
        description="How much panorama width the source image should occupy on control canvas",
        default=0.60,
        min=0.15,
        max=0.85,
    )
    seam_fix: BoolProperty(
        name="Seam Fix",
        description="Enable seam smoothing/fix step in worker",
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


class HDRI_OT_apply_from_api(Operator):
    bl_idname = "hdri.apply_from_api"
    bl_label = "Generate & Apply HDRI"
    bl_options = {"REGISTER"}

    @staticmethod
    def _resolution_pair(value: str) -> tuple[int, int]:
        try:
            w_s, h_s = value.lower().split("x", 1)
            return int(w_s), int(h_s)
        except Exception:
            return 2048, 1024

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
        payload["reference_coverage"] = float(s.reference_coverage)
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

        poll_url = f"{base}/v1/jobs/{job_id}"
        poll_interval_s = 2.0
        deadline = time.time() + float(prefs.timeout_s)
        resp = None
        while time.time() < deadline:
            try:
                status_resp = _http_get_json(poll_url, headers=headers, timeout_s=min(20, int(prefs.timeout_s)))
            except Exception as e:
                self.report({"ERROR"}, f"Polling failed: {e}")
                return {"CANCELLED"}
            if not isinstance(status_resp, dict):
                self.report({"ERROR"}, "Invalid job status response.")
                return {"CANCELLED"}
            status = str(status_resp.get("status", "")).strip().lower()
            if status:
                s.current_job_status = status
            if status in {"queued", "running"}:
                time.sleep(poll_interval_s)
                continue
            if status == "failed":
                err = str(status_resp.get("error", "Job failed without details."))
                s.last_job_error = err
                self.report({"ERROR"}, f"Job failed: {err[:300]}")
                acct = _safe_get_account(base, headers, timeout_s=10)
                if acct and "tokens_remaining" in acct:
                    try:
                        s.tokens_remaining = int(acct["tokens_remaining"])
                    except Exception:
                        pass
                return {"CANCELLED"}
            if status == "succeeded":
                resp = status_resp
                break
            self.report({"ERROR"}, f"Unexpected job status: {status or '(missing)'}")
            return {"CANCELLED"}

        if resp is None:
            self.report({"ERROR"}, "Job polling timed out.")
            return {"CANCELLED"}

        # Safety net: block accidental resize responses from misconfigured servers.
        if isinstance(resp, dict) and str(resp.get("panorama_mode", "")).strip().lower() == "resize":
            s.last_panorama_mode = "resize"
            self.report(
                {"ERROR"},
                "API returned panorama_mode=resize (stretched source image). Fix server mode to http_json and retry.",
            )
            return {"CANCELLED"}

        # Prefer signed URL: hdri_url (Radiance .hdr) or exr_url (same URL or .exr)
        download_url = None
        if isinstance(resp, dict):
            download_url = resp.get("hdri_url") or resp.get("exr_url")

        file_bytes = None
        if download_url:
            try:
                file_bytes = _download_bytes(download_url, headers=headers, timeout_s=int(prefs.timeout_s))
            except Exception as e:
                self.report({"ERROR"}, f"Failed to download HDRI: {e}")
                return {"CANCELLED"}
        elif isinstance(resp, dict) and resp.get("exr_base64"):
            try:
                file_bytes = base64.b64decode(resp["exr_base64"])
            except Exception as e:
                self.report({"ERROR"}, f"Bad exr_base64: {e}")
                return {"CANCELLED"}
        else:
            self.report({"ERROR"}, "API response missing hdri_url, exr_url, or exr_base64.")
            return {"CANCELLED"}

        # Infer extension from URL path (Radiance .hdr vs OpenEXR .exr)
        suffix = ".hdr"
        if download_url:
            path_part = download_url.split("?", 1)[0].lower()
            if path_part.endswith(".exr"):
                suffix = ".exr"
            elif path_part.endswith(".hdr"):
                suffix = ".hdr"
        elif isinstance(resp, dict) and resp.get("exr_base64"):
            suffix = ".exr"

        # Blender needs a file path for Environment Texture. We'll write to a temp file.
        # (Not “saving to library”; just a temp cache file for this session.)
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="hdri_api_", suffix=suffix)
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to write temp HDRI file: {e}")
            return {"CANCELLED"}

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

            # Load image into Blender and assign to env texture
            img = bpy.data.images.load(tmp_path, check_existing=True)
            # Blender 4.x+ renamed "Linear" — use scene-linear / linear Rec.709 for HDRI lighting
            _set_env_image_colorspace(img)
            env_node.image = img
            env_blur_node.image = img

            # Apply rotation/look controls and keep fake ground in sync.
            _apply_look_controls_to_nodes(
                s,
                mapping_node,
                mix_node,
                tint_mix_node,
                hue_sat_node,
                bg_node,
            )

            if s.add_preview_sphere:
                _ensure_preview_sphere()

            if s.fake_ground:
                _apply_fake_ground(
                    context,
                    s,
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
            self.report({"ERROR"}, f"Failed to apply HDRI: {e}")
            return {"CANCELLED"}

        mode = resp.get("panorama_mode", "") if isinstance(resp, dict) else ""
        if mode:
            s.last_panorama_mode = str(mode)
        s.current_job_status = "succeeded"
        acct = _safe_get_account(base, headers, timeout_s=10)
        if acct and "tokens_remaining" in acct:
            try:
                s.tokens_remaining = int(acct["tokens_remaining"])
            except Exception:
                pass
        if mode:
            if mode == "resize":
                self.report(
                    {"INFO"},
                    "HDRI applied (panorama_mode=resize — photo stretched to 2:1; prompts unused until API uses PANORAMA_MODE=http_json).",
                )
            else:
                self.report({"INFO"}, f"HDRI applied (panorama_mode={mode}).")
        else:
            self.report({"INFO"}, "HDRI applied to World.")
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
        if s.current_job_id:
            box.label(text=f"Current job: {s.current_job_id}", icon="TIME")
        if s.current_job_status:
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
        col2.prop(s, "reference_coverage")
        col2.prop(s, "seam_fix")
        row2 = col2.row(align=True)
        row2.prop(s, "erp_canvas_width")
        row2.prop(s, "erp_canvas_height")
        col2.prop(s, "panorama_extra_json")
        col2.prop(s, "hdr_reconstruction_mode")
        col2.prop(s, "hdr_exposure_bias")

        col.separator()
        col.operator(HDRI_OT_apply_from_api.bl_idname, icon="WORLD")


classes = (
    HDRI_API_Preferences,
    HDRI_API_Settings,
    HDRI_OT_refresh_server_config,
    HDRI_OT_apply_from_api,
    HDRI_PT_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.hdri_api_settings = PointerProperty(type=HDRI_API_Settings)


def unregister():
    if hasattr(bpy.types.Scene, "hdri_api_settings"):
        del bpy.types.Scene.hdri_api_settings
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

