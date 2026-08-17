#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _import_obj(path: Path):
    import bpy

    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))
    objects = [obj for obj in bpy.data.objects if obj not in before and getattr(obj, "type", "") == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh objects were imported from {path}")
    return objects


def _object_bounds(objects):
    from mathutils import Vector

    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world.x)
            mins.y = min(mins.y, world.y)
            mins.z = min(mins.z, world.z)
            maxs.x = max(maxs.x, world.x)
            maxs.y = max(maxs.y, world.y)
            maxs.z = max(maxs.z, world.z)
    return mins, maxs


def _normalize_group(objects, target_x: float, target_height: float = 2.2) -> None:
    from mathutils import Matrix, Vector

    mins, maxs = _object_bounds(objects)
    center = (mins + maxs) * 0.5
    height = max(maxs.z - mins.z, 1e-6)
    scale = target_height / height
    target_center = Vector((target_x, 0.0, 0.0))
    transform = Matrix.Translation(target_center) @ Matrix.Scale(scale, 4) @ Matrix.Translation(-center)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world


def _make_material(name: str, color: tuple[float, float, float, float]):
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.55
    return mat


def _assign_material(objects, material) -> None:
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def _add_label(text: str, x: float) -> None:
    import bpy

    bpy.ops.object.text_add(location=(x - 0.42, -0.03, -1.42), rotation=(math.radians(78), 0.0, 0.0))
    obj = bpy.context.object
    obj.name = f"label_{text}"
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.size = 0.14
    mat = _make_material(f"mat_{text}", (0.02, 0.02, 0.02, 1.0))
    obj.data.materials.append(mat)


def _look_at(obj, target) -> None:
    from mathutils import Vector

    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a PNG preview of a prepared THuman appearance asset.")
    parser.add_argument("--asset-dir", required=True, help="Prepared subject directory containing asset_manifest.json.")
    parser.add_argument("--output-png", required=True, help="PNG path to write.")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    import bpy

    args = parse_args(argv)
    asset_dir = Path(args.asset_dir).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(asset_dir / "asset_manifest.json")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    smplx_objects = _import_obj(Path(manifest["smplx_mesh"]))
    _assign_material(smplx_objects, _make_material("mat_smplx_fit", (0.74, 0.77, 0.82, 1.0)))
    _normalize_group(smplx_objects, target_x=-1.35)
    _add_label("SMPL-X fit", -1.35)

    textured_mesh = manifest.get("textured_mesh") or ""
    if textured_mesh:
        textured_objects = _import_obj(Path(textured_mesh))
        _normalize_group(textured_objects, target_x=1.35)
        _add_label("THuman textured scan", 1.35)

    bpy.ops.mesh.primitive_plane_add(size=6.2, location=(0.0, 0.0, -1.18))
    plane = bpy.context.object
    plane.name = "matte_floor"
    plane.data.materials.append(_make_material("mat_floor", (0.82, 0.84, 0.82, 1.0)))

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 4.0))
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = 600.0
    light.data.size = 4.0

    bpy.ops.object.camera_add(location=(0.0, -5.2, 0.55))
    camera = bpy.context.object
    _look_at(camera, (0.0, 0.0, 0.0))
    camera.data.lens = 45
    camera.data.dof.use_dof = False
    bpy.context.scene.camera = camera

    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
    if hasattr(bpy.context.scene, "eevee"):
        bpy.context.scene.eevee.taa_render_samples = 64
    bpy.context.scene.render.resolution_x = int(args.width)
    bpy.context.scene.render.resolution_y = int(args.height)
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.world.color = (1.0, 1.0, 1.0)
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"
    bpy.context.scene.render.filepath = str(output_png)
    bpy.ops.render.render(write_still=True)
    print(json.dumps({"output_png": str(output_png), "asset_dir": str(asset_dir)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    raise SystemExit(main(argv))
