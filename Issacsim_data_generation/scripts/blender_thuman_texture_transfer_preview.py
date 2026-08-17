#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _import_obj(path: Path, name_prefix: str):
    import bpy

    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))
    objects = [obj for obj in bpy.data.objects if obj not in before and getattr(obj, "type", "") == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh objects were imported from {path}")
    for obj in objects:
        obj.name = f"{name_prefix}_{obj.name}"
    return objects


def _single_mesh(objects, label: str):
    if len(objects) != 1:
        raise RuntimeError(f"{label} expected one mesh object, got {len(objects)}")
    return objects[0]


def _coords(obj):
    import numpy as np

    return np.asarray([obj.matrix_world @ vertex.co for vertex in obj.data.vertices], dtype=float)


def _find_texture_file(textured_mesh: Path) -> Path:
    for name in ("material0.jpeg", "material0.jpg", "material0.png"):
        path = textured_mesh.parent / name
        if path.exists():
            return path
    images = sorted(path for path in textured_mesh.parent.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"No texture image found beside {textured_mesh}")
    return images[0]


def _load_image_pixels(image_path: Path):
    import bpy
    import numpy as np

    image = bpy.data.images.load(str(image_path))
    width, height = image.size
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    return pixels, int(width), int(height)


def _sample_image_pixels(pixels, width: int, height: int, uv):
    u = float(uv[0]) % 1.0
    v = float(uv[1]) % 1.0
    x = int(round(u * (width - 1)))
    y = int(round(v * (height - 1)))
    y = max(0, min(height - 1, y))
    x = max(0, min(width - 1, x))
    return pixels[y, x, :4]


def _source_scan_vertex_colors(source_scan, texture_path: Path):
    import numpy as np

    uv_layer = source_scan.data.uv_layers.active
    if uv_layer is None:
        raise RuntimeError("Source scan has no UV layer; cannot sample texture.")
    pixels, width, height = _load_image_pixels(texture_path)
    sums = np.zeros((len(source_scan.data.vertices), 4), dtype=np.float64)
    counts = np.zeros(len(source_scan.data.vertices), dtype=np.int32)
    for loop in source_scan.data.loops:
        uv = uv_layer.data[loop.index].uv
        color = _sample_image_pixels(pixels, width, height, uv)
        sums[loop.vertex_index] += color
        counts[loop.vertex_index] += 1
    counts[counts == 0] = 1
    colors = sums / counts[:, None]
    colors[:, 3] = 1.0
    return colors.astype(np.float32)


def _transfer_colors_to_target_smplx(source_scan, source_smplx, target_smplx, texture_path: Path):
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.kdtree import KDTree

    scan_xyz = _coords(source_scan)
    source_smplx_xyz = _coords(source_smplx)
    scan_colors = _source_scan_vertex_colors(source_scan, texture_path)
    kd = KDTree(len(scan_xyz))
    for idx, xyz in enumerate(scan_xyz):
        kd.insert(Vector(xyz), idx)
    kd.balance()

    vertex_colors = np.zeros((len(target_smplx.data.vertices), 4), dtype=np.float32)
    nearest_distance = np.zeros(len(target_smplx.data.vertices), dtype=np.float32)
    for idx, xyz in enumerate(source_smplx_xyz):
        _co, nearest_idx, dist = kd.find(Vector(xyz))
        vertex_colors[idx] = scan_colors[int(nearest_idx)]
        nearest_distance[idx] = float(dist)

    color_attr = target_smplx.data.color_attributes.new(name="thuman_transfer_color", type="BYTE_COLOR", domain="CORNER")
    for polygon in target_smplx.data.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = target_smplx.data.loops[loop_index].vertex_index
            color_attr.data[loop_index].color = vertex_colors[vertex_index]

    mat = bpy.data.materials.new("mat_transferred_thuman_vertex_color")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    color_node = nodes.new(type="ShaderNodeVertexColor")
    color_node.layer_name = "thuman_transfer_color"
    if bsdf is not None:
        mat.node_tree.links.new(color_node.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.58
    target_smplx.data.materials.clear()
    target_smplx.data.materials.append(mat)

    return {
        "source_scan_vertices": int(len(scan_xyz)),
        "target_smplx_vertices": int(len(target_smplx.data.vertices)),
        "source_smplx_to_scan_distance_m": _quantiles(nearest_distance),
        "texture": str(texture_path),
    }


def _quantiles(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


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


def _robust_bounds(objects):
    import numpy as np
    from mathutils import Vector

    coords = []
    for obj in objects:
        coords.extend([obj.matrix_world @ vertex.co for vertex in obj.data.vertices])
    arr = np.asarray(coords, dtype=float)
    return Vector(np.quantile(arr, 0.01, axis=0)), Vector(np.quantile(arr, 0.99, axis=0))


def _normalize_group(objects, target_x: float, target_height: float = 1.85) -> None:
    from mathutils import Matrix, Vector

    mins, maxs = _robust_bounds(objects)
    center = (mins + maxs) * 0.5
    height = max(maxs.z - mins.z, 1e-6)
    transform = Matrix.Translation(Vector((target_x, 0.0, 0.0))) @ Matrix.Scale(target_height / height, 4) @ Matrix.Translation(-center)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world


def _add_label(text: str, x: float) -> None:
    import bpy

    bpy.ops.object.text_add(location=(x, -0.05, -1.38), rotation=(math.radians(75), 0.0, 0.0))
    obj = bpy.context.object
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.size = 0.13
    obj.data.materials.append(_make_material(f"mat_label_{text}", (0.02, 0.02, 0.02, 1.0)))


def _look_at(obj, target) -> None:
    from mathutils import Vector

    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer one THuman scan texture/color to another SMPL-X pose.")
    parser.add_argument("--source-asset-dir", required=True)
    parser.add_argument("--target-asset-dir", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    import bpy

    args = parse_args(argv)
    source_dir = Path(args.source_asset_dir).expanduser().resolve()
    target_dir = Path(args.target_asset_dir).expanduser().resolve()
    output_png = Path(args.output_png).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    source_manifest = _load_json(source_dir / "asset_manifest.json")
    target_manifest = _load_json(target_dir / "asset_manifest.json")
    textured_mesh = Path(source_manifest.get("textured_mesh") or "")
    if not textured_mesh.exists():
        raise RuntimeError(f"Source asset has no textured mesh: {source_dir}")
    texture_path = _find_texture_file(textured_mesh)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    source_scan = _single_mesh(_import_obj(textured_mesh, "source_scan"), "source scan")
    source_smplx = _single_mesh(_import_obj(Path(source_manifest["smplx_mesh"]), "source_smplx"), "source SMPL-X")
    target_plain = _single_mesh(_import_obj(Path(target_manifest["smplx_mesh"]), "target_plain"), "target SMPL-X plain")
    target_transfer = _single_mesh(_import_obj(Path(target_manifest["smplx_mesh"]), "target_transfer"), "target SMPL-X transfer")

    _assign_material([target_plain], _make_material("mat_plain_target", (0.78, 0.80, 0.84, 1.0)))
    transfer_stats = _transfer_colors_to_target_smplx(source_scan, source_smplx, target_transfer, texture_path)
    source_smplx.hide_render = True
    source_smplx.hide_viewport = True

    _normalize_group([source_scan], target_x=-2.1)
    _normalize_group([target_transfer], target_x=0.0)
    _normalize_group([target_plain], target_x=2.1)
    _add_label("0000 textured scan", -2.1)
    _add_label("texture transfer to 0001", 0.0)
    _add_label("0001 SMPL-X target", 2.1)

    bpy.ops.mesh.primitive_plane_add(size=7.0, location=(0.0, 0.0, -1.18))
    floor = bpy.context.object
    floor.data.materials.append(_make_material("mat_floor", (0.82, 0.84, 0.82, 1.0)))

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.4, 4.0))
    light = bpy.context.object
    light.data.energy = 750.0
    light.data.size = 4.8

    bpy.ops.object.camera_add(location=(0.0, -6.8, 0.42))
    camera = bpy.context.object
    _look_at(camera, (0.0, 0.0, 0.0))
    camera.data.lens = 34
    bpy.context.scene.camera = camera

    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
    if hasattr(bpy.context.scene, "eevee"):
        bpy.context.scene.eevee.taa_render_samples = 64
    bpy.context.scene.render.resolution_x = int(args.width)
    bpy.context.scene.render.resolution_y = int(args.height)
    bpy.context.scene.world.color = (1.0, 1.0, 1.0)
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"
    bpy.context.scene.render.filepath = str(output_png)
    bpy.ops.render.render(write_still=True)

    report = {
        "source_asset_dir": str(source_dir),
        "target_asset_dir": str(target_dir),
        "output_png": str(output_png),
        "method": "nearest scan vertex color transfer from source scan to target SMPL-X via source SMPL-X correspondence",
        "stats": transfer_stats,
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    raise SystemExit(main(argv))
