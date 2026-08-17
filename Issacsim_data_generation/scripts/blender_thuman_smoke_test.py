#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    after = [obj for obj in bpy.data.objects if obj not in before and getattr(obj, "type", "") == "MESH"]
    if not after:
        raise RuntimeError(f"Blender did not import any mesh objects from {path}")
    return after


def _mesh_stats(objects) -> dict[str, object]:
    vertex_count = sum(len(obj.data.vertices) for obj in objects)
    face_count = sum(len(obj.data.polygons) for obj in objects)
    material_count = sum(len(obj.data.materials) for obj in objects)
    return {
        "objects": len(objects),
        "vertices": vertex_count,
        "faces": face_count,
        "materials": material_count,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless Blender import smoke test for a prepared THuman appearance asset.")
    parser.add_argument("--asset-dir", required=True, help="Prepared subject directory containing asset_manifest.json.")
    parser.add_argument("--output-json", default="", help="Optional smoke-test report path.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    asset_dir = Path(args.asset_dir).expanduser().resolve()
    manifest_path = asset_dir / "asset_manifest.json"
    manifest = _load_json(manifest_path)

    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    smplx_mesh = Path(manifest["smplx_mesh"])
    smplx_objects = _import_obj(smplx_mesh)
    report = {
        "asset_dir": str(asset_dir),
        "smplx_mesh": str(smplx_mesh),
        "smplx": _mesh_stats(smplx_objects),
    }
    if report["smplx"]["vertices"] <= 0 or report["smplx"]["faces"] <= 0:
        raise RuntimeError(f"Imported SMPL-X mesh is empty: {smplx_mesh}")

    textured_mesh = manifest.get("textured_mesh") or ""
    if textured_mesh:
        textured_objects = _import_obj(Path(textured_mesh))
        report["textured_mesh"] = textured_mesh
        report["textured"] = _mesh_stats(textured_objects)
        if report["textured"]["vertices"] <= 0 or report["textured"]["faces"] <= 0:
            raise RuntimeError(f"Imported textured mesh is empty: {textured_mesh}")

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else asset_dir / "blender_smoke_test.json"
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
