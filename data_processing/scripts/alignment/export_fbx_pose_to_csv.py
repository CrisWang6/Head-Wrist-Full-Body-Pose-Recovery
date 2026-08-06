"""
Export FBX armature pose animation to CSV.

Run with Blender, for example:

blender --background --python export_fbx_pose_to_csv.py -- ^
  --fbx "C:\\Users\\hand\\Desktop\\Dataset\\0714\\002\\SIK_Actor_01_20260714_121232.fbx" ^
  --outdir "C:\\Users\\hand\\Desktop\\Dataset\\0714\\002\\fbx_pose_export"

Outputs:
  - pose_frames.csv: one row per frame per bone.
  - skeleton_bones.csv: rest-pose hierarchy and bone endpoints.
  - metadata.json: frame range, fps, armature names, source path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    import bpy
except ImportError as exc:  # Allows a clear message if run with normal Python.
    raise SystemExit(
        "This script must be run by Blender's Python, not normal python.exe.\n"
        "Example: blender --background --python export_fbx_pose_to_csv.py -- --fbx input.fbx"
    ) from exc


DEFAULT_FBX = r"C:\Users\hand\Desktop\Dataset\0714\002\SIK_Actor_01_20260714_121232.fbx"


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", default=DEFAULT_FBX, help="Input FBX file path.")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: <fbx parent>/fbx_pose_export",
    )
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--only-deform-bones",
        action="store_true",
        help="Export only bones marked as deform bones.",
    )
    return parser.parse_args(argv)


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_fbx(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    bpy.ops.import_scene.fbx(filepath=str(path))


def get_armatures() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]


def frame_range_from_scene(args: argparse.Namespace) -> tuple[int, int]:
    scene = bpy.context.scene
    start = args.frame_start if args.frame_start is not None else int(scene.frame_start)
    end = args.frame_end if args.frame_end is not None else int(scene.frame_end)
    if end < start:
        raise ValueError(f"Invalid frame range: {start}..{end}")
    return start, end


def fps_from_scene() -> float:
    scene = bpy.context.scene
    return float(scene.render.fps) / float(scene.render.fps_base)


def vec3(v) -> tuple[float, float, float]:
    return (float(v.x), float(v.y), float(v.z))


def quat4(q) -> tuple[float, float, float, float]:
    return (float(q.w), float(q.x), float(q.y), float(q.z))


def euler_deg(e) -> tuple[float, float, float]:
    return (math.degrees(float(e.x)), math.degrees(float(e.y)), math.degrees(float(e.z)))


def matrix_flat(m) -> list[float]:
    return [float(m[r][c]) for r in range(4) for c in range(4)]


def fmt(x: float) -> str:
    return f"{x:.9f}"


def should_export_bone(pose_bone, only_deform: bool) -> bool:
    if not only_deform:
        return True
    return bool(pose_bone.bone.use_deform)


def write_skeleton_csv(path: Path, armatures, only_deform: bool) -> int:
    fieldnames = [
        "armature",
        "bone",
        "parent",
        "use_deform",
        "rest_head_local_x",
        "rest_head_local_y",
        "rest_head_local_z",
        "rest_tail_local_x",
        "rest_tail_local_y",
        "rest_tail_local_z",
        "rest_length",
    ] + [f"rest_matrix_local_{i:02d}" for i in range(16)]

    rows = 0
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for arm in armatures:
            for bone in arm.data.bones:
                if only_deform and not bone.use_deform:
                    continue
                row = {
                    "armature": arm.name,
                    "bone": bone.name,
                    "parent": bone.parent.name if bone.parent else "",
                    "use_deform": int(bool(bone.use_deform)),
                    "rest_length": fmt(float(bone.length)),
                }
                for prefix, values in [
                    ("rest_head_local", vec3(bone.head_local)),
                    ("rest_tail_local", vec3(bone.tail_local)),
                ]:
                    row[f"{prefix}_x"] = fmt(values[0])
                    row[f"{prefix}_y"] = fmt(values[1])
                    row[f"{prefix}_z"] = fmt(values[2])
                for i, value in enumerate(matrix_flat(bone.matrix_local)):
                    row[f"rest_matrix_local_{i:02d}"] = fmt(value)
                writer.writerow(row)
                rows += 1
    return rows


def write_pose_csv(
    path: Path,
    armatures,
    frame_start: int,
    frame_end: int,
    frame_step: int,
    fps: float,
    only_deform: bool,
) -> int:
    fieldnames = [
        "frame",
        "time_sec",
        "armature",
        "bone",
        "parent",
        "use_deform",
        "world_head_x",
        "world_head_y",
        "world_head_z",
        "world_tail_x",
        "world_tail_y",
        "world_tail_z",
        "world_loc_x",
        "world_loc_y",
        "world_loc_z",
        "world_quat_w",
        "world_quat_x",
        "world_quat_y",
        "world_quat_z",
        "world_euler_xyz_deg_x",
        "world_euler_xyz_deg_y",
        "world_euler_xyz_deg_z",
        "local_basis_loc_x",
        "local_basis_loc_y",
        "local_basis_loc_z",
        "local_basis_quat_w",
        "local_basis_quat_x",
        "local_basis_quat_y",
        "local_basis_quat_z",
        "local_basis_euler_xyz_deg_x",
        "local_basis_euler_xyz_deg_y",
        "local_basis_euler_xyz_deg_z",
    ] + [f"world_matrix_{i:02d}" for i in range(16)]

    rows = 0
    scene = bpy.context.scene
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frame in range(frame_start, frame_end + 1, frame_step):
            scene.frame_set(frame)
            time_sec = (frame - frame_start) / fps if fps else 0.0

            for arm in armatures:
                arm_world = arm.matrix_world.copy()
                for pose_bone in arm.pose.bones:
                    if not should_export_bone(pose_bone, only_deform):
                        continue

                    bone = pose_bone.bone
                    world_matrix = arm_world @ pose_bone.matrix
                    world_loc = world_matrix.to_translation()
                    world_quat = world_matrix.to_quaternion()
                    world_euler = world_quat.to_euler("XYZ")
                    local_loc = pose_bone.matrix_basis.to_translation()
                    local_quat = pose_bone.matrix_basis.to_quaternion()
                    local_euler = local_quat.to_euler("XYZ")
                    world_head = arm_world @ pose_bone.head
                    world_tail = arm_world @ pose_bone.tail

                    row = {
                        "frame": frame,
                        "time_sec": fmt(time_sec),
                        "armature": arm.name,
                        "bone": pose_bone.name,
                        "parent": pose_bone.parent.name if pose_bone.parent else "",
                        "use_deform": int(bool(bone.use_deform)),
                    }

                    for prefix, values in [
                        ("world_head", vec3(world_head)),
                        ("world_tail", vec3(world_tail)),
                        ("world_loc", vec3(world_loc)),
                    ]:
                        row[f"{prefix}_x"] = fmt(values[0])
                        row[f"{prefix}_y"] = fmt(values[1])
                        row[f"{prefix}_z"] = fmt(values[2])

                    for prefix, values in [
                        ("world_quat", quat4(world_quat)),
                        ("local_basis_quat", quat4(local_quat)),
                    ]:
                        row[f"{prefix}_w"] = fmt(values[0])
                        row[f"{prefix}_x"] = fmt(values[1])
                        row[f"{prefix}_y"] = fmt(values[2])
                        row[f"{prefix}_z"] = fmt(values[3])

                    for prefix, values in [
                        ("world_euler_xyz_deg", euler_deg(world_euler)),
                        ("local_basis_euler_xyz_deg", euler_deg(local_euler)),
                        ("local_basis_loc", vec3(local_loc)),
                    ]:
                        row[f"{prefix}_x"] = fmt(values[0])
                        row[f"{prefix}_y"] = fmt(values[1])
                        row[f"{prefix}_z"] = fmt(values[2])

                    for i, value in enumerate(matrix_flat(world_matrix)):
                        row[f"world_matrix_{i:02d}"] = fmt(value)

                    writer.writerow(row)
                    rows += 1
    return rows


def write_metadata(
    path: Path,
    source_fbx: Path,
    outdir: Path,
    armatures,
    frame_start: int,
    frame_end: int,
    frame_step: int,
    fps: float,
    skeleton_rows: int,
    pose_rows: int,
    only_deform: bool,
) -> None:
    payload = {
        "source_fbx": str(source_fbx),
        "output_dir": str(outdir),
        "fps": fps,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_step": frame_step,
        "armatures": [arm.name for arm in armatures],
        "skeleton_rows": skeleton_rows,
        "pose_rows": pose_rows,
        "only_deform_bones": only_deform,
        "pose_csv_semantics": {
            "world_*": "final evaluated bone transform in world coordinates",
            "local_basis_*": "pose_bone.matrix_basis, local pose delta relative to rest/parent",
            "time_sec": "(frame - frame_start) / scene_fps",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_fbx = Path(args.fbx)
    outdir = Path(args.outdir) if args.outdir else source_fbx.parent / "fbx_pose_export"
    outdir.mkdir(parents=True, exist_ok=True)

    clean_scene()
    import_fbx(source_fbx)
    armatures = get_armatures()
    if not armatures:
        raise RuntimeError("No armature object found in FBX.")

    frame_start, frame_end = frame_range_from_scene(args)
    fps = fps_from_scene()

    skeleton_rows = write_skeleton_csv(
        outdir / "skeleton_bones.csv", armatures, args.only_deform_bones
    )
    pose_rows = write_pose_csv(
        outdir / "pose_frames.csv",
        armatures,
        frame_start,
        frame_end,
        max(1, args.frame_step),
        fps,
        args.only_deform_bones,
    )
    write_metadata(
        outdir / "metadata.json",
        source_fbx,
        outdir,
        armatures,
        frame_start,
        frame_end,
        max(1, args.frame_step),
        fps,
        skeleton_rows,
        pose_rows,
        args.only_deform_bones,
    )

    print(f"source_fbx={source_fbx}")
    print(f"output_dir={outdir}")
    print(f"armatures={[arm.name for arm in armatures]}")
    print(f"fps={fps}")
    print(f"frame_range={frame_start}..{frame_end}")
    print(f"skeleton_rows={skeleton_rows}")
    print(f"pose_rows={pose_rows}")


if __name__ == "__main__":
    main()
