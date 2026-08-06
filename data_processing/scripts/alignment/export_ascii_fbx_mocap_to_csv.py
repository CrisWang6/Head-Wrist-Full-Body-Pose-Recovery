"""
Export mocap animation channels from an ASCII FBX file to CSV.

This is intended for Noitom-style ASCII FBX files where Blender's built-in FBX
importer refuses the file with "ASCII FBX files are not supported".

Outputs:
  - mocap_joints_long.csv: one row per frame per model/bone, including world pose.
  - mocap_joints_wide.csv: one row per frame, world/local columns grouped by bone.
  - mocap_pose_wide.csv: one row per frame, columns grouped by bone/channel.
  - mocap_skeleton.csv: model hierarchy and default local transforms.
  - mocap_metadata.json: parse summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_FBX = r"C:\Users\hand\Desktop\Dataset\0714\002\SIK_Actor_01_20260714_121232.fbx"
FBX_TIME_UNIT = 46186158000.0  # FBX KTime ticks per second.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", default=DEFAULT_FBX)
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--include-end-bones",
        action="store_true",
        help="Include bones whose name ends with _End in pose CSV outputs.",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Also write wide per-frame CSV. Long CSV is always written.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escape = False
    for i in range(open_index, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"No matching brace found at offset {open_index}")


def iter_blocks(text: str, header_re: re.Pattern[str]):
    for match in header_re.finditer(text):
        brace = text.find("{", match.end())
        if brace < 0:
            continue
        end = find_matching_brace(text, brace)
        yield match, text[brace + 1 : end]


def parse_float_triplet_from_property(block: str, prop_name: str) -> tuple[float, float, float]:
    pattern = re.compile(
        r'P:\s*"'
        + re.escape(prop_name)
        + r'"\s*,\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*'
        + r"([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)"
    )
    match = pattern.search(block)
    if not match:
        return (0.0, 0.0, 0.0)
    return tuple(float(match.group(i)) for i in range(1, 4))  # type: ignore[return-value]


def parse_time_settings(text: str) -> dict:
    def prop_number(name: str, default=None):
        m = re.search(r'P:\s*"' + re.escape(name) + r'".*?,\s*([-+0-9.eE]+)\s*$', text, re.M)
        return float(m.group(1)) if m else default

    return {
        "custom_frame_rate": prop_number("CustomFrameRate"),
        "timespan_start_ticks": prop_number("TimeSpanStart", 0.0),
        "timespan_stop_ticks": prop_number("TimeSpanStop", 0.0),
    }


def parse_models(text: str) -> dict[int, dict]:
    header = re.compile(r'Model:\s*(\d+),\s*"Model::([^"]*)",\s*"([^"]*)"')
    models: dict[int, dict] = {}
    for match, block in iter_blocks(text, header):
        model_id = int(match.group(1))
        name = match.group(2)
        kind = match.group(3)
        models[model_id] = {
            "id": model_id,
            "name": name,
            "kind": kind,
            "default_t": parse_float_triplet_from_property(block, "Lcl Translation"),
            "default_r": parse_float_triplet_from_property(block, "Lcl Rotation"),
            "parent_id": None,
            "parent_name": "",
        }
    return models


def parse_curve_nodes(text: str) -> dict[int, dict]:
    header = re.compile(r'AnimationCurveNode:\s*(\d+),\s*"AnimCurveNode::([^"]*)",\s*"[^"]*"')
    nodes: dict[int, dict] = {}
    for match, _block in iter_blocks(text, header):
        node_id = int(match.group(1))
        nodes[node_id] = {
            "id": node_id,
            "name": match.group(2),
            "model_id": None,
            "property": "",
            "curves": {},
        }
    return nodes


def parse_number_array(block: str, name: str, cast=float) -> list:
    match = re.search(name + r":\s*\*\d+\s*\{\s*a:\s*(.*?)\n\s*\}", block, re.S)
    if not match:
        return []
    payload = match.group(1).replace("\n", "")
    if not payload.strip():
        return []
    return [cast(x) for x in payload.split(",") if x.strip()]


def parse_curves(text: str) -> dict[int, dict]:
    header = re.compile(r'AnimationCurve:\s*(\d+),\s*"AnimCurve::[^"]*",\s*"[^"]*"')
    curves: dict[int, dict] = {}
    for match, block in iter_blocks(text, header):
        curve_id = int(match.group(1))
        ticks = parse_number_array(block, "KeyTime", int)
        values = parse_number_array(block, "KeyValueFloat", float)
        if len(values) != len(ticks):
            min_len = min(len(values), len(ticks))
            ticks = ticks[:min_len]
            values = values[:min_len]
        curves[curve_id] = {
            "id": curve_id,
            "ticks": ticks,
            "values": values,
            "node_id": None,
            "axis": "",
        }
    return curves


def parse_connections(text: str, models: dict[int, dict], nodes: dict[int, dict], curves: dict[int, dict]) -> None:
    conn_re = re.compile(r'C:\s*"([^"]+)",\s*(\d+),\s*(\d+)(?:,\s*"([^"]*)")?')
    for kind, child_s, parent_s, prop in conn_re.findall(text):
        child = int(child_s)
        parent = int(parent_s)
        prop = prop or ""
        if kind == "OO" and child in models and parent in models:
            models[child]["parent_id"] = parent
        elif kind == "OP" and child in nodes and parent in models:
            nodes[child]["model_id"] = parent
            nodes[child]["property"] = prop
        elif kind == "OP" and child in curves and parent in nodes:
            curves[child]["node_id"] = parent
            curves[child]["axis"] = prop.split("|")[-1] if "|" in prop else prop
            nodes[parent]["curves"][curves[child]["axis"]] = child

    for model in models.values():
        parent_id = model["parent_id"]
        if parent_id in models:
            model["parent_name"] = models[parent_id]["name"]


def sanitize_column_name(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    return name.strip("_")


def mat4_identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def local_matrix(tx: float, ty: float, tz: float, rx_deg: float, ry_deg: float, rz_deg: float) -> list[list[float]]:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    mx = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    my = [
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    mz = [
        [cz, -sz, 0.0, 0.0],
        [sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    m = mat4_mul(mat4_mul(mz, my), mx)
    m[0][3] = tx
    m[1][3] = ty
    m[2][3] = tz
    return m


def matrix_translation(m: list[list[float]]) -> tuple[float, float, float]:
    return (m[0][3], m[1][3], m[2][3])


def matrix_quaternion(m: list[list[float]]) -> tuple[float, float, float, float]:
    r00, r01, r02 = m[0][0], m[0][1], m[0][2]
    r10, r11, r12 = m[1][0], m[1][1], m[1][2]
    r20, r21, r22 = m[2][0], m[2][1], m[2][2]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s
    return (qw, qx, qy, qz)


def matrix_euler_xyz_deg(m: list[list[float]]) -> tuple[float, float, float]:
    # Inverse of R = Rz * Ry * Rx.
    r20 = max(-1.0, min(1.0, m[2][0]))
    ry = math.asin(-r20)
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rx = math.atan2(m[2][1], m[2][2])
        rz = math.atan2(m[1][0], m[0][0])
    else:
        rx = math.atan2(-m[1][2], m[1][1])
        rz = 0.0
    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def build_channels(models: dict[int, dict], nodes: dict[int, dict], curves: dict[int, dict]) -> dict[int, dict]:
    channels: dict[int, dict] = {
        model_id: {
            "T": {"X": None, "Y": None, "Z": None},
            "R": {"X": None, "Y": None, "Z": None},
        }
        for model_id in models
    }
    for node in nodes.values():
        model_id = node.get("model_id")
        if model_id not in models:
            continue
        node_name = node.get("name")
        if node_name not in ("T", "R"):
            continue
        for axis, curve_id in node.get("curves", {}).items():
            if axis in ("X", "Y", "Z") and curve_id in curves:
                channels[model_id][node_name][axis] = curve_id
    return channels


def collect_frame_ticks(curves: dict[int, dict]) -> list[int]:
    ticks = set()
    for curve in curves.values():
        ticks.update(curve["ticks"])
    return sorted(ticks)


def curve_value_at_tick(curve: dict | None, tick: int, default: float) -> float:
    if not curve:
        return default
    ticks = curve["ticks"]
    values = curve["values"]
    if not ticks:
        return default
    # Noitom export has dense matching keyframes. Use exact match map lazily.
    value_map = curve.get("_value_map")
    if value_map is None:
        value_map = dict(zip(ticks, values))
        curve["_value_map"] = value_map
    return float(value_map.get(tick, default))


def local_values_for_model(
    model: dict, channels: dict[int, dict], curves: dict[int, dict], tick: int
) -> dict:
    model_id = model["id"]
    ch = channels.get(model_id, {})
    t_defaults = model["default_t"]
    r_defaults = model["default_r"]
    return {
        "tx": curve_value_at_tick(curves.get(ch.get("T", {}).get("X")), tick, t_defaults[0]),
        "ty": curve_value_at_tick(curves.get(ch.get("T", {}).get("Y")), tick, t_defaults[1]),
        "tz": curve_value_at_tick(curves.get(ch.get("T", {}).get("Z")), tick, t_defaults[2]),
        "rx": curve_value_at_tick(curves.get(ch.get("R", {}).get("X")), tick, r_defaults[0]),
        "ry": curve_value_at_tick(curves.get(ch.get("R", {}).get("Y")), tick, r_defaults[1]),
        "rz": curve_value_at_tick(curves.get(ch.get("R", {}).get("Z")), tick, r_defaults[2]),
    }


def model_depth(model_id: int, models: dict[int, dict]) -> int:
    depth = 0
    seen = set()
    parent_id = models[model_id].get("parent_id")
    while parent_id in models and parent_id not in seen:
        seen.add(parent_id)
        depth += 1
        parent_id = models[parent_id].get("parent_id")
    return depth


def compute_world_pose_for_tick(
    models: dict[int, dict], channels: dict[int, dict], curves: dict[int, dict], tick: int
) -> dict[int, dict]:
    world: dict[int, dict] = {}
    ordered_ids = sorted(models, key=lambda model_id: model_depth(model_id, models))
    for model_id in ordered_ids:
        model = models[model_id]
        values = local_values_for_model(model, channels, curves, tick)
        local = local_matrix(values["tx"], values["ty"], values["tz"], values["rx"], values["ry"], values["rz"])
        parent_id = model.get("parent_id")
        if parent_id in world:
            world_matrix = mat4_mul(world[parent_id]["world_matrix"], local)
        else:
            world_matrix = local
        wx, wy, wz = matrix_translation(world_matrix)
        qw, qx, qy, qz = matrix_quaternion(world_matrix)
        erx, ery, erz = matrix_euler_xyz_deg(world_matrix)
        world[model_id] = {
            **values,
            "world_matrix": world_matrix,
            "world_x": wx,
            "world_y": wy,
            "world_z": wz,
            "world_qw": qw,
            "world_qx": qx,
            "world_qy": qy,
            "world_qz": qz,
            "world_rx": erx,
            "world_ry": ery,
            "world_rz": erz,
        }
    return world


def should_include_model(model: dict, include_end_bones: bool) -> bool:
    if model["kind"] not in ("LimbNode", "Null"):
        return False
    if not include_end_bones and model["name"].endswith("_End"):
        return False
    return True


def write_skeleton(out_path: Path, models: dict[int, dict]) -> int:
    rows = 0
    fields = [
        "model_id",
        "name",
        "kind",
        "parent_id",
        "parent_name",
        "default_tx",
        "default_ty",
        "default_tz",
        "default_rx_deg",
        "default_ry_deg",
        "default_rz_deg",
    ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model in models.values():
            row = {
                "model_id": model["id"],
                "name": model["name"],
                "kind": model["kind"],
                "parent_id": model["parent_id"] or "",
                "parent_name": model["parent_name"],
                "default_tx": f"{model['default_t'][0]:.9f}",
                "default_ty": f"{model['default_t'][1]:.9f}",
                "default_tz": f"{model['default_t'][2]:.9f}",
                "default_rx_deg": f"{model['default_r'][0]:.9f}",
                "default_ry_deg": f"{model['default_r'][1]:.9f}",
                "default_rz_deg": f"{model['default_r'][2]:.9f}",
            }
            writer.writerow(row)
            rows += 1
    return rows


def write_joints_long(
    out_path: Path,
    models: dict[int, dict],
    channels: dict[int, dict],
    curves: dict[int, dict],
    frame_ticks: list[int],
    include_end_bones: bool,
) -> int:
    fields = [
        "frame_index",
        "fbx_tick",
        "time_sec",
        "model_id",
        "bone",
        "parent",
        "kind",
        "world_x",
        "world_y",
        "world_z",
        "world_qw",
        "world_qx",
        "world_qy",
        "world_qz",
        "world_rx_deg",
        "world_ry_deg",
        "world_rz_deg",
        "local_tx",
        "local_ty",
        "local_tz",
        "local_rx_deg",
        "local_ry_deg",
        "local_rz_deg",
    ]
    export_models = [m for m in models.values() if should_include_model(m, include_end_bones)]
    rows = 0
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame_index, tick in enumerate(frame_ticks):
            world = compute_world_pose_for_tick(models, channels, curves, tick)
            for model in export_models:
                model_id = model["id"]
                pose = world[model_id]
                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "fbx_tick": tick,
                        "time_sec": f"{tick / FBX_TIME_UNIT:.9f}",
                        "model_id": model_id,
                        "bone": model["name"],
                        "parent": model["parent_name"],
                        "kind": model["kind"],
                        "world_x": f"{pose['world_x']:.9f}",
                        "world_y": f"{pose['world_y']:.9f}",
                        "world_z": f"{pose['world_z']:.9f}",
                        "world_qw": f"{pose['world_qw']:.9f}",
                        "world_qx": f"{pose['world_qx']:.9f}",
                        "world_qy": f"{pose['world_qy']:.9f}",
                        "world_qz": f"{pose['world_qz']:.9f}",
                        "world_rx_deg": f"{pose['world_rx']:.9f}",
                        "world_ry_deg": f"{pose['world_ry']:.9f}",
                        "world_rz_deg": f"{pose['world_rz']:.9f}",
                        "local_tx": f"{pose['tx']:.9f}",
                        "local_ty": f"{pose['ty']:.9f}",
                        "local_tz": f"{pose['tz']:.9f}",
                        "local_rx_deg": f"{pose['rx']:.9f}",
                        "local_ry_deg": f"{pose['ry']:.9f}",
                        "local_rz_deg": f"{pose['rz']:.9f}",
                    }
                )
                rows += 1
    return rows


def write_joints_wide(
    out_path: Path,
    models: dict[int, dict],
    channels: dict[int, dict],
    curves: dict[int, dict],
    frame_ticks: list[int],
    include_end_bones: bool,
) -> tuple[int, int]:
    export_models = [m for m in models.values() if should_include_model(m, include_end_bones)]
    base_fields = ["frame_index", "fbx_tick", "time_sec"]
    value_fields = []
    for model in export_models:
        name = sanitize_column_name(model["name"])
        for suffix in (
            "world_x",
            "world_y",
            "world_z",
            "world_qw",
            "world_qx",
            "world_qy",
            "world_qz",
            "world_rx_deg",
            "world_ry_deg",
            "world_rz_deg",
            "local_tx",
            "local_ty",
            "local_tz",
            "local_rx_deg",
            "local_ry_deg",
            "local_rz_deg",
        ):
            value_fields.append(f"{name}_{suffix}")
    fields = base_fields + value_fields
    rows = 0
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame_index, tick in enumerate(frame_ticks):
            world = compute_world_pose_for_tick(models, channels, curves, tick)
            row = {
                "frame_index": frame_index,
                "fbx_tick": tick,
                "time_sec": f"{tick / FBX_TIME_UNIT:.9f}",
            }
            for model in export_models:
                model_id = model["id"]
                name = sanitize_column_name(model["name"])
                pose = world[model_id]
                values = {
                    "world_x": pose["world_x"],
                    "world_y": pose["world_y"],
                    "world_z": pose["world_z"],
                    "world_qw": pose["world_qw"],
                    "world_qx": pose["world_qx"],
                    "world_qy": pose["world_qy"],
                    "world_qz": pose["world_qz"],
                    "world_rx_deg": pose["world_rx"],
                    "world_ry_deg": pose["world_ry"],
                    "world_rz_deg": pose["world_rz"],
                    "local_tx": pose["tx"],
                    "local_ty": pose["ty"],
                    "local_tz": pose["tz"],
                    "local_rx_deg": pose["rx"],
                    "local_ry_deg": pose["ry"],
                    "local_rz_deg": pose["rz"],
                }
                for suffix, value in values.items():
                    row[f"{name}_{suffix}"] = f"{value:.9f}"
            writer.writerow(row)
            rows += 1
    return rows, len(fields)


def main() -> None:
    args = parse_args()
    fbx_path = Path(args.fbx)
    outdir = Path(args.outdir) if args.outdir else fbx_path.parent / "fbx_mocap_csv"
    outdir.mkdir(parents=True, exist_ok=True)

    text = read_text(fbx_path)
    settings = parse_time_settings(text)
    models = parse_models(text)
    nodes = parse_curve_nodes(text)
    curves = parse_curves(text)
    parse_connections(text, models, nodes, curves)
    channels = build_channels(models, nodes, curves)
    frame_ticks = collect_frame_ticks(curves)

    skeleton_rows = write_skeleton(outdir / "mocap_skeleton.csv", models)
    long_rows = write_joints_long(
        outdir / "mocap_joints_long.csv",
        models,
        channels,
        curves,
        frame_ticks,
        args.include_end_bones,
    )
    wide_rows = 0
    wide_cols = 0
    if args.wide:
        wide_rows, wide_cols = write_joints_wide(
            outdir / "mocap_joints_wide.csv",
            models,
            channels,
            curves,
            frame_ticks,
            args.include_end_bones,
        )

    animated_models = sorted(
        {
            nodes[curve["node_id"]]["model_id"]
            for curve in curves.values()
            if curve.get("node_id") in nodes and nodes[curve["node_id"]].get("model_id") in models
        }
    )
    metadata = {
        "source_fbx": str(fbx_path),
        "output_dir": str(outdir),
        "fbx_time_unit_ticks_per_second": FBX_TIME_UNIT,
        "settings": settings,
        "model_count": len(models),
        "curve_node_count": len(nodes),
        "curve_count": len(curves),
        "animated_model_count": len(animated_models),
        "frame_count": len(frame_ticks),
        "duration_sec_from_keys": (frame_ticks[-1] - frame_ticks[0]) / FBX_TIME_UNIT if frame_ticks else 0.0,
        "skeleton_rows": skeleton_rows,
        "joints_long_rows": long_rows,
        "joints_wide_rows": wide_rows,
        "joints_wide_columns": wide_cols,
        "include_end_bones": bool(args.include_end_bones),
        "notes": [
            "world_* fields are forward-kinematics results computed from the FBX model hierarchy and local T/R channels.",
            "local_t* and local_r*_deg are original FBX local animation channels.",
            "Euler conversion assumes XYZ local rotation order using R = Rz * Ry * Rx.",
        ],
    }
    (outdir / "mocap_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"source_fbx={fbx_path}")
    print(f"output_dir={outdir}")
    print(f"models={len(models)} curves={len(curves)} frames={len(frame_ticks)}")
    print(f"duration_sec={metadata['duration_sec_from_keys']:.6f}")
    print(f"skeleton_csv={outdir / 'mocap_skeleton.csv'} rows={skeleton_rows}")
    print(f"joints_long_csv={outdir / 'mocap_joints_long.csv'} rows={long_rows}")
    if args.wide:
        print(f"joints_wide_csv={outdir / 'mocap_joints_wide.csv'} rows={wide_rows} cols={wide_cols}")


if __name__ == "__main__":
    main()
