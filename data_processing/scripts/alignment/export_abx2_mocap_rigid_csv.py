from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import re
import struct
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from preprocess_9cam_imu_mocap import (
    euler_to_quat,
    export_bvh_wide,
    load_bvh_motion,
    parse_bvh,
    q_mul,
    q_normalize,
    quat_to_euler_xyz_deg,
    rotate_vec,
    sanitize,
)


DEFAULT_SENSOR_IDS = (301, 307, 308)
SLIM_JOINTS = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Neck1",
    "Head",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
)
SLIM_MOCAP_SUFFIXES = ("world_x", "world_y", "world_z", "world_qw", "world_qx", "world_qy", "world_qz")


def read_abx2_header(abx2_path: Path) -> tuple[dict, dict]:
    with abx2_path.open("rb") as f:
        info_len = struct.unpack("<Q", f.read(8))[0]
        info = json.loads(f.read(info_len).decode("utf-8"))
        config_len = struct.unpack("<Q", f.read(8))[0]
        config = json.loads(f.read(config_len).decode("utf-8"))
    return info, config


def find_candidate_bvh(abx2_path: Path, frame_count: int | None = None) -> Path | None:
    candidates = sorted(abx2_path.parent.glob("*.bvh"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    if frame_count is None:
        return candidates[0]
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("Frames:"):
                        if int(line.split(":", 1)[1].strip()) == frame_count:
                            return path
                        break
        except Exception:
            continue
    return candidates[0]


def pwr_map_for_sensors(config: dict, sensor_ids: tuple[int, ...]) -> list[dict]:
    wanted = set(sensor_ids)
    out = []
    for item in config.get("pwrs", []):
        try:
            sensor_id = int(item.get("sensorId"))
        except (TypeError, ValueError):
            continue
        if sensor_id in wanted:
            out.append(item)
    found = {int(item["sensorId"]) for item in out}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"sensor IDs not found in ABX2 PWR table: {missing}")
    return sorted(out, key=lambda item: int(item["sensorId"]))


def extract_ch3_rigids(abx2_path: Path, pwrs: list[dict], fps: float) -> dict[int, list[dict]]:
    by_sensor: dict[int, list[dict]] = {int(item["sensorId"]): [] for item in pwrs}
    with abx2_path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for pwr in pwrs:
            pwr_id = int(pwr["pwrId"])
            sensor_id = int(pwr["sensorId"])
            pattern = struct.pack("<II", pwr_id, sensor_id)
            start = 0
            while True:
                idx = mm.find(pattern, start)
                if idx < 0:
                    break
                start = idx + 1
                if idx < 16 or idx + 44 > len(mm):
                    continue
                pwr_id2, sensor_id2, status, raw_tick = struct.unpack_from("<IIII", mm, idx)
                vals = struct.unpack_from("<7f", mm, idx + 16)
                if pwr_id2 != pwr_id or sensor_id2 != sensor_id:
                    continue
                if status not in (0, 1, 2, 3):
                    continue
                if not all(math.isfinite(v) and abs(v) < 1e6 for v in vals):
                    continue
                frame_index = len(by_sensor[sensor_id])
                by_sensor[sensor_id].append(
                    {
                        "sensor_id": sensor_id,
                        "pwr_id": pwr_id,
                        "name": pwr.get("name", f"sensor_{sensor_id}"),
                        "rigid_type": pwr.get("rigidType", ""),
                        "pac_key": pwr.get("pacKey", ""),
                        "frame_index": frame_index,
                        "time_sec": frame_index / fps if fps else "",
                        "status": status,
                        "raw_tick": raw_tick,
                        "x": vals[0],
                        "y": vals[1],
                        "z": vals[2],
                        "qw": vals[3],
                        "qx": vals[4],
                        "qy": vals[5],
                        "qz": vals[6],
                        "quat_order": "wxyz_inferred",
                        "file_offset": idx,
                    }
                )
        mm.close()
    return by_sensor


def add_rigid_euler(by_sensor: dict[int, list[dict]]) -> None:
    for rows in by_sensor.values():
        if not rows:
            continue
        q = np.array([[r["qw"], r["qx"], r["qy"], r["qz"]] for r in rows], dtype=np.float64)
        e = quat_to_euler_xyz_deg(q)
        for row, erow in zip(rows, e):
            row["rx_deg"] = float(erow[0])
            row["ry_deg"] = float(erow[1])
            row["rz_deg"] = float(erow[2])


def write_rigid_sidecars(outdir: Path, by_sensor: dict[int, list[dict]], pwrs: list[dict]) -> dict:
    rigids_dir = outdir / "rigids"
    rigids_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sensor_id",
        "pwr_id",
        "name",
        "rigid_type",
        "pac_key",
        "frame_index",
        "time_sec",
        "status",
        "raw_tick",
        "x",
        "y",
        "z",
        "qw",
        "qx",
        "qy",
        "qz",
        "rx_deg",
        "ry_deg",
        "rz_deg",
        "quat_order",
        "file_offset",
    ]
    combined = rigids_dir / "ch3_rigids_wide_source.csv"
    with combined.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sensor_id in sorted(by_sensor):
            writer.writerows(by_sensor[sensor_id])
    for sensor_id, rows in sorted(by_sensor.items()):
        path = rigids_dir / f"ch3_{sensor_id}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return {
        "combined_rigids_csv": str(combined),
        "per_sensor_counts": {str(sensor_id): len(rows) for sensor_id, rows in sorted(by_sensor.items())},
        "pwrs": pwrs,
    }


def bvh_slim_fields_and_indices(bvh: Path, joints_to_export: tuple[str, ...]) -> tuple[list, list[int], list[str], list[str], int, float, int]:
    joints, channel_count, frame_time, frame_count = parse_bvh(bvh)
    joint_by_name = {j.name: i for i, j in enumerate(joints)}
    export_indices = [joint_by_name[name] for name in joints_to_export if name in joint_by_name]
    missing_joints = [name for name in joints_to_export if name not in joint_by_name]
    if missing_joints:
        print(f"missing slim joints ignored: {missing_joints}")

    fields = ["frame_index", "time_sec"]
    for i in export_indices:
        name = sanitize(joints[i].name)
        fields.extend(f"{name}_{suffix}" for suffix in SLIM_MOCAP_SUFFIXES)
    return joints, export_indices, fields, missing_joints, channel_count, frame_time, frame_count


def slim_mocap_row_from_world(
    frame_index: int,
    time_sec: float,
    export_indices: list[int],
    world_pos: list[np.ndarray],
    world_q: list[np.ndarray],
    local_row_index: int,
) -> list[str | int]:
    row: list[str | int] = [frame_index, f"{time_sec:.9f}"]
    for i in export_indices:
        pos = world_pos[i][local_row_index]
        quat = world_q[i][local_row_index]
        row.extend(
            [
                f"{pos[0]:.9f}",
                f"{pos[1]:.9f}",
                f"{pos[2]:.9f}",
                f"{quat[0]:.9f}",
                f"{quat[1]:.9f}",
                f"{quat[2]:.9f}",
                f"{quat[3]:.9f}",
            ]
        )
    return row


def iter_bvh_world_chunks(
    bvh: Path,
    joints: list,
    channel_count: int,
    frame_count: int,
    chunk: int = 2048,
):
    motion = load_bvh_motion(bvh, channel_count, frame_count)
    for start in range(0, frame_count, chunk):
        end = min(frame_count, start + chunk)
        n = end - start
        world_pos: list[np.ndarray] = []
        world_q: list[np.ndarray] = []
        for joint in joints:
            vals = (
                motion[start:end, joint.channel_start : joint.channel_start + len(joint.channels)]
                if joint.channels
                else np.empty((n, 0))
            )
            tx = np.full(n, joint.offset[0], dtype=np.float64)
            ty = np.full(n, joint.offset[1], dtype=np.float64)
            tz = np.full(n, joint.offset[2], dtype=np.float64)
            rx = np.zeros(n, dtype=np.float64)
            ry = np.zeros(n, dtype=np.float64)
            rz = np.zeros(n, dtype=np.float64)
            rot_order = ""
            for ci, channel in enumerate(joint.channels):
                if channel == "Xposition":
                    tx += vals[:, ci]
                elif channel == "Yposition":
                    ty += vals[:, ci]
                elif channel == "Zposition":
                    tz += vals[:, ci]
                elif channel == "Xrotation":
                    rx = np.deg2rad(vals[:, ci].astype(np.float64))
                    rot_order += "X"
                elif channel == "Yrotation":
                    ry = np.deg2rad(vals[:, ci].astype(np.float64))
                    rot_order += "Y"
                elif channel == "Zrotation":
                    rz = np.deg2rad(vals[:, ci].astype(np.float64))
                    rot_order += "Z"
            local_q = euler_to_quat(rx, ry, rz, rot_order or "XYZ")
            local_t = np.stack((tx, ty, tz), axis=-1)
            if joint.parent is None:
                pos = local_t
                quat = local_q
            else:
                parent_q = world_q[joint.parent]
                pos = world_pos[joint.parent] + rotate_vec(parent_q, local_t)
                quat = q_mul(parent_q, local_q)
            world_pos.append(pos)
            world_q.append(q_normalize(quat))
        yield start, end, world_pos, world_q


def export_bvh_slim_combined(
    bvh: Path,
    output_csv: Path,
    by_sensor: dict[int, list[dict]],
    pwrs: list[dict],
    joints_to_export: tuple[str, ...],
) -> tuple[int, int, dict]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    joints, export_indices, fields, missing_joints, channel_count, frame_time, frame_count = bvh_slim_fields_and_indices(
        bvh, joints_to_export
    )
    appended = rigid_columns(pwrs)
    time = np.arange(frame_count, dtype=np.float64) * frame_time
    rows = 0
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(fields + appended)
        for start, end, world_pos, world_q in iter_bvh_world_chunks(bvh, joints, channel_count, frame_count):
            for r in range(end - start):
                frame_index = start + r
                writer.writerow(
                    slim_mocap_row_from_world(frame_index, time[frame_index], export_indices, world_pos, world_q, r)
                    + rigid_values_for_frame(by_sensor, pwrs, frame_index)
                )
                rows += 1
            print(f"exported combined slim rows: {end}")
    metadata = {
        "source_bvh": str(bvh),
        "frame_count": frame_count,
        "frame_time_sec": frame_time,
        "fps": 1.0 / frame_time if frame_time else None,
        "export_joint_count": len(export_indices),
        "export_joints": [joints[i].name for i in export_indices],
        "missing_joints": missing_joints,
        "mocap_columns": len(fields),
        "profile": "slim",
        "notes": [
            "Direct slim export: no persisted mocap intermediate CSV is written.",
            "Each exported joint keeps world position and world quaternion only.",
        ],
    }
    return rows, len(fields) + len(appended), metadata


def slim_mocap_columns_from_header(header: list[str], joints: tuple[str, ...]) -> list[str]:
    columns = ["frame_index", "time_sec"]
    for joint in joints:
        name = sanitize(joint)
        for suffix in SLIM_MOCAP_SUFFIXES:
            col = f"{name}_{suffix}"
            if col in header:
                columns.append(col)
    missing = [col for col in columns if col not in header]
    if missing:
        raise ValueError(f"required mocap columns missing: {missing[:10]}")
    return columns


def export_bvh_slim_wide(bvh: Path, outdir: Path, joints_to_export: tuple[str, ...]) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    joints, export_indices, fields, missing_joints, channel_count, frame_time, frame_count = bvh_slim_fields_and_indices(
        bvh, joints_to_export
    )
    time = np.arange(frame_count, dtype=np.float64) * frame_time

    wide_path = outdir / "mocap_joints_slim.csv"
    with wide_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for start, end, world_pos, world_q in iter_bvh_world_chunks(bvh, joints, channel_count, frame_count):
            for r in range(end - start):
                frame_index = start + r
                writer.writerow(slim_mocap_row_from_world(frame_index, time[frame_index], export_indices, world_pos, world_q, r))
            print(f"exported slim mocap rows: {end}")

    metadata = {
        "source_bvh": str(bvh),
        "output_csv": str(wide_path),
        "frame_count": frame_count,
        "frame_time_sec": frame_time,
        "fps": 1.0 / frame_time if frame_time else None,
        "export_joint_count": len(export_indices),
        "export_joints": [joints[i].name for i in export_indices],
        "columns": len(fields),
        "profile": "slim",
        "notes": [
            "Slim mocap export keeps main body joints only.",
            "Each joint keeps world position and world quaternion only.",
        ],
    }
    (outdir / "mocap_slim_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def rigid_columns(pwrs: list[dict]) -> list[str]:
    fields = []
    for pwr in sorted(pwrs, key=lambda item: int(item["sensorId"])):
        name = sanitize(pwr.get("name") or f"sensor_{pwr['sensorId']}")
        fields.extend(
            [
                f"{name}_world_x",
                f"{name}_world_y",
                f"{name}_world_z",
                f"{name}_world_qw",
                f"{name}_world_qx",
                f"{name}_world_qy",
                f"{name}_world_qz",
                f"{name}_world_rx_deg",
                f"{name}_world_ry_deg",
                f"{name}_world_rz_deg",
            ]
        )
    return fields


def rigid_values_for_frame(by_sensor: dict[int, list[dict]], pwrs: list[dict], frame_index: int) -> list[str]:
    values = []
    for pwr in sorted(pwrs, key=lambda item: int(item["sensorId"])):
        sensor_id = int(pwr["sensorId"])
        rows = by_sensor.get(sensor_id, [])
        if frame_index >= len(rows):
            values.extend([""] * 10)
            continue
        row = rows[frame_index]
        values.extend(
            [
                f"{row['x']:.9f}",
                f"{row['y']:.9f}",
                f"{row['z']:.9f}",
                f"{row['qw']:.9f}",
                f"{row['qx']:.9f}",
                f"{row['qy']:.9f}",
                f"{row['qz']:.9f}",
                f"{row['rx_deg']:.9f}",
                f"{row['ry_deg']:.9f}",
                f"{row['rz_deg']:.9f}",
            ]
        )
    return values


def infer_record_date(abx2: Path, bvh: Path | None = None) -> str:
    paths = [p for p in (bvh, abx2) if p is not None]
    for path in paths:
        match = re.search(r"(20\d{6})", path.name)
        if match:
            return match.group(1)

    timestamp_year = datetime.fromtimestamp(abx2.stat().st_mtime).year
    for path in paths:
        for part in reversed(path.parent.parts):
            if re.fullmatch(r"\d{4}", part):
                month = int(part[:2])
                day = int(part[2:])
                try:
                    datetime(timestamp_year, month, day)
                except ValueError:
                    continue
                return f"{timestamp_year}{part}"

    return datetime.fromtimestamp(abx2.stat().st_mtime).strftime("%Y%m%d")


def default_output_name(abx2: Path, bvh: Path | None = None) -> str:
    return f"mocap_rigid_{infer_record_date(abx2, bvh)}.csv"


def combine_mocap_and_rigids(
    mocap_wide: Path,
    output_csv: Path,
    by_sensor: dict[int, list[dict]],
    pwrs: list[dict],
    mocap_columns: list[str] | None = None,
) -> tuple[int, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    appended = rigid_columns(pwrs)
    rows = 0
    with mocap_wide.open("r", newline="", encoding="utf-8-sig") as src, output_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        if mocap_columns is None:
            mocap_columns = header
        column_indices = [header.index(col) for col in mocap_columns]
        writer.writerow(mocap_columns + appended)
        frame_index_col = header.index("frame_index")
        for row in reader:
            frame_index = int(row[frame_index_col])
            writer.writerow([row[i] for i in column_indices] + rigid_values_for_frame(by_sensor, pwrs, frame_index))
            rows += 1
            if rows % 5000 == 0:
                print(f"merged rows: {rows}")
    return rows, len(mocap_columns) + len(appended)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one wide CSV from a Noitom ABX2 recording plus its BVH avatar export and CH3 rigid bodies."
    )
    parser.add_argument("abx2", type=Path, help="Source .abx2 file.")
    parser.add_argument("--bvh", type=Path, default=None, help="BVH exported from the same ABX2. If omitted, find sibling .bvh.")
    parser.add_argument(
        "--mocap-wide",
        type=Path,
        default=None,
        help="Existing mocap_joints_wide.csv to reuse instead of regenerating from BVH.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--output-name", default=None, help="Combined CSV file name.")
    parser.add_argument(
        "--profile",
        choices=("slim", "full"),
        default="slim",
        help="slim keeps main body joints world pose only; full keeps all mocap_joints_wide columns.",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=None,
        help="Override slim joint list. Names must match BVH joints, e.g. Hips Head LeftHand RightHand.",
    )
    parser.add_argument("--sensor-ids", type=int, nargs="+", default=list(DEFAULT_SENSOR_IDS), help="CH3 sensor IDs to extract.")
    parser.add_argument("--include-end-bones", action="store_true", help="Include BVH End Site bones in mocap columns.")
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Also save helper CSVs such as per-sensor rigid files and generated mocap source CSVs. Default: only final CSV + metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    abx2 = args.abx2
    if not abx2.exists():
        raise FileNotFoundError(abx2)
    outdir = args.output_dir or (abx2.parent / "abx2_mocap_rigid_csv")
    outdir.mkdir(parents=True, exist_ok=True)

    info, config = read_abx2_header(abx2)
    abx_info = info.get("ABXInfo", {})
    fps = float(abx_info.get("fps") or 60.0)
    frame_count = int(abx_info.get("totalFrameNum") or 0)
    print(f"ABX2 frames={frame_count} fps={fps}")

    sensor_ids = tuple(int(v) for v in args.sensor_ids)
    pwrs = pwr_map_for_sensors(config, sensor_ids)
    print("PWR mapping:")
    for pwr in pwrs:
        print(f"  sensor={pwr['sensorId']} pwr={pwr['pwrId']} name={pwr.get('name')} pac={pwr.get('pacKey')}")

    print("extracting CH3 rigid bodies from ABX2...")
    by_sensor = extract_ch3_rigids(abx2, pwrs, fps)
    add_rigid_euler(by_sensor)
    sidecar_meta = {}
    if args.keep_intermediates:
        sidecar_meta = write_rigid_sidecars(outdir, by_sensor, pwrs)

    bvh_export_meta = None
    bvh = args.bvh
    mocap_wide = None
    if args.mocap_wide is not None:
        mocap_wide = args.mocap_wide
        if not mocap_wide.exists():
            raise FileNotFoundError(mocap_wide)
    else:
        bvh = args.bvh or find_candidate_bvh(abx2, frame_count)
        if bvh is None or not bvh.exists():
            raise FileNotFoundError(
                "No BVH was provided or found next to ABX2. ABX2 avatar motion is stored in Noitom private binary; "
                "provide --bvh exported from the same ABX2, or --mocap-wide."
            )

    output_name = args.output_name or default_output_name(abx2, bvh)
    output_csv = outdir / output_name

    if args.mocap_wide is None:
        if args.profile == "slim":
            joints = tuple(args.joints or SLIM_JOINTS)
            print(f"direct-exporting slim BVH + CH3 rigids into {output_csv} ...")
            rows, cols, bvh_export_meta = export_bvh_slim_combined(bvh, output_csv, by_sensor, pwrs, joints)
            mocap_wide = None
        else:
            if args.keep_intermediates:
                mocap_dir = outdir / "mocap_source_csv"
                print(f"exporting full BVH wide from {bvh} ...")
                export_bvh_wide(bvh, mocap_dir, include_end_bones=args.include_end_bones)
                mocap_wide = mocap_dir / "mocap_joints_wide.csv"
                print(f"combining into {output_csv} ...")
                rows, cols = combine_mocap_and_rigids(mocap_wide, output_csv, by_sensor, pwrs)
            else:
                with tempfile.TemporaryDirectory(prefix="abx2_mocap_full_") as tmp:
                    mocap_dir = Path(tmp)
                    print(f"exporting full BVH wide to temporary storage from {bvh} ...")
                    export_bvh_wide(bvh, mocap_dir, include_end_bones=args.include_end_bones)
                    mocap_wide = mocap_dir / "mocap_joints_wide.csv"
                    print(f"combining into {output_csv} ...")
                    rows, cols = combine_mocap_and_rigids(mocap_wide, output_csv, by_sensor, pwrs)
                mocap_wide = None

    if args.mocap_wide is not None:
        print(f"combining into {output_csv} ...")
        mocap_columns = None
        if args.profile == "slim":
            with mocap_wide.open("r", newline="", encoding="utf-8-sig") as f:
                header = next(csv.reader(f))
            mocap_columns = slim_mocap_columns_from_header(header, tuple(args.joints or SLIM_JOINTS))
        rows, cols = combine_mocap_and_rigids(mocap_wide, output_csv, by_sensor, pwrs, mocap_columns=mocap_columns)

    status_counts = {
        str(sensor_id): {
            str(status): sum(1 for row in rows_for_sensor if row["status"] == status)
            for status in sorted({row["status"] for row in rows_for_sensor})
        }
        for sensor_id, rows_for_sensor in sorted(by_sensor.items())
    }
    metadata = {
        "source_abx2": str(abx2),
        "source_bvh": None if bvh is None else str(bvh),
        "source_mocap_wide": None if mocap_wide is None else str(mocap_wide),
        "output_csv": str(output_csv),
        "frame_count_from_abx2": frame_count,
        "fps_from_abx2": fps,
        "rows": rows,
        "columns": cols,
        "profile": args.profile,
        "keep_intermediates": bool(args.keep_intermediates),
        "direct_slim_export": bool(args.profile == "slim" and args.mocap_wide is None),
        "slim_joints": list(args.joints or SLIM_JOINTS) if args.profile == "slim" else None,
        "slim_mocap_suffixes": list(SLIM_MOCAP_SUFFIXES) if args.profile == "slim" else None,
        "sensor_ids": list(sensor_ids),
        "rigid_quat_order": "wxyz",
        "rigid_position_units": "ABX2 native units, observed as meters for CH3 PWR records",
        "rigid_status_counts": status_counts,
        "bvh_export": bvh_export_meta,
        **sidecar_meta,
    }
    meta_path = outdir / "abx2_mocap_rigid_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
