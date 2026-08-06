from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import deque
from pathlib import Path

import numpy as np


CAMERAS = ("CAM_A", "CAM_B", "CAM_C")
CAMERA_ALIGNMENT_TS_FIELD = "device_ts_ms"
MODULE_LABELS = {1: "module01", 2: "module02", 3: "module03"}
WRISTS = ("LeftHand", "RightHand")
IMU_NEAREST_THRESHOLD_MS = 20.0
LONG_CAMERA_GAP_DELETE_THRESHOLD_PERIODS = 2.5
DEFAULT_MOCAP_TIME_OFFSET_SEC = 0.0
MOCAP_VALUE_FIELDS = (
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
)


def ffloat(value: str | float | int | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    return float(value)


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")


def q_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n <= 1e-12] = 1.0
    return q / n


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def q_inv(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1
    return q_normalize(out)


def euler_to_quat(rx: np.ndarray, ry: np.ndarray, rz: np.ndarray, order: str) -> np.ndarray:
    def axis_quat(axis: str, angle: np.ndarray) -> np.ndarray:
        half = angle * 0.5
        c = np.cos(half)
        s = np.sin(half)
        z = np.zeros_like(c)
        if axis == "X":
            return np.stack((c, s, z, z), axis=-1)
        if axis == "Y":
            return np.stack((c, z, s, z), axis=-1)
        return np.stack((c, z, z, s), axis=-1)

    angles = {"X": rx, "Y": ry, "Z": rz}
    q = np.zeros((len(rx), 4), dtype=np.float64)
    q[:, 0] = 1.0
    for axis in order:
        q = q_mul(q, axis_quat(axis, angles[axis]))
    return q_normalize(q)


def rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # q * [0, v] * q^-1, vectorized over q rows.
    vq = np.zeros((len(q), 4), dtype=np.float64)
    vq[:, 1:] = v
    return q_mul(q_mul(q, vq), q_inv(q))[:, 1:]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    m = np.empty((len(q), 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def quat_to_euler_xyz_deg(q: np.ndarray) -> np.ndarray:
    m = quat_to_matrix(q)
    r20 = np.clip(m[:, 2, 0], -1.0, 1.0)
    ry = np.arcsin(-r20)
    cy = np.cos(ry)
    rx = np.where(np.abs(cy) > 1e-8, np.arctan2(m[:, 2, 1], m[:, 2, 2]), np.arctan2(-m[:, 1, 2], m[:, 1, 1]))
    rz = np.where(np.abs(cy) > 1e-8, np.arctan2(m[:, 1, 0], m[:, 0, 0]), 0.0)
    return np.rad2deg(np.stack((rx, ry, rz), axis=-1))


def angular_velocity_from_quat(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = len(q)
    out = np.zeros((n, 3), dtype=np.float64)
    if n < 3:
        return out
    prev_i = np.arange(n) - 1
    next_i = np.arange(n) + 1
    prev_i[0] = 0
    next_i[-1] = n - 1
    dt = t[next_i] - t[prev_i]
    valid = dt > 1e-12
    dq = q_mul(q[next_i], q_inv(q[prev_i]))
    dq = q_normalize(dq)
    neg = dq[:, 0] < 0
    dq[neg] *= -1
    w = np.clip(dq[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    ok = valid & (s > 1e-9) & (angle > 1e-12)
    out[ok] = dq[ok, 1:] / s[ok, None] * (angle[ok] / dt[ok])[:, None]
    return out


class Joint:
    def __init__(self, name: str, parent: int | None, offset: tuple[float, float, float], channels: list[str]):
        self.name = name
        self.parent = parent
        self.offset = np.array(offset, dtype=np.float64)
        self.channels = channels
        self.channel_start = 0


def parse_bvh(path: Path) -> tuple[list[Joint], int, float, int]:
    joints: list[Joint] = []
    stack: list[int] = []
    pending_name: str | None = None
    pending_end = False
    channel_count = 0
    frames = 0
    frame_time = 0.0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if parts[0] in ("ROOT", "JOINT"):
                pending_name = parts[1]
                pending_end = False
            elif parts[0] == "End":
                pending_name = f"{joints[stack[-1]].name}_End"
                pending_end = True
            elif parts[0] == "{":
                pass
            elif parts[0] == "}":
                if stack:
                    stack.pop()
            elif parts[0] == "OFFSET" and pending_name is not None:
                parent = stack[-1] if stack else None
                offset = (float(parts[1]), float(parts[2]), float(parts[3]))
                channels: list[str] = []
                joint = Joint(pending_name, parent, offset, channels)
                joints.append(joint)
                idx = len(joints) - 1
                stack.append(idx)
                if pending_end:
                    pending_name = None
            elif parts[0] == "CHANNELS" and stack:
                count = int(parts[1])
                channels = parts[2 : 2 + count]
                joints[stack[-1]].channels = channels
                joints[stack[-1]].channel_start = channel_count
                channel_count += count
                pending_name = None
            elif parts[0] == "Frames:":
                frames = int(parts[1])
            elif parts[0] == "Frame" and parts[1] == "Time:":
                frame_time = float(parts[2])
                break
    return joints, channel_count, frame_time, frames


def load_bvh_motion(path: Path, channel_count: int, frames: int) -> np.ndarray:
    data = np.empty((frames, channel_count), dtype=np.float32)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().startswith("Frame Time:"):
                break
        for i, line in enumerate(f):
            if i >= frames:
                break
            data[i] = np.fromstring(line, sep=" ", dtype=np.float32, count=channel_count)
    return data


def export_bvh_wide(bvh: Path, outdir: Path, include_end_bones: bool = False) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    joints, channel_count, frame_time, frame_count = parse_bvh(bvh)
    motion = load_bvh_motion(bvh, channel_count, frame_count)
    time = np.arange(frame_count, dtype=np.float64) * frame_time

    export_indices = [
        i for i, j in enumerate(joints) if (include_end_bones or not j.name.endswith("_End"))
    ]
    fields = ["frame_index", "time_sec"]
    for i in export_indices:
        name = sanitize(joints[i].name)
        fields += [f"{name}_{suffix}" for suffix in MOCAP_VALUE_FIELDS]

    wide_path = outdir / "mocap_joints_wide.csv"
    skel_path = outdir / "mocap_skeleton.csv"
    meta_path = outdir / "mocap_metadata.json"

    with skel_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["joint_index", "name", "parent_index", "parent_name", "offset_x", "offset_y", "offset_z", "channels"])
        w.writeheader()
        for i, j in enumerate(joints):
            w.writerow(
                {
                    "joint_index": i,
                    "name": j.name,
                    "parent_index": "" if j.parent is None else j.parent,
                    "parent_name": "" if j.parent is None else joints[j.parent].name,
                    "offset_x": f"{j.offset[0]:.9f}",
                    "offset_y": f"{j.offset[1]:.9f}",
                    "offset_z": f"{j.offset[2]:.9f}",
                    "channels": " ".join(j.channels),
                }
            )

    # Stream rows to avoid keeping the full 1000-column table in memory.
    chunk = 2048
    with wide_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for start in range(0, frame_count, chunk):
            end = min(frame_count, start + chunk)
            n = end - start
            world_pos: list[np.ndarray] = []
            world_q: list[np.ndarray] = []
            local_vals: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for j in joints:
                vals = motion[start:end, j.channel_start : j.channel_start + len(j.channels)] if j.channels else np.empty((n, 0))
                tx = np.full(n, j.offset[0], dtype=np.float64)
                ty = np.full(n, j.offset[1], dtype=np.float64)
                tz = np.full(n, j.offset[2], dtype=np.float64)
                rx = np.zeros(n, dtype=np.float64)
                ry = np.zeros(n, dtype=np.float64)
                rz = np.zeros(n, dtype=np.float64)
                rot_order = ""
                for ci, ch in enumerate(j.channels):
                    if ch == "Xposition":
                        tx += vals[:, ci]
                    elif ch == "Yposition":
                        ty += vals[:, ci]
                    elif ch == "Zposition":
                        tz += vals[:, ci]
                    elif ch == "Xrotation":
                        rx = np.deg2rad(vals[:, ci].astype(np.float64))
                        rot_order += "X"
                    elif ch == "Yrotation":
                        ry = np.deg2rad(vals[:, ci].astype(np.float64))
                        rot_order += "Y"
                    elif ch == "Zrotation":
                        rz = np.deg2rad(vals[:, ci].astype(np.float64))
                        rot_order += "Z"
                local_q = euler_to_quat(rx, ry, rz, rot_order or "XYZ")
                local_t = np.stack((tx, ty, tz), axis=-1)
                if j.parent is None:
                    pos = local_t
                    q = local_q
                else:
                    pq = world_q[j.parent]
                    pos = world_pos[j.parent] + rotate_vec(pq, local_t)
                    q = q_mul(pq, local_q)
                world_pos.append(pos)
                world_q.append(q_normalize(q))
                local_vals.append((tx, ty, tz, np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)))

            world_euler = {i: quat_to_euler_xyz_deg(world_q[i]) for i in export_indices}
            for r in range(n):
                row: list[str | int] = [start + r, f"{time[start + r]:.9f}"]
                for i in export_indices:
                    tx, ty, tz, rx, ry, rz = local_vals[i]
                    p = world_pos[i][r]
                    q = world_q[i][r]
                    e = world_euler[i][r]
                    row.extend(
                        [
                            f"{p[0]:.9f}",
                            f"{p[1]:.9f}",
                            f"{p[2]:.9f}",
                            f"{q[0]:.9f}",
                            f"{q[1]:.9f}",
                            f"{q[2]:.9f}",
                            f"{q[3]:.9f}",
                            f"{e[0]:.9f}",
                            f"{e[1]:.9f}",
                            f"{e[2]:.9f}",
                            f"{tx[r]:.9f}",
                            f"{ty[r]:.9f}",
                            f"{tz[r]:.9f}",
                            f"{rx[r]:.9f}",
                            f"{ry[r]:.9f}",
                            f"{rz[r]:.9f}",
                        ]
                    )
                writer.writerow(row)

    metadata = {
        "source_bvh": str(bvh),
        "output_dir": str(outdir),
        "frame_count": frame_count,
        "frame_time_sec": frame_time,
        "fps": 1.0 / frame_time if frame_time else None,
        "duration_sec": (frame_count - 1) * frame_time if frame_count else 0.0,
        "joint_count": len(joints),
        "export_joint_count": len(export_indices),
        "joints_wide_rows": frame_count,
        "joints_wide_columns": len(fields),
        "include_end_bones": include_end_bones,
        "notes": [
            "world_* fields are forward-kinematics results computed from BVH hierarchy and local channels.",
            "BVH channel rotation order is applied in the channel order declared for each joint.",
        ],
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def write_wrist_derivatives(mocap_wide: Path, outdir: Path) -> dict:
    cols = ["frame_index", "time_sec"]
    for wrist in WRISTS:
        cols += [f"{wrist}_world_{suffix}" for suffix in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    data = {c: [] for c in cols}
    with mocap_wide.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in cols:
                data[c].append(float(row[c]) if c != "frame_index" else int(row[c]))
    t = np.array(data["time_sec"], dtype=np.float64)
    fields = ["frame_index", "time_sec"]
    arrays: dict[str, np.ndarray] = {}
    for wrist in WRISTS:
        q = np.stack([np.array(data[f"{wrist}_world_{s}"], dtype=np.float64) for s in ("qw", "qx", "qy", "qz")], axis=-1)
        gyro = angular_velocity_from_quat(q, t)
        arrays[f"{wrist}_gyro_x_rad_s"] = gyro[:, 0]
        arrays[f"{wrist}_gyro_y_rad_s"] = gyro[:, 1]
        arrays[f"{wrist}_gyro_z_rad_s"] = gyro[:, 2]
        arrays[f"{wrist}_gyro_norm_rad_s"] = np.linalg.norm(gyro, axis=1)
        fields += [
            f"{wrist}_gyro_x_rad_s",
            f"{wrist}_gyro_y_rad_s",
            f"{wrist}_gyro_z_rad_s",
            f"{wrist}_gyro_norm_rad_s",
        ]
    path = outdir / "wrist_mocap_derivatives_wide.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(len(t)):
            row = [int(data["frame_index"][i]), f"{t[i]:.9f}"]
            for c in fields[2:]:
                row.append(f"{arrays[c][i]:.9f}")
            w.writerow(row)
    meta = {"derivative_csv": str(path), "rows": len(t), "bones": list(WRISTS)}
    (outdir / "wrist_mocap_derivatives_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def read_imu(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    out = {k: np.array([ffloat(r.get(k)) for r in rows], dtype=np.float64) for k in rows[0].keys() if k not in ("module", "mxid")}
    out["raw_rows"] = np.array(rows, dtype=object)
    return out


def make_camera_master(root: Path) -> tuple[list[dict], dict]:
    per_key: dict[tuple[int, str], list[dict]] = {}
    with (root / "timestamps.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            module = int(row["module"])
            cam = row["camera"]
            per_key.setdefault((module, cam), []).append(row)
    filled: dict[tuple[int, str], list[dict | None]] = {}
    stats = {}
    for key, rows in per_key.items():
        rows.sort(key=lambda r: int(r["seq"]))
        times = [float(r[CAMERA_ALIGNMENT_TS_FIELD]) for r in rows]
        small_gaps = [b - a for a, b in zip(times, times[1:]) if 15.0 <= b - a <= 45.0]
        period = float(np.median(small_gaps)) if small_gaps else 33.333
        out: list[dict | None] = []
        inserted = 0
        suppressed = 0
        filled_gaps = []
        deleted_long_gaps = []
        for i, row in enumerate(rows):
            out.append(row)
            if i == len(rows) - 1:
                continue
            dt = times[i + 1] - times[i]
            missing = max(0, int(round(dt / period)) - 1)
            if missing:
                gap_info = {
                    "after_seq": int(row["seq"]),
                    "before_seq": int(rows[i + 1]["seq"]),
                    "gap_ms": dt,
                    "estimated_missing_slots": missing,
                }
                if dt > period * LONG_CAMERA_GAP_DELETE_THRESHOLD_PERIODS:
                    gap_info["suppressed_slots"] = missing
                    deleted_long_gaps.append(gap_info)
                    suppressed += missing
                    continue
                gap_info["inserted_slots"] = missing
                filled_gaps.append(gap_info)
                for j in range(missing):
                    out.append(
                        {
                            "module": str(key[0]),
                            "camera": key[1],
                            "seq": "",
                            "device_ts_ms": "",
                            "_gap_filled": "1",
                            "_slot_device_ts_ms": f"{times[i] + period * (j + 1):.9f}",
                        }
                    )
                inserted += missing
        filled[key] = out
        stats[f"module{key[0]:02d}_{key[1]}_rows"] = len(rows)
        stats[f"module{key[0]:02d}_{key[1]}_period_ms"] = period
        stats[f"module{key[0]:02d}_{key[1]}_inserted_missing_slots"] = inserted
        stats[f"module{key[0]:02d}_{key[1]}_suppressed_long_gap_slots"] = suppressed
        stats[f"module{key[0]:02d}_{key[1]}_filled_gap_count"] = len(filled_gaps)
        stats[f"module{key[0]:02d}_{key[1]}_deleted_long_gap_count"] = len(deleted_long_gaps)
        stats[f"module{key[0]:02d}_{key[1]}_first_filled_gaps"] = filled_gaps[:10]
        stats[f"module{key[0]:02d}_{key[1]}_first_deleted_long_gaps"] = deleted_long_gaps[:10]

    max_slots = max(len(v) for v in filled.values())
    rows_out = []
    for out_seq in range(max_slots):
        row = {"seq": out_seq}
        for module in (1, 2, 3):
            for cam in CAMERAS:
                slots = filled[(module, cam)]
                src = slots[out_seq] if out_seq < len(slots) else None
                prefix = f"module{module:02d}_{cam}"
                if not src:
                    row[f"{prefix}_device_ts_ms"] = ""
                    row[f"{prefix}_gap_filled"] = "1"
                    row[f"{prefix}_slot_device_ts_ms"] = ""
                    continue
                row[f"{prefix}_device_ts_ms"] = src["device_ts_ms"]
                row[f"{prefix}_gap_filled"] = src.get("_gap_filled", "0")
                row[f"{prefix}_slot_device_ts_ms"] = src.get(
                    "_slot_device_ts_ms", src["device_ts_ms"]
                )

        rows_out.append(row)

    # Keep internal missing-camera slots, but trim trailing incomplete rows so
    # the final frame has all 9 camera timestamps.
    def has_all_cameras(row: dict) -> bool:
        return all(row.get(f"module{m:02d}_{cam}_device_ts_ms", "") != "" for m in (1, 2, 3) for cam in CAMERAS)

    tail_removed = 0
    while rows_out and not has_all_cameras(rows_out[-1]):
        rows_out.pop()
        tail_removed += 1
    stats["camera_rows_before_gap_fill"] = {str(k): len(v) for k, v in per_key.items()}
    stats["gap_filled_rows"] = len(rows_out)
    stats["gap_fill_tail_removed_rows"] = tail_removed
    stats["long_camera_gap_delete_threshold_periods"] = LONG_CAMERA_GAP_DELETE_THRESHOLD_PERIODS
    stats["gap_fill_rule"] = (
        "Per camera, insert blank slots only for short device_ts_ms gaps up to "
        f"{LONG_CAMERA_GAP_DELETE_THRESHOLD_PERIODS} frame periods; longer gaps are treated as "
        "recording interruptions and suppressed instead of being expanded into many blank rows. "
        "Keep internal short-gap blanks, trim only trailing incomplete rows."
    )
    return rows_out, stats


def nearest_index(sorted_values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_values, targets)
    idx = np.clip(idx, 1, len(sorted_values) - 1)
    left = idx - 1
    choose_left = np.abs(sorted_values[left] - targets) <= np.abs(sorted_values[idx] - targets)
    return np.where(choose_left, left, idx)


def zscore(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    if s < 1e-12:
        return x * 0.0
    return (x - float(np.mean(x))) / s


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(zscore(a) * zscore(b)))


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x.copy()
    kernel = np.ones(n, dtype=np.float64) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), kernel, mode="valid")


def find_active_segments(signal: np.ndarray, sample_rate: float, count: int = 10) -> list[tuple[int, int]]:
    win = max(120, int(sample_rate * 16))
    step = max(30, int(sample_rate * 4))
    hp = np.abs(signal - moving_average(signal, max(3, int(sample_rate * 0.5))))
    candidates = []
    for start in range(0, max(1, len(signal) - win), step):
        end = start + win
        score = float(np.percentile(hp[start:end], 95) + np.std(hp[start:end]))
        candidates.append((score, start, end))
    candidates.sort(reverse=True)
    picked = []
    for _score, start, end in candidates:
        if all(abs(start - s) > win // 2 for s, _e in picked):
            picked.append((start, end))
        if len(picked) >= count:
            break
    return sorted(picked)


def read_mocap_derivatives(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    out = {"time_sec": np.array([float(r["time_sec"]) for r in rows], dtype=np.float64)}
    for wrist in WRISTS:
        for axis in ("x", "y", "z", "norm"):
            out[f"{wrist}_g{axis}"] = np.array([float(r[f"{wrist}_gyro_{axis}_rad_s"]) for r in rows], dtype=np.float64)
    return out


def search_alignment(
    imus: dict[str, dict[str, np.ndarray]], mocap: dict[str, np.ndarray], scale_range: tuple[float, float, float]
) -> dict:
    # Use module02 and module03 wrists as the most useful pair. Search obvious wave segments instead of assuming T-pose.
    module_axes = {
        "module02": ("gx_rad_s", "gy_rad_s", "gz_rad_s"),
        "module03": ("gx_rad_s", "gy_rad_s", "gz_rad_s"),
    }
    mocap_axes = [(w, f"{w}_g{a}") for w in WRISTS for a in ("x", "y", "z")]
    imu_segments = {}
    for mod in module_axes:
        imu = imus[mod]
        rate = 1000.0 / np.median(np.diff(imu["gyro_device_ts_ms"]))
        strength = np.sqrt(imu["gx_rad_s"] ** 2 + imu["gy_rad_s"] ** 2 + imu["gz_rad_s"] ** 2)
        imu_segments[mod] = find_active_segments(strength, rate, count=8)

    scale_min, scale_max, scale_step = scale_range
    scales = np.arange(scale_min, scale_max + scale_step * 0.5, scale_step)
    best = {"score": -999.0}
    for scale in scales:
        rows = []
        for mod, axes in module_axes.items():
            imu = imus[mod]
            imu_t = imu["gyro_device_ts_ms"] / 1000.0
            imu_t0 = float(imu_t[0])
            for seg_start, seg_end in imu_segments[mod]:
                seg_t_rel_base = imu_t[seg_start:seg_end] - imu_t[seg_start]
                for imu_axis in axes:
                    ds = 4
                    seg_t_rel = seg_t_rel_base[::ds]
                    imu_sig = moving_average(imu[imu_axis][seg_start:seg_end:ds], 3)
                    duration = seg_t_rel[-1] * scale
                    max_start = mocap["time_sec"][-1] - duration
                    if max_start <= 0:
                        continue
                    expected_start = (float(imu_t[seg_start]) - imu_t0) * scale
                    lo = max(0.0, expected_start - 90.0)
                    hi = min(max_start, expected_start + 90.0)
                    if hi <= lo:
                        continue
                    candidate_times = np.arange(lo, hi, 1.0)
                    for wrist, mocap_key in mocap_axes:
                        mocap_sig_full = moving_average(mocap[mocap_key], 9)
                        best_seg = (0.0, 0.0, 1)
                        for start_t in candidate_times:
                            vals = np.interp(start_t + seg_t_rel * scale, mocap["time_sec"], mocap_sig_full)
                            c = corr(imu_sig, vals)
                            if abs(c) > abs(best_seg[0]):
                                best_seg = (c, start_t, 1 if c >= 0 else -1)
                        rows.append(
                            {
                                "module": mod,
                                "seg_start": seg_start,
                                "seg_end": seg_end,
                                "imu_axis": imu_axis,
                                "wrist": wrist,
                                "mocap_axis": mocap_key,
                                "corr": best_seg[0],
                                "mocap_start": best_seg[1],
                                "sign": best_seg[2],
                                "imu_start": imu_t[seg_start],
                            }
                        )
        by_module = {}
        for mod in module_axes:
            mod_rows = []
            imu0 = float(imus[mod]["gyro_device_ts_ms"][0]) / 1000.0
            for r in rows:
                if r["module"] != mod:
                    continue
                rr = dict(r)
                rr["offset_sec"] = rr["mocap_start"] - (rr["imu_start"] - imu0) * scale
                mod_rows.append(rr)
            by_module[mod] = sorted(mod_rows, key=lambda r: abs(r["corr"]), reverse=True)[:80]
        if any(len(v) == 0 for v in by_module.values()):
            continue

        best_pair = None
        for r2 in by_module["module02"]:
            for r3 in by_module["module03"]:
                offsets = [r2["offset_sec"], r3["offset_sec"]]
                offset_diff = abs(offsets[0] - offsets[1])
                mean_corr = float(np.mean([abs(r2["corr"]), abs(r3["corr"])]))
                score = mean_corr - 0.08 * offset_diff
                if best_pair is None or score > best_pair["score"]:
                    best_pair = {
                        "score": score,
                        "mean_abs_corr": mean_corr,
                        "offset_diff_sec": offset_diff,
                        "global_mocap_offset_sec": float(np.mean(offsets)),
                        "selected": {"module02": r2, "module03": r3},
                        "top_matches": [r2, r3],
                    }
        if best_pair and best_pair["score"] > best["score"]:
            best = {
                "score": best_pair["score"],
                "mean_abs_corr": best_pair["mean_abs_corr"],
                "offset_diff_sec": best_pair["offset_diff_sec"],
                "global_scale": float(scale),
                "global_mocap_offset_sec": best_pair["global_mocap_offset_sec"],
                "selected": best_pair["selected"],
                "top_matches": best_pair["top_matches"],
                "all_candidate_rows": rows,
            }

    best["scale_search_range"] = f"[{scale_min}, {scale_max}]"
    return best


def load_mocap_wide_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    return fields, rows


def build_aligned(root: Path, mocap_wide: Path, align: dict, outdir: Path, mocap_offset_sec: float = 0.0) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    camera_rows, cam_stats = make_camera_master(root)
    imus = {
        "module01": read_imu(next(root.glob("module01_*_imu.csv"))),
        "module02": read_imu(next(root.glob("module02_*_imu.csv"))),
        "module03": read_imu(next(root.glob("module03_*_imu.csv"))),
    }
    imu_raw_keys = {
        module: list(imu["raw_rows"][0].keys()) if len(imu["raw_rows"]) else []
        for module, imu in imus.items()
    }
    mocap_fields, mocap_rows = load_mocap_wide_rows(mocap_wide)
    mocap_t = np.array([float(r["time_sec"]) for r in mocap_rows], dtype=np.float64)
    mocap_rate_hz = 1.0 / float(np.median(np.diff(mocap_t)))
    mocap_half_frame_ms = 500.0 / mocap_rate_hz

    output_rows = []
    missing_imu = 0
    mocap_dt = []
    for row in camera_rows:
        out = {
            k: v
            for k, v in row.items()
            if not (k.endswith("_gap_filled") or k.endswith("_slot_device_ts_ms"))
        }
        module_targets = {}
        for module in (1, 2, 3):
            prefix_base = f"module{module:02d}"
            cam_times = [
                ffloat(row[f"{prefix_base}_{cam}_slot_device_ts_ms"])
                for cam in CAMERAS
                if row.get(f"{prefix_base}_{cam}_slot_device_ts_ms", "") != ""
            ]
            if not cam_times:
                module_targets[prefix_base] = float("nan")
                missing_imu += 1
                continue
            # The median is robust to one camera stream being temporarily shifted
            # by a reconstructed missing-frame slot.
            target = float(np.median(cam_times))
            module_targets[prefix_base] = target
            imu = imus[prefix_base]
            idx = int(nearest_index(imu["gyro_device_ts_ms"], np.array([target]))[0])
            raw = imu["raw_rows"][idx]
            imu_dt = abs(float(raw["gyro_device_ts_ms"]) - target)
            if imu_dt > IMU_NEAREST_THRESHOLD_MS:
                missing_imu += 1
                for k in imu_raw_keys[prefix_base]:
                    out[f"{prefix_base}_imu_{k}"] = ""
            else:
                for k, v in raw.items():
                    out[f"{prefix_base}_imu_{k}"] = v

        # Map both wrist modules through the same camera-elapsed-time model.
        mocap_targets = []
        for mod in ("module02", "module03"):
            if not math.isfinite(module_targets.get(mod, float("nan"))):
                continue
            cam_avg_ms = module_targets[mod]
            camera_origins = align.get("camera_time_origins_ms", {})
            if mod in camera_origins:
                camera_elapsed_sec = (cam_avg_ms - float(camera_origins[mod])) / 1000.0
                mocap_targets.append(
                    align["global_mocap_offset_sec"] + camera_elapsed_sec * align["global_scale"]
                )
            elif "global_mocap_offset_sec" in align:
                imu0_sec = float(imus[mod]["gyro_device_ts_ms"][0]) / 1000.0
                mocap_targets.append(align["global_mocap_offset_sec"] + ((cam_avg_ms / 1000.0) - imu0_sec) * align["global_scale"])
            else:
                sel = align["selected"][mod]
                mocap_targets.append(sel["mocap_start"] + ((cam_avg_ms / 1000.0) - sel["imu_start"]) * align["global_scale"])
        if not mocap_targets:
            out["mocap_time_sec_target"] = ""
            out["mocap_nearest_time_sec"] = ""
            out["mocap_nearest_dt_ms"] = ""
            out["mocap_valid"] = 0
            output_rows.append(out)
            continue
        mocap_target_raw = float(np.mean(mocap_targets))
        mocap_target = mocap_target_raw + mocap_offset_sec
        mi = int(nearest_index(mocap_t, np.array([mocap_target]))[0])
        dt = float(mocap_t[mi] - mocap_target)
        valid = 1 if (mocap_t[0] <= mocap_target <= mocap_t[-1]) else 0
        out["mocap_time_sec_target"] = f"{mocap_target:.9f}"
        out["mocap_time_sec_target_before_manual_offset"] = f"{mocap_target_raw:.9f}"
        out["mocap_nearest_time_sec"] = f"{mocap_t[mi]:.9f}" if valid else ""
        out["mocap_nearest_dt_ms"] = f"{dt * 1000.0:.9f}" if valid else ""
        out["mocap_valid"] = valid
        if valid:
            mocap_dt.append(dt * 1000.0)
            for k in mocap_fields:
                out[f"mocap_{k}"] = mocap_rows[mi][k]
        output_rows.append(out)

    before = len(output_rows)
    output_rows = [r for r in output_rows if r["mocap_valid"] == 1]
    # Reindex seq after overlap trim. The pre-trim sequence is intentionally
    # omitted from the final delivery because it is only an internal index.
    for i, r in enumerate(output_rows):
        r["seq"] = i

    fields = list(output_rows[0].keys())
    out_csv = outdir / "aligned_30hz.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(output_rows)

    selected = align.get("selected", {})
    pairing = align.get("pairing", {})
    module_results = align.get("modules", {})
    report = {
        "output_csv": str(out_csv),
        "rows": len(output_rows),
        "columns": len(fields),
        "rows_before_overlap_trim": before,
        "overlap_trim_removed_rows": before - len(output_rows),
        "mocap_valid_rows": len(output_rows),
        "mocap_invalid_rows": 0,
        "global_scale": align["global_scale"],
        "alignment_parameters": {
            "method": align.get("method", "legacy segment search"),
            "global_scale": align["global_scale"],
            "module02": selected.get("module02", {"wrist": pairing.get("module02"), **module_results.get("module02", {})}),
            "module03": selected.get("module03", {"wrist": pairing.get("module03"), **module_results.get("module03", {})}),
            "scale_search_range": align.get("scale_search_range", ""),
            "global_mocap_offset_sec": align.get("global_mocap_offset_sec"),
            "camera_time_origins_ms": align.get("camera_time_origins_ms"),
            "global_vector_corr": align.get("global_vector_corr"),
            "cross_validation": align.get("cross_validation"),
            "local_lag_validation": align.get("local_lag_validation"),
            "module_offset_diff_sec": align.get("offset_diff_sec"),
            "mean_abs_corr": align.get("mean_abs_corr"),
            "manual_mocap_time_offset_sec": mocap_offset_sec,
            "manual_mocap_time_offset_frames_at_30hz": mocap_offset_sec * 30.0,
            "manual_offset_reason": "Optional explicit correction only; the global optimizer normally leaves this at zero.",
            "note": "The accepted mapping uses both wrists and all three body-frame angular-velocity axes over the full recording.",
        },
        "mocap_dt_ms_valid_rows": {
            "count": len(mocap_dt),
            "median": float(np.median(mocap_dt)) if mocap_dt else None,
            "max_abs": float(np.max(np.abs(mocap_dt))) if mocap_dt else None,
        },
        "mocap_source_rate_hz": mocap_rate_hz,
        "mocap_half_frame_limit_ms": mocap_half_frame_ms,
        "missing_imu_count": missing_imu,
        "imu_nearest_threshold_ms": IMU_NEAREST_THRESHOLD_MS,
        "camera_alignment_timestamp_field": CAMERA_ALIGNMENT_TS_FIELD,
        "seq_reindexed_after_overlap_trim": True,
        "first_seq_after_reindex": output_rows[0]["seq"] if output_rows else None,
        "last_seq_after_reindex": output_rows[-1]["seq"] if output_rows else None,
        "camera_stats": cam_stats,
        "source_files": {
            "dataset_root": str(root),
            "mocap_wide": str(mocap_wide),
            "timestamps": str(root / "timestamps.csv"),
            "summary": str(root / "summary.json"),
        },
    }
    (outdir / "aligned_30hz_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset recording folder, e.g. ...\\0715\\001")
    parser.add_argument("--bvh", default=None)
    parser.add_argument("--mocap-wide", default=None, help="Existing mocap wide CSV to use directly.")
    parser.add_argument("--mocap-offset-sec", type=float, default=DEFAULT_MOCAP_TIME_OFFSET_SEC)
    parser.add_argument("--global-alignment-report", default=None)
    args = parser.parse_args()

    root = Path(args.dataset)
    bvh = Path(args.bvh) if args.bvh else root.parent / f"{root.name}.bvh"
    aligned = root / "aligned_data"
    mocap_dir = aligned / "mocap_source_csv"
    print(f"dataset={root}")
    print(f"bvh={bvh}")
    if args.mocap_wide:
        mocap_wide = Path(args.mocap_wide)
        if not mocap_wide.exists():
            raise FileNotFoundError(mocap_wide)
        print(f"using existing mocap wide={mocap_wide}")
    elif (mocap_dir / "mocap_joints_wide.csv").exists():
        print("reusing existing mocap wide...")
        mocap_meta = json.loads((mocap_dir / "mocap_metadata.json").read_text(encoding="utf-8"))
        mocap_wide = mocap_dir / "mocap_joints_wide.csv"
        print(f"mocap rows={mocap_meta['joints_wide_rows']} cols={mocap_meta['joints_wide_columns']}")
    else:
        print("exporting mocap wide...")
        mocap_meta = export_bvh_wide(bvh, mocap_dir)
        mocap_wide = mocap_dir / "mocap_joints_wide.csv"
        print(f"mocap rows={mocap_meta['joints_wide_rows']} cols={mocap_meta['joints_wide_columns']}")
    global_report_path = (
        Path(args.global_alignment_report)
        if args.global_alignment_report
        else aligned / "global_imu_mocap_alignment_report.json"
    )
    if not global_report_path.exists():
        raise FileNotFoundError(
            "Global IMU-mocap report is required. Run global_imu_mocap_alignment.py first: "
            f"{global_report_path}"
        )
    global_report = json.loads(global_report_path.read_text(encoding="utf-8"))
    align = {
        "method": global_report["method"],
        "score": global_report["global_vector_corr"],
        "global_vector_corr": global_report["global_vector_corr"],
        "global_scale": global_report["global_scale"],
        "global_mocap_offset_sec": global_report["global_offset_sec"],
        "camera_time_origins_ms": global_report["camera_time_origins_ms"],
        "pairing": global_report["pairing"],
        "modules": global_report["modules"],
        "cross_validation": global_report["cross_validation"],
        "local_lag_validation": global_report["local_lag_validation"],
    }
    (aligned / "alignment_search_report.json").write_text(json.dumps(align, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"using global scale={align['global_scale']:.8f} "
        f"offset={align['global_mocap_offset_sec']:.4f}s score={align['score']:.4f}"
    )
    print("building final aligned csv...")
    report = build_aligned(root, mocap_wide, align, aligned, mocap_offset_sec=args.mocap_offset_sec)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
