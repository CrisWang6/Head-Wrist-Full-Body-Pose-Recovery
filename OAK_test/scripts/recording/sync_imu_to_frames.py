#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path


def parse_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_imu_rows(session: Path) -> dict[tuple[int, str], list[dict]]:
    by_module: dict[tuple[int, str], list[dict]] = {}
    for path in sorted(session.glob("module*_imu.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            module_key: tuple[int, str] | None = None
            for row in reader:
                module = int(row["module"])
                mxid = row["mxid"]
                rotation_ts = parse_float(row.get("rotation_device_ts_ms"))
                accel_ts = parse_float(row.get("accel_device_ts_ms"))
                gyro_ts = parse_float(row.get("gyro_device_ts_ms"))
                if rotation_ts is not None:
                    imu_ts = rotation_ts
                elif accel_ts is None and gyro_ts is None:
                    continue
                elif accel_ts is None:
                    imu_ts = gyro_ts
                elif gyro_ts is None:
                    imu_ts = accel_ts
                else:
                    imu_ts = 0.5 * (accel_ts + gyro_ts)
                row["_imu_ts_ms"] = imu_ts
                rows.append(row)
                module_key = (module, mxid)
            if module_key is not None:
                rows.sort(key=lambda item: item["_imu_ts_ms"])
                by_module[module_key] = rows
    return by_module


def nearest_row(rows: list[dict], target_ts_ms: float) -> dict | None:
    if not rows:
        return None
    times = [row["_imu_ts_ms"] for row in rows]
    pos = bisect.bisect_left(times, target_ts_ms)
    candidates = []
    if pos < len(rows):
        candidates.append(rows[pos])
    if pos > 0:
        candidates.append(rows[pos - 1])
    return min(candidates, key=lambda row: abs(row["_imu_ts_ms"] - target_ts_ms))


def frame_timestamp(row: dict) -> float | None:
    for key in ("exposure_middle_ts_ms", "device_ts_ms", "exposure_start_ts_ms", "exposure_end_ts_ms"):
        value = parse_float(row.get(key))
        if value is not None:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Match recorded OAK frame timestamps to nearest same-device IMU samples.")
    parser.add_argument("session", type=Path, help="Recording session folder containing timestamps.csv and module*_imu.csv.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    session = args.session
    timestamps_path = session / "timestamps.csv"
    if not timestamps_path.exists():
        raise SystemExit(f"missing {timestamps_path}")

    imu_by_module = load_imu_rows(session)
    out_path = args.out or (session / "camera_imu_sync.csv")
    with timestamps_path.open("r", newline="", encoding="utf-8") as src, out_path.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        fieldnames = [
            *reader.fieldnames,
            "matched_imu_ts_ms",
            "imu_dt_ms",
            "accel_seq",
            "accel_device_ts_ms",
            "ax_m_s2",
            "ay_m_s2",
            "az_m_s2",
            "gyro_seq",
            "gyro_device_ts_ms",
            "gx_rad_s",
            "gy_rad_s",
            "gz_rad_s",
            "rotation_seq",
            "rotation_device_ts_ms",
            "quat_i",
            "quat_j",
            "quat_k",
            "quat_real",
            "rotation_accuracy_rad",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        matched = 0
        total = 0
        for frame in reader:
            total += 1
            module_key = (int(frame["module"]), frame["mxid"])
            target = frame_timestamp(frame)
            imu = nearest_row(imu_by_module.get(module_key, []), target) if target is not None else None
            out = dict(frame)
            if imu is not None and target is not None:
                matched += 1
                out.update(
                    {
                        "matched_imu_ts_ms": imu["_imu_ts_ms"],
                        "imu_dt_ms": imu["_imu_ts_ms"] - target,
                        "accel_seq": imu["accel_seq"],
                        "accel_device_ts_ms": imu["accel_device_ts_ms"],
                        "ax_m_s2": imu["ax_m_s2"],
                        "ay_m_s2": imu["ay_m_s2"],
                        "az_m_s2": imu["az_m_s2"],
                        "gyro_seq": imu["gyro_seq"],
                        "gyro_device_ts_ms": imu["gyro_device_ts_ms"],
                        "gx_rad_s": imu["gx_rad_s"],
                        "gy_rad_s": imu["gy_rad_s"],
                        "gz_rad_s": imu["gz_rad_s"],
                        "rotation_seq": imu.get("rotation_seq", ""),
                        "rotation_device_ts_ms": imu.get("rotation_device_ts_ms", ""),
                        "quat_i": imu.get("quat_i", ""),
                        "quat_j": imu.get("quat_j", ""),
                        "quat_k": imu.get("quat_k", ""),
                        "quat_real": imu.get("quat_real", ""),
                        "rotation_accuracy_rad": imu.get("rotation_accuracy_rad", ""),
                        "roll_deg": imu.get("roll_deg", ""),
                        "pitch_deg": imu.get("pitch_deg", ""),
                        "yaw_deg": imu.get("yaw_deg", ""),
                    }
                )
            writer.writerow(out)
    print(f"wrote {out_path} matched={matched}/{total}")


if __name__ == "__main__":
    main()
