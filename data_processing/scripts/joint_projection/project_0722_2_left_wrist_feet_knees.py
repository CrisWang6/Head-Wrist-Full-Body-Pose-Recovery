#!/usr/bin/env python3
"""Project BVH knees/feet into the three CH3_01-anchored left-wrist cameras."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import project_0722_abx2_subject_scaled as kin
import project_joints as base
from compare_0722_rigid_wrist_replacement import R_WRIST_RIGID, WRIST_TO_RIGID_MM


HERE = Path(__file__).resolve().parent
RECORDING = Path(r"C:\Users\hand\Desktop\Dataset\0722_2\0711_035935")
ALIGNED = RECORDING / "aligned_data" / "aligned_30hz.csv"
TIMESTAMPS = RECORDING / "timestamps.csv"
CONFIG = HERE / "projection_config.json"
PAIR_EXTRINSICS = (
    HERE.parent
    / "calibrate"
    / "parameters"
    / "extrinsics"
    / "left_wrist_extrinsics_pairs_1920x1200.json"
)
OUTPUT = HERE / "validation_0722_2_left_wrist_ch301_feet_knees"
FFMPEG = Path(
    r"C:\Users\hand\miniconda3\Lib\site-packages\imageio_ffmpeg"
    r"\binaries\ffmpeg-win-x86_64-v7.1.exe"
)
CAMERAS = ("CAM_A", "CAM_B", "CAM_C")
VIDEO_PREFIX = "module02_13652E00"
WIDTH, HEIGHT = 1920, 1200
VISIBLE_SAMPLES = 6
EMPTY_SAMPLES = 0

TARGETS = ("LeftLeg", "RightLeg", "LeftFoot", "RightFoot")
COLORS = {
    "LeftLeg": (0, 230, 255),
    "RightLeg": (0, 230, 255),
    "LeftFoot": (255, 0, 255),
    "RightFoot": (255, 0, 255),
}
LABELS = {
    "LeftLeg": "L knee",
    "RightLeg": "R knee",
    "LeftFoot": "L foot",
    "RightFoot": "R foot",
}
PTS_PATTERN = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)")


def timestamp_key(value: str | float) -> float:
    return round(float(value), 6)


def read_aligned() -> list[dict[str, str]]:
    with ALIGNED.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_timestamp_ordinals() -> dict[str, dict[float, int]]:
    values: dict[str, list[float]] = {camera: [] for camera in CAMERAS}
    with TIMESTAMPS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["module"] == "2" and row["camera"] in values:
                values[row["camera"]].append(timestamp_key(row["device_ts_ms"]))
    return {
        camera: {timestamp: ordinal for ordinal, timestamp in enumerate(rows)}
        for camera, rows in values.items()
    }


def calibrated_camera_poses(
    models: dict[str, dict[str, object]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return camera origin/axes in wrist coordinates, anchored by CAM_A CAD pose."""
    anchor = models["module02_CAM_A"]
    p_wrist_a = np.asarray(anchor["position_anchor_mm"], dtype=np.float64)
    r_wrist_a = np.asarray(anchor["R_anchor_camera"], dtype=np.float64)
    pairs = base.load_json(PAIR_EXTRINSICS)["pairs"]
    poses = {"CAM_A": (p_wrist_a, r_wrist_a)}
    for camera in ("CAM_B", "CAM_C"):
        transform = np.asarray(
            pairs[f"CAM_A_to_{camera}"]["T_dst_src"], dtype=np.float64
        )
        r_camera_a = transform[:3, :3]
        t_camera_a_mm = transform[:3, 3] * 1000.0
        r_wrist_camera = r_wrist_a @ r_camera_a.T
        p_wrist_camera = p_wrist_a - r_wrist_camera @ t_camera_a_mm
        poses[camera] = (p_wrist_camera, r_wrist_camera)
    return poses


def valid_row(
    row: dict[str, str],
    ordinals: dict[str, dict[float, int]],
) -> bool:
    required = [
        *(f"mocap_{joint}_world_{axis}" for joint in (*TARGETS, "Head") for axis in "xyz"),
        *(
            f"mocap_{rigid}_world_{suffix}"
            for rigid in ("CH3_01_Rigid_K", "CH3_08_Rigid_K")
            for suffix in ("x", "y", "z", "qw", "qx", "qy", "qz")
        ),
    ]
    if row.get("mocap_valid") not in {"1", "1.0", "True", "true"}:
        return False
    if not all(row.get(field) for field in required):
        return False
    for camera in CAMERAS:
        field = f"module02_{camera}_device_ts_ms"
        if not row.get(field) or timestamp_key(row[field]) not in ordinals[camera]:
            return False
    return True


def xyz(row: dict[str, str], prefix: str, scale: float) -> np.ndarray:
    return np.asarray(
        [float(row[f"{prefix}_{axis}"]) * scale for axis in "xyz"],
        dtype=np.float64,
    )


def quaternion(row: dict[str, str], prefix: str) -> np.ndarray:
    value = np.asarray(
        [float(row[f"{prefix}_q{axis}"]) for axis in "wxyz"], dtype=np.float64
    )
    return value / np.linalg.norm(value)


def geometry(
    row: dict[str, str],
    camera: str,
    model: dict[str, object],
    pose: tuple[np.ndarray, np.ndarray],
) -> dict[str, object]:
    target_world = np.asarray(
        [xyz(row, f"mocap_{joint}_world", 10.0) for joint in TARGETS]
    )

    # Estimate only the global translation between PWR rigid coordinates and
    # the BVH skeleton world. Camera orientation/translation still comes from
    # CH3_01 and the measured wrist/camera transforms.
    bvh_head = xyz(row, "mocap_Head_world", 10.0)
    head_rigid_position = xyz(row, "mocap_CH3_08_Rigid_K_world", 1000.0)
    r_world_head_rigid = kin.q_to_matrix(
        quaternion(row, "mocap_CH3_08_Rigid_K_world")
    )
    head_axes = np.column_stack(
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    pwr_head_joint = (
        head_rigid_position + r_world_head_rigid @ head_joint_in_rigid_mm
    )
    pwr_to_bvh_translation = bvh_head - pwr_head_joint

    rigid_position = xyz(row, "mocap_CH3_01_Rigid_K_world", 1000.0)
    r_world_rigid = kin.q_to_matrix(
        quaternion(row, "mocap_CH3_01_Rigid_K_world")
    )
    r_world_wrist = r_world_rigid @ R_WRIST_RIGID["Left"].T
    p_wrist_camera, r_wrist_camera = pose
    camera_position = (
        rigid_position
        + r_world_wrist @ (p_wrist_camera - WRIST_TO_RIGID_MM["Left"])
        + pwr_to_bvh_translation
    )
    camera_rotation = r_world_wrist @ r_wrist_camera
    target_camera = (camera_rotation.T @ (target_world - camera_position).T).T
    uv, model_valid = base.omni_project(target_camera, model)
    in_image = (
        model_valid
        & np.all(np.isfinite(uv), axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < WIDTH)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < HEIGHT)
    )
    return {
        "uv": uv,
        "model_valid": model_valid,
        "in_image": in_image,
        "depth_mm": target_camera[:, 2],
        "camera_position_mm": camera_position,
        "camera_rotation": camera_rotation,
    }


def stratified(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count <= 0:
        return []
    if len(rows) <= count:
        return rows
    bins = np.array_split(np.arange(len(rows)), count)
    return [rows[int(indices[len(indices) // 2])] for indices in bins if len(indices)]


def inspect_timeline(camera: str) -> tuple[np.ndarray, dict[str, object]]:
    source = RECORDING / f"{VIDEO_PREFIX}_{camera}.h265"
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-i",
        str(source),
        "-vf",
        "showinfo",
        "-an",
        "-f",
        "null",
        "NUL",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    pts: list[int] = []
    events: list[str] = []
    assert process.stderr is not None
    for line in process.stderr:
        match = PTS_PATTERN.search(line)
        if match:
            pts.append(int(match.group(2)))
        elif any(
            token in line
            for token in (
                "Could not find ref",
                "Duplicate POC",
                "Skipping invalid undecodable",
            )
        ):
            events.append(line.strip()[-180:])
    return_code = process.wait()
    if return_code != 0 or not pts:
        raise RuntimeError(f"Timeline decode failed for {camera}: {return_code}")
    steps = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    nominal = min(steps)
    capture = np.rint((np.asarray(pts) - pts[0]) / nominal).astype(np.int64)
    return capture, {
        "decoded_frames": len(pts),
        "last_capture_index": int(capture[-1]),
        "missing_capture_indices": int(capture[-1] + 1 - len(capture)),
        "decoder_event_count": len(events),
        "decoder_events": events,
    }


def extract_images(
    camera: str,
    selected: list[dict[str, object]],
    ordinals: dict[str, dict[float, int]],
    capture_map: np.ndarray,
) -> tuple[dict[int, Path], list[dict[str, int]]]:
    requests: list[dict[str, int]] = []
    for item in selected:
        row = item["row"]
        assert isinstance(row, dict)
        seq = int(row["seq"])
        timestamp = timestamp_key(row[f"module02_{camera}_device_ts_ms"])
        capture_ordinal = ordinals[camera][timestamp]
        decoded_index = int(np.argmin(np.abs(capture_map - capture_ordinal)))
        requests.append(
            {
                "seq": seq,
                "capture_ordinal": capture_ordinal,
                "decoded_index": decoded_index,
            }
        )
    requests.sort(key=lambda item: item["decoded_index"])
    expression = "+".join(f"eq(n\\,{item['decoded_index']})" for item in requests)
    source = RECORDING / f"{VIDEO_PREFIX}_{camera}.h265"
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"select={expression}",
        "-vsync",
        "0",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    destination = OUTPUT / "source_frames" / camera
    destination.mkdir(parents=True, exist_ok=True)
    images: dict[int, Path] = {}
    frame_bytes = WIDTH * HEIGHT * 3
    for item in requests:
        buffer = process.stdout.read(frame_bytes)
        if len(buffer) != frame_bytes:
            break
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        path = destination / f"seq_{item['seq']:06d}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 96])
        images[item["seq"]] = path
    process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0 or len(images) != len(requests):
        raise RuntimeError(
            f"Extraction failed for {camera}: {len(images)}/{len(requests)}, "
            f"return={return_code}, stderr={stderr[-500:]}"
        )
    return images, requests


def draw_overlay(
    source: np.ndarray,
    item: dict[str, object],
    camera: str,
) -> np.ndarray:
    image = source.copy()
    uv = np.asarray(item["uv"])
    inside = np.asarray(item["in_image"], dtype=bool)
    seq = int(item["seq"])
    visible_names = [TARGETS[i] for i in np.flatnonzero(inside)]

    for knee, foot in ((0, 2), (1, 3)):
        if inside[knee] and inside[foot]:
            cv2.line(
                image,
                tuple(np.rint(uv[knee]).astype(int)),
                tuple(np.rint(uv[foot]).astype(int)),
                (0, 255, 120),
                4,
                cv2.LINE_AA,
            )
    for index, joint in enumerate(TARGETS):
        if not inside[index]:
            continue
        point = tuple(np.rint(uv[index]).astype(int))
        cv2.circle(image, point, 10, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, point, 7, COLORS[joint], -1, cv2.LINE_AA)
        cv2.putText(
            image,
            LABELS[joint],
            (point[0] + 12, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            LABELS[joint],
            (point[0] + 12, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            COLORS[joint],
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(image, (0, 0), (WIDTH, 78), (0, 0, 0), -1)
    status = (
        f"visible {len(visible_names)}/4: "
        + (", ".join(LABELS[name] for name in visible_names) if visible_names else "NONE")
    )
    cv2.putText(
        image,
        f"Left wrist {camera} | CH3_01 pose | seq={seq:06d}",
        (24, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        status,
        (24, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (60, 220, 255) if visible_names else (80, 80, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def contact_sheet(paths: list[Path], destination: Path) -> None:
    thumbs: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        thumbs.append(cv2.resize(image, (480, 300), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    columns = 2
    rows = []
    for start in range(0, len(thumbs), columns):
        group = thumbs[start : start + columns]
        if len(group) < columns:
            group.append(np.zeros_like(group[0]))
        rows.append(np.hstack(group))
    cv2.imwrite(str(destination), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])


def remove_empty_visualizations() -> int:
    root = OUTPUT.resolve()
    targets = list(OUTPUT.glob("CAM_?/empty_*.jpg"))
    for path in targets:
        resolved = path.resolve()
        if root not in resolved.parents:
            raise RuntimeError(f"Refusing to delete path outside output: {resolved}")
        resolved.unlink()
    return len(targets)


def render_full_visible(
    camera: str,
    items: list[dict[str, object]],
    ordinals: dict[str, dict[float, int]],
    capture_map: np.ndarray,
) -> list[dict[str, object]]:
    visible = [item for item in items if np.any(item["in_image"])]
    by_decoded_index: dict[int, list[dict[str, object]]] = {}
    for item in visible:
        row = item["row"]
        assert isinstance(row, dict)
        timestamp = timestamp_key(row[f"module02_{camera}_device_ts_ms"])
        capture_ordinal = ordinals[camera][timestamp]
        decoded_index = int(np.argmin(np.abs(capture_map - capture_ordinal)))
        by_decoded_index.setdefault(decoded_index, []).append(item)

    destination = OUTPUT / "full_visible" / camera
    destination.mkdir(parents=True, exist_ok=True)
    source = RECORDING / f"{VIDEO_PREFIX}_{camera}.h265"
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None

    manifest: list[dict[str, object]] = []
    maximum = max(by_decoded_index)
    decoded_index = 0
    frame_bytes = WIDTH * HEIGHT * 3
    while decoded_index <= maximum:
        buffer = process.stdout.read(frame_bytes)
        if len(buffer) != frame_bytes:
            break
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        for item in by_decoded_index.get(decoded_index, []):
            seq = int(item["seq"])
            overlay = draw_overlay(frame, item, camera)
            path = destination / f"seq_{seq:06d}.jpg"
            cv2.imwrite(str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest.append(
                {
                    "camera": camera,
                    "seq": seq,
                    "decoded_index": decoded_index,
                    "visible_count": int(np.count_nonzero(item["in_image"])),
                    "output": str(path),
                }
            )
        decoded_index += 1
        if decoded_index % 1000 == 0:
            print(
                f"{camera}: decoded {decoded_index}, "
                f"saved {len(manifest)}/{len(visible)}",
                flush=True,
            )
    process.stdout.close()
    process.wait()
    if len(manifest) != len(visible):
        raise RuntimeError(
            f"Full render incomplete for {camera}: {len(manifest)}/{len(visible)}"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-visible",
        action="store_true",
        help="Render every aligned frame containing at least one projected target",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    removed_empty = remove_empty_visualizations()
    if removed_empty:
        print(f"Removed {removed_empty} empty visualization images", flush=True)
    config = base.load_json(CONFIG)
    models = base.load_camera_models(config)
    poses = calibrated_camera_poses(models)
    ordinals = load_timestamp_ordinals()
    rows = [row for row in read_aligned() if valid_row(row, ordinals)]
    if not rows:
        raise RuntimeError("No valid aligned rows")

    per_camera: dict[str, list[dict[str, object]]] = {camera: [] for camera in CAMERAS}
    csv_records: list[dict[str, object]] = []
    for row in rows:
        for camera in CAMERAS:
            result = geometry(
                row,
                camera,
                models[f"module02_{camera}"],
                poses[camera],
            )
            item = {
                "seq": int(row["seq"]),
                "row": row,
                **result,
            }
            per_camera[camera].append(item)
            record: dict[str, object] = {
                "seq": int(row["seq"]),
                "camera": camera,
                "visible_count": int(np.count_nonzero(result["in_image"])),
                "no_projection_in_image": int(not np.any(result["in_image"])),
            }
            for index, joint in enumerate(TARGETS):
                record[f"{joint}_x"] = float(result["uv"][index, 0])
                record[f"{joint}_y"] = float(result["uv"][index, 1])
                record[f"{joint}_model_valid"] = int(result["model_valid"][index])
                record[f"{joint}_in_image"] = int(result["in_image"][index])
                record[f"{joint}_camera_z_mm"] = float(result["depth_mm"][index])
            csv_records.append(record)

    selected_by_camera: dict[str, list[dict[str, object]]] = {}
    for camera, items in per_camera.items():
        visible = [item for item in items if np.any(item["in_image"])]
        empty = [item for item in items if not np.any(item["in_image"])]
        selected = stratified(visible, VISIBLE_SAMPLES) + stratified(empty, EMPTY_SAMPLES)
        selected_by_camera[camera] = sorted(selected, key=lambda item: int(item["seq"]))

    with (OUTPUT / "projection_records.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_records[0]))
        writer.writeheader()
        writer.writerows(csv_records)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            camera: executor.submit(inspect_timeline, camera) for camera in CAMERAS
        }
        timelines = {camera: future.result() for camera, future in futures.items()}

    reports: dict[str, object] = {}
    full_manifest: list[dict[str, object]] = []
    for camera in CAMERAS:
        capture_map, timeline_report = timelines[camera]
        selected = selected_by_camera[camera]
        source_images, requests = extract_images(
            camera, selected, ordinals, capture_map
        )
        camera_dir = OUTPUT / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[Path] = []
        samples: list[dict[str, object]] = []
        for item in selected:
            seq = int(item["seq"])
            source = cv2.imread(str(source_images[seq]), cv2.IMREAD_COLOR)
            if source is None:
                raise RuntimeError(f"Missing extracted image {camera} seq={seq}")
            overlay = draw_overlay(source, item, camera)
            status = "visible" if np.any(item["in_image"]) else "empty"
            path = camera_dir / f"{status}_seq_{seq:06d}.jpg"
            cv2.imwrite(str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 96])
            rendered.append(path)
            samples.append(
                {
                    "seq": seq,
                    "status": status,
                    "visible_joints": [
                        TARGETS[index]
                        for index in np.flatnonzero(item["in_image"])
                    ],
                    "output": str(path),
                }
            )
        contact_sheet(rendered, OUTPUT / f"{camera}_contact_sheet.jpg")

        all_items = per_camera[camera]
        counts = np.asarray(
            [np.count_nonzero(item["in_image"]) for item in all_items], dtype=int
        )
        reports[camera] = {
            "valid_aligned_frames": len(all_items),
            "frames_with_any_projection": int(np.count_nonzero(counts > 0)),
            "frames_without_projection": int(np.count_nonzero(counts == 0)),
            "any_projection_rate": float(np.mean(counts > 0)),
            "visible_joint_count_histogram": {
                str(value): int(np.count_nonzero(counts == value))
                for value in range(5)
            },
            "camera_pose_in_wrist": {
                "position_mm": poses[camera][0].tolist(),
                "R_wrist_camera": poses[camera][1].tolist(),
            },
            "timeline": timeline_report,
            "decode_requests": requests,
            "samples": samples,
        }
        if args.full_visible:
            camera_manifest = render_full_visible(
                camera, all_items, ordinals, capture_map
            )
            full_manifest.extend(camera_manifest)
            reports[camera]["full_visible_outputs"] = len(camera_manifest)

    if args.full_visible:
        with (OUTPUT / "full_visible_manifest.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(full_manifest[0]))
            writer.writeheader()
            writer.writerows(full_manifest)

    summary = {
        "schema": "0722_2.left_wrist_ch301_feet_knees.v1",
        "recording": str(RECORDING),
        "rigid_anchor": "CH3_01_Rigid_K",
        "targets": list(TARGETS),
        "world_alignment": "CH3_08 head rigid used only for PWR-to-BVH global translation",
        "camera_extrinsics": (
            "CAM_A CAD wrist pose; CAM_B/C derived from calibrated "
            "left-wrist A-to-B/A-to-C transforms"
        ),
        "empty_visualizations_removed": removed_empty,
        "full_visible_render": bool(args.full_visible),
        "intrinsics": str(
            HERE.parent
            / "calibrate"
            / "parameters"
            / "intrinsics"
            / "left_wrist"
            / "left_wrist_intrinsics_kalibr_omni_1920x1200.json"
        ),
        "reports": reports,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                camera: {
                    "with_projection": report["frames_with_any_projection"],
                    "without_projection": report["frames_without_projection"],
                    "rate": report["any_projection_rate"],
                }
                for camera, report in reports.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
