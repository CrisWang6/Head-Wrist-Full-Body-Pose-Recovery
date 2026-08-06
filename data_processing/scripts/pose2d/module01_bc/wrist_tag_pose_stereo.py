#!/usr/bin/env python3
"""Recover a left-wrist pose from AprilTags in synchronized CAM_B/C H.265 streams."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np


TARGETS = {
    # T_tag_wrist: wrist origin and wrist basis expressed in the tag frame.
    9: {
        "name": "side",
        "t_mm": [16.0, -8.0, -53.0],
        "R": [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    },
    22: {
        "name": "top",
        "t_mm": [0.0, -16.0, -58.0],
        "R": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    },
    32: {
        "name": "dorsal",
        "t_mm": [7.3, -15.7, -53.0],
        "R": [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    },
}

COLORS = {
    "tag": ((0, 0, 255), (0, 255, 0), (255, 0, 0)),
    "wrist": ((255, 0, 255), (0, 215, 255), (255, 255, 0)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--camchain", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--start-offset", type=float, default=0.0)
    p.add_argument("--tag-size", type=float, default=0.0352)
    p.add_argument("--axis-length", type=float, default=0.045)
    p.add_argument("--module", type=int, default=1)
    p.add_argument("--preview-scale", type=float, default=0.5)
    p.add_argument("--max-sync-ms", type=float, default=8.0)
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


def make_transform(R=None, t=None):
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def target_transforms():
    out = {}
    for tag_id, spec in TARGETS.items():
        R = np.asarray(spec["R"], dtype=np.float64)
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-9) or np.linalg.det(R) < 0.999:
            raise ValueError(f"Invalid tag-to-wrist rotation for tag {tag_id}")
        out[tag_id] = make_transform(R, np.asarray(spec["t_mm"]) / 1000.0)
    return out


def load_cameras(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name in ("CAM_B", "CAM_C"):
        c = data["cameras"][name]
        out[name] = {
            "K": np.asarray(c["K"], dtype=np.float64),
            "D": np.asarray(c["D"], dtype=np.float64).reshape(4, 1),
            "xi": float(c["xi"]),
            "size": tuple(c.get("resolution", c.get("image_size"))),
        }
    return out


def load_t_b_c(path: Path):
    # In this exported chain, cam1.T_cn_cnm1 is stored as T_CAM_B_CAM_C.
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return np.asarray(data["cam1"]["T_cn_cnm1"], dtype=np.float64)


def load_timestamps(path: Path, module: int):
    rows = {"CAM_B": [], "CAM_C": []}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if int(row["module"]) == module and row["camera"] in rows:
                row["_middle"] = float(row["exposure_middle_ts_ms"])
                rows[row["camera"]].append(row)
    if not rows["CAM_B"] or not rows["CAM_C"]:
        raise RuntimeError("CAM_B/C timestamp rows were not found")
    return rows


def select_duration(rows, duration_s, start_offset_s):
    recording_start_ms = max(
        rows["CAM_B"][0]["_middle"], rows["CAM_C"][0]["_middle"]
    )
    start_ms = recording_start_ms + start_offset_s * 1000.0
    end_ms = start_ms + duration_s * 1000.0
    selected = {}
    for cam in ("CAM_B", "CAM_C"):
        selected[cam] = [
            (index, row)
            for index, row in enumerate(rows[cam])
            if start_ms <= row["_middle"] < end_ms
        ]
    return start_ms, end_ms, selected


def load_decode_map(showinfo_path: Path, timestamp_rows):
    starts = []
    total = 0
    for row in timestamp_rows:
        starts.append(total)
        total += int(row["bytes"])
    row_to_decoded = {}
    pattern = re.compile(r"n:\s*(\d+).*?pos:\s*(\d+)")
    with showinfo_path.open(errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            decoded_index, position = map(int, match.groups())
            at = bisect.bisect_left(starts, position)
            choices = [i for i in (at - 1, at) if 0 <= i < len(starts)]
            row_index = min(choices, key=lambda i: abs(starts[i] - position))
            row_to_decoded[row_index] = decoded_index
    if not row_to_decoded:
        raise RuntimeError(f"No frame mappings found in {showinfo_path}")
    return row_to_decoded


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_range(path: Path, first_index: int, last_index: int, width: int, height: int):
    # FFmpeg's decoded index is the index used by the precomputed showinfo maps.
    # OpenCV VideoCapture drifts on CAM_B after damaged initial reference frames.
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-err_detect", "ignore_err", "-i", str(path),
        "-vf", f"select=between(n\\,{first_index}\\,{last_index})",
        "-frames:v", str(last_index - first_index + 1), "-pix_fmt", "bgr24",
        "-vsync", "0", "-f", "rawvideo", "-",
    ]
    decoder = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    frame_bytes = width * height * 3
    frames = {}
    for index in range(first_index, last_index + 1):
        raw = read_exact(decoder.stdout, frame_bytes)
        if len(raw) != frame_bytes:
            decoder.kill()
            raise RuntimeError(f"Decode stopped at frame {index} in {path}")
        frames[index] = np.frombuffer(raw, dtype=np.uint8).reshape(
            height, width, 3
        ).copy()
    if decoder.wait() != 0:
        raise RuntimeError(f"FFmpeg failed while decoding {path}")
    return frames


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.006
    params.maxMarkerPerimeterRate = 4.0
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return dictionary, params


def detect(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if isinstance(detector, tuple):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, detector[0], parameters=detector[1])
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    result = {}
    if ids is None:
        return result
    for tag_id, points in zip(ids.reshape(-1), corners):
        tag_id = int(tag_id)
        if tag_id in TARGETS:
            result[tag_id] = points.reshape(4, 2).astype(np.float64)
    return result


def tag_object_points(size):
    s = size / 2.0
    return np.asarray(
        [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
        dtype=np.float64,
    )


def project_omni(points, camera, T_cam_obj):
    rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
    # OpenCV's omnidir binding misreads a strided column view; force contiguous.
    tvec = np.ascontiguousarray(T_cam_obj[:3, 3]).reshape(3, 1)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
    image, _ = cv2.omnidir.projectPoints(
        pts, rvec, tvec, camera["K"], camera["xi"], camera["D"]
    )
    return image.reshape(-1, 2)


def residual(params, obj, image, camera):
    R, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
    T = make_transform(R, params[3:])
    return (project_omni(obj, camera, T) - image).reshape(-1)


def refine_pose(params, obj, image, camera, iterations=18):
    params = np.asarray(params, dtype=np.float64).copy()
    damping = 1e-3
    step_sizes = np.full(6, 1e-5, dtype=np.float64)
    best = float(np.linalg.norm(residual(params, obj, image, camera)))
    for _ in range(iterations):
        r = residual(params, obj, image, camera)
        J = np.empty((r.size, 6), dtype=np.float64)
        for col, eps in enumerate(step_sizes):
            d = np.zeros(6)
            d[col] = eps
            J[:, col] = (
                residual(params + d, obj, image, camera)
                - residual(params - d, obj, image, camera)
            ) / (2.0 * eps)
        try:
            delta = np.linalg.solve(J.T @ J + damping * np.eye(6), -J.T @ r)
        except np.linalg.LinAlgError:
            break
        candidate = params + delta
        error = float(np.linalg.norm(residual(candidate, obj, image, camera)))
        if error < best:
            params, best = candidate, error
            damping *= 0.4
        else:
            damping *= 6.0
        if np.linalg.norm(delta) < 1e-9:
            break
    return params, best / math.sqrt(len(obj))


def solve_tag_pose(corners, camera, obj):
    normalized = cv2.omnidir.undistortPoints(
        corners.reshape(-1, 1, 2), camera["K"], camera["D"],
        np.asarray([camera["xi"]], dtype=np.float64),
        np.eye(3, dtype=np.float64),
    ).reshape(-1, 2)
    result = cv2.solvePnPGeneric(
        obj, normalized, np.eye(3, dtype=np.float64), None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not result[0]:
        return None
    candidates = []
    for rvec, tvec in zip(result[1], result[2]):
        params, rmse = refine_pose(
            np.r_[rvec.reshape(3), tvec.reshape(3)], obj, corners, camera
        )
        R, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
        T = make_transform(R, params[3:])
        distance = float(np.linalg.norm(T[:3, 3]))
        if T[2, 3] > 0 and distance < 5.0 and np.isfinite(rmse):
            candidates.append((rmse, T))
    if not candidates:
        return None
    rmse, T = min(candidates, key=lambda item: item[0])
    if rmse > 5.0:
        return None
    return T, rmse


def filter_isolated_tracks(solved, max_gap_frames=3):
    """Drop one-frame target IDs, which are commonly background-board hits."""
    keep = set()
    for cam in ("CAM_B", "CAM_C"):
        for tag_id in TARGETS:
            indices = sorted(
                frame_index
                for (name, frame_index), poses in solved.items()
                if name == cam and tag_id in poses
            )
            for pos, frame_index in enumerate(indices):
                previous_ok = pos > 0 and frame_index - indices[pos - 1] <= max_gap_frames
                next_ok = (
                    pos + 1 < len(indices)
                    and indices[pos + 1] - frame_index <= max_gap_frames
                )
                if previous_ok or next_ok:
                    keep.add((cam, frame_index, tag_id))
    removed = {str(tag_id): 0 for tag_id in TARGETS}
    for (cam, frame_index), poses in solved.items():
        for tag_id in list(poses):
            if (cam, frame_index, tag_id) not in keep:
                del poses[tag_id]
                removed[str(tag_id)] += 1
    return removed


def matrix_to_quaternion(R):
    # Returns w, x, y, z.
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = np.array([(R[0, 2] - R[2, 0]) / s,
                          (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                          (R[1, 2] + R[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = np.array([(R[1, 0] - R[0, 1]) / s,
                          (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quaternion_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def fuse_transforms(observations):
    weights = np.asarray([1.0 / max(0.2, o["rmse"]) ** 2 for o in observations])
    weights /= weights.sum()
    t = sum(w * o["T_b_wrist"][:3, 3] for w, o in zip(weights, observations))
    quats = [matrix_to_quaternion(o["T_b_wrist"][:3, :3]) for o in observations]
    reference = quats[0]
    quats = [q if np.dot(q, reference) >= 0 else -q for q in quats]
    q = sum(w * q for w, q in zip(weights, quats))
    q /= np.linalg.norm(q)
    return make_transform(quaternion_to_matrix(q), t)


def draw_axes(frame, camera, T, length, colors, thickness):
    points = np.asarray(
        [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]],
        dtype=np.float64,
    )
    try:
        image = np.round(project_omni(points, camera, T)).astype(np.int32)
    except cv2.error:
        return
    origin = tuple(image[0])
    for endpoint, color in zip(image[1:], colors):
        cv2.line(frame, origin, tuple(endpoint), color, thickness, cv2.LINE_AA)


def annotate(frame, cam_name, camera, detections, tag_poses, T_cam_wrist, axis_length):
    out = frame.copy()
    for tag_id, corners in detections.items():
        cv2.polylines(out, [np.round(corners).astype(np.int32)], True, (0, 255, 255), 3)
        center = np.mean(corners, axis=0).astype(int)
        cv2.putText(out, f"id{tag_id}", (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        if tag_id in tag_poses:
            draw_axes(out, camera, tag_poses[tag_id][0], axis_length,
                      COLORS["tag"], 3)
    if T_cam_wrist is not None:
        draw_axes(out, camera, T_cam_wrist, axis_length * 1.25,
                  COLORS["wrist"], 4)
    cv2.rectangle(out, (0, 0), (out.shape[1], 54), (15, 15, 15), -1)
    status = "WRIST FUSED" if T_cam_wrist is not None else "NO TARGET TAG"
    cv2.putText(out, f"{cam_name} | {status}", (18, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255) if T_cam_wrist is not None else (0, 180, 255),
                2, cv2.LINE_AA)
    return out


def nearest_index(entries, target_ms):
    times = np.asarray([row["_middle"] for _, row in entries])
    pos = int(np.searchsorted(times, target_ms))
    choices = [p for p in (pos - 1, pos) if 0 <= p < len(entries)]
    return min(choices, key=lambda p: abs(times[p] - target_ms))


def estimate_stereo_scale(solved, selected, T_b_c, max_sync_ms):
    """Infer the physical tag scale from the known stereo baseline."""
    R_b_c = T_b_c[:3, :3]
    t_b_c = T_b_c[:3, 3]
    samples = []
    raw_samples = []
    common_pair_count = 0
    for b_index, b_row in selected["CAM_B"]:
        c_pos = nearest_index(selected["CAM_C"], b_row["_middle"])
        c_index, c_row = selected["CAM_C"][c_pos]
        if abs(c_row["_middle"] - b_row["_middle"]) > max_sync_ms:
            continue
        common = set(solved[("CAM_B", b_index)]) & set(solved[("CAM_C", c_index)])
        for tag_id in common:
            common_pair_count += 1
            t_b = solved[("CAM_B", b_index)][tag_id][0][:3, 3]
            t_c = solved[("CAM_C", c_index)][tag_id][0][:3, 3]
            a = t_c - R_b_c @ t_b
            denom = float(a @ a)
            if denom > 1e-10:
                scale = float((a @ t_b_c) / denom)
                raw_samples.append(scale)
                if 0.2 < scale < 10.0:
                    samples.append((scale, a))
    if len(samples) < 5:
        return 1.0, {
            "common_pair_count": common_pair_count,
            "valid_scale_pair_count": len(samples),
            "raw_scale_median": (
                float(np.median(raw_samples)) if raw_samples else None
            ),
            "raw_scale_min": float(np.min(raw_samples)) if raw_samples else None,
            "raw_scale_max": float(np.max(raw_samples)) if raw_samples else None,
            "status": "insufficient_pairs",
        }
    raw = np.asarray([item[0] for item in samples])
    median = float(np.median(raw))
    mad = float(np.median(np.abs(raw - median)))
    tolerance = max(0.15 * median, 3.5 * mad)
    inliers = [a for scale, a in samples if abs(scale - median) <= tolerance]
    numerator = sum(float(a @ t_b_c) for a in inliers)
    denominator = sum(float(a @ a) for a in inliers)
    scale = numerator / denominator
    residuals = [float(np.linalg.norm(scale * a - t_b_c)) for a in inliers]
    return scale, {
        "common_pair_count": common_pair_count,
        "valid_scale_pair_count": len(samples),
        "inlier_count": len(inliers),
        "raw_scale_median": median,
        "scale": scale,
        "baseline_residual_m_median": float(np.median(residuals)),
        "baseline_residual_m_max": float(np.max(residuals)),
        "status": "ok",
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras = load_cameras(args.intrinsics)
    T_b_c = load_t_b_c(args.camchain)
    T_c_b = np.linalg.inv(T_b_c)
    T_tag_wrist = target_transforms()
    timestamp_rows = load_timestamps(args.input_dir / "timestamps.csv", args.module)
    start_ms, end_ms, selected = select_duration(
        timestamp_rows, args.duration, args.start_offset
    )
    if not selected["CAM_B"] or not selected["CAM_C"]:
        raise RuntimeError("No frames in requested time interval")

    video_paths = {
        "CAM_B": next(args.input_dir.glob("module01_*_CAM_B.h265")),
        "CAM_C": next(args.input_dir.glob("module01_*_CAM_C.h265")),
    }
    showinfo_paths = {
        "CAM_B": args.input_dir.parent / "output" / "cam_b_showinfo.log",
        "CAM_C": args.input_dir.parent / "output" / "cam_c_showinfo.log",
    }
    decode_maps = {
        cam: load_decode_map(showinfo_paths[cam], timestamp_rows[cam])
        for cam in ("CAM_B", "CAM_C")
    }
    selected_decoded = {}
    frames = {}
    for cam, entries in selected.items():
        selected_decoded[cam] = [
            (decode_maps[cam][row_index], row)
            for row_index, row in entries
            if row_index in decode_maps[cam]
        ]
        decoded_indices = [index for index, _ in selected_decoded[cam]]
        frames[cam] = decode_range(
            video_paths[cam], min(decoded_indices), max(decoded_indices),
            cameras[cam]["size"][0], cameras[cam]["size"][1],
        )
    detector = make_detector()
    obj = tag_object_points(args.tag_size)

    detected = {}
    solved = {}
    for cam in ("CAM_B", "CAM_C"):
        for frame_index, row in selected_decoded[cam]:
            key = (cam, frame_index)
            detected[key] = detect(detector, frames[cam][frame_index])
            solved[key] = {}
            for tag_id, corners in detected[key].items():
                pose = solve_tag_pose(corners, cameras[cam], obj)
                if pose is not None:
                    solved[key][tag_id] = pose

    isolated_removed = filter_isolated_tracks(solved, max_gap_frames=3)
    stereo_scale, stereo_scale_report = estimate_stereo_scale(
        solved, selected_decoded, T_b_c, args.max_sync_ms
    )
    # PnP translation is proportional to the assumed tag edge length.  Recover
    # the physical scale from simultaneous B/C observations and the calibrated
    # stereo baseline, while leaving the tag orientation unchanged.
    for poses in solved.values():
        for tag_id, (T_cam_tag, rmse) in poses.items():
            T_cam_tag[:3, 3] *= stereo_scale

    scale = args.preview_scale
    source_h, source_w = next(iter(frames["CAM_B"].values())).shape[:2]
    cell_size = (round(source_w * scale), round(source_h * scale))
    output_size = (cell_size[0] * 2, cell_size[1])
    b_times = [r["_middle"] for _, r in selected["CAM_B"]]
    fps = (len(b_times) - 1) * 1000.0 / (b_times[-1] - b_times[0])
    video_out = None
    writer = None
    if not args.no_video:
        video_out = args.output_dir / f"wrist_pose_preview_{args.duration:g}s.mp4"
        writer = cv2.VideoWriter(
            str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create {video_out}")

    csv_out = args.output_dir / f"wrist_pose_CAM_B_{args.duration:g}s.csv"
    fieldnames = [
        "timestamp_ms", "CAM_B_seq", "CAM_B_device_ts_ms",
        "CAM_B_exposure_middle_ts_ms", "CAM_B_host_ts_ms",
        "CAM_C_seq", "CAM_C_exposure_middle_ts_ms", "sync_delta_ms",
        "detected_tag_ids", "observation_sources", "mean_reprojection_error_px",
        "wrist_CAM_B_x_m", "wrist_CAM_B_y_m", "wrist_CAM_B_z_m",
        "wrist_CAM_B_qw", "wrist_CAM_B_qx", "wrist_CAM_B_qy", "wrist_CAM_B_qz",
    ]
    stats = {"frames": 0, "frames_with_pose": 0, "observations": 0,
             "detections_by_id": {str(k): 0 for k in TARGETS}}

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        for b_row_index, b_row in selected["CAM_B"]:
            c_pos = nearest_index(selected["CAM_C"], b_row["_middle"])
            c_row_index, c_row = selected["CAM_C"][c_pos]
            b_index = decode_maps["CAM_B"].get(b_row_index)
            c_index = decode_maps["CAM_C"].get(c_row_index)
            sync_delta = c_row["_middle"] - b_row["_middle"]
            pair = {"CAM_B": (b_index, b_row), "CAM_C": (c_index, c_row)}
            observations = []
            all_ids = set()
            for cam, (frame_index, _) in pair.items():
                if frame_index is None:
                    continue
                if cam == "CAM_C" and abs(sync_delta) > args.max_sync_ms:
                    continue
                key = (cam, frame_index)
                for tag_id, (T_cam_tag, rmse) in solved[key].items():
                    all_ids.add(tag_id)
                    T_cam_wrist = T_cam_tag @ T_tag_wrist[tag_id]
                    T_b_wrist = T_cam_wrist if cam == "CAM_B" else T_c_b @ T_cam_wrist
                    observations.append({
                        "cam": cam, "tag_id": tag_id, "rmse": rmse,
                        "T_b_wrist": T_b_wrist,
                    })
                    stats["detections_by_id"][str(tag_id)] += 1
            T_b_wrist = fuse_transforms(observations) if observations else None
            T_c_wrist = T_b_c @ T_b_wrist if T_b_wrist is not None else None

            if writer is not None:
                views = []
                for cam, (frame_index, _) in pair.items():
                    if frame_index is None:
                        view = np.zeros((source_h, source_w, 3), dtype=np.uint8)
                        cv2.putText(
                            view, f"{cam} | UNDECODABLE FRAME", (18, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 255), 2,
                            cv2.LINE_AA,
                        )
                        views.append(
                            cv2.resize(view, cell_size, interpolation=cv2.INTER_AREA)
                        )
                        continue
                    T_cam_wrist = T_b_wrist if cam == "CAM_B" else T_c_wrist
                    view = annotate(
                        frames[cam][frame_index], cam, cameras[cam],
                        detected[(cam, frame_index)], solved[(cam, frame_index)],
                        T_cam_wrist, args.axis_length,
                    )
                    views.append(
                        cv2.resize(view, cell_size, interpolation=cv2.INTER_AREA)
                    )
                grid = np.hstack(views)
                elapsed = (b_row["_middle"] - start_ms) / 1000.0
                cv2.rectangle(
                    grid, (0, grid.shape[0] - 42),
                    (grid.shape[1], grid.shape[0]), (15, 15, 15), -1
                )
                legend = (
                    f"t={elapsed:06.2f}s | tag XYZ=red/green/blue | "
                    "wrist XYZ=magenta/orange/cyan"
                )
                cv2.putText(
                    grid, legend, (18, grid.shape[0] - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255),
                    2, cv2.LINE_AA
                )
                writer.write(grid)

            out_row = {
                "timestamp_ms": b_row["exposure_middle_ts_ms"],
                "CAM_B_seq": b_row["seq"],
                "CAM_B_device_ts_ms": b_row["device_ts_ms"],
                "CAM_B_exposure_middle_ts_ms": b_row["exposure_middle_ts_ms"],
                "CAM_B_host_ts_ms": b_row["host_ts_ms"],
                "CAM_C_seq": c_row["seq"],
                "CAM_C_exposure_middle_ts_ms": c_row["exposure_middle_ts_ms"],
                "sync_delta_ms": f"{sync_delta:.6f}",
            }
            if observations:
                q = matrix_to_quaternion(T_b_wrist[:3, :3])
                t = T_b_wrist[:3, 3]
                out_row.update({
                    "detected_tag_ids": "|".join(map(str, sorted(all_ids))),
                    "observation_sources": "|".join(
                        f"{o['cam']}:id{o['tag_id']}" for o in observations
                    ),
                    "mean_reprojection_error_px": f"{np.mean([o['rmse'] for o in observations]):.6f}",
                    "wrist_CAM_B_x_m": f"{t[0]:.9f}",
                    "wrist_CAM_B_y_m": f"{t[1]:.9f}",
                    "wrist_CAM_B_z_m": f"{t[2]:.9f}",
                    "wrist_CAM_B_qw": f"{q[0]:.9f}",
                    "wrist_CAM_B_qx": f"{q[1]:.9f}",
                    "wrist_CAM_B_qy": f"{q[2]:.9f}",
                    "wrist_CAM_B_qz": f"{q[3]:.9f}",
                })
                stats["frames_with_pose"] += 1
                stats["observations"] += len(observations)
            csv_writer.writerow(out_row)
            stats["frames"] += 1
    if writer is not None:
        writer.release()

    report = {
        "schema": "wrist_tag_pose_stereo.v1",
        "input_dir": str(args.input_dir),
        "time_interval_exposure_middle_ms": [start_ms, end_ms],
        "duration_s": args.duration,
        "start_offset_s": args.start_offset,
        "tag_family": "tag36h11",
        "tag_size_m": args.tag_size,
        "effective_tag_size_m": args.tag_size * stereo_scale,
        "stereo_translation_scale": stereo_scale_report,
        "pose_frame": "CAM_B",
        "timestamp_policy": "CAM_B exposure_middle timeline; CAM_C nearest neighbor",
        "max_sync_ms_requested": args.max_sync_ms,
        "temporal_support_max_gap_frames": 3,
        "isolated_detections_removed": isolated_removed,
        "T_CAM_B_CAM_C": T_b_c.tolist(),
        "targets": TARGETS,
        "stats": stats,
        "outputs": {
            "video": str(video_out) if video_out is not None else None,
            "csv": str(csv_out),
        },
    }
    report_out = args.output_dir / f"wrist_pose_report_{args.duration:g}s.json"
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
