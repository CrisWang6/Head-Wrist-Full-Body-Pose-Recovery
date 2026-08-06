#!/usr/bin/env python3
"""Render the strict first third with head-view upper-body replacement and filtering."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
REPLACE = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}
FACE = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}
TORSO = {"nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"}
MID = {"left_elbow", "right_elbow", "left_knee", "right_knee"}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--head-a", type=Path, required=True)
    p.add_argument("--head-d", type=Path, required=True)
    p.add_argument("--candidate-a", type=Path, required=True)
    p.add_argument("--candidate-d", type=Path, required=True)
    p.add_argument("--hand-a", type=Path, required=True)
    p.add_argument("--hand-d", type=Path, required=True)
    p.add_argument("--projection-csv", type=Path, required=True)
    p.add_argument("--timestamps", type=Path, required=True)
    p.add_argument("--strict-sequences", type=Path, required=True)
    p.add_argument("--strict-external", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fraction", type=float, default=1.0 / 3.0)
    p.add_argument("--median-window", type=int, default=5)
    p.add_argument("--no-upper-replacement", action="store_true", help="Keep projected shoulders/elbows; still use the hand model")
    return p.parse_args()


def load_timestamp_maps(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for camera in ("CAM_A", "CAM_D"):
        cr = [r for r in rows if r["module"] == "1" and r["camera"] == camera]
        out[camera] = {int(r["seq"]): i for i, r in enumerate(cr)}
    return out, min(len(v) for v in out.values())


def load_candidates(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_hand_detections(path: Path):
    return {int(r["sequence"]): r for r in load_candidates(path)}


def load_projection(path: Path):
    data = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            seq = int(row["sequence"]); joint = row["joint"]
            item = data.setdefault(seq, {})
            for camera, prefix in (("CAM_A", "head_A"), ("CAM_D", "head_D")):
                try:
                    uv = np.asarray([float(row[f"{prefix}_u_px"]), float(row[f"{prefix}_v_px"])], np.float32)
                except (ValueError, TypeError):
                    continue
                if np.all(np.isfinite(uv)):
                    item.setdefault(camera, {})[joint] = uv
    return data


def choose_candidate(record, projected):
    best = None
    anchors = {"nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"}
    for candidate in record.get("candidates", []):
        distances = []
        for name in anchors:
            kp = candidate.get("keypoints", {}).get(name)
            if name not in projected or kp is None or float(kp[2]) < 0.10:
                continue
            distances.append(float(np.linalg.norm(np.asarray(kp[:2], np.float32) - projected[name])))
        if len(distances) < 3:
            continue
        median = float(np.median(distances))
        score = median - 35.0 * float(candidate.get("box_confidence", 0.0))
        if best is None or score < best[0]:
            best = (score, median, candidate)
    if best is None or best[1] > 420.0:
        return None, None
    return best[2], best[1]


def replacement_threshold(name):
    return 0.18 if "shoulder" in name or "elbow" in name else 0.20


def median_filter(values, window):
    window = max(1, int(window))
    if window % 2 == 0: window += 1
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.asarray([np.median(padded[i:i + window], axis=0) for i in range(len(values))], np.float32)


def step_limit(name, confidence):
    base = 30.0 if name in TORSO else (40.0 if name in MID else 52.0)
    if confidence < 0.20: base *= 0.55
    return base


def smooth_joint(values, confidence, name, window):
    measurements = median_filter(values, window)
    output = np.empty_like(measurements); output[0] = measurements[0]
    for i in range(1, len(measurements)):
        residual = measurements[i] - output[i - 1]
        distance = float(np.linalg.norm(residual)); score = float(confidence[i])
        if score < 0.06:
            alpha = 0.03
        else:
            confidence_scale = float(np.clip((score - 0.05) / 0.80, 0.2, 1.0))
            alpha = float(np.clip((0.18 + 0.0042 * distance) * confidence_scale, 0.12, 0.68))
        candidate = output[i - 1] + alpha * residual
        delta = candidate - output[i - 1]; norm = float(np.linalg.norm(delta)); limit = step_limit(name, score)
        if norm > limit: candidate = output[i - 1] + delta * (limit / norm)
        output[i] = candidate
    return output


def build_camera_pose(sequences, frame_indices, projected, candidates, hands, camera, window, replace_upper=True):
    n = len(sequences); raw = np.full((n, len(JOINTS), 2), np.nan, np.float32)
    confidence = np.zeros((n, len(JOINTS)), np.float32); sources = [["missing"] * len(JOINTS) for _ in range(n)]
    match_distance = np.full(n, np.nan, np.float32); replaced = 0
    for i, (seq, fi) in enumerate(zip(sequences, frame_indices)):
        proj = projected.get(seq, {}).get(camera, {})
        for j, name in enumerate(JOINTS):
            if name in proj:
                raw[i, j] = proj[name]; confidence[i, j] = 0.70; sources[i][j] = "stereo_projection"
        if replace_upper:
            selected, md = choose_candidate(candidates[fi], proj)
            if selected is not None:
                match_distance[i] = md
                for j, name in enumerate(JOINTS):
                    if name not in REPLACE or "wrist" in name: continue
                    kp = selected.get("keypoints", {}).get(name)
                    if kp is None or float(kp[2]) < replacement_threshold(name): continue
                    uv = np.asarray(kp[:2], np.float32)
                    if not (0 <= uv[0] < 1920 and 0 <= uv[1] < 1200): continue
                    if np.all(np.isfinite(raw[i, j])) and float(np.linalg.norm(uv - raw[i, j])) > 340.0: continue
                    raw[i, j] = uv; confidence[i, j] = float(kp[2]); sources[i][j] = "head_pose_replacement"; replaced += 1
        # Hand endpoints come from the dedicated 21-landmark hand model.  The
        # detector was run on projection-guided crops, so side assignment is
        # inherited from the external elbow/wrist ROI rather than image-wide
        # handedness classification.
        hand_record = hands.get(seq, {}).get("hands", {})
        for side in ("left", "right"):
            detection = hand_record.get(side)
            if detection is None: continue
            name = f"{side}_wrist"; j = JOINTS.index(name); wrist = np.asarray(detection["wrist"], np.float32)
            if np.all(np.isfinite(raw[i, j])) and float(np.linalg.norm(wrist - raw[i, j])) > 320.0: continue
            raw[i, j] = wrist; confidence[i, j] = float(detection.get("confidence", .5)); sources[i][j] = "mediapipe_hand_wrist"; replaced += 1
    filtered = np.full_like(raw, np.nan)
    for j, name in enumerate(JOINTS):
        values = raw[:, j].copy(); valid = np.all(np.isfinite(values), axis=1)
        if not np.any(valid): continue
        valid_i = np.flatnonzero(valid)
        for axis in range(2): values[:, axis] = np.interp(np.arange(n), valid_i, values[valid_i, axis])
        filtered[:, j] = smooth_joint(values, confidence[:, j], name, window)
    return raw, filtered, confidence, sources, match_distance, replaced


def draw_pose(frame, points, sources, seq, camera, replace_upper=True):
    for a, b in EDGES:
        ia, ib = JOINTS.index(a), JOINTS.index(b)
        if np.all(np.isfinite(points[ia])) and np.all(np.isfinite(points[ib])):
            cv2.line(frame, tuple(np.rint(points[ia]).astype(int)), tuple(np.rint(points[ib]).astype(int)), (0, 230, 255), 4, cv2.LINE_AA)
    for j, name in enumerate(JOINTS):
        if name in FACE or not np.all(np.isfinite(points[j])): continue
        source = sources[j]
        color = (255, 0, 255) if source == "mediapipe_hand_wrist" else ((255, 255, 0) if source == "head_pose_replacement" else (0, 230, 255))
        p = tuple(np.rint(points[j]).astype(int)); cv2.circle(frame, p, 7, (0, 0, 0), -1, cv2.LINE_AA); cv2.circle(frame, p, 5, color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (1150, 66), (0, 0, 0), -1)
    cv2.putText(frame, f"{camera} seq={seq} | hybrid filtered | first 1/3", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .64, (255, 255, 255), 2, cv2.LINE_AA)
    legend = "cyan=all projected body  magenta=hand-model wrist" if not replace_upper else "cyan=projected body  yellow=head shoulder/elbow  magenta=hand-model wrist"
    cv2.putText(frame, legend, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 230, 255), 2, cv2.LINE_AA)


def expand_to_source_time(input_path, output_path, source_indices, output_frames, size):
    """Preserve 50 Hz source duration without ever displaying a rejected frame."""
    cap = cv2.VideoCapture(str(input_path))
    ok, current = cap.read()
    if not ok: raise RuntimeError(f"empty event video: {input_path}")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 50, size)
    next_event = 1
    for source_frame in range(output_frames):
        while next_event < len(source_indices) and source_indices[next_event] <= source_frame:
            ok, candidate = cap.read()
            if not ok: raise RuntimeError(f"event video ended at event {next_event}: {input_path}")
            current = candidate; next_event += 1
        writer.write(current)
    cap.release(); writer.release()


def main():
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    maps, frame_count = load_timestamp_maps(a.timestamps); cutoff = int(np.floor(frame_count * a.fraction))
    strict_obj = json.loads(a.strict_sequences.read_text(encoding="utf-8")); strict = [int(x) for x in strict_obj["kept_sequences"]]
    selected = [s for s in strict if s in maps["CAM_A"] and s in maps["CAM_D"] and maps["CAM_A"][s] < cutoff and maps["CAM_D"][s] < cutoff]
    idx_a = [maps["CAM_A"][s] for s in selected]; idx_d = [maps["CAM_D"][s] for s in selected]
    projection = load_projection(a.projection_csv); cand_a = load_candidates(a.candidate_a); cand_d = load_candidates(a.candidate_d); hand_a = load_hand_detections(a.hand_a); hand_d = load_hand_detections(a.hand_d)
    results = {}
    for camera, indices, candidates, hands in (("CAM_A", idx_a, cand_a, hand_a), ("CAM_D", idx_d, cand_d, hand_d)):
        results[camera] = build_camera_pose(selected, indices, projection, candidates, hands, camera, a.median_window, replace_upper=not a.no_upper_replacement)

    caps = {"CAM_A": cv2.VideoCapture(str(a.head_a)), "CAM_D": cv2.VideoCapture(str(a.head_d))}
    writers = {
        c: cv2.VideoWriter(str(a.output_dir / f"{c}_hybrid_filtered_first_third.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 50, (1920, 1200)) for c in caps
    }
    stereo = cv2.VideoWriter(str(a.output_dir / "head_stereo_hybrid_filtered_first_third.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 50, (1920, 600))
    external = cv2.VideoCapture(str(a.strict_external)); four = cv2.VideoWriter(str(a.output_dir / "four_view_hybrid_filtered_first_third.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 50, (1920, 1200))
    current = {"CAM_A": -1, "CAM_D": -1}; frames = {"CAM_A": None, "CAM_D": None}
    csv_rows = []
    for i, seq in enumerate(selected):
        rendered = {}
        for camera, target in (("CAM_A", idx_a[i]), ("CAM_D", idx_d[i])):
            while current[camera] < target:
                ok, frame = caps[camera].read(); current[camera] += 1
                if not ok: raise RuntimeError(f"decode failed {camera} frame {target}")
                frames[camera] = frame
            frame = frames[camera].copy(); raw, filt, conf, sources, _, _ = results[camera]
            draw_pose(frame, filt[i], sources[i], seq, camera, replace_upper=not a.no_upper_replacement); writers[camera].write(frame); rendered[camera] = frame
            for j, name in enumerate(JOINTS):
                csv_rows.append({"sequence": seq, "camera": camera, "source_frame_index": target, "joint": name, "source": sources[i][j], "confidence": float(conf[i, j]), "raw_x": float(raw[i, j, 0]) if np.isfinite(raw[i, j, 0]) else "", "raw_y": float(raw[i, j, 1]) if np.isfinite(raw[i, j, 1]) else "", "filtered_x": float(filt[i, j, 0]) if np.isfinite(filt[i, j, 0]) else "", "filtered_y": float(filt[i, j, 1]) if np.isfinite(filt[i, j, 1]) else ""})
        head = np.hstack([cv2.resize(rendered["CAM_A"], (960, 600)), cv2.resize(rendered["CAM_D"], (960, 600))]); stereo.write(head)
        ok, ext = external.read()
        if not ok: raise RuntimeError(f"strict external ended at output frame {i}")
        four.write(np.vstack([ext, head]))
    for cap in caps.values(): cap.release()
    external.release()
    for writer in writers.values(): writer.release()
    stereo.release(); four.release()
    with (a.output_dir / "hybrid_filtered_2d.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    event_outputs = [
        (a.output_dir / "CAM_A_hybrid_filtered_first_third.mp4", a.output_dir / "CAM_A_hybrid_filtered_first_third_source_time.mp4", (1920, 1200)),
        (a.output_dir / "CAM_D_hybrid_filtered_first_third.mp4", a.output_dir / "CAM_D_hybrid_filtered_first_third_source_time.mp4", (1920, 1200)),
        (a.output_dir / "head_stereo_hybrid_filtered_first_third.mp4", a.output_dir / "head_stereo_hybrid_filtered_first_third_source_time.mp4", (1920, 600)),
        (a.output_dir / "four_view_hybrid_filtered_first_third.mp4", a.output_dir / "four_view_hybrid_filtered_first_third_source_time.mp4", (1920, 1200)),
    ]
    for input_path, output_path, size in event_outputs:
        expand_to_source_time(input_path, output_path, idx_a, cutoff, size)
    summary = {"schema": "hearwristcam.0711_175408.hybrid_first_third.v3", "source_frame_count": frame_count, "source_cutoff_exclusive": cutoff, "fraction": a.fraction, "strict_frames_in_first_third": len(selected), "source_time_output_frames": cutoff, "source_time_duration_seconds": cutoff / 50.0, "rejected_trigger_policy": "event outputs delete rejected frames; source_time outputs hold the last complete accepted multi-camera frame, preserving 50 Hz duration without showing rejected frames", "first_sequence": selected[0], "last_sequence": selected[-1], "replacement": {"shoulders_elbows": "external stereo projection only" if a.no_upper_replacement else "selected matching head-view YOLO pose keypoints", "hands": "MediaPipe Hand Landmarker, 21 landmarks, projection-guided elbow/wrist crop; skeleton endpoint uses hand landmark 0 (wrist), with external projection fallback", "candidate_match_max_median_px": None if a.no_upper_replacement else 420, "shoulder_elbow_min_confidence": None if a.no_upper_replacement else .18, "hand_to_projected_wrist_max_px": 320}, "filter": {"median_window_frames": a.median_window, "adaptive_ema_alpha": "clip((0.18 + 0.0042*pixel_residual)*confidence_scale, 0.12, 0.68)", "per_frame_step_limits_px": {"torso": 30, "elbow_knee": 40, "wrist_ankle": 52}}, "replacement_counts": {camera: results[camera][-1] for camera in results}, "candidate_match_median_px": {camera: (None if a.no_upper_replacement else float(np.nanmedian(results[camera][-2]))) for camera in results}}
    (a.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()
