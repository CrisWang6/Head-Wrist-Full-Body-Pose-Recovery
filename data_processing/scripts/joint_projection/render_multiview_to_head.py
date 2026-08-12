#!/usr/bin/env python3
"""Reproject filtered multiview 3D skeletons onto the 0806 head dual cameras.

Pose chain (no nose/shoulder soft optimization):
  T_world_head_mocap  <- aligned mocap_CH3_08 quat/xyz
  T_rigid_camera      <- xy_swap head basis + file optical centers
                        (same convention as render_fivepoint_clean_to_head.py)
  T_world_camera      = T_world_head_mocap @ T_rigid_camera
  uv                  = OmniCamera.project(T_camera_world @ p_world)

Discrete fix: head_camera_rotation_basis=xy_swap for body ~Z-90 vs head video.
T_mocap_rigid_head is identity (mirror_y reverted — it reversed L/R hands).

Source video: decode elementary .h265 via timestamps.csv packet sizes.
OpenCV on remuxed .mp4 presents capture index N+4 as frame N (exact match),
which makes the skeleton look ~4 frames behind the person.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import csv
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from multiview_geometry import (
    CameraPose,
    OmniCamera,
    homogeneous,
    load_json,
    rigid_world_transform,
)
FACE_HIDE = ("left_eye", "right_eye", "left_ear", "right_ear")
BODY_EDGES = (
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_big_toe"),
    ("left_ankle", "left_small_toe"),
    ("left_ankle", "left_heel"),
    ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_big_toe"),
    ("right_ankle", "right_small_toe"),
    ("right_ankle", "right_heel"),
    ("right_big_toe", "right_small_toe"),
    ("left_ankle", "left_toe"),
    ("right_ankle", "right_toe"),
)


def draw_skeleton_nose_only_face(
    image: np.ndarray,
    points: dict[str, np.ndarray],
    color: tuple[int, int, int],
    *,
    radius: int = 4,
) -> None:
    visible = {name: uv for name, uv in points.items() if name not in FACE_HIDE}
    for first, second in BODY_EDGES:
        if first in visible and second in visible:
            cv2.line(
                image,
                tuple(np.rint(visible[first]).astype(int)),
                tuple(np.rint(visible[second]).astype(int)),
                color,
                3,
                cv2.LINE_AA,
            )
    for point in visible.values():
        cv2.circle(
            image,
            tuple(np.rint(point).astype(int)),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "0806_dual_external_mocap.json"
DEFAULT_HEAD_RIGID = SCRIPT_DIR / "head_stereo_rigid_extrinsics.json"
DEFAULT_HEAD_INTRINSICS = (
    REPO_ROOT
    / "test_code"
    / "calibrate"
    / "parameters"
    / "intrinsics"
    / "head"
    / "head_intrinsics_kalibr_omni_1920x1200.json"
)
CAMERA_ORDER = ("CAM_A", "CAM_D")
SIDE_BY_KEY = {"CAM_A": "left", "CAM_D": "right"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="0806 recording root containing aligned_data/ and head capture dir.",
    )
    parser.add_argument(
        "--skeleton-playback",
        type=Path,
        help="Compressed methods.filtered.multiview xyz from full run "
        "(default: <data-root>/multiview_3d_results/full/skeleton_playback.json).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Optional multiview_3d_results.jsonl; used only when "
        "--skeleton-playback is omitted and this file exists.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--head-rigid", type=Path, default=DEFAULT_HEAD_RIGID)
    parser.add_argument("--head-intrinsics", type=Path, default=DEFAULT_HEAD_INTRINSICS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <data-root>/multiview_3d_results/full/head_reprojection",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-seq", type=int)
    parser.add_argument("--end-seq", type=int, help="Inclusive aligned sequence")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--snapshot-seq",
        type=int,
        action="append",
        default=[],
        help="Also write stereo JPEG snapshot(s) at these aligned seq values.",
    )
    parser.add_argument(
        "--min-depth-m",
        type=float,
        default=0.01,
        help="Drop projections with camera-frame Z below this (behind/near lens).",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write head_reprojection_2d CSV + report.json only; skip MP4/JPEG encode.",
    )
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def discover_head_dir(data_root: Path, config: dict) -> Path:
    head_cfg = config.get("head", {})
    recording = head_cfg.get("recording_directory")
    if recording:
        candidate = data_root / recording
        if candidate.is_dir():
            return candidate
    matches = sorted(
        {
            path.parent
            for path in data_root.glob("*/module01_*_CAM_A.h265")
        }
        | {
            path.parent
            for path in data_root.glob("*/module01_*_CAM_A.mp4")
        }
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one head recording under {data_root}, found {matches}"
        )
    return matches[0]


def find_head_video(head_dir: Path, socket: str, config: dict) -> Path:
    head_cfg = config.get("head", {})
    camera_cfg = head_cfg.get(f"camera_{socket[-1].lower()}", {})
    pattern = camera_cfg.get("video_glob", f"module01_*_{socket}.h265")
    matches = sorted(head_dir.glob(pattern))
    if len(matches) != 1:
        # Prefer elementary H.265 over remuxed MP4 when glob still points at mp4.
        h265_matches = sorted(head_dir.glob(f"module01_*_{socket}.h265"))
        if len(h265_matches) == 1:
            return h265_matches[0]
        raise RuntimeError(
            f"Expected one head video matching {head_dir / pattern}: {matches}"
        )
    return matches[0]


def infer_head_module_from_video(path: Path) -> str:
    """Map elementary head video filename to timestamps.csv module column."""
    name = path.name.lower()
    if "module01" in name:
        return "1"
    if "module02" in name:
        return "2"
    if "module03" in name:
        return "3"
    return "1"


class HeadTimestampIndex:
    """Map exposure-end timestamps (ms) to capture-row index for one head camera."""

    def __init__(self, path: Path, camera: str, module: str = "1"):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if row["camera"] == camera and row["module"] == module
            ]
        if not rows:
            raise RuntimeError(f"No module={module} {camera} rows in {path}")
        self.module = module
        self.camera = camera
        self.rows = rows
        self.timestamps_ms = [float(row["exposure_end_ts_ms"]) for row in rows]
        self.by_timestamp = {
            float(row["exposure_end_ts_ms"]): (index, row)
            for index, row in enumerate(rows)
        }

    def nearest(
        self, timestamp_ms: float, tolerance_ms: float = 1.0
    ) -> tuple[int, dict, float]:
        exact = self.by_timestamp.get(timestamp_ms)
        if exact is not None:
            return exact[0], exact[1], 0.0
        position = bisect.bisect_left(self.timestamps_ms, timestamp_ms)
        candidates = self.timestamps_ms[max(0, position - 1) : position + 1]
        if not candidates:
            raise KeyError(f"No head timestamps near {timestamp_ms}")
        nearest = min(candidates, key=lambda value: abs(value - timestamp_ms))
        error = nearest - timestamp_ms
        if abs(error) > tolerance_ms:
            raise KeyError(
                f"Nearest head timestamp to {timestamp_ms} differs by {error} ms"
            )
        index, row = self.by_timestamp[nearest]
        return index, row, error


class H265CaptureReader:
    """Sequential elementary-H.265 reader keyed by timestamps.csv capture row."""

    def __init__(self, path: Path, rows: list[dict]):
        if path.suffix.lower() != ".h265":
            raise ValueError(
                f"Expected elementary .h265 source, got {path}. "
                "Remuxed .mp4 is +4 capture frames ahead under OpenCV."
            )
        self.path = path
        self.rows = rows
        self._stream = path.open("rb")
        self._codec = av.CodecContext.create("hevc", "r")
        self._next_row = 0
        self._buffer: dict[int, np.ndarray] = {}
        self._last_image: np.ndarray | None = None
        self._flushed = False
        self.missing_fallbacks = 0
        self._eof = False

    def close(self) -> None:
        self._stream.close()

    def _decode_packet(self, packet: av.Packet | None) -> None:
        try:
            frames = self._codec.decode(packet)
        except av.error.InvalidDataError:
            self.missing_fallbacks += 1
            return
        except av.error.FFmpegError:
            self.missing_fallbacks += 1
            return
        for frame in frames:
            self._buffer[int(frame.pts)] = frame.to_ndarray(format="bgr24")

    def _feed_row(self, row_index: int) -> None:
        row = self.rows[row_index]
        expected = int(row["bytes"])
        data = self._stream.read(expected)
        if len(data) < expected:
            # timestamps.csv can list more capture rows than the elementary file
            # contains (tail mismatch). Treat as EOF and use holdover in read().
            self.missing_fallbacks += 1
            self._eof = True
            return
        packet = av.Packet(data)
        packet.pts = row_index
        packet.dts = row_index
        packet.time_base = Fraction(1, 30)
        self._decode_packet(packet)

    def _flush(self) -> None:
        if self._flushed:
            return
        self._decode_packet(None)
        self._flushed = True

    def read(self, capture_index: int) -> np.ndarray:
        if capture_index < 0 or capture_index >= len(self.rows):
            raise RuntimeError(
                f"Capture index {capture_index} out of range for {self.path}"
            )
        # HEVC reordering: a capture may only appear after several later packets.
        max_lookahead = 64
        while capture_index not in self._buffer:
            if self._eof:
                break
            if self._next_row < len(self.rows):
                self._feed_row(self._next_row)
                self._next_row += 1
            else:
                self._flush()
                break
            stale = [key for key in self._buffer if key < capture_index]
            for key in stale:
                # Keep the newest past frame as holdover for true gaps.
                self._last_image = self._buffer.pop(key)
            if (
                self._next_row > capture_index + max_lookahead
                and capture_index not in self._buffer
            ):
                break
        if capture_index in self._buffer:
            image = self._buffer.pop(capture_index)
            self._last_image = image
            return image
        self.missing_fallbacks += 1
        if self._last_image is None:
            raise RuntimeError(
                f"No decodable H.265 frame at capture {capture_index} in {self.path}"
            )
        return self._last_image.copy()


def load_skeleton_playback(path: Path) -> tuple[list[int], list[str], list[dict]]:
    payload = load_json(path)
    joints = list(payload["joints"])
    seqs = [int(value) for value in payload["seqs"]]
    miss = int(payload["missing_sentinel"])
    array = np.frombuffer(
        base64.b64decode(payload["xyz_i16_b64"]), dtype=np.int16
    ).reshape(len(seqs), len(joints), 3)
    frames: list[dict] = []
    for frame_index, sequence in enumerate(seqs):
        points = {}
        for joint_index, name in enumerate(joints):
            sample = array[frame_index, joint_index].astype(np.float64)
            if (sample == miss).any():
                continue
            points[name] = (sample / 1000.0).tolist()
        frames.append({"seq": sequence, "xyz_world_m": points})
    return seqs, joints, frames


def load_skeleton_jsonl(path: Path) -> tuple[list[int], list[str], list[dict]]:
    frames: list[dict] = []
    joint_names: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            points = {
                name: payload["xyz_world_m"]
                for name, payload in record["methods"]["filtered"]["multiview"].items()
            }
            if not joint_names and points:
                joint_names = list(points)
            frames.append({"seq": int(record["seq"]), "xyz_world_m": points})
    seqs = [frame["seq"] for frame in frames]
    return seqs, joint_names, frames


XY_SWAP_R_RIGID_CAMERA = np.asarray(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)


def rigid_camera_mounts(
    head_rigid: dict,
    head_intrinsics: dict,
    *,
    basis: str = "file",
) -> dict[str, np.ndarray]:
    """Build T_rigid_camera for each head socket.

    basis:
      - file: R/t from head_stereo_rigid_extrinsics.json
      - xy_swap: approved fivepoint/head convention — CAM_A uses xy_swap+z_flip
        rotation with file left center; CAM_D uses xy_swap @ R_DA.T with file
        right center (Kalibr stereo orientation, mechanical positions).
    """
    positions = {
        "CAM_A": np.asarray(
            head_rigid["cameras"]["left"]["p_rigid_camera_mm"], dtype=np.float64
        )
        / 1000.0,
        "CAM_D": np.asarray(
            head_rigid["cameras"]["right"]["p_rigid_camera_mm"], dtype=np.float64
        )
        / 1000.0,
    }
    if basis == "file":
        mounts = {}
        for socket, side in SIDE_BY_KEY.items():
            camera = head_rigid["cameras"][side]
            mounts[socket] = homogeneous(
                np.asarray(camera["R_rigid_camera"], dtype=np.float64),
                positions[socket],
            )
        return mounts
    if basis != "xy_swap":
        raise ValueError(f"Unknown head camera rotation basis: {basis}")
    rotation_da = np.asarray(
        head_intrinsics["stereo_extrinsics"]["T_CAM_D_CAM_A"], dtype=np.float64
    )[:3, :3]
    return {
        "CAM_A": homogeneous(XY_SWAP_R_RIGID_CAMERA, positions["CAM_A"]),
        "CAM_D": homogeneous(XY_SWAP_R_RIGID_CAMERA @ rotation_da.T, positions["CAM_D"]),
    }


def head_mocap_correction(head_cfg: dict, config: dict) -> np.ndarray:
    """Return T_mocap_rigid_head (identity if omitted)."""
    if "T_mocap_rigid_head" in head_cfg:
        return np.asarray(head_cfg["T_mocap_rigid_head"], dtype=np.float64).reshape(4, 4)
    if head_cfg.get("apply_T_mocap_rigid_rigid_k"):
        return np.asarray(config["T_mocap_rigid_rigid_k"], dtype=np.float64).reshape(4, 4)
    return np.eye(4, dtype=np.float64)


def draw_laterality_labels(
    image: np.ndarray, projected: dict[str, np.ndarray]
) -> None:
    """Mark left/right wrists so L/R can be checked against the video."""
    for name, color in (("left_wrist", (255, 128, 0)), ("right_wrist", (0, 128, 255))):
        if name not in projected:
            continue
        point = tuple(np.rint(projected[name]).astype(int))
        cv2.circle(image, point, 8, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            "L" if name.startswith("left") else "R",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )


def project_joints(
    pose: CameraPose,
    joints_world: dict[str, list[float]],
    *,
    min_depth_m: float,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    projected = {}
    for name, xyz in joints_world.items():
        point_camera = pose.point_camera(xyz)
        if float(point_camera[2]) < min_depth_m:
            continue
        uv = pose.camera.project(point_camera)
        if uv is None:
            continue
        if -200.0 <= float(uv[0]) < width + 200.0 and -200.0 <= float(uv[1]) < height + 200.0:
            projected[name] = np.asarray(uv, dtype=np.float64)
    return projected


def open_writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    return writer


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    config = load_json(args.config)
    head_cfg = config.get("head", {})
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_root / "multiview_3d_results" / "full" / "head_reprojection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton_path = args.skeleton_playback
    if skeleton_path is None:
        default_playback = (
            data_root / "multiview_3d_results" / "full" / "skeleton_playback.json"
        )
        if default_playback.is_file():
            skeleton_path = default_playback
        elif args.results is not None:
            skeleton_path = args.results
        else:
            raise FileNotFoundError(
                "Provide --skeleton-playback or --results; "
                f"missing {default_playback}"
            )
    skeleton_path = skeleton_path.resolve()
    if skeleton_path.suffix.lower() == ".jsonl":
        seqs, joint_names, frames = load_skeleton_jsonl(skeleton_path)
        skeleton_source = "multiview_3d_results.jsonl methods.filtered.multiview"
    else:
        seqs, joint_names, frames = load_skeleton_playback(skeleton_path)
        skeleton_source = "skeleton_playback.json methods.filtered.multiview"

    if args.start_seq is not None or args.end_seq is not None:
        frames = [
            frame
            for frame in frames
            if (args.start_seq is None or int(frame["seq"]) >= args.start_seq)
            and (args.end_seq is None or int(frame["seq"]) <= args.end_seq)
        ]
        seqs = [int(frame["seq"]) for frame in frames]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
        seqs = seqs[: args.max_frames]

    aligned_path = data_root / "aligned_data" / "aligned_30hz.csv"
    with aligned_path.open("r", encoding="utf-8-sig", newline="") as stream:
        aligned_by_seq = {int(row["seq"]): row for row in csv.DictReader(stream)}

    head_dir = discover_head_dir(data_root, config)
    rigid_prefix = head_cfg.get("rigid_prefix", "mocap_CH3_08")
    head_intrinsics_path = resolve_repo_path(
        head_cfg.get("intrinsics", args.head_intrinsics)
    )
    head_rigid_path = resolve_repo_path(
        head_cfg.get("rigid_extrinsics", args.head_rigid)
    )
    head_intrinsics = load_json(head_intrinsics_path)
    head_rigid = load_json(head_rigid_path)
    camera_basis = str(head_cfg.get("head_camera_rotation_basis", "file"))
    mounts = rigid_camera_mounts(
        head_rigid, head_intrinsics, basis=camera_basis
    )
    mocap_to_head = head_mocap_correction(head_cfg, config)
    models = {
        socket: OmniCamera.from_calibration(head_intrinsics, socket, name=socket)
        for socket in CAMERA_ORDER
    }
    videos = {
        socket: find_head_video(head_dir, socket, config) for socket in CAMERA_ORDER
    }
    indexes = {
        socket: HeadTimestampIndex(
            head_dir / "timestamps.csv",
            socket,
            module=infer_head_module_from_video(videos[socket]),
        )
        for socket in CAMERA_ORDER
    }
    tolerance_ms = float(head_cfg.get("timestamp_match_tolerance_ms", 1.0))

    # Precompute per-aligned-seq head frames and projections.
    records = []
    projection_rows = []
    fallback_matches = 0
    for frame in frames:
        sequence = int(frame["seq"])
        aligned = aligned_by_seq.get(sequence)
        if aligned is None:
            continue
        if int(float(aligned.get(f"{rigid_prefix}_status", "0"))) != 1:
            continue
        world_rigid = rigid_world_transform(aligned, rigid_prefix) @ mocap_to_head
        item = {
            "seq": sequence,
            "frames": {},
            "camera_poses": {},
            "projected": {},
        }
        for socket in CAMERA_ORDER:
            column = f"head_{socket}_exposure_end_timestamp_ms"
            target_ms = float(aligned[column])
            frame_index, ts_row, error = indexes[socket].nearest(
                target_ms, tolerance_ms
            )
            if error:
                fallback_matches += 1
            world_camera = world_rigid @ mounts[socket]
            pose = CameraPose(models[socket], world_camera)
            projected = project_joints(
                pose,
                frame["xyz_world_m"],
                min_depth_m=args.min_depth_m,
                width=models[socket].width,
                height=models[socket].height,
            )
            projected = {
                name: uv for name, uv in projected.items() if name not in FACE_HIDE
            }
            item["frames"][socket] = frame_index
            item["camera_poses"][socket] = world_camera.tolist()
            item["projected"][socket] = {
                name: uv.tolist() for name, uv in projected.items()
            }
            for name, xyz in frame["xyz_world_m"].items():
                uv = projected.get(name)
                projection_rows.append(
                    {
                        "seq": sequence,
                        "joint": name,
                        "camera": socket,
                        "frame_index": frame_index,
                        "capture_sequence": int(ts_row["seq"]),
                        "timestamp_match_error_ms": error,
                        "world_x_m": xyz[0],
                        "world_y_m": xyz[1],
                        "world_z_m": xyz[2],
                        "u_px": float("nan") if uv is None else float(uv[0]),
                        "v_px": float("nan") if uv is None else float(uv[1]),
                    }
                )
        records.append(item)

    if not records:
        raise RuntimeError("No aligned head frames available for projection")

    # Filename tag: no-mirror / identity T_mocap_rigid_head (user keyword).
    # Use underscore form; slash in "w/o" is invalid on Windows paths.
    out_tag = "wo_calibration_proj"

    with (output_dir / f"head_reprojection_2d_{out_tag}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(projection_rows[0]))
        writer.writeheader()
        writer.writerows(projection_rows)

    if args.csv_only:
        report = {
            "schema": "joint_projection.head_multiview_reprojection.v1",
            "data_root": str(data_root),
            "skeleton_source_path": str(skeleton_path),
            "skeleton_source": skeleton_source,
            "csv_only": True,
            "csv": str(output_dir / f"head_reprojection_2d_{out_tag}.csv"),
            "projection_rows": len(projection_rows),
            "aligned_frames": len(records),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    readers = {
        socket: H265CaptureReader(videos[socket], indexes[socket].rows)
        for socket in CAMERA_ORDER
    }

    writers = {
        socket: open_writer(
            output_dir / f"head_{socket}_multiview_reprojection_{out_tag}.mp4",
            (models[socket].width, models[socket].height),
            args.fps,
        )
        for socket in CAMERA_ORDER
    }
    stereo_writer = open_writer(
        output_dir / f"head_stereo_multiview_reprojection_{out_tag}.mp4",
        (1920, 600),
        args.fps,
    )

    try:
        for record_index, record in enumerate(records):
            panels = []
            for socket in CAMERA_ORDER:
                image = readers[socket].read(int(record["frames"][socket]))
                projected = {
                    name: np.asarray(uv, dtype=np.float64)
                    for name, uv in record["projected"][socket].items()
                }
                draw_skeleton_nose_only_face(image, projected, (255, 0, 255), radius=4)
                draw_laterality_labels(image, projected)
                title = (
                    f"HEAD {socket} seq={record['seq']} "
                    f"capture={record['frames'][socket]} "
                    f"h265 magenta=multiview->head L/R labeled"
                )
                cv2.putText(
                    image,
                    title,
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                writers[socket].write(image)
                panels.append(
                    cv2.resize(image, (960, 600), interpolation=cv2.INTER_AREA)
                )
            stereo = np.hstack(panels)
            stereo_writer.write(stereo)
            snapshot_seqs = set(args.snapshot_seq)
            if record_index == 0:
                snapshot_seqs.add(int(record["seq"]))
            if int(record["seq"]) in snapshot_seqs:
                snapshot = (
                    output_dir
                    / (
                        f"head_stereo_multiview_reprojection_{out_tag}"
                        f"_seq{int(record['seq']):04d}.jpg"
                    )
                )
                ok, encoded = cv2.imencode(".jpg", stereo)
                if ok:
                    encoded.tofile(str(snapshot))
                if record_index == 0:
                    first = (
                        output_dir
                        / (
                            f"head_stereo_multiview_reprojection_{out_tag}"
                            "_first_frame.jpg"
                        )
                    )
                    if ok:
                        encoded.tofile(str(first))
    finally:
        for writer in writers.values():
            writer.release()
        stereo_writer.release()
        for reader in readers.values():
            reader.close()

    report = {
        "schema": "joint_projection.head_multiview_reprojection.v1",
        "data_root": str(data_root),
        "skeleton_source_path": str(skeleton_path),
        "skeleton_source": skeleton_source,
        "aligned_csv": str(aligned_path),
        "head_directory": str(head_dir),
        "head_intrinsics": str(head_intrinsics_path.resolve()),
        "head_rigid_extrinsics": str(head_rigid_path.resolve()),
        "head_rigid_schema": head_rigid.get("schema"),
        "rigid_prefix": rigid_prefix,
        "output_tag": out_tag,
        "output_tag_note": (
            "User keyword w/o_calibration_proj (filesystem: wo_calibration_proj). "
            "Identity T_mocap_rigid_head; no mirror_y."
        ),
        "pose_chain": (
            "p_world (methods.filtered.multiview) -> "
            "T_world_head_mocap(mocap_CH3_08) -> "
            f"T_rigid_camera(basis={camera_basis}) -> "
            "Omni project"
        ),
        "soft_optimization": "none",
        "source_decode": "elementary_h265_packet_by_timestamps_bytes",
        "head_camera_rotation_basis": camera_basis,
        "T_mocap_rigid_head": mocap_to_head.tolist(),
        "T_mocap_rigid_head_note": head_cfg.get(
            "T_mocap_rigid_head_note",
            "Identity; mirror_y reverted.",
        ),
        "fixes": {
            "z90_body_orientation": (
                "head_camera_rotation_basis=xy_swap "
                "(CAM_A R=[[0,1,0],[1,0,0],[0,0,-1]], CAM_D=R@R_DA.T; "
                "same as approved fivepoint/head projection)"
            ),
            "left_right_laterality": (
                "No mirror_y. User confirmed mirror_y reversed L/R; "
                "keep xy_swap orientation only with identity T_mocap_rigid_head."
            ),
            "mp4_decode_lead": (
                "OpenCV on remuxed .mp4 shows capture N+4 as frame N (MAE=0 vs "
                "elementary .h265). Skeleton appeared ~4 frames behind video. "
                "Fixed by decoding .h265 with timestamps.csv packet sizes."
            ),
        },
        "videos": {socket: str(path) for socket, path in videos.items()},
        "counts": {
            "skeleton_frames": len(frames),
            "rendered_frames": len(records),
            "projection_rows": len(projection_rows),
            "timestamp_fallback_matches": fallback_matches,
            "h265_missing_fallbacks": {
                socket: readers[socket].missing_fallbacks for socket in CAMERA_ORDER
            },
            "joint_names": joint_names,
            "sequence_range": [seqs[0], seqs[-1]] if seqs else None,
        },
        "outputs": {
            "cam_a": str(
                output_dir / f"head_CAM_A_multiview_reprojection_{out_tag}.mp4"
            ),
            "cam_d": str(
                output_dir / f"head_CAM_D_multiview_reprojection_{out_tag}.mp4"
            ),
            "stereo": str(
                output_dir / f"head_stereo_multiview_reprojection_{out_tag}.mp4"
            ),
            "csv": str(output_dir / f"head_reprojection_2d_{out_tag}.csv"),
            "first_frame": str(
                output_dir
                / f"head_stereo_multiview_reprojection_{out_tag}_first_frame.jpg"
            ),
        },
        "min_depth_m": args.min_depth_m,
        "fps": args.fps,
    }
    (output_dir / f"report_{out_tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
