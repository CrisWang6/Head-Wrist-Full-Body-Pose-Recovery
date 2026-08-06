#!/usr/bin/env python3
"""Render CAM_C with BVH body, Sapiens shoulders/elbows, and rigid wrists."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import project_joints as base


WIDTH = 1920
HEIGHT = 1200
FRAME_BYTES = WIDTH * HEIGHT * 3
JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Neck1", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
]
WRIST_TO_RIGID_MM = {
    "Left": np.asarray([53.5, 76.5, 2.2], dtype=np.float64),
    "Right": np.asarray([53.5, 76.5, -2.2], dtype=np.float64),
}
R_WRIST_RIGID = {
    "Left": np.column_stack(
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
    ),
    "Right": np.column_stack(
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    ),
}
RTMPOSE_FIELDS = {
    "LeftArm": ("left_shoulder_x", "left_shoulder_y"),
    "RightArm": ("right_shoulder_x", "right_shoulder_y"),
    "LeftForeArm": ("left_elbow_x", "left_elbow_y"),
    "RightForeArm": ("right_elbow_x", "right_elbow_y"),
}
EDGES = [
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Neck"),
    ("Neck", "Neck1"),
    ("Neck1", "Head"),
    ("Spine2", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
]
HIDDEN = {"LeftShoulder", "RightShoulder"}


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def in_image(point: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(point))
        and 0.0 <= point[0] < WIDTH
        and 0.0 <= point[1] < HEIGHT
    )


def camera_geometry(
    row: dict[str, str],
    model: dict[str, object],
    calibration: dict[str, object],
) -> dict[str, object]:
    joint_index = {name: index for index, name in enumerate(JOINT_NAMES)}
    points = np.asarray(
        [
            [
                float(row[f"mocap_{name}_world_{axis}"]) * 10.0
                for axis in "xyz"
            ]
            for name in JOINT_NAMES
        ],
        dtype=np.float64,
    )
    head_position = np.asarray(
        [
            float(row[f"mocap_CH3_08_Rigid_K_world_{axis}"]) * 1000.0
            for axis in "xyz"
        ]
    )
    head_quaternion = np.asarray(
        [
            float(row[f"mocap_CH3_08_Rigid_K_world_q{axis}"])
            for axis in "wxyz"
        ]
    )
    head_quaternion /= np.linalg.norm(head_quaternion)
    r_world_head = quaternion_to_matrix(head_quaternion)

    head_axes = np.column_stack(
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    bvh_head = points[joint_index["Head"]]

    r_rigid_camera = np.asarray(
        calibration["R_rigid_cam_C"], dtype=np.float64
    )
    p_rigid_camera = np.asarray(
        calibration["p_rigid_cam_C_mm"], dtype=np.float64
    )
    camera_position = (
        bvh_head
        + r_world_head @ (p_rigid_camera - head_joint_in_rigid_mm)
    )
    camera_rotation = r_world_head @ r_rigid_camera

    rigid_head_joint = (
        head_position + r_world_head @ head_joint_in_rigid_mm
    )
    rigid_to_bvh_translation = bvh_head - rigid_head_joint

    points_camera = (
        camera_rotation.T @ (points - camera_position).T
    ).T
    uv_raw, valid_raw = base.omni_project(points_camera, model)

    rigid_wrist_uv: dict[str, np.ndarray] = {}
    for side, rigid_code in (("Left", "01"), ("Right", "07")):
        rigid_name = f"CH3_{rigid_code}_Rigid_K"
        rigid_position = np.asarray(
            [
                float(row[f"mocap_{rigid_name}_world_{axis}"]) * 1000.0
                for axis in "xyz"
            ]
        )
        rigid_quaternion = np.asarray(
            [
                float(row[f"mocap_{rigid_name}_world_q{axis}"])
                for axis in "wxyz"
            ]
        )
        rigid_quaternion /= np.linalg.norm(rigid_quaternion)
        r_world_rigid = quaternion_to_matrix(rigid_quaternion)
        r_world_wrist = r_world_rigid @ R_WRIST_RIGID[side].T
        wrist_world = (
            rigid_position
            - r_world_wrist @ WRIST_TO_RIGID_MM[side]
            + rigid_to_bvh_translation
        )
        wrist_camera = camera_rotation.T @ (
            wrist_world - camera_position
        )
        wrist_uv, wrist_valid = base.omni_project(
            wrist_camera[None, :], model
        )
        rigid_wrist_uv[side] = (
            wrist_uv[0]
            if wrist_valid[0]
            else np.asarray([np.nan, np.nan])
        )

    return {
        "uv_raw": uv_raw,
        "valid_raw": valid_raw,
        "rigid_wrist_uv": rigid_wrist_uv,
    }


def draw_skeleton(
    frame: np.ndarray,
    uv: np.ndarray,
    visible: np.ndarray,
    index: dict[str, int],
) -> None:
    edge_color = (0, 220, 255)
    for first, second in EDGES:
        first_index = index[first]
        second_index = index[second]
        if visible[first_index] and visible[second_index]:
            cv2.line(
                frame,
                tuple(np.rint(uv[first_index]).astype(int)),
                tuple(np.rint(uv[second_index]).astype(int)),
                edge_color,
                4,
                cv2.LINE_AA,
            )

    for name in JOINT_NAMES:
        if name in HIDDEN:
            continue
        joint_index = index[name]
        if not visible[joint_index]:
            continue
        point = tuple(np.rint(uv[joint_index]).astype(int))
        if name in RTMPOSE_FIELDS:
            color, radius = (255, 255, 0), 7
        elif name in {"LeftHand", "RightHand"}:
            color, radius = (255, 0, 255), 8
        elif name == "Spine2":
            color, radius = (0, 255, 0), 8
        else:
            color, radius = (0, 150, 255), 5
        cv2.circle(
            frame, point, radius + 2, (0, 0, 0), -1, cv2.LINE_AA
        )
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    recording = args.recording.resolve()
    project_dir = args.project_dir.resolve()
    aligned_rows = {
        int(row["seq"]): row
        for row in load_csv(
            recording / "aligned_data" / "aligned_30hz.csv"
        )
    }
    pose_rows_list = load_csv(
        recording
        / "aligned_data"
        / "module01_cam_c_shoulder_elbow_2d.csv"
    )
    pose_rows = {int(row["seq"]): row for row in pose_rows_list}
    decoded_to_seq = {
        int(row["decoded_frame_index"]): int(row["seq"])
        for row in pose_rows_list
        if row["decoded_frame_index"].strip()
    }

    config = base.load_json(
        project_dir / "projection_config_0722_head_ch3_08.json"
    )
    calibration = base.load_json(
        project_dir
        / "validation_0722_h265_fixed_time_calibration"
        / "calibration_fixed_time.json"
    )
    model = base.load_camera_models(config)["module01_CAM_C"]
    index = {name: i for i, name in enumerate(JOINT_NAMES)}

    output = (
        recording
        / "module01_CAM_C_hybrid_sapiens_rigid_bvh_skeleton.mp4"
    )
    report_path = (
        recording
        / "aligned_data"
        / "cam_c_hybrid_sapiens_rigid_bvh_report.json"
    )
    decoder = subprocess.Popen(
        [
            str(args.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            str(recording / "module01_D45D2E00_CAM_C.h265"),
            "-vsync",
            "0",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            str(args.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "19",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )

    decoded_index = 0
    frame_count = 0
    rtmpose_counts = {name: 0 for name in RTMPOSE_FIELDS}
    wrist_counts = {"LeftHand": 0, "RightHand": 0}
    while True:
        raw = read_exact(decoder.stdout, FRAME_BYTES)
        if not raw:
            break
        if len(raw) != FRAME_BYTES:
            raise RuntimeError(
                f"Partial decoded frame: {len(raw)} bytes"
            )
        frame = (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(HEIGHT, WIDTH, 3)
            .copy()
        )
        seq = decoded_to_seq.get(decoded_index)
        if seq is not None and seq in aligned_rows:
            geometry = camera_geometry(
                aligned_rows[seq], model, calibration
            )
            uv = np.asarray(geometry["uv_raw"]).copy()
            visible = np.asarray(geometry["valid_raw"], dtype=bool)
            visible &= np.asarray([in_image(point) for point in uv])
            visible[index["LeftShoulder"]] = False
            visible[index["RightShoulder"]] = False

            pose = pose_rows[seq]
            for name, (x_field, y_field) in RTMPOSE_FIELDS.items():
                joint_index = index[name]
                visible[joint_index] = False
                if (
                    pose["status"] == "ok"
                    and pose[x_field].strip()
                    and pose[y_field].strip()
                ):
                    point = np.asarray(
                        [float(pose[x_field]), float(pose[y_field])]
                    )
                    if in_image(point):
                        uv[joint_index] = point
                        visible[joint_index] = True
                        rtmpose_counts[name] += 1

            for side, name in (
                ("Left", "LeftHand"),
                ("Right", "RightHand"),
            ):
                joint_index = index[name]
                point = geometry["rigid_wrist_uv"][side]
                uv[joint_index] = point
                visible[joint_index] = in_image(point)
                if visible[joint_index]:
                    wrist_counts[name] += 1

            draw_skeleton(frame, uv, visible, index)
            frame_count += 1

        cv2.rectangle(frame, (0, 0), (980, 82), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"CAM_C hybrid skeleton  seq={seq}",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "BVH+CH3_08 | cyan=Sapiens2-0.4B shoulder/elbow | magenta=CH3_01/07 wrist",
            (20, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.61,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        encoder.stdin.write(frame.tobytes())
        decoded_index += 1
        if decoded_index % 1000 == 0:
            print(f"rendered {decoded_index} frames", flush=True)

    decoder_code = decoder.wait()
    encoder.stdin.close()
    encoder_code = encoder.wait()
    if decoder_code or encoder_code:
        raise RuntimeError(
            f"ffmpeg failed: decoder={decoder_code}, encoder={encoder_code}"
        )

    report = {
        "schema": "cam_c_hybrid_sapiens_rigid_bvh.v1",
        "video": str(output),
        "decoded_frames": decoded_index,
        "frames_with_aligned_seq": frame_count,
        "camera_pose": "CH3_08 rigid",
        "other_joints": "original BVH",
        "shoulders_elbows": "Sapiens2-0.4B module01_cam_c_shoulder_elbow_2d.csv",
        "wrists": "CH3_01/CH3_07 rigid-derived",
        "hidden_joints": sorted(HIDDEN),
        "arm_edges": [
            "Spine2->LeftArm->LeftForeArm->LeftHand",
            "Spine2->RightArm->RightForeArm->RightHand",
        ],
        "rtmpose_visible_counts": rtmpose_counts,
        "rigid_wrist_visible_counts": wrist_counts,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
