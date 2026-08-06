#!/usr/bin/env python3
"""Retarget ABX2/BVH motion by subject/default bone-length ratios and project it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
TOOLS = Path(r"C:\Users\hand\Desktop\Dataset\tools")
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from export_abx2_mocap_rigid_csv import (  # noqa: E402
    extract_ch3_rigids,
    pwr_map_for_sensors,
    read_abx2_header,
)
from preprocess_9cam_imu_mocap import (  # noqa: E402
    euler_to_quat,
    load_bvh_motion,
    parse_bvh,
    q_mul,
    q_normalize,
    rotate_vec,
)

import project_joints as base  # noqa: E402


ABX2 = Path(r"C:\Users\hand\Desktop\Dataset\0722\005.abx2")
BVH = Path(r"C:\Users\hand\Desktop\Dataset\0722\SIK_Actor_01_20260722_100548.bvh")
SUBJECT_JSON = HERE / "subject_001_skeleton_parameters.json"
CONFIG = HERE / "projection_config_0722_head_ch3_08.json"
CALIBRATION = HERE / "validation_0722_h265_fixed_time_calibration" / "calibration_fixed_time.json"
H265_MAPPING = HERE / "validation_0722_h265_random100" / "h265_frame_mapping.json"
IMAGES = HERE / "validation_0722_h265_random100" / "source_frames" / "module01"
OUTPUT = HERE / "validation_0722_abx2_subject_exact_kinematics_ch08"

CAMERA_KEYS = {"CAM_B": "module01_CAM_B", "CAM_C": "module01_CAM_C"}
JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Neck1", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
]
EDGES = [
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
    ("Spine2", "Neck"), ("Neck", "Neck1"), ("Neck1", "Head"),
    ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"),
]


def subject_values(document: dict[str, object]) -> dict[str, float]:
    return {
        "head": float(document["头部长度(mm)"]),
        "neck": float(document["脖子长度(mm)"]),
        "body": float(document["躯干长度(mm)"]),
        "shoulderWidth": float(document["肩宽长度(mm)"]),
        "leftUpperArm": float(document["左上臂长度(mm)"]),
        "rightUpperArm": float(document["右上臂长度(mm)"]),
        "leftForeArm": float(document["左小臂长度(mm)"]),
        "rightForeArm": float(document["右小臂长度(mm)"]),
        "leftHand": float(document["左手掌长度(mm)"]),
        "rightHand": float(document["右手掌长度(mm)"]),
        "hipWidth": float(document["胯宽长度(mm)"]),
        "leftUpperLeg": float(document["左大腿长度(mm)"]),
        "rightUpperLeg": float(document["右大腿长度(mm)"]),
        "leftLeg": float(document["左小腿长度(mm)"]),
        "rightLeg": float(document["右小腿长度(mm)"]),
        "leftAnkleHeight": float(document["左脚踝高度(mm)"]),
        "rightAnkleHeight": float(document["右脚踝高度(mm)"]),
        "leftFootLength": float(document["左脚掌长度(mm)"]),
        "rightFootLength": float(document["右脚掌长度(mm)"]),
    }


def scale_table(default_m: dict[str, float], subject: dict[str, float]) -> dict[str, dict[str, float]]:
    default_mm = {key: float(value) * 1000.0 for key, value in default_m.items()}
    default_key = {
        "head": "head", "neck": "neck", "body": "body", "shoulderWidth": "shoulderWidth",
        "leftUpperArm": "upperArm", "rightUpperArm": "upperArm",
        "leftForeArm": "foreArm", "rightForeArm": "foreArm",
        "leftHand": "hand", "rightHand": "hand", "hipWidth": "hipWidth",
        "leftUpperLeg": "upperLeg", "rightUpperLeg": "upperLeg",
        "leftLeg": "leg", "rightLeg": "leg",
        "leftAnkleHeight": "ankleHeight", "rightAnkleHeight": "ankleHeight",
        "leftFootLength": "footLength", "rightFootLength": "footLength",
    }
    return {
        key: {
            "default_mm": default_mm[default_key[key]],
            "subject_mm": value,
            "scale": value / default_mm[default_key[key]],
        }
        for key, value in subject.items()
    }


def exact_subject_offsets(joints: list[object], subject: dict[str, float]) -> list[np.ndarray]:
    """Build a rest skeleton whose semantic segment lengths equal the JSON."""
    offsets = [np.asarray(joint.offset, dtype=np.float64).copy() for joint in joints]
    by_name = {joint.name: i for i, joint in enumerate(joints)}

    def set_length(name: str, length_mm: float) -> None:
        index = by_name[name]
        norm = float(np.linalg.norm(offsets[index]))
        if norm <= 1e-12:
            raise ValueError(f"Cannot set length of zero offset: {name}")
        offsets[index] *= (length_mm / 10.0) / norm

    def distribute_chain(names: tuple[str, ...], total_mm: float) -> None:
        total_cm = sum(float(np.linalg.norm(offsets[by_name[name]])) for name in names)
        if total_cm <= 1e-12:
            raise ValueError(f"Cannot distribute zero-length chain: {names}")
        ratio = (total_mm / 10.0) / total_cm
        for name in names:
            offsets[by_name[name]] *= ratio

    # Pelvis and legs.
    for side in ("Left", "Right"):
        set_length(f"{side}UpLeg", subject["hipWidth"] / 2.0)
    for side in ("Left", "Right"):
        set_length(f"{side}Leg", subject[f"{side.lower()}UpperLeg"])
        set_length(f"{side}Foot", subject[f"{side.lower()}Leg"])
        end = offsets[by_name[f"{side}Foot_End"]]
        end[0] = 0.0
        end[1] = -subject[f"{side.lower()}AnkleHeight"] / 10.0
        end[2] = subject[f"{side.lower()}FootLength"] / 10.0

    # Exact axial chain lengths. Torso is Hips->Neck; neck is Neck->Head.
    distribute_chain(("Spine", "Spine1", "Spine2", "Neck"), subject["body"])
    distribute_chain(("Neck1", "Head"), subject["neck"])
    set_length("Head_End", subject["head"])

    # Set the neutral LeftArm-to-RightArm shoulder-center distance exactly.
    half_shoulder_cm = subject["shoulderWidth"] / 20.0
    for side in ("Left", "Right"):
        shoulder_name, arm_name = f"{side}Shoulder", f"{side}Arm"
        lateral_cm = abs(float(offsets[by_name[shoulder_name]][0])) + abs(float(offsets[by_name[arm_name]][0]))
        lateral_ratio = half_shoulder_cm / lateral_cm
        offsets[by_name[shoulder_name]][0] *= lateral_ratio
        offsets[by_name[arm_name]][0] *= lateral_ratio
        set_length(f"{side}ForeArm", subject[f"{side.lower()}UpperArm"])
        set_length(f"{side}Hand", subject[f"{side.lower()}ForeArm"])

        # Hand length is represented by the palm/finger descendants, which are
        # not among the 21 projected joints but remain exact in the full model.
        middle_chain = (
            f"{side}HandMiddle", f"{side}HandMiddle1", f"{side}HandMiddle2",
            f"{side}HandMiddle3", f"{side}HandMiddle3_End",
        )
        current_hand_cm = sum(float(np.linalg.norm(offsets[by_name[name]])) for name in middle_chain)
        hand_scale = (subject[f"{side.lower()}Hand"] / 10.0) / current_hand_cm
        hand_index = by_name[f"{side}Hand"]
        for i, joint in enumerate(joints):
            parent = joint.parent
            while parent is not None and parent != hand_index:
                parent = joints[parent].parent
            if parent == hand_index:
                offsets[i] *= hand_scale
    return offsets


def realized_lengths_mm(joints: list[object], offsets: list[np.ndarray]) -> dict[str, float]:
    by_name = {joint.name: i for i, joint in enumerate(joints)}
    length = lambda name: float(np.linalg.norm(offsets[by_name[name]]) * 10.0)
    chain = lambda names: float(sum(np.linalg.norm(offsets[by_name[name]]) for name in names) * 10.0)
    shoulder_half = abs(float(offsets[by_name["LeftShoulder"]][0])) + abs(float(offsets[by_name["LeftArm"]][0]))
    return {
        "head": length("Head_End"),
        "neck": chain(("Neck1", "Head")),
        "body": chain(("Spine", "Spine1", "Spine2", "Neck")),
        "shoulderWidth": 2.0 * shoulder_half * 10.0,
        "leftUpperArm": length("LeftForeArm"), "rightUpperArm": length("RightForeArm"),
        "leftForeArm": length("LeftHand"), "rightForeArm": length("RightHand"),
        "leftHand": chain(("LeftHandMiddle", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle3_End")),
        "rightHand": chain(("RightHandMiddle", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle3_End")),
        "hipWidth": length("LeftUpLeg") + length("RightUpLeg"),
        "leftUpperLeg": length("LeftLeg"), "rightUpperLeg": length("RightLeg"),
        "leftLeg": length("LeftFoot"), "rightLeg": length("RightFoot"),
        "leftAnkleHeight": abs(float(offsets[by_name["LeftFoot_End"]][1]) * 10.0),
        "rightAnkleHeight": abs(float(offsets[by_name["RightFoot_End"]][1]) * 10.0),
        "leftFootLength": abs(float(offsets[by_name["LeftFoot_End"]][2]) * 10.0),
        "rightFootLength": abs(float(offsets[by_name["RightFoot_End"]][2]) * 10.0),
    }


def local_pose_at(joint: object, motion: np.ndarray, frame_float: float) -> tuple[np.ndarray, np.ndarray]:
    i0 = int(np.clip(np.floor(frame_float), 0, len(motion) - 1))
    i1 = min(i0 + 1, len(motion) - 1)
    alpha = float(frame_float - i0)
    values = motion[[i0, i1], joint.channel_start:joint.channel_start + len(joint.channels)] if joint.channels else np.empty((2, 0))
    translation = np.zeros((2, 3), dtype=np.float64)
    rotations = np.zeros((2, 3), dtype=np.float64)
    order = ""
    for ci, channel in enumerate(joint.channels):
        if channel.endswith("position"):
            translation[:, "XYZ".index(channel[0])] = values[:, ci]
        elif channel.endswith("rotation"):
            rotations[:, "XYZ".index(channel[0])] = np.deg2rad(values[:, ci])
            order += channel[0]
    quat = euler_to_quat(rotations[:, 0], rotations[:, 1], rotations[:, 2], order or "XYZ")
    if np.dot(quat[0], quat[1]) < 0:
        quat[1] *= -1
    q = q_normalize(((1.0 - alpha) * quat[0] + alpha * quat[1])[None, :])[0]
    t = (1.0 - alpha) * translation[0] + alpha * translation[1]
    return t, q


def forward_kinematics_at(joints: list[object], offsets: list[np.ndarray], motion: np.ndarray, frame_float: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    world_pos: list[np.ndarray] = []
    world_q: list[np.ndarray] = []
    for i, joint in enumerate(joints):
        channel_t, local_q = local_pose_at(joint, motion, frame_float)
        local_t = offsets[i] + channel_t
        if joint.parent is None:
            pos, quat = local_t, local_q
        else:
            pos = world_pos[joint.parent] + rotate_vec(world_q[joint.parent][None, :], local_t[None, :])[0]
            quat = q_mul(world_q[joint.parent][None, :], local_q[None, :])[0]
        world_pos.append(pos)
        world_q.append(q_normalize(quat[None, :])[0])
    return world_pos, world_q


def interpolate_rigid(rows: list[dict[str, object]], frame_float: float) -> tuple[np.ndarray, np.ndarray]:
    i0 = int(np.clip(np.floor(frame_float), 0, len(rows) - 1)); i1 = min(i0 + 1, len(rows) - 1)
    alpha = float(frame_float - i0)
    p0 = np.asarray([rows[i0][axis] for axis in "xyz"], dtype=np.float64) * 1000.0
    p1 = np.asarray([rows[i1][axis] for axis in "xyz"], dtype=np.float64) * 1000.0
    q0 = np.asarray([rows[i0][f"q{axis}"] for axis in "wxyz"], dtype=np.float64)
    q1 = np.asarray([rows[i1][f"q{axis}"] for axis in "wxyz"], dtype=np.float64)
    if np.dot(q0, q1) < 0: q1 *= -1
    return (1 - alpha) * p0 + alpha * p1, q_normalize(((1 - alpha) * q0 + alpha * q1)[None, :])[0]


def q_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def draw(image: np.ndarray, points_world_mm: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray, model: dict[str, object], color: tuple[int, int, int], thickness: int) -> None:
    points_camera = (camera_rotation.T @ (points_world_mm - camera_position).T).T
    uv, valid = base.omni_project(points_camera, model)
    visible = base.in_image(uv, valid, int(model["width"]), int(model["height"]))
    index = {name: i for i, name in enumerate(JOINT_NAMES)}
    for first, second in EDGES:
        a, b = index[first], index[second]
        if visible[a] and visible[b]:
            cv2.line(image, tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int)), color, thickness, cv2.LINE_AA)
    for point in uv[visible]:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    info, abx_config = read_abx2_header(ABX2)
    avatar = abx_config["sik"]["avatars"][0]
    defaults_m = {key: float(value) for key, value in avatar["BoneLength"].items()}
    subject_doc = json.loads(SUBJECT_JSON.read_text(encoding="utf-8"))
    subject = subject_values(subject_doc)
    scales = scale_table(defaults_m, subject)
    (OUTPUT / "abx2_default_bone_lengths.json").write_text(json.dumps({"source": str(ABX2), "subject": subject_doc["受试者姓名"], "unit": "mm", "bone_lengths": {key: value * 1000 for key, value in defaults_m.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "subject_bone_length_scales.json").write_text(json.dumps({"source_default": str(ABX2), "source_subject": str(SUBJECT_JSON), "subject": subject_doc["受试者姓名"], "scales": scales}, ensure_ascii=False, indent=2), encoding="utf-8")

    joints, channel_count, frame_time, frame_count = parse_bvh(BVH)
    motion = load_bvh_motion(BVH, channel_count, frame_count)
    original_offsets = [np.asarray(joint.offset, dtype=np.float64).copy() for joint in joints]
    subject_offsets = exact_subject_offsets(joints, subject)
    realized = realized_lengths_mm(joints, subject_offsets)
    length_validation = {
        key: {"target_mm": float(value), "realized_mm": realized.get(key), "error_mm": (None if key not in realized else realized[key] - float(value))}
        for key, value in subject.items()
    }
    (OUTPUT / "exact_bone_length_validation.json").write_text(
        json.dumps({"subject": subject_doc["受试者姓名"], "lengths": length_validation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joint_index = {joint.name: i for i, joint in enumerate(joints)}

    pwrs = pwr_map_for_sensors(abx_config, (308,))
    rigid_rows = extract_ch3_rigids(ABX2, pwrs, float(info["ABXInfo"]["fps"]))[308]
    mapping = base.load_json(H265_MAPPING)
    calibration = base.load_json(CALIBRATION)
    timing = calibration["timing"]
    models = base.load_camera_models(base.load_json(CONFIG))
    reports: dict[str, object] = {}
    head_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm

    for camera, camera_key in CAMERA_KEYS.items():
        model = models[camera_key]
        rc = np.asarray(calibration[f"R_rigid_cam_{camera[-1]}"])
        pc = np.asarray(calibration[f"p_rigid_cam_{camera[-1]}_mm"])
        head_to_camera_rigid = pc - head_joint_in_rigid_mm
        destination = OUTPUT / camera_key
        comparison_destination = OUTPUT / "default_vs_subject" / camera_key
        destination.mkdir(parents=True, exist_ok=True); comparison_destination.mkdir(parents=True, exist_ok=True)
        records = {int(row["seq"]): row for row in mapping[camera]["mapping"]}
        for seq, row in sorted(records.items()):
            sample_time = float(timing["global_offset_sec"]) + float(timing["global_scale"]) * ((float(row["device_ts_ms"]) - float(timing["origin_ms"])) / 1000.0) + float(timing["head_imu_delta_sec"]) + float(timing["exposure_center_shift_sec"])
            frame_float = sample_time / frame_time
            original_pos, _ = forward_kinematics_at(joints, original_offsets, motion, frame_float)
            scaled_pos, _ = forward_kinematics_at(joints, subject_offsets, motion, frame_float)
            _, head_q = interpolate_rigid(rigid_rows, frame_float)
            head_r = q_to_matrix(head_q)
            source = cv2.imread(str(IMAGES / camera / f"seq_{seq:06d}.jpg"))
            original_mm = np.asarray([original_pos[joint_index[name]] for name in JOINT_NAMES]) * 10.0
            scaled_mm = np.asarray([scaled_pos[joint_index[name]] for name in JOINT_NAMES]) * 10.0
            # Match the previously validated CH08 visualization convention:
            # instantaneous CH08 orientation, but translation anchored to the
            # retargeted BVH Head. ABX2 PWR and SIK/BVH use different absolute
            # translation origins in this recording (~1 m apart).
            camera_position = scaled_mm[JOINT_NAMES.index("Head")] + head_r @ head_to_camera_rigid
            camera_rotation = head_r @ rc
            scaled_image = source.copy()
            draw(scaled_image, scaled_mm, camera_position, camera_rotation, model, (255, 0, 255), 4)
            cv2.putText(scaled_image, f"{camera_key} CH08 exact-kinematics seq={seq:06d}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, .85, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite(str(destination / f"seq_{seq:06d}_joints.jpg"), scaled_image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            comparison = source.copy()
            draw(comparison, original_mm, camera_position, camera_rotation, model, (255, 255, 0), 2)
            draw(comparison, scaled_mm, camera_position, camera_rotation, model, (255, 0, 255), 4)
            cv2.putText(comparison, "cyan=ABX2 default  magenta=subject exact kinematics", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite(str(comparison_destination / f"seq_{seq:06d}_comparison.jpg"), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94])
        reports[camera] = {"images": len(records), "output": str(destination)}

    report = {"schema": "abx2_subject_exact_kinematics_projection.v1", "source_abx2": str(ABX2), "source_motion": str(BVH), "subject_json": str(SUBJECT_JSON), "frame_count": frame_count, "frame_time_sec": frame_time, "reference_scale_file": str(OUTPUT / "subject_bone_length_scales.json"), "exact_length_validation": str(OUTPUT / "exact_bone_length_validation.json"), "reports": reports, "notes": ["Local joint rotations and root translation are unchanged.", "Static offsets are reconstructed so semantic chain lengths equal the subject JSON exactly; this is not a uniform default-skeleton scale.", "CH3_08 instantaneous orientation is read directly from ABX2; translation is anchored to the retargeted BVH Head, matching the previous CH08 comparison convention because ABX2 PWR and SIK/BVH absolute translation origins differ by about 1 m.", "No YOLO, wrist IK, or cross-camera 2-D transfer is used."]}
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
