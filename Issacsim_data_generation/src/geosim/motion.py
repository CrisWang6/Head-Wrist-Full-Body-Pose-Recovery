from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geosim.linalg import ensure_rotation_stack, frame_from_forearm, frame_from_shoulders, rotz


@dataclass(frozen=True)
class MotionSequence:
    name: str
    fps: float
    head_pos: np.ndarray
    head_rot: np.ndarray
    left_elbow_pos: np.ndarray
    left_wrist_pos: np.ndarray
    left_wrist_rot: np.ndarray
    right_elbow_pos: np.ndarray | None = None
    right_wrist_pos: np.ndarray | None = None
    right_wrist_rot: np.ndarray | None = None
    source_path: Path | None = None
    betas: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return int(self.head_pos.shape[0])


def load_motion_npz(path: str | Path, smplx_model_path: str | Path | None = None) -> MotionSequence:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    required = {"head_pos", "left_wrist_pos", "left_elbow_pos"}
    if not required.issubset(keys):
        if {"poses", "trans"}.issubset(keys):
            if smplx_model_path is None:
                raise ValueError(f"{path} is raw AMASS data; pass smplx_model_path to extract joints.")
            return load_amass_smplx_motion(path, smplx_model_path)
        missing = ", ".join(sorted(required - keys))
        raise ValueError(f"{path} is missing required motion arrays: {missing}")

    head_pos = np.asarray(data["head_pos"], dtype=float)
    left_wrist_pos = np.asarray(data["left_wrist_pos"], dtype=float)
    left_elbow_pos = np.asarray(data["left_elbow_pos"], dtype=float)
    right_wrist_pos = np.asarray(data["right_wrist_pos"], dtype=float) if "right_wrist_pos" in keys else None
    right_elbow_pos = np.asarray(data["right_elbow_pos"], dtype=float) if "right_elbow_pos" in keys else None
    frames = head_pos.shape[0]
    _validate_positions(head_pos, frames, "head_pos")
    _validate_positions(left_wrist_pos, frames, "left_wrist_pos")
    _validate_positions(left_elbow_pos, frames, "left_elbow_pos")
    if right_wrist_pos is not None:
        _validate_positions(right_wrist_pos, frames, "right_wrist_pos")
    if right_elbow_pos is not None:
        _validate_positions(right_elbow_pos, frames, "right_elbow_pos")

    head_rot = ensure_rotation_stack(data["head_rot"] if "head_rot" in keys else np.eye(3), frames, "head_rot")
    if "left_wrist_rot" in keys:
        left_wrist_rot = ensure_rotation_stack(data["left_wrist_rot"], frames, "left_wrist_rot")
    else:
        left_wrist_rot = np.stack(
            [frame_from_forearm(left_elbow_pos[i], left_wrist_pos[i]) for i in range(frames)],
            axis=0,
        )
    right_wrist_rot = None
    if right_wrist_pos is not None and right_elbow_pos is not None:
        if "right_wrist_rot" in keys:
            right_wrist_rot = ensure_rotation_stack(data["right_wrist_rot"], frames, "right_wrist_rot")
        else:
            right_wrist_rot = np.stack(
                [frame_from_forearm(right_elbow_pos[i], right_wrist_pos[i]) for i in range(frames)],
                axis=0,
            )
    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in keys else 30.0
    return MotionSequence(
        name=path.stem,
        fps=fps,
        head_pos=head_pos,
        head_rot=head_rot,
        left_elbow_pos=left_elbow_pos,
        left_wrist_pos=left_wrist_pos,
        left_wrist_rot=left_wrist_rot,
        right_elbow_pos=right_elbow_pos,
        right_wrist_pos=right_wrist_pos,
        right_wrist_rot=right_wrist_rot,
        source_path=path,
        betas=np.asarray(data["betas"], dtype=float) if "betas" in keys else None,
    )


def load_motion_dir(path: str | Path, smplx_model_path: str | Path | None = None) -> list[MotionSequence]:
    path = Path(path)
    motions = []
    for motion_path in sorted(path.rglob("*.npz")):
        try:
            motions.append(load_motion_npz(motion_path, smplx_model_path=smplx_model_path))
        except ValueError:
            continue
    return motions


def load_amass_smplx_motion(path: str | Path, smplx_model_path: str | Path) -> MotionSequence:
    from geosim.smplx_numpy import load_smplx_model

    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if not {"poses", "trans"}.issubset(data.files):
        raise ValueError(f"{path} is not a usable AMASS motion file.")

    model = load_smplx_model(smplx_model_path)
    poses = np.asarray(data["poses"], dtype=float)
    trans = np.asarray(data["trans"], dtype=float)
    betas = np.asarray(data["betas"], dtype=float) if "betas" in data.files else None
    joints, _ = model.forward_joints(poses, trans, betas)
    joint_names = model.joint2num
    head_idx = joint_names["Head"]
    left_shoulder_idx = joint_names["L_Shoulder"]
    right_shoulder_idx = joint_names["R_Shoulder"]
    left_elbow_idx = joint_names["L_Elbow"]
    left_wrist_idx = joint_names["L_Wrist"]
    right_elbow_idx = joint_names["R_Elbow"]
    right_wrist_idx = joint_names["R_Wrist"]
    head_rot = np.stack(
        [frame_from_shoulders(joints[i, left_shoulder_idx, :], joints[i, right_shoulder_idx, :]) for i in range(len(joints))],
        axis=0,
    )
    left_wrist_rot = np.stack(
        [frame_from_forearm(joints[i, left_elbow_idx, :], joints[i, left_wrist_idx, :]) for i in range(len(joints))],
        axis=0,
    )
    right_wrist_rot = np.stack(
        [frame_from_forearm(joints[i, right_elbow_idx, :], joints[i, right_wrist_idx, :]) for i in range(len(joints))],
        axis=0,
    )
    fps = float(np.asarray(data["mocap_frame_rate"]).reshape(-1)[0]) if "mocap_frame_rate" in data.files else 30.0
    relative_name = path.with_suffix("").as_posix()
    return MotionSequence(
        name=relative_name,
        fps=fps,
        head_pos=joints[:, head_idx, :],
        head_rot=head_rot,
        left_elbow_pos=joints[:, left_elbow_idx, :],
        left_wrist_pos=joints[:, left_wrist_idx, :],
        left_wrist_rot=left_wrist_rot,
        right_elbow_pos=joints[:, right_elbow_idx, :],
        right_wrist_pos=joints[:, right_wrist_idx, :],
        right_wrist_rot=right_wrist_rot,
        source_path=path,
        betas=betas,
    )


def synthetic_motion(frames: int = 120, fps: float = 30.0) -> MotionSequence:
    t = np.arange(frames, dtype=float) / fps
    head_pos = np.column_stack(
        [
            0.03 * np.sin(2.0 * np.pi * 0.2 * t),
            0.02 * np.cos(2.0 * np.pi * 0.17 * t),
            np.full(frames, 1.62),
        ]
    )
    head_rot = np.stack([rotz(0.12 * np.sin(2.0 * np.pi * 0.15 * ti)) for ti in t], axis=0)

    left_wrist_pos = np.column_stack(
        [
            -0.24 + 0.12 * np.sin(2.0 * np.pi * 0.35 * t),
            0.10 + 0.24 * np.sin(2.0 * np.pi * 0.21 * t + 0.4),
            1.06 + 0.13 * np.sin(2.0 * np.pi * 0.27 * t + 1.1),
        ]
    )
    forearm_axis = np.column_stack(
        [
            -0.25 + 0.03 * np.sin(2.0 * np.pi * 0.25 * t),
            0.05 * np.cos(2.0 * np.pi * 0.29 * t),
            -0.04 + 0.04 * np.sin(2.0 * np.pi * 0.31 * t),
        ]
    )
    forearm_axis /= np.linalg.norm(forearm_axis, axis=1, keepdims=True)
    left_elbow_pos = left_wrist_pos - 0.27 * forearm_axis
    left_wrist_rot = np.stack(
        [frame_from_forearm(left_elbow_pos[i], left_wrist_pos[i]) for i in range(frames)],
        axis=0,
    )
    right_wrist_pos = left_wrist_pos.copy()
    right_wrist_pos[:, 0] *= -1.0
    right_elbow_pos = left_elbow_pos.copy()
    right_elbow_pos[:, 0] *= -1.0
    right_wrist_rot = np.stack(
        [frame_from_forearm(right_elbow_pos[i], right_wrist_pos[i]) for i in range(frames)],
        axis=0,
    )
    return MotionSequence(
        name="synthetic_reach",
        fps=fps,
        head_pos=head_pos,
        head_rot=head_rot,
        left_elbow_pos=left_elbow_pos,
        left_wrist_pos=left_wrist_pos,
        left_wrist_rot=left_wrist_rot,
        right_elbow_pos=right_elbow_pos,
        right_wrist_pos=right_wrist_pos,
        right_wrist_rot=right_wrist_rot,
        source_path=None,
        betas=None,
    )


def _validate_positions(value: np.ndarray, frames: int, name: str) -> None:
    if value.shape != (frames, 3):
        raise ValueError(f"{name} must have shape ({frames}, 3).")
