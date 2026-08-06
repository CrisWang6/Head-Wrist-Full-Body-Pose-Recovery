from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from geosim.linalg import axis_angle_to_matrix


@dataclass(frozen=True)
class SmplxFrame:
    joints: np.ndarray
    joint_rotations: np.ndarray
    vertices: np.ndarray | None = None


class SmplxNumpyModel:
    """Small NumPy-only SMPL-X adapter for AMASS geometry tests.

    This intentionally ignores pose blend shapes and dynamic landmarks. It uses
    shape blend shapes, the SMPL-X kinematic tree, and linear blend skinning,
    which is enough for approximate joint paths and lightweight visualization.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        data = np.load(self.model_path, allow_pickle=True)
        self.v_template = np.asarray(data["v_template"], dtype=float)
        self.shapedirs = np.asarray(data["shapedirs"], dtype=float)
        self.j_regressor = np.asarray(data["J_regressor"], dtype=float)
        self.parents = np.asarray(data["kintree_table"], dtype=np.int64)[0].copy()
        self.parents[0] = -1
        self.weights = np.asarray(data["weights"], dtype=float)
        self.faces = np.asarray(data["f"], dtype=np.int32)
        self.joint2num = data["joint2num"].item()

    @property
    def joint_count(self) -> int:
        return int(self.j_regressor.shape[0])

    def shaped_vertices_and_joints(self, betas: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        betas_arr = np.zeros(0, dtype=float) if betas is None else np.asarray(betas, dtype=float).reshape(-1)
        n_betas = min(len(betas_arr), self.shapedirs.shape[2])
        v_shaped = self.v_template.copy()
        if n_betas:
            v_shaped = v_shaped + np.einsum("vcn,n->vc", self.shapedirs[:, :, :n_betas], betas_arr[:n_betas])
        joints = self.j_regressor @ v_shaped
        return v_shaped, joints

    def forward_frame(
        self,
        pose_axis_angle: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray | None = None,
        include_vertices: bool = False,
    ) -> SmplxFrame:
        v_shaped, joints_rest = self.shaped_vertices_and_joints(betas)
        pose = np.asarray(pose_axis_angle, dtype=float).reshape(-1, 3)
        if pose.shape[0] < self.joint_count:
            padded = np.zeros((self.joint_count, 3), dtype=float)
            padded[: pose.shape[0]] = pose
            pose = padded
        local_rot = axis_angle_to_matrix(pose[: self.joint_count])
        global_rot, global_pos, transforms = self._forward_kinematics(local_rot, joints_rest, np.asarray(trans, dtype=float))
        vertices = None
        if include_vertices:
            rest_inv = self._rest_inverse_transforms(joints_rest)
            skinning_transforms = transforms @ rest_inv
            vertices_h = np.column_stack([v_shaped, np.ones(len(v_shaped))])
            vertices = np.einsum("vj,jab,vb->va", self.weights, skinning_transforms, vertices_h)[:, :3]
        return SmplxFrame(joints=global_pos, joint_rotations=global_rot, vertices=vertices)

    def forward_joints(
        self,
        poses_axis_angle: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        poses = np.asarray(poses_axis_angle, dtype=float)
        translations = np.asarray(trans, dtype=float)
        if poses.ndim != 2:
            raise ValueError("poses_axis_angle must have shape (F, J*3).")
        frames = poses.shape[0]
        all_joints = np.zeros((frames, self.joint_count, 3), dtype=float)
        all_rot = np.zeros((frames, self.joint_count, 3, 3), dtype=float)
        _, joints_rest = self.shaped_vertices_and_joints(betas)
        for idx in range(frames):
            pose = poses[idx].reshape(-1, 3)
            if pose.shape[0] < self.joint_count:
                padded = np.zeros((self.joint_count, 3), dtype=float)
                padded[: pose.shape[0]] = pose
                pose = padded
            local_rot = axis_angle_to_matrix(pose[: self.joint_count])
            global_rot, global_pos, _ = self._forward_kinematics(local_rot, joints_rest, translations[idx])
            all_joints[idx] = global_pos
            all_rot[idx] = global_rot
        return all_joints, all_rot

    def _forward_kinematics(
        self,
        local_rot: np.ndarray,
        joints_rest: np.ndarray,
        trans: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        global_rot = np.zeros((self.joint_count, 3, 3), dtype=float)
        global_pos = np.zeros((self.joint_count, 3), dtype=float)
        transforms = np.zeros((self.joint_count, 4, 4), dtype=float)
        transforms[:, 3, 3] = 1.0

        for joint_idx in range(self.joint_count):
            parent = int(self.parents[joint_idx])
            if parent < 0:
                global_rot[joint_idx] = local_rot[joint_idx]
                global_pos[joint_idx] = trans + local_rot[joint_idx] @ joints_rest[joint_idx]
            else:
                offset = joints_rest[joint_idx] - joints_rest[parent]
                global_rot[joint_idx] = global_rot[parent] @ local_rot[joint_idx]
                global_pos[joint_idx] = global_pos[parent] + global_rot[parent] @ offset
            transforms[joint_idx, :3, :3] = global_rot[joint_idx]
            transforms[joint_idx, :3, 3] = global_pos[joint_idx]
        return global_rot, global_pos, transforms

    def _rest_inverse_transforms(self, joints_rest: np.ndarray) -> np.ndarray:
        transforms = np.repeat(np.eye(4)[None, :, :], self.joint_count, axis=0)
        transforms[:, :3, 3] = joints_rest
        return np.linalg.inv(transforms)


@lru_cache(maxsize=4)
def load_smplx_model(model_path: str | Path) -> SmplxNumpyModel:
    return SmplxNumpyModel(Path(model_path).resolve())
