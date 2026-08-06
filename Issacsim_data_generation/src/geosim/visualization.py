from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from geosim.camera import make_default_camera_rig
from geosim.config import SimulationConfig
from geosim.linalg import rotx, rotz
from geosim.motion import MotionSequence
from geosim.smplx_numpy import SmplxNumpyModel
from geosim.tag_rig import make_wrist_tag_rig


@dataclass(frozen=True)
class ViewProjector:
    center: np.ndarray
    view_rot: np.ndarray
    scale: float
    width: int
    height: int

    def project(self, points_world: np.ndarray) -> np.ndarray:
        view = (np.asarray(points_world, dtype=float) - self.center) @ self.view_rot.T
        x = self.width * 0.5 + view[:, 0] * self.scale
        y = self.height * 0.55 - view[:, 2] * self.scale
        return np.column_stack([x, y, view[:, 1]])


def render_motion_video(
    motion: MotionSequence,
    config: SimulationConfig,
    smplx_model: SmplxNumpyModel,
    output_path: str | Path,
    output_fps: float = 10.0,
    width: int = 1280,
    height: int = 720,
    max_frames: int = 0,
) -> Path:
    if motion.source_path is None:
        raise ValueError("SMPL-X visualization requires a motion loaded from an AMASS source file.")
    source = np.load(motion.source_path, allow_pickle=True)
    if not {"poses", "trans"}.issubset(source.files):
        raise ValueError(f"{motion.source_path} does not contain AMASS poses/trans.")

    poses = np.asarray(source["poses"], dtype=float)
    trans = np.asarray(source["trans"], dtype=float)
    betas = np.asarray(source["betas"], dtype=float) if "betas" in source.files else motion.betas
    frame_count = motion.frames if max_frames <= 0 else min(motion.frames, max_frames)
    stride = max(1, int(round(motion.fps / output_fps)))
    frame_indices = np.arange(0, frame_count, stride, dtype=int)
    if len(frame_indices) == 0:
        frame_indices = np.array([0], dtype=int)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    bound_points = _sample_body_bounds(smplx_model, poses, trans, betas, frame_count)
    projector = _make_projector(motion, config, width, height, frame_count, bound_points)
    cameras = make_default_camera_rig(config.camera_rig)
    tag_rig = make_wrist_tag_rig(config.tag_rig)
    faces = smplx_model.faces[::8]
    parents = smplx_model.parents

    for out_idx, frame_idx in enumerate(frame_indices):
        canvas = np.full((height, width, 3), 248, dtype=np.uint8)
        _draw_grid(canvas, projector)
        _draw_path(canvas, projector, motion.head_pos[: frame_idx + 1], (80, 150, 245))
        _draw_path(canvas, projector, motion.left_wrist_pos[: frame_idx + 1], (230, 100, 45))

        frame = smplx_model.forward_frame(poses[frame_idx], trans[frame_idx], betas, include_vertices=True)
        assert frame.vertices is not None
        _draw_mesh(canvas, projector, frame.vertices, faces)
        _draw_skeleton(canvas, projector, frame.joints, parents)
        _draw_cameras(canvas, projector, cameras, motion.head_pos[frame_idx], motion.head_rot[frame_idx])
        _draw_tags(canvas, projector, tag_rig, motion.left_wrist_pos[frame_idx], motion.left_wrist_rot[frame_idx])

        label = f"{Path(motion.name).name}  frame {frame_idx}/{frame_count - 1}  {output_fps:.0f} Hz"
        cv2.putText(canvas, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, "orange: head path   blue: wrist path   red: head-mounted fisheye rig", (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (55, 55, 55), 1, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    return output_path


def _sample_body_bounds(
    smplx_model: SmplxNumpyModel,
    poses: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray | None,
    frame_count: int,
) -> np.ndarray:
    sample_count = min(frame_count, 80)
    sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)
    joints = []
    for idx in sample_indices:
        frame = smplx_model.forward_frame(poses[idx], trans[idx], betas, include_vertices=False)
        joints.append(frame.joints)
    return np.vstack(joints)


def _make_projector(
    motion: MotionSequence,
    config: SimulationConfig,
    width: int,
    height: int,
    frame_count: int,
    extra_points: np.ndarray | None = None,
) -> ViewProjector:
    cameras = make_default_camera_rig(config.camera_rig)
    cam_points = []
    sample_count = min(frame_count, 200)
    sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)
    for idx in sample_indices:
        for camera in cameras:
            cam_pos, _ = camera.world_pose(motion.head_pos[idx], motion.head_rot[idx])
            cam_points.append(cam_pos)
    base_points = [motion.head_pos[:frame_count], motion.left_wrist_pos[:frame_count], np.asarray(cam_points)]
    if extra_points is not None and len(extra_points):
        base_points.append(extra_points)
    points = np.vstack(base_points)
    center = points.mean(axis=0)
    view_rot = rotx(math.radians(62.0)) @ rotz(math.radians(-38.0))
    view = (points - center) @ view_rot.T
    span_x = max(float(np.ptp(view[:, 0])), 1e-3)
    span_z = max(float(np.ptp(view[:, 2])), 1e-3)
    scale = 0.82 * min((width - 140) / span_x, (height - 140) / span_z)
    return ViewProjector(center=center, view_rot=view_rot, scale=scale, width=width, height=height)


def _draw_grid(canvas: np.ndarray, projector: ViewProjector) -> None:
    extent = 2.0
    lines = []
    for v in np.linspace(-extent, extent, 9):
        lines.append((np.array([[-extent, v, 0.0], [extent, v, 0.0]]), (222, 222, 222)))
        lines.append((np.array([[v, -extent, 0.0], [v, extent, 0.0]]), (222, 222, 222)))
    for points, color in lines:
        pts = projector.project(points)[:, :2].astype(int)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), color, 1, cv2.LINE_AA)


def _draw_path(canvas: np.ndarray, projector: ViewProjector, points: np.ndarray, color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    projected = projector.project(points)[:, :2].astype(int)
    for start, end in zip(projected[:-1], projected[1:]):
        cv2.line(canvas, tuple(start), tuple(end), color, 2, cv2.LINE_AA)


def _draw_mesh(canvas: np.ndarray, projector: ViewProjector, vertices: np.ndarray, faces: np.ndarray) -> None:
    projected = projector.project(vertices)
    tri_depth = projected[faces, 2].mean(axis=1)
    order = np.argsort(tri_depth)
    light = np.array([0.25, -0.45, 0.86])
    light = light / np.linalg.norm(light)
    for face_idx in order:
        face = faces[face_idx]
        pts = projected[face, :2]
        if np.any(pts[:, 0] < -80) or np.any(pts[:, 0] > canvas.shape[1] + 80):
            continue
        if np.any(pts[:, 1] < -80) or np.any(pts[:, 1] > canvas.shape[0] + 80):
            continue
        v0, v1, v2 = vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        shade = 0.65 if norm < 1e-12 else 0.52 + 0.28 * max(0.0, float(np.dot(normal / norm, light)))
        gray = int(np.clip(190 * shade + 45, 80, 215))
        cv2.fillConvexPoly(canvas, np.round(pts).astype(np.int32), (gray, gray, gray), cv2.LINE_AA)


def _draw_skeleton(canvas: np.ndarray, projector: ViewProjector, joints: np.ndarray, parents: np.ndarray) -> None:
    projected = projector.project(joints)[:, :2].astype(int)
    for joint_idx, parent in enumerate(parents):
        if parent < 0:
            continue
        cv2.line(canvas, tuple(projected[joint_idx]), tuple(projected[parent]), (70, 70, 70), 1, cv2.LINE_AA)


def _draw_cameras(canvas: np.ndarray, projector: ViewProjector, cameras, head_pos: np.ndarray, head_rot: np.ndarray) -> None:
    for camera in cameras:
        cam_pos, cam_rot = camera.world_pose(head_pos, head_rot)
        direction = cam_rot @ np.array([0.0, 0.0, 1.0])
        pts = projector.project(np.vstack([cam_pos, cam_pos + 0.16 * direction]))[:, :2].astype(int)
        cv2.circle(canvas, tuple(pts[0]), 6, (25, 40, 220), -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, tuple(pts[0]), tuple(pts[1]), (25, 40, 220), 2, cv2.LINE_AA, tipLength=0.32)


def _draw_tags(canvas: np.ndarray, projector: ViewProjector, tag_rig, wrist_pos: np.ndarray, wrist_rot: np.ndarray) -> None:
    points = tag_rig.world_points(wrist_pos, wrist_rot)
    for tag_name, color in (("tag0", (40, 190, 80)), ("tag1", (40, 165, 210))):
        ordered = np.stack([points[f"{tag_name}_c{i}"] for i in range(4)], axis=0)
        projected = projector.project(ordered)[:, :2]
        cv2.fillConvexPoly(canvas, np.round(projected).astype(np.int32), color, cv2.LINE_AA)
        cv2.polylines(canvas, [np.round(projected).astype(np.int32)], True, (20, 70, 20), 2, cv2.LINE_AA)
