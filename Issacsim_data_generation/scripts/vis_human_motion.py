#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geosim.smplx_numpy import SmplxNumpyModel, load_smplx_model


DEFAULT_MOTION = ROOT / "test_motion/HumanEva/S1/Walking_3_stageii.npz"
DEFAULT_MODEL = ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"
DEFAULT_RECOVERY_POSE_KEYS = (
    ("left_wrist_est_pos", "left_wrist_est_rot"),
    ("right_wrist_est_pos", "right_wrist_est_rot"),
    ("left_ankle_est_pos", "left_ankle_est_rot"),
    ("right_ankle_est_pos", "right_ankle_est_rot"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview an AMASS/SMPL-X human motion with an Open3D realtime viewer.")
    parser.add_argument("motion_name", nargs="?", default="", help="Motion name such as S1_Walking_3_stageii.npz, or a motion path.")
    parser.add_argument("--motion", default="", help="Input .npz motion file with poses/trans. Overrides motion_name.")
    parser.add_argument("--smplx-model", default=str(DEFAULT_MODEL), help="SMPL-X neutral model .npz.")
    parser.add_argument("--playback-fps", type=float, default=0.0, help="Override playback FPS. 0 uses motion FPS.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means play to the end.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier; 0.25 is quarter-speed.")
    parser.add_argument("--skeleton-only", action="store_true", help="Hide SMPL-X mesh and draw only the joint skeleton.")
    parser.add_argument("--no-precompute", action="store_true", help="Compute SMPL-X frames during playback instead of caching first.")
    parser.add_argument("--simple", action="store_true", help="Open immediately and play by computing frames online, GMR-style.")
    parser.add_argument("--pose-track", default="", help="Optional pose-track .npz to overlay. If omitted, the matching wrist_ankle_recovery output is used when present.")
    parser.add_argument("--pose-pos-key", default="", help="Position array key inside --pose-track.")
    parser.add_argument("--pose-rot-key", default="", help="Rotation array key inside --pose-track.")
    parser.add_argument("--pose-axis-length", type=float, default=0.14)
    parser.add_argument("--no-auto-pose-track", action="store_true", help="Do not auto-load the matching outputs/wrist_ankle_recovery file.")
    return parser.parse_args()


def resolve_motion_and_overlay_args(args: argparse.Namespace) -> tuple[Path, str, tuple[tuple[str, str], ...]]:
    motion_path = Path(args.motion) if args.motion else _resolve_motion_name(args.motion_name)
    pose_track = args.pose_track
    if not pose_track and not args.no_auto_pose_track:
        candidate = _default_recovery_path(motion_path)
        if candidate.exists():
            pose_track = str(candidate)
    pose_keys = _resolve_pose_keys(pose_track, args.pose_pos_key, args.pose_rot_key)
    return motion_path, pose_track, pose_keys


def _resolve_motion_name(name: str) -> Path:
    if not name:
        return DEFAULT_MOTION
    raw = Path(name)
    if raw.exists():
        return raw
    candidate = ROOT / name
    if candidate.exists():
        return candidate
    candidate = ROOT / "test_motion/HumanEva" / name
    if candidate.exists():
        return candidate

    stem = raw.name
    if stem.endswith(".npz"):
        stem = stem[:-4]
    if stem.startswith("HumanEva_"):
        stem = stem[len("HumanEva_") :]
    if stem.endswith("_wrist_ankle_recovery"):
        stem = stem[: -len("_wrist_ankle_recovery")]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].startswith("S"):
        subject = parts[0]
        motion_stem = "_".join(parts[1:])
        motion_path = ROOT / "test_motion/HumanEva" / subject / f"{motion_stem}.npz"
        if motion_path.exists():
            return motion_path
    raise FileNotFoundError(
        f"Could not resolve motion name '{name}'. Try a full path or a name like S1_Walking_3_stageii.npz."
    )


def _default_recovery_path(motion_path: Path) -> Path:
    try:
        rel = motion_path.resolve().relative_to((ROOT / "test_motion").resolve())
        safe = "_".join(rel.with_suffix("").parts)
    except ValueError:
        safe = motion_path.with_suffix("").name
    return ROOT / "outputs/wrist_ankle_recovery" / f"{safe}_wrist_ankle_recovery.npz"


def _resolve_pose_keys(track_path: str, pos_key: str, rot_key: str) -> tuple[tuple[str, str], ...]:
    if not track_path:
        return ((pos_key, rot_key),) if pos_key and rot_key else ()
    if pos_key and rot_key:
        return ((pos_key, rot_key),)
    data = np.load(track_path, allow_pickle=False)
    recovery_keys = tuple((pos, rot) for pos, rot in DEFAULT_RECOVERY_POSE_KEYS if pos in data.files and rot in data.files)
    if recovery_keys:
        return recovery_keys
    candidates = (
        ("left_wrist_est_pos", "left_wrist_est_rot"),
        ("wrist_est_pos", "wrist_est_rot"),
        ("left_ankle_est_pos", "left_ankle_est_rot"),
    )
    for candidate_pos, candidate_rot in candidates:
        if candidate_pos in data.files and candidate_rot in data.files:
            return ((pos_key or candidate_pos, rot_key or candidate_rot),)
    if not pos_key or not rot_key:
        raise ValueError(f"Could not infer pose keys from {track_path}; pass --pose-pos-key and --pose-rot-key.")
    return ((pos_key, rot_key),)


class HumanMotion:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Motion file not found: {self.path}")
        data = np.load(self.path, allow_pickle=True)
        if not {"poses", "trans"}.issubset(data.files):
            raise ValueError(f"{self.path} must contain AMASS/SMPL-X arrays 'poses' and 'trans'.")
        self.poses = np.asarray(data["poses"], dtype=float)
        self.trans = np.asarray(data["trans"], dtype=float)
        self.betas = np.asarray(data["betas"], dtype=float) if "betas" in data.files else None
        self.fps = self._read_fps(data)
        if self.poses.ndim != 2 or self.trans.shape != (len(self.poses), 3):
            raise ValueError("Expected poses shape (F, J*3) and trans shape (F, 3).")

    @staticmethod
    def _read_fps(data: np.lib.npyio.NpzFile) -> float:
        for key in ("mocap_frame_rate", "fps"):
            if key in data.files:
                return float(np.asarray(data[key]).reshape(-1)[0])
        return 30.0

    @property
    def frames(self) -> int:
        return int(self.poses.shape[0])


class SmplxFrameCache:
    def __init__(
        self,
        motion: HumanMotion,
        model: SmplxNumpyModel,
        frame_indices: np.ndarray,
        include_vertices: bool,
        precompute: bool,
    ):
        self.motion = motion
        self.model = model
        self.frame_indices = np.asarray(frame_indices, dtype=int)
        self.include_vertices = include_vertices
        self.vertices: np.ndarray | None = None
        self.joints: np.ndarray | None = None
        self.rotations: np.ndarray | None = None
        if precompute:
            self._precompute()

    def _precompute(self) -> None:
        joints = []
        rotations = []
        vertices = [] if self.include_vertices else None
        total = len(self.frame_indices)
        print(f"Precomputing {total} SMPL-X frames for smooth playback...")
        last_report = -1
        for idx, frame_idx in enumerate(self.frame_indices):
            frame = self._forward(int(frame_idx))
            joints.append(frame.joints.astype(np.float32))
            rotations.append(frame.joint_rotations.astype(np.float32))
            if vertices is not None:
                if frame.vertices is None:
                    raise RuntimeError("SMPL-X frame did not include vertices.")
                vertices.append(frame.vertices.astype(np.float32))
            report = int((idx + 1) * 100 / total)
            if report >= last_report + 10 or idx == total - 1:
                print(f"  {idx + 1}/{total} ({report}%)", flush=True)
                last_report = report
        self.joints = np.stack(joints, axis=0)
        self.rotations = np.stack(rotations, axis=0)
        self.vertices = np.stack(vertices, axis=0) if vertices is not None else None

    def get(self, local_idx: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        if self.joints is not None:
            vertices = None if self.vertices is None else self.vertices[local_idx]
            return vertices, self.joints[local_idx], self.rotations[local_idx]
        frame = self._forward(int(self.frame_indices[local_idx]))
        return frame.vertices, frame.joints, frame.joint_rotations

    def _forward(self, frame_idx: int):
        return self.model.forward_frame(
            self.motion.poses[frame_idx],
            self.motion.trans[frame_idx],
            self.motion.betas,
            include_vertices=self.include_vertices,
        )


class PoseTrackOverlay:
    def __init__(self, path: str | Path, pose_keys: tuple[tuple[str, str], ...], axis_length: float):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=False)
        if "frame_indices" not in data.files:
            raise ValueError(f"{self.path} is missing frame_indices.")
        if not pose_keys:
            raise ValueError("At least one pose key pair is required.")
        self.frame_indices = np.asarray(data["frame_indices"], dtype=int)
        self.pose_keys = pose_keys
        positions = []
        rotations = []
        for pos_key, rot_key in pose_keys:
            if pos_key not in data.files:
                raise ValueError(f"{self.path} is missing {pos_key}.")
            if rot_key not in data.files:
                raise ValueError(f"{self.path} is missing {rot_key}.")
            pos = np.asarray(data[pos_key], dtype=float)
            rot = np.asarray(data[rot_key], dtype=float)
            if pos.shape != (len(self.frame_indices), 3):
                raise ValueError(f"{pos_key} must have shape (F, 3).")
            if rot.shape != (len(self.frame_indices), 3, 3):
                raise ValueError(f"{rot_key} must have shape (F, 3, 3).")
            positions.append(pos)
            rotations.append(rot)
        self.positions = np.stack(positions, axis=1)
        self.rotations = np.stack(rotations, axis=1)
        self.axis_length = float(axis_length)
        self.valid = np.isfinite(self.positions).all(axis=2) & np.isfinite(self.rotations).all(axis=(2, 3))

    def nearest_index(self, motion_frame_idx: int) -> int | None:
        if len(self.frame_indices) == 0:
            return None
        insert = int(np.searchsorted(self.frame_indices, int(motion_frame_idx)))
        candidates = []
        if insert < len(self.frame_indices):
            candidates.append(insert)
        if insert > 0:
            candidates.append(insert - 1)
        if not candidates:
            return None
        return min(candidates, key=lambda idx: abs(int(self.frame_indices[idx]) - int(motion_frame_idx)))

    def poses_for_frame(self, motion_frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        idx = self.nearest_index(motion_frame_idx)
        if idx is None:
            return None
        return self.positions[idx], self.rotations[idx], self.valid[idx]

    def make_axes(self, o3d, motion_frame_idx: int):
        line_set = o3d.geometry.LineSet()
        axis_count = len(self.pose_keys) * 3
        line_set.lines = o3d.utility.Vector2iVector(np.asarray([(2 * i, 2 * i + 1) for i in range(axis_count)], dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([[1.0, 0.0, 0.0], [0.0, 0.75, 0.0], [0.0, 0.15, 1.0]], dtype=float), (len(self.pose_keys), 1)))
        self.update_axes(line_set, o3d, motion_frame_idx)
        return line_set

    def update_axes(self, line_set, o3d, motion_frame_idx: int) -> None:
        poses = self.poses_for_frame(motion_frame_idx)
        points = []
        if poses is not None:
            positions, rotations, valid = poses
            for pose_idx in range(len(self.pose_keys)):
                pos = positions[pose_idx]
                rot = rotations[pose_idx]
                if not valid[pose_idx]:
                    pos = np.zeros(3, dtype=float)
                    rot = np.zeros((3, 3), dtype=float)
                for axis_idx in range(3):
                    points.append(pos)
                    points.append(pos + rot[:, axis_idx] * self.axis_length)
        if not points:
            points = np.zeros((len(self.pose_keys) * 6, 3), dtype=float)
        else:
            points = np.asarray(points, dtype=float)
        line_set.points = o3d.utility.Vector3dVector(points)


class Open3DHumanMotionViewer:
    def __init__(
        self,
        motion: HumanMotion,
        model: SmplxNumpyModel,
        frame_indices: np.ndarray,
        playback_fps: float,
        speed: float,
        width: int,
        height: int,
        skeleton_only: bool,
        precompute: bool,
        pose_overlay: PoseTrackOverlay | None = None,
    ):
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("Open3D is required for this viewer. Install it with: python -m pip install open3d") from exc

        self.o3d = o3d
        self.motion = motion
        self.model = model
        self.frame_indices = np.asarray(frame_indices, dtype=int)
        self.playback_fps = float(playback_fps)
        self.speed = max(float(speed), 1e-6)
        self.frame_dt = 1.0 / max(self.playback_fps * self.speed, 1e-6)
        self.skeleton_only = skeleton_only
        self.pose_overlay = pose_overlay
        self.cache = SmplxFrameCache(
            motion=motion,
            model=model,
            frame_indices=self.frame_indices,
            include_vertices=not skeleton_only,
            precompute=precompute,
        )

        self.cursor = 0
        self.playing = True
        self.last_step_time = time.perf_counter()
        self.running = True

        first_vertices, first_joints, _first_rotations = self.cache.get(0)
        self.mesh = self._make_mesh(first_vertices) if first_vertices is not None else None
        self.skeleton = self._make_skeleton(first_joints)
        self.grid = self._make_grid(first_joints)
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=first_joints[0])
        self.pose_axes = None if pose_overlay is None else pose_overlay.make_axes(o3d, int(self.frame_indices[0]))

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name=f"SMPL-X human motion: {motion.path.name}",
            width=int(width),
            height=int(height),
        )
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.96, 0.96, 0.93])
        render_option.mesh_show_back_face = True
        render_option.line_width = 2.0

        if self.mesh is not None:
            self.vis.add_geometry(self.mesh)
        self.vis.add_geometry(self.skeleton)
        self.vis.add_geometry(self.grid)
        self.vis.add_geometry(self.axes)
        if self.pose_axes is not None:
            self.vis.add_geometry(self.pose_axes)
        self._set_camera(first_joints)
        self._register_keys()

    def run(self) -> None:
        print("Controls: mouse rotates/pans/zooms | Space play/pause | Left/Right step | R reset camera | Q/Esc quit")
        while self.running and self.vis.poll_events():
            now = time.perf_counter()
            if self.playing and now - self.last_step_time >= self.frame_dt:
                elapsed_frames = max(1, int((now - self.last_step_time) / self.frame_dt))
                self.cursor = int((self.cursor + elapsed_frames) % len(self.frame_indices))
                self.last_step_time += elapsed_frames * self.frame_dt
                self._update_geometry()
            self.vis.update_renderer()
            time.sleep(0.001)
        self.vis.destroy_window()

    def _register_keys(self) -> None:
        self.vis.register_key_callback(ord(" "), self._toggle_play)
        self.vis.register_key_callback(262, self._step_forward)
        self.vis.register_key_callback(263, self._step_back)
        self.vis.register_key_callback(ord("R"), self._reset_camera)
        self.vis.register_key_callback(ord("Q"), self._quit)
        self.vis.register_key_callback(256, self._quit)

    def _toggle_play(self, _vis) -> bool:
        self.playing = not self.playing
        self.last_step_time = time.perf_counter()
        return False

    def _step_forward(self, _vis) -> bool:
        self.playing = False
        self.cursor = int((self.cursor + 1) % len(self.frame_indices))
        self._update_geometry()
        return False

    def _step_back(self, _vis) -> bool:
        self.playing = False
        self.cursor = int((self.cursor - 1) % len(self.frame_indices))
        self._update_geometry()
        return False

    def _reset_camera(self, _vis) -> bool:
        _, joints, _ = self.cache.get(self.cursor)
        self._set_camera(joints)
        return False

    def _quit(self, _vis) -> bool:
        self.running = False
        return False

    def _make_mesh(self, vertices: np.ndarray | None):
        if vertices is None:
            return None
        mesh = self.o3d.geometry.TriangleMesh()
        mesh.vertices = self.o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
        mesh.triangles = self.o3d.utility.Vector3iVector(self.model.faces.astype(np.int32))
        mesh.paint_uniform_color([0.78, 0.76, 0.70])
        mesh.compute_vertex_normals()
        return mesh

    def _make_skeleton(self, joints: np.ndarray):
        lines = [(int(parent), child) for child, parent in enumerate(self.model.parents) if parent >= 0]
        skeleton = self.o3d.geometry.LineSet()
        skeleton.points = self.o3d.utility.Vector3dVector(np.asarray(joints, dtype=np.float64))
        skeleton.lines = self.o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        skeleton.colors = self.o3d.utility.Vector3dVector(np.tile(np.array([[0.05, 0.25, 0.55]]), (len(lines), 1)))
        return skeleton

    def _make_grid(self, joints: np.ndarray):
        center = joints.mean(axis=0)
        ground_z = float(np.nanmin(joints[:, 2]))
        extent = 1.8
        line_points = []
        lines = []
        for offset in np.linspace(-extent, extent, 13):
            base = len(line_points)
            line_points.extend(
                [
                    [center[0] - extent, center[1] + offset, ground_z],
                    [center[0] + extent, center[1] + offset, ground_z],
                    [center[0] + offset, center[1] - extent, ground_z],
                    [center[0] + offset, center[1] + extent, ground_z],
                ]
            )
            lines.extend([(base, base + 1), (base + 2, base + 3)])
        grid = self.o3d.geometry.LineSet()
        grid.points = self.o3d.utility.Vector3dVector(np.asarray(line_points, dtype=np.float64))
        grid.lines = self.o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        grid.colors = self.o3d.utility.Vector3dVector(np.tile(np.array([[0.72, 0.72, 0.72]]), (len(lines), 1)))
        return grid

    def _update_geometry(self) -> None:
        vertices, joints, _rotations = self.cache.get(self.cursor)
        if self.mesh is not None and vertices is not None:
            self.mesh.vertices = self.o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
            self.mesh.compute_vertex_normals()
            self.vis.update_geometry(self.mesh)
        self.skeleton.points = self.o3d.utility.Vector3dVector(np.asarray(joints, dtype=np.float64))
        self.vis.update_geometry(self.skeleton)
        self.axes.translate(joints[0] - np.asarray(self.axes.get_center()), relative=True)
        self.vis.update_geometry(self.axes)
        if self.pose_overlay is not None and self.pose_axes is not None:
            self.pose_overlay.update_axes(self.pose_axes, self.o3d, int(self.frame_indices[self.cursor]))
            self.vis.update_geometry(self.pose_axes)

    def _set_camera(self, joints: np.ndarray) -> None:
        ctr = self.vis.get_view_control()
        center = joints.mean(axis=0)
        ctr.set_lookat(center)
        ctr.set_front([0.8, -1.2, 0.55])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.65)

def _frame_indices(motion: HumanMotion, start_frame: int, max_frames: int) -> np.ndarray:
    start = int(np.clip(start_frame, 0, motion.frames - 1))
    stop = motion.frames if max_frames <= 0 else min(motion.frames, start + max_frames)
    return np.arange(start, max(start + 1, stop), dtype=int)


def run_simple_viewer(
    motion: HumanMotion,
    model: SmplxNumpyModel,
    frame_indices: np.ndarray,
    playback_fps: float,
    speed: float,
    width: int,
    height: int,
    pose_overlay: PoseTrackOverlay | None = None,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for this viewer. Install it with: python -m pip install open3d") from exc

    frame_dt = 1.0 / max(float(playback_fps) * max(float(speed), 1e-6), 1e-6)
    first_idx = int(frame_indices[0])
    first_frame = model.forward_frame(motion.poses[first_idx], motion.trans[first_idx], motion.betas, include_vertices=True)
    if first_frame.vertices is None:
        raise RuntimeError("SMPL-X frame did not include vertices.")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(first_frame.vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(model.faces.astype(np.int32))
    mesh.paint_uniform_color([0.78, 0.76, 0.70])
    mesh.compute_vertex_normals()

    grid = _make_simple_grid(o3d, first_frame.joints)
    pose_axes = None if pose_overlay is None else pose_overlay.make_axes(o3d, first_idx)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"SMPL-X human motion simple: {motion.path.name}",
        width=int(width),
        height=int(height),
    )
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.96, 0.96, 0.93])
    render_option.mesh_show_back_face = True
    vis.add_geometry(mesh)
    vis.add_geometry(grid)
    if pose_axes is not None:
        vis.add_geometry(pose_axes)
    ctr = vis.get_view_control()
    ctr.set_lookat(first_frame.joints.mean(axis=0))
    ctr.set_front([0.8, -1.2, 0.55])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.65)

    playing = True

    def toggle_play(_vis):
        nonlocal playing, last_step_time
        playing = not playing
        last_step_time = time.perf_counter()
        return False

    vis.register_key_callback(ord(" "), toggle_play)

    print("Simple mode: online SMPL-X forward + Open3D mesh update. Space pauses/resumes; close the window to stop.")
    cursor = 0
    last_step_time = time.perf_counter()
    while vis.poll_events():
        now = time.perf_counter()
        if playing and now - last_step_time >= frame_dt:
            elapsed_frames = max(1, int((now - last_step_time) / frame_dt))
            cursor = int((cursor + elapsed_frames) % len(frame_indices))
            last_step_time += elapsed_frames * frame_dt
            frame_idx = int(frame_indices[cursor])
            frame = model.forward_frame(motion.poses[frame_idx], motion.trans[frame_idx], motion.betas, include_vertices=True)
            if frame.vertices is not None:
                mesh.vertices = o3d.utility.Vector3dVector(np.asarray(frame.vertices, dtype=np.float64))
                mesh.compute_vertex_normals()
                vis.update_geometry(mesh)
            if pose_overlay is not None and pose_axes is not None:
                pose_overlay.update_axes(pose_axes, o3d, frame_idx)
                vis.update_geometry(pose_axes)
        vis.update_renderer()
        time.sleep(0.001)
    vis.destroy_window()


def _make_simple_grid(o3d, joints: np.ndarray):
    center = joints.mean(axis=0)
    ground_z = float(np.nanmin(joints[:, 2]))
    extent = 1.8
    points = []
    lines = []
    for offset in np.linspace(-extent, extent, 13):
        base = len(points)
        points.extend(
            [
                [center[0] - extent, center[1] + offset, ground_z],
                [center[0] + extent, center[1] + offset, ground_z],
                [center[0] + offset, center[1] - extent, ground_z],
                [center[0] + offset, center[1] + extent, ground_z],
            ]
        )
        lines.extend([(base, base + 1), (base + 2, base + 3)])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.72, 0.72, 0.72]]), (len(lines), 1)))
    return grid


def main() -> int:
    args = parse_args()
    motion_path, pose_track, pose_keys = resolve_motion_and_overlay_args(args)
    print(f"motion: {motion_path}")
    if pose_track:
        key_text = ", ".join(f"{pos}/{rot}" for pos, rot in pose_keys)
        print(f"pose track: {pose_track} ({key_text})")
    motion = HumanMotion(motion_path)
    model = load_smplx_model(args.smplx_model)
    playback_fps = float(args.playback_fps) if args.playback_fps > 0 else motion.fps
    frames = _frame_indices(motion, args.start_frame, args.max_frames)
    pose_overlay = (
        PoseTrackOverlay(pose_track, pose_keys, args.pose_axis_length)
        if pose_track
        else None
    )
    if args.simple:
        run_simple_viewer(
            motion=motion,
            model=model,
            frame_indices=frames,
            playback_fps=playback_fps,
            speed=args.speed,
            width=args.width,
            height=args.height,
            pose_overlay=pose_overlay,
        )
        return 0
    viewer = Open3DHumanMotionViewer(
        motion=motion,
        model=model,
        frame_indices=frames,
        playback_fps=playback_fps,
        speed=args.speed,
        width=args.width,
        height=args.height,
        skeleton_only=args.skeleton_only,
        precompute=not args.no_precompute,
        pose_overlay=pose_overlay,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
