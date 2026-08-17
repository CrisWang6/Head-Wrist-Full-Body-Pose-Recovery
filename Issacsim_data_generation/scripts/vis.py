#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import tkinter as tk

import numpy as np


DEFAULT_TRACK = Path("outputs/camera_views_realistic/S1_walk_head_front_left_right_pose_tracks.npz")


def parse_pose_tracks_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively visualize truth and recovered wrist/tag pose tracks.")
    parser.add_argument("track_file", nargs="?", default=str(DEFAULT_TRACK))
    parser.add_argument("--axis-step", type=int, default=30, help="Draw one pose frame every N output frames.")
    parser.add_argument("--axis-length", type=float, default=0.08)
    parser.add_argument("--show-tags", action="store_true", help="Also draw tag center trajectories.")
    return parser.parse_args()


class PoseTrackViewer:
    def __init__(self, root: tk.Tk, data: np.lib.npyio.NpzFile, axis_step: int, axis_length: float, show_tags: bool):
        self.root = root
        self.data = data
        self.axis_step = max(1, axis_step)
        self.axis_length = float(axis_length)
        self.show_tags = show_tags
        self.width = 1280
        self.height = 800
        self.azimuth = math.radians(-35.0)
        self.elevation = math.radians(28.0)
        self.zoom = 1.0
        self.pan = np.array([0.0, 0.0], dtype=float)
        self.last_mouse: tuple[int, int] | None = None
        self.drag_mode = "rotate"

        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="#f7f7f4", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._start_rotate)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._drag)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(1.08))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(1.0 / 1.08))

        self.center, self.base_scale = self._scene_bounds()
        root.title("Pose tracks: truth vs recovered")
        self.draw()

    def _scene_bounds(self) -> tuple[np.ndarray, float]:
        points = [self.data["wrist_truth_pos"], self.data["wrist_est_pos"]]
        if self.show_tags and "tag_truth_pos" in self.data:
            points.extend([self.data["tag_truth_pos"].reshape(-1, 3), self.data["tag_est_pos"].reshape(-1, 3)])
        valid = np.vstack([arr.reshape(-1, 3) for arr in points])
        valid = valid[np.isfinite(valid).all(axis=1)]
        if len(valid) == 0:
            return np.zeros(3), 200.0
        center = valid.mean(axis=0)
        span = np.ptp(valid, axis=0)
        scale = 0.78 * min(self.width, self.height) / max(float(np.max(span)), 1e-3)
        return center, scale

    def _on_resize(self, event: tk.Event) -> None:
        self.width = max(1, event.width)
        self.height = max(1, event.height)
        self.center, self.base_scale = self._scene_bounds()
        self.draw()

    def _start_rotate(self, event: tk.Event) -> None:
        self.drag_mode = "rotate"
        self.last_mouse = (event.x, event.y)

    def _start_pan(self, event: tk.Event) -> None:
        self.drag_mode = "pan"
        self.last_mouse = (event.x, event.y)

    def _drag(self, event: tk.Event) -> None:
        if self.last_mouse is None:
            self.last_mouse = (event.x, event.y)
            return
        dx = event.x - self.last_mouse[0]
        dy = event.y - self.last_mouse[1]
        self.last_mouse = (event.x, event.y)
        if self.drag_mode == "pan":
            self.pan += np.array([dx, dy], dtype=float)
        else:
            self.azimuth += dx * 0.008
            self.elevation = float(np.clip(self.elevation + dy * 0.006, math.radians(-85.0), math.radians(85.0)))
        self.draw()

    def _wheel(self, event: tk.Event) -> None:
        self._zoom_at(1.08 if event.delta > 0 else 1.0 / 1.08)

    def _zoom_at(self, factor: float) -> None:
        self.zoom = float(np.clip(self.zoom * factor, 0.08, 30.0))
        self.draw()

    def _view_coords(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float) - self.center
        ca = math.cos(self.azimuth)
        sa = math.sin(self.azimuth)
        ce = math.cos(self.elevation)
        se = math.sin(self.elevation)
        x0 = ca * pts[:, 0] - sa * pts[:, 1]
        depth = sa * pts[:, 0] + ca * pts[:, 1]
        y0 = -se * depth + ce * pts[:, 2]
        return np.column_stack([x0, y0, depth])

    def _project(self, points: np.ndarray) -> np.ndarray:
        view = self._view_coords(points)
        scale = self.base_scale * self.zoom
        x = self.width * 0.5 + self.pan[0] + view[:, 0] * scale
        y = self.height * 0.53 + self.pan[1] - view[:, 1] * scale
        return np.column_stack([x, y, view[:, 2]])

    def draw(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        if self.show_tags and "tag_names" in self.data:
            self._draw_tag_tracks()
        self._draw_track(self.data["wrist_truth_pos"], "#1f77b4", width=3)
        self._draw_track(self.data["wrist_est_pos"], "#d62728", width=3)
        self._draw_pose_axes(self.data["wrist_truth_pos"], self.data["wrist_truth_rot"], truth=True)
        self._draw_pose_axes(self.data["wrist_est_pos"], self.data["wrist_est_rot"], truth=False)
        self._draw_overlay_text()

    def _draw_grid(self) -> None:
        truth = self.data["wrist_truth_pos"]
        valid = truth[np.isfinite(truth).all(axis=1)]
        if len(valid) == 0:
            return
        z = float(np.nanmin(valid[:, 2]))
        center_xy = valid[:, :2].mean(axis=0)
        extent = max(float(np.ptp(valid[:, 0])), float(np.ptp(valid[:, 1])), 1.0) * 0.7
        lines = []
        for offset in np.linspace(-extent, extent, 9):
            lines.append(np.array([[center_xy[0] - extent, center_xy[1] + offset, z], [center_xy[0] + extent, center_xy[1] + offset, z]]))
            lines.append(np.array([[center_xy[0] + offset, center_xy[1] - extent, z], [center_xy[0] + offset, center_xy[1] + extent, z]]))
        for line in lines:
            pts = self._project(line)
            self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1], fill="#dddddd")

    def _draw_track(self, points: np.ndarray, color: str, width: int = 2) -> None:
        valid = np.isfinite(points).all(axis=1)
        start = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = idx
            if start is not None and (not is_valid or idx == len(valid) - 1):
                end = idx + 1 if is_valid and idx == len(valid) - 1 else idx
                if end - start >= 2:
                    projected = self._project(points[start:end])
                    coords = []
                    for point in projected:
                        coords.extend([point[0], point[1]])
                    self.canvas.create_line(*coords, fill=color, width=width, smooth=True)
                start = None

    def _draw_tag_tracks(self) -> None:
        truth = self.data["tag_truth_pos"]
        estimate = self.data["tag_est_pos"]
        colors = [("#78aadd", "#ff9999"), ("#87c98b", "#ffbf66")]
        for tag_idx in range(truth.shape[1]):
            self._draw_track(truth[:, tag_idx, :], colors[tag_idx % len(colors)][0], width=1)
            self._draw_track(estimate[:, tag_idx, :], colors[tag_idx % len(colors)][1], width=1)

    def _draw_pose_axes(self, positions: np.ndarray, rotations: np.ndarray, truth: bool) -> None:
        dash = () if truth else (4, 3)
        width = 2 if truth else 1
        axis_colors = ("#cc3333", "#2a9d55", "#3366cc")
        for frame_idx in range(0, len(positions), self.axis_step):
            pos = positions[frame_idx]
            rot = rotations[frame_idx]
            if not (np.isfinite(pos).all() and np.isfinite(rot).all()):
                continue
            for axis_idx, color in enumerate(axis_colors):
                segment = np.vstack([pos, pos + rot[:, axis_idx] * self.axis_length])
                pts = self._project(segment)
                self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1], fill=color, width=width, dash=dash)

    def _draw_overlay_text(self) -> None:
        frame_count = len(self.data["frame_indices"])
        fps = float(self.data["output_fps"][0]) if np.asarray(self.data["output_fps"]).ndim else float(self.data["output_fps"])
        lines = [
            f"frames: {frame_count}   fps: {fps:.1f}",
            "blue: truth wrist trajectory   red: recovered wrist trajectory",
            "solid axes: truth pose   dashed axes: recovered pose",
            "mouse: left drag rotate, right drag pan, wheel zoom",
        ]
        for idx, text in enumerate(lines):
            self.canvas.create_text(18, 20 + idx * 22, anchor="w", text=text, fill="#222222", font=("TkDefaultFont", 11, "bold" if idx == 0 else "normal"))


def pose_tracks_main() -> int:
    args = parse_pose_tracks_args()
    data = np.load(args.track_file, allow_pickle=False)
    root = tk.Tk()
    PoseTrackViewer(root, data, axis_step=args.axis_step, axis_length=args.axis_length, show_tags=args.show_tags)
    root.mainloop()
    return 0


# Foot trajectory viewer
import argparse
import math
from pathlib import Path
import tkinter as tk

import numpy as np


DEFAULT_TRACK = Path("outputs/four_view_foot_pipeline/S1_walk_four_view_foot_tracks_realistic_imu.npz")


def parse_foot_tracks_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively visualize truth and four-view estimated foot tracks.")
    parser.add_argument("track_file", nargs="?", default=str(DEFAULT_TRACK))
    parser.add_argument("--axis-step", type=int, default=30)
    parser.add_argument("--axis-length", type=float, default=0.08)
    parser.add_argument("--show-landmarks", action="store_true")
    return parser.parse_args()


class FootTrackViewer:
    def __init__(self, root: tk.Tk, data: np.lib.npyio.NpzFile, axis_step: int, axis_length: float, show_landmarks: bool):
        self.root = root
        self.data = data
        self.axis_step = max(1, axis_step)
        self.axis_length = float(axis_length)
        self.show_landmarks = show_landmarks
        self.width = 1280
        self.height = 820
        self.azimuth = math.radians(-35.0)
        self.elevation = math.radians(25.0)
        self.zoom = 1.0
        self.pan = np.array([0.0, 0.0], dtype=float)
        self.last_mouse: tuple[int, int] | None = None
        self.drag_mode = "rotate"

        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="#f7f7f4", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._start_rotate)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._drag)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda _event: self._zoom_at(1.08))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom_at(1.0 / 1.08))
        root.bind("r", lambda _event: self._reset())

        self.center, self.base_scale = self._scene_bounds()
        root.title("Four-view foot tracks: truth vs estimate")
        self.draw()

    def _scene_bounds(self) -> tuple[np.ndarray, float]:
        points = [self.data["foot_truth_pos"].reshape(-1, 3), self.data["foot_est_pos"].reshape(-1, 3)]
        if self.show_landmarks:
            points.extend([self.data["foot_truth_landmarks"].reshape(-1, 3), self.data["foot_est_landmarks"].reshape(-1, 3)])
        valid = np.vstack(points)
        valid = valid[np.isfinite(valid).all(axis=1)]
        if len(valid) == 0:
            return np.zeros(3), 200.0
        center = valid.mean(axis=0)
        span = np.ptp(valid, axis=0)
        scale = 0.78 * min(self.width, self.height) / max(float(np.max(span)), 1e-3)
        return center, scale

    def _on_resize(self, event: tk.Event) -> None:
        self.width = max(1, event.width)
        self.height = max(1, event.height)
        self.center, self.base_scale = self._scene_bounds()
        self.draw()

    def _start_rotate(self, event: tk.Event) -> None:
        self.drag_mode = "rotate"
        self.last_mouse = (event.x, event.y)

    def _start_pan(self, event: tk.Event) -> None:
        self.drag_mode = "pan"
        self.last_mouse = (event.x, event.y)

    def _drag(self, event: tk.Event) -> None:
        if self.last_mouse is None:
            self.last_mouse = (event.x, event.y)
            return
        dx = event.x - self.last_mouse[0]
        dy = event.y - self.last_mouse[1]
        self.last_mouse = (event.x, event.y)
        if self.drag_mode == "pan":
            self.pan += np.array([dx, dy], dtype=float)
        else:
            self.azimuth += dx * 0.008
            self.elevation = float(np.clip(self.elevation + dy * 0.006, math.radians(-85.0), math.radians(85.0)))
        self.draw()

    def _wheel(self, event: tk.Event) -> None:
        self._zoom_at(1.08 if event.delta > 0 else 1.0 / 1.08)

    def _zoom_at(self, factor: float) -> None:
        self.zoom = float(np.clip(self.zoom * factor, 0.08, 30.0))
        self.draw()

    def _reset(self) -> None:
        self.azimuth = math.radians(-35.0)
        self.elevation = math.radians(25.0)
        self.zoom = 1.0
        self.pan[:] = 0.0
        self.draw()

    def _project(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3) - self.center
        ca = math.cos(self.azimuth)
        sa = math.sin(self.azimuth)
        ce = math.cos(self.elevation)
        se = math.sin(self.elevation)
        x0 = ca * pts[:, 0] - sa * pts[:, 1]
        depth = sa * pts[:, 0] + ca * pts[:, 1]
        y0 = -se * depth + ce * pts[:, 2]
        scale = self.base_scale * self.zoom
        x = self.width * 0.5 + self.pan[0] + x0 * scale
        y = self.height * 0.55 + self.pan[1] - y0 * scale
        return np.column_stack([x, y, depth])

    def draw(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        truth_colors = ("#1f77b4", "#2ca02c")
        est_colors = ("#d62728", "#ff7f0e")
        foot_names = [str(x) for x in self.data["foot_names"]]
        for foot_idx, foot_name in enumerate(foot_names):
            self._draw_track(self.data["foot_truth_pos"][:, foot_idx], truth_colors[foot_idx % 2], width=3)
            self._draw_track(self.data["foot_est_pos"][:, foot_idx], est_colors[foot_idx % 2], width=3)
            self._draw_axes(self.data["foot_truth_pos"][:, foot_idx], self.data["foot_truth_rot"][:, foot_idx], truth=True)
            self._draw_axes(self.data["foot_est_pos"][:, foot_idx], self.data["foot_est_rot"][:, foot_idx], truth=False)
            if self.show_landmarks:
                self._draw_landmarks(self.data["foot_truth_landmarks"][:, foot_idx], truth_colors[foot_idx % 2])
                self._draw_landmarks(self.data["foot_est_landmarks"][:, foot_idx], est_colors[foot_idx % 2])
        self._draw_overlay()

    def _draw_grid(self) -> None:
        truth = self.data["foot_truth_pos"].reshape(-1, 3)
        valid = truth[np.isfinite(truth).all(axis=1)]
        if len(valid) == 0:
            return
        z = float(np.nanmin(valid[:, 2]))
        center_xy = valid[:, :2].mean(axis=0)
        extent = max(float(np.ptp(valid[:, 0])), float(np.ptp(valid[:, 1])), 1.0) * 0.7
        for offset in np.linspace(-extent, extent, 9):
            self._line3d(np.array([[center_xy[0] - extent, center_xy[1] + offset, z], [center_xy[0] + extent, center_xy[1] + offset, z]]), fill="#dddddd")
            self._line3d(np.array([[center_xy[0] + offset, center_xy[1] - extent, z], [center_xy[0] + offset, center_xy[1] + extent, z]]), fill="#dddddd")

    def _draw_track(self, points: np.ndarray, color: str, width: int) -> None:
        valid = np.isfinite(points).all(axis=1)
        start = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = idx
            if start is not None and (not is_valid or idx == len(valid) - 1):
                end = idx + 1 if is_valid and idx == len(valid) - 1 else idx
                if end - start >= 2:
                    projected = self._project(points[start:end])
                    coords = [coord for point in projected[:, :2] for coord in point]
                    self.canvas.create_line(*coords, fill=color, width=width, smooth=True)
                start = None

    def _draw_axes(self, positions: np.ndarray, rotations: np.ndarray, truth: bool) -> None:
        dash = () if truth else (4, 3)
        width = 2 if truth else 1
        colors = ("#cc3333", "#2a9d55", "#3366cc")
        for frame_idx in range(0, len(positions), self.axis_step):
            pos = positions[frame_idx]
            rot = rotations[frame_idx]
            if not (np.isfinite(pos).all() and np.isfinite(rot).all()):
                continue
            for axis_idx, color in enumerate(colors):
                pts = self._project(np.vstack([pos, pos + rot[:, axis_idx] * self.axis_length]))
                self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1], fill=color, width=width, dash=dash)

    def _draw_landmarks(self, landmarks: np.ndarray, color: str) -> None:
        for frame_idx in range(0, len(landmarks), self.axis_step):
            pts3 = landmarks[frame_idx]
            if not np.isfinite(pts3).all():
                continue
            pts = self._project(pts3)
            for point in pts:
                self.canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill=color, outline="")

    def _line3d(self, points: np.ndarray, **kwargs) -> None:
        pts = self._project(points)
        self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1], **kwargs)

    def _draw_overlay(self) -> None:
        frame_count = len(self.data["frame_indices"])
        fps = float(np.asarray(self.data["output_fps"]).reshape(-1)[0])
        foot_names = ", ".join(str(name) for name in self.data["foot_names"])
        lines = [
            f"frames: {frame_count}   fps: {fps:.1f}",
            f"feet: {foot_names}   cool colors: truth   warm colors: estimated",
            "solid axes: truth pose   dashed axes: estimated pose",
            "mouse: left drag rotate, right drag pan, wheel zoom, r reset",
        ]
        for idx, text in enumerate(lines):
            self.canvas.create_text(18, 20 + idx * 22, anchor="w", text=text, fill="#222222", font=("TkDefaultFont", 11, "bold" if idx == 0 else "normal"))


def foot_tracks_main() -> int:
    args = parse_foot_tracks_args()
    data = np.load(args.track_file, allow_pickle=False)
    root = tk.Tk()
    FootTrackViewer(root, data, args.axis_step, args.axis_length, args.show_landmarks)
    root.mainloop()
    return 0


# Single-frame camera rig viewer
import argparse
import math
from pathlib import Path
import sys
import tkinter as tk

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geosim.camera import make_default_camera_rig, make_default_wrist_camera_rig
from geosim.config import load_config
from geosim.motion import load_motion_npz
from geosim.smplx_numpy import load_smplx_model
from geosim.tag_rig import make_wrist_tag_rig


def parse_rig_frame_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively inspect one SMPL-X frame with head/wrist camera rigs.")
    parser.add_argument("--motion", default=str(ROOT / "test_motion/HumanEva/S1/Walking_3_stageii.npz"))
    parser.add_argument("--smplx-model", default=str(ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--config", default=str(ROOT / "config/default_geometry.json"))
    parser.add_argument("--frame", type=int, default=1600)
    parser.add_argument("--mesh-stride", type=int, default=3)
    parser.add_argument("--history", type=int, default=120)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


class CameraRigFrameViewer:
    def __init__(
        self,
        root: tk.Tk,
        *,
        vertices: np.ndarray,
        faces: np.ndarray,
        joints: np.ndarray,
        parents: np.ndarray,
        head_path: np.ndarray,
        wrist_path: np.ndarray,
        head_cameras: list[tuple[str, np.ndarray, np.ndarray]],
        wrist_cameras: list[tuple[str, np.ndarray, np.ndarray]],
        tag_quads: list[tuple[str, np.ndarray]],
        frame_idx: int,
        frame_count: int,
        width: int,
        height: int,
    ):
        self.root = root
        self.vertices = vertices
        self.faces = faces
        self.joints = joints
        self.parents = parents
        self.head_path = head_path
        self.wrist_path = wrist_path
        self.head_cameras = head_cameras
        self.wrist_cameras = wrist_cameras
        self.tag_quads = tag_quads
        self.frame_idx = frame_idx
        self.frame_count = frame_count
        self.width = int(width)
        self.height = int(height)

        self.azimuth = math.radians(-42.0)
        self.elevation = math.radians(18.0)
        self.zoom = 1.0
        self.pan = np.zeros(2, dtype=float)
        self.last_mouse: tuple[int, int] | None = None
        self.drag_mode = "rotate"
        self.show_mesh = True
        self.show_skeleton = True
        self.show_cameras = True
        self.show_tags = True

        self.center, self.base_scale = self._scene_bounds()
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="#f6f6f2", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._bind_events()

        root.title(f"Camera rig frame {self.frame_idx}/{self.frame_count - 1}")
        self.draw()

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._start_rotate)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._drag)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda _event: self._zoom_at(1.08))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom_at(1.0 / 1.08))
        self.root.bind("r", lambda _event: self._reset_view())
        self.root.bind("m", lambda _event: self._toggle("show_mesh"))
        self.root.bind("b", lambda _event: self._toggle("show_skeleton"))
        self.root.bind("c", lambda _event: self._toggle("show_cameras"))
        self.root.bind("t", lambda _event: self._toggle("show_tags"))

    def _scene_bounds(self) -> tuple[np.ndarray, float]:
        points = [self.vertices[:: max(1, len(self.vertices) // 2500)], self.joints, self.head_path, self.wrist_path]
        for _name, pos, direction in self.head_cameras + self.wrist_cameras:
            points.append(np.vstack([pos, pos + 0.28 * direction]))
        for _name, corners in self.tag_quads:
            points.append(corners)
        valid = np.vstack([np.asarray(arr, dtype=float).reshape(-1, 3) for arr in points])
        valid = valid[np.isfinite(valid).all(axis=1)]
        center = valid.mean(axis=0) if len(valid) else np.zeros(3)
        span = np.ptp(valid, axis=0) if len(valid) else np.ones(3)
        scale = 0.72 * min(self.width, self.height) / max(float(np.max(span)), 1e-3)
        return center, scale

    def _on_resize(self, event: tk.Event) -> None:
        self.width = max(1, event.width)
        self.height = max(1, event.height)
        self.center, self.base_scale = self._scene_bounds()
        self.draw()

    def _start_rotate(self, event: tk.Event) -> None:
        self.drag_mode = "rotate"
        self.last_mouse = (event.x, event.y)

    def _start_pan(self, event: tk.Event) -> None:
        self.drag_mode = "pan"
        self.last_mouse = (event.x, event.y)

    def _drag(self, event: tk.Event) -> None:
        if self.last_mouse is None:
            self.last_mouse = (event.x, event.y)
            return
        dx = event.x - self.last_mouse[0]
        dy = event.y - self.last_mouse[1]
        self.last_mouse = (event.x, event.y)
        if self.drag_mode == "pan":
            self.pan += np.array([dx, dy], dtype=float)
        else:
            self.azimuth += dx * 0.008
            self.elevation = float(np.clip(self.elevation + dy * 0.006, math.radians(-85.0), math.radians(85.0)))
        self.draw()

    def _wheel(self, event: tk.Event) -> None:
        self._zoom_at(1.08 if event.delta > 0 else 1.0 / 1.08)

    def _zoom_at(self, factor: float) -> None:
        self.zoom = float(np.clip(self.zoom * factor, 0.06, 35.0))
        self.draw()

    def _reset_view(self) -> None:
        self.azimuth = math.radians(-42.0)
        self.elevation = math.radians(18.0)
        self.zoom = 1.0
        self.pan[:] = 0.0
        self.draw()

    def _toggle(self, attr: str) -> None:
        setattr(self, attr, not getattr(self, attr))
        self.draw()

    def _view_coords(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3) - self.center
        ca = math.cos(self.azimuth)
        sa = math.sin(self.azimuth)
        ce = math.cos(self.elevation)
        se = math.sin(self.elevation)
        x0 = ca * pts[:, 0] - sa * pts[:, 1]
        depth = sa * pts[:, 0] + ca * pts[:, 1]
        y0 = -se * depth + ce * pts[:, 2]
        return np.column_stack([x0, y0, depth])

    def _project(self, points: np.ndarray) -> np.ndarray:
        view = self._view_coords(points)
        scale = self.base_scale * self.zoom
        x = self.width * 0.5 + self.pan[0] + view[:, 0] * scale
        y = self.height * 0.54 + self.pan[1] - view[:, 1] * scale
        return np.column_stack([x, y, view[:, 2]])

    def draw(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        if self.show_mesh:
            self._draw_mesh()
        if self.show_skeleton:
            self._draw_skeleton()
        self._draw_path(self.head_path, "#4b8fd9", width=2)
        self._draw_path(self.wrist_path, "#e3863f", width=2)
        if self.show_tags:
            self._draw_tags()
        if self.show_cameras:
            self._draw_cameras()
        self._draw_frame_label()

    def _draw_grid(self) -> None:
        z = float(np.nanmin(self.vertices[:, 2]))
        center_xy = np.nanmean(self.vertices[:, :2], axis=0)
        extent = max(float(np.ptp(self.vertices[:, 0])), float(np.ptp(self.vertices[:, 1])), 1.2) * 0.7
        for offset in np.linspace(-extent, extent, 9):
            self._line3d(
                np.array([[center_xy[0] - extent, center_xy[1] + offset, z], [center_xy[0] + extent, center_xy[1] + offset, z]]),
                fill="#deded8",
            )
            self._line3d(
                np.array([[center_xy[0] + offset, center_xy[1] - extent, z], [center_xy[0] + offset, center_xy[1] + extent, z]]),
                fill="#deded8",
            )

    def _draw_mesh(self) -> None:
        projected = self._project(self.vertices)
        face_depth = projected[self.faces, 2].mean(axis=1)
        order = np.argsort(face_depth)
        for rank, face_idx in enumerate(order):
            pts = projected[self.faces[face_idx], :2]
            if np.any(pts[:, 0] < -80) or np.any(pts[:, 0] > self.width + 80):
                continue
            if np.any(pts[:, 1] < -80) or np.any(pts[:, 1] > self.height + 80):
                continue
            shade = 205 + int(22 * rank / max(1, len(order) - 1))
            fill = f"#{shade:02x}{shade:02x}{max(190, shade - 9):02x}"
            coords = [coord for point in pts for coord in point]
            self.canvas.create_polygon(*coords, fill=fill, outline="#aaa7a0", width=1)

    def _draw_skeleton(self) -> None:
        projected = self._project(self.joints)
        for joint_idx, parent in enumerate(self.parents):
            if parent >= 0:
                self.canvas.create_line(
                    projected[joint_idx, 0],
                    projected[joint_idx, 1],
                    projected[parent, 0],
                    projected[parent, 1],
                    fill="#333333",
                    width=2,
                )

    def _draw_path(self, points: np.ndarray, color: str, width: int) -> None:
        if len(points) < 2:
            return
        projected = self._project(points)
        coords = [coord for point in projected[:, :2] for coord in point]
        self.canvas.create_line(*coords, fill=color, width=width, smooth=True)

    def _draw_tags(self) -> None:
        for name, corners in self.tag_quads:
            projected = self._project(corners)
            coords = [coord for point in projected[:, :2] for coord in point]
            fill = "#48b86f" if name == "tag0" else "#37a5c7"
            self.canvas.create_polygon(*coords, fill=fill, outline="#0d4d2c", width=2)
            center = projected[:, :2].mean(axis=0)
            self.canvas.create_text(center[0], center[1], text=name, fill="#082718", font=("TkDefaultFont", 10, "bold"))

    def _draw_cameras(self) -> None:
        for name, pos, direction in self.head_cameras:
            self._draw_camera(name, pos, direction, "#c83c3c", length=0.22)
        for name, pos, direction in self.wrist_cameras:
            self._draw_camera(name, pos, direction, "#9b3aa6", length=0.18)

    def _draw_camera(self, name: str, pos: np.ndarray, direction: np.ndarray, color: str, length: float) -> None:
        segment = np.vstack([pos, pos + length * direction])
        projected = self._project(segment)
        self.canvas.create_oval(
            projected[0, 0] - 6,
            projected[0, 1] - 6,
            projected[0, 0] + 6,
            projected[0, 1] + 6,
            fill=color,
            outline="#ffffff",
            width=2,
        )
        self.canvas.create_line(
            projected[0, 0],
            projected[0, 1],
            projected[1, 0],
            projected[1, 1],
            fill=color,
            width=4,
            arrow=tk.LAST,
            arrowshape=(14, 17, 6),
        )
        label_offset = np.array([8.0, -8.0])
        self.canvas.create_text(
            projected[1, 0] + label_offset[0],
            projected[1, 1] + label_offset[1],
            anchor="w",
            text=name,
            fill=color,
            font=("TkDefaultFont", 10, "bold"),
        )

    def _line3d(self, points: np.ndarray, **kwargs) -> None:
        projected = self._project(points)
        self.canvas.create_line(projected[0, 0], projected[0, 1], projected[1, 0], projected[1, 1], **kwargs)

    def _draw_frame_label(self) -> None:
        self.canvas.create_text(
            18,
            22,
            anchor="w",
            text=f"frame {self.frame_idx}/{self.frame_count - 1}",
            fill="#222222",
            font=("TkDefaultFont", 12, "bold"),
        )


def rig_frame_main() -> int:
    args = parse_rig_frame_args()
    config = load_config(args.config)
    motion = load_motion_npz(args.motion, smplx_model_path=args.smplx_model)
    model = load_smplx_model(args.smplx_model)
    frame_idx = int(np.clip(args.frame, 0, motion.frames - 1))

    source = np.load(motion.source_path, allow_pickle=True)
    betas = np.asarray(source["betas"], dtype=float) if "betas" in source.files else motion.betas
    frame = model.forward_frame(source["poses"][frame_idx], source["trans"][frame_idx], betas, include_vertices=True)
    if frame.vertices is None:
        raise RuntimeError("SMPL-X frame did not include vertices.")

    start_idx = max(0, frame_idx - max(0, args.history))
    head_cameras = [
        _camera_pose_tuple(camera.name, *camera.world_pose(motion.head_pos[frame_idx], motion.head_rot[frame_idx]))
        for camera in make_default_camera_rig(config.camera_rig)
    ]
    wrist_cameras = [
        _camera_pose_tuple(camera.name, *camera.world_pose(motion.left_wrist_pos[frame_idx], motion.left_wrist_rot[frame_idx]))
        for camera in make_default_wrist_camera_rig(config.camera_rig)
    ]
    tag_points = make_wrist_tag_rig(config.tag_rig).world_points(motion.left_wrist_pos[frame_idx], motion.left_wrist_rot[frame_idx])
    tag_quads = [
        (tag_name, np.stack([tag_points[f"{tag_name}_c{i}"] for i in range(4)], axis=0))
        for tag_name in ("tag0", "tag1")
    ]

    root = tk.Tk()
    CameraRigFrameViewer(
        root,
        vertices=frame.vertices,
        faces=model.faces[:: max(1, args.mesh_stride)],
        joints=frame.joints,
        parents=model.parents,
        head_path=motion.head_pos[start_idx : frame_idx + 1],
        wrist_path=motion.left_wrist_pos[start_idx : frame_idx + 1],
        head_cameras=head_cameras,
        wrist_cameras=wrist_cameras,
        tag_quads=tag_quads,
        frame_idx=frame_idx,
        frame_count=motion.frames,
        width=args.width,
        height=args.height,
    )
    root.mainloop()
    return 0


def _camera_pose_tuple(name: str, pos: np.ndarray, rot: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    direction = rot @ np.array([0.0, 0.0, 1.0], dtype=float)
    return name, np.asarray(pos, dtype=float), direction / max(float(np.linalg.norm(direction)), 1e-12)


COMMANDS = {
    'pose-tracks': pose_tracks_main,
    'foot-tracks': foot_tracks_main,
    'rig-frame': rig_frame_main,
}


def main() -> int:
    parser = argparse.ArgumentParser(description='Open interactive visualizations.')
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, rest = parser.parse_known_args()
    sys.argv = [f"{Path(sys.argv[0]).name} {args.command}", *rest]
    return int(COMMANDS[args.command]() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
