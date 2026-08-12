#!/usr/bin/env python3
"""Render four-view reprojection diagnostics and an interactive 3-D viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from multiview_geometry import CameraPose, OmniCamera, load_json, stereo_transform_d_a


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "0806_dual_external_mocap.json"
CAMERA_ORDER = (
    "module01_CAM_A", "module01_CAM_D",
    "module02_CAM_A", "module02_CAM_D",
)
# Delivery skeleton: nose-only face, one big toe per foot.
EDGES = (
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_big_toe"), ("right_ankle", "right_big_toe"),
    ("left_ankle", "left_toe"), ("right_ankle", "right_toe"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Override manifest path recorded in the report (useful on a remote host).",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        help="Override videos with <root>/<module01|module02>/<original filename>.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_models(config: dict) -> dict[str, OmniCamera]:
    models = {}
    for module_name, module in config["modules"].items():
        calibration = load_json(resolve_repo_path(module["intrinsics"]))
        for socket in ("CAM_A", "CAM_D"):
            name = f"{module_name}_{socket}"
            models[name] = OmniCamera.from_calibration(
                calibration, socket, name=name
            )
    return models


def display_uv(uv, width: int, height: int, rotate: bool) -> tuple[int, int]:
    x, y = map(float, uv)
    if rotate:
        x, y = width - x, height - y
    return int(round(x)), int(round(y))


def draw_skeleton(
    image: np.ndarray,
    points: dict[str, np.ndarray],
    color: tuple[int, int, int],
    *,
    radius: int,
) -> None:
    for first, second in EDGES:
        if first in points and second in points:
            cv2.line(
                image,
                tuple(np.rint(points[first]).astype(int)),
                tuple(np.rint(points[second]).astype(int)),
                color,
                3,
                cv2.LINE_AA,
            )
    for point in points.values():
        cv2.circle(
            image,
            tuple(np.rint(point).astype(int)),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )


def read_frame(
    capture: cv2.VideoCapture,
    target: int,
    current: dict[str, int],
    name: str,
    *,
    path: Path | None = None,
) -> np.ndarray:
    """Read absolute frame_index without CAP_PROP_POS_FRAMES.

    OpenCV seek on these MJPEG streams is unreliable and often returns frame 0,
    which makes overlays look one or more frames ahead of the video.
    """
    if target < 0:
        raise RuntimeError(f"Invalid frame index {target} for {name}")
    if current[name] > target:
        if path is None:
            raise RuntimeError(
                f"Cannot rewind {name} to frame {target} without reopening the video"
            )
        capture.release()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not reopen {path}")
        current[name] = -1
    while current[name] < target:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read {name} frame {target}")
        current[name] += 1
        if current[name] == target:
            return capture, frame
    raise RuntimeError(f"Failed to reach {name} frame {target}")


def video_paths(report: dict, video_root: Path | None = None) -> dict[str, Path]:
    output = {}
    for module_name in ("module01", "module02"):
        videos = report["mapping"][module_name]["videos"]
        for socket in ("CAM_A", "CAM_D"):
            original = Path(videos[socket])
            output[f"{module_name}_{socket}"] = (
                video_root / module_name / PureWindowsPath(videos[socket]).name
                if video_root is not None
                else original
            )
    return output


def render_video(
    records: list[dict],
    paths: dict[str, Path],
    models: dict[str, OmniCamera],
    output: Path,
    rotate: bool,
) -> None:
    captures = {name: cv2.VideoCapture(str(path)) for name, path in paths.items()}
    if not all(capture.isOpened() for capture in captures.values()):
        failed = [name for name, capture in captures.items() if not capture.isOpened()]
        raise RuntimeError(f"Could not open videos: {failed}")
    current = {name: -1 for name in captures}
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1920, 1200)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output}")
    for record_index, record in enumerate(records):
        panels = []
        for camera_name in CAMERA_ORDER:
            captures[camera_name], frame = read_frame(
                captures[camera_name],
                int(record["frames"][camera_name]),
                current,
                camera_name,
                path=paths[camera_name],
            )
            height, width = frame.shape[:2]
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            observed = {
                joint: np.asarray(
                    display_uv(payload["filtered_uv"], width, height, rotate)
                )
                for joint, payload in record["observations"].get(camera_name, {}).items()
                if payload["confidence"] >= 0.16
            }
            pose = CameraPose(
                models[camera_name],
                np.asarray(record["camera_poses"][camera_name], dtype=np.float64),
            )
            projected = {}
            for joint, payload in record["methods"]["filtered"]["multiview"].items():
                uv = pose.project_world(payload["xyz_world_m"])
                if uv is not None:
                    projected[joint] = np.asarray(
                        display_uv(uv, width, height, rotate)
                    )
            draw_skeleton(frame, observed, (0, 255, 255), radius=5)
            draw_skeleton(frame, projected, (255, 0, 255), radius=3)
            for joint in set(observed) & set(projected):
                cv2.line(
                    frame,
                    tuple(observed[joint].astype(int)),
                    tuple(projected[joint].astype(int)),
                    (80, 80, 255),
                    1,
                    cv2.LINE_AA,
                )
            title = (
                f"{camera_name} seq={record['seq']} "
                "cyan=RTMPose magenta=4-view"
            )
            cv2.putText(
                frame, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
            panels.append(cv2.resize(frame, (960, 600), interpolation=cv2.INTER_AREA))
        grid = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
        writer.write(grid)
        if record_index == 0:
            snapshot = output.with_name(f"{output.stem}_first_frame.jpg")
            ok, encoded = cv2.imencode(".jpg", grid)
            if ok:
                encoded.tofile(str(snapshot))
    writer.release()
    for capture in captures.values():
        capture.release()


def skeleton_trace_payload(record: dict) -> tuple[list, list, list]:
    joints = record["methods"]["filtered"]["multiview"]
    x, y, z = [], [], []
    for first, second in EDGES:
        if first in joints and second in joints:
            for joint in (first, second):
                point = joints[joint]["xyz_world_m"]
                x.append(point[0])
                y.append(point[1])
                z.append(point[2])
            x.append(None)
            y.append(None)
            z.append(None)
    return x, y, z


def transform_from_mechanical_camera(camera: dict) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(camera["R_rigid_camera"], dtype=np.float64)
    transform[:3, 3] = (
        np.asarray(camera["p_rigid_camera_mm"], dtype=np.float64) / 1000.0
    )
    return transform


def axes_trace(
    transforms: dict[str, np.ndarray],
    axis: int,
    scale: float,
    name: str,
    *,
    dash: str = "solid",
    opacity: float = 1.0,
) -> dict:
    colors = ("#ef5350", "#66bb6a", "#42a5f5")
    axis_names = ("X", "Y", "Z")
    x, y, z = [], [], []
    for transform in transforms.values():
        origin = transform[:3, 3]
        endpoint = origin + scale * transform[:3, axis]
        for point in (origin, endpoint):
            x.append(float(point[0]))
            y.append(float(point[1]))
            z.append(float(point[2]))
        x.append(None)
        y.append(None)
        z.append(None)
    return {
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "line": {"width": 6, "color": colors[axis], "dash": dash},
        "opacity": opacity,
        "name": f"{name} {axis_names[axis]}",
        "legendgroup": name,
    }


def baseline_trace(camera_poses: dict[str, np.ndarray]) -> dict:
    x, y, z = [], [], []
    for module_name in ("module01", "module02"):
        for socket in ("CAM_A", "CAM_D"):
            point = camera_poses[f"{module_name}_{socket}"][:3, 3]
            x.append(float(point[0]))
            y.append(float(point[1]))
            z.append(float(point[2]))
        x.append(None)
        y.append(None)
        z.append(None)
    return {
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "line": {"width": 8, "color": "#ffffff"},
        "name": "A-D calibrated baselines",
    }


def point_trace(
    points: dict[str, np.ndarray],
    name: str,
    color: str,
    symbol: str,
    *,
    size: int = 5,
) -> dict:
    return {
        "type": "scatter3d",
        "mode": "markers+text",
        "x": [float(point[0]) for point in points.values()],
        "y": [float(point[1]) for point in points.values()],
        "z": [float(point[2]) for point in points.values()],
        "text": list(points),
        "textposition": "top center",
        "marker": {
            "size": size,
            "color": color,
            "symbol": symbol,
            "line": {"width": 2, "color": color},
        },
        "name": name,
    }


def load_stereo_transforms(config: dict) -> dict[str, np.ndarray]:
    transforms = {}
    for module_name, module in config["modules"].items():
        calibration = load_json(resolve_repo_path(module["intrinsics"]))
        transforms[module_name] = stereo_transform_d_a(calibration)
    return transforms


def ankle_heights(record: dict) -> tuple[float | None, float | None]:
    joints = record["methods"]["filtered"]["multiview"]
    left = joints.get("left_ankle")
    right = joints.get("right_ankle")
    left_z = None if left is None else float(left["xyz_world_m"][2])
    right_z = None if right is None else float(right["xyz_world_m"][2])
    return left_z, right_z


def write_viewer(records: list[dict], output: Path) -> None:
    """Skeleton-only interactive viewer with playback, view presets, and foot HUD."""
    frames = []
    all_points = []
    ankle_z_samples: list[float] = []
    foot_heights: list[dict[str, float | None]] = []
    for record in records:
        x, y, z = skeleton_trace_payload(record)
        all_points.extend(point for point in zip(x, y, z) if point[0] is not None)
        left_z, right_z = ankle_heights(record)
        for value in (left_z, right_z):
            if value is not None:
                ankle_z_samples.append(value)
        foot_heights.append(
            {
                "seq": int(record["seq"]),
                "left_z": left_z,
                "right_z": right_z,
            }
        )
        frames.append(
            {
                "name": str(record["seq"]),
                "data": [
                    {
                        "type": "scatter3d",
                        "mode": "lines+markers",
                        "x": x,
                        "y": y,
                        "z": z,
                        "line": {"width": 6, "color": "#00bcd4"},
                        "marker": {"size": 3, "color": "#ffeb3b"},
                        "hoverinfo": "skip",
                        "showlegend": False,
                    }
                ],
            }
        )
    if not frames:
        raise RuntimeError("No records to render")
    if not ankle_z_samples:
        raise RuntimeError("No ankle joints available to estimate ground Z")

    ground_z = float(np.percentile(ankle_z_samples, 5))
    for item in foot_heights:
        left_z = item["left_z"]
        right_z = item["right_z"]
        item["left_above_m"] = None if left_z is None else float(left_z - ground_z)
        item["right_above_m"] = None if right_z is None else float(right_z - ground_z)

    points = np.asarray(all_points, dtype=np.float64)
    center = np.nanmedian(points, axis=0)
    low = np.nanpercentile(points, 2, axis=0)
    high = np.nanpercentile(points, 98, axis=0)
    span = max(float(np.max(high - low)), 1.0)
    half = span * 0.65
    ranges = [
        [float(center[0] - half), float(center[0] + half)],
        [float(center[1] - half), float(center[1] + half)],
        [float(center[2] - half), float(center[2] + half)],
    ]
    cx, cy, cz = map(float, center)
    eye_distance = span * 1.35
    views = {
        "iso": {"eye": {"x": cx + eye_distance, "y": cy - eye_distance, "z": cz + eye_distance * 0.75}},
        "front": {"eye": {"x": cx, "y": cy - eye_distance * 1.35, "z": cz + eye_distance * 0.15}},
        "side": {"eye": {"x": cx + eye_distance * 1.35, "y": cy, "z": cz + eye_distance * 0.15}},
        "top": {"eye": {"x": cx, "y": cy, "z": cz + eye_distance * 1.6}},
    }
    slider_steps = [
        {
            "label": frame["name"],
            "method": "animate",
            "args": [
                [frame["name"]],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for frame in frames
    ]
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>0806 four-view skeleton</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html,body {{ margin:0; height:100%; background:#111; color:#eee; font:14px/1.4 Segoe UI,sans-serif; }}
  #plot {{ width:100vw; height:100vh; }}
  #toolbar {{
    position:fixed; left:16px; top:16px; z-index:5; display:flex; gap:8px; flex-wrap:wrap;
    background:rgba(18,18,18,0.72); border:1px solid #333; border-radius:10px; padding:10px;
  }}
  #toolbar button {{
    background:#222; color:#eee; border:1px solid #444; border-radius:8px; padding:6px 10px; cursor:pointer;
  }}
  #toolbar button.active, #toolbar button:hover {{ background:#333; border-color:#777; }}
  #hud {{
    position:fixed; right:16px; top:16px; z-index:5; min-width:220px;
    background:rgba(18,18,18,0.72); border:1px solid #333; border-radius:10px; padding:12px 14px;
  }}
  #hud .label {{ color:#aaa; font-size:12px; }}
  #hud .value {{ font-size:18px; margin:2px 0 8px; font-variant-numeric:tabular-nums; }}
</style>
</head>
<body>
<div id="toolbar">
  <button id="play">Play</button>
  <button id="pause">Pause</button>
  <button data-view="iso" class="active">Iso</button>
  <button data-view="front">Front</button>
  <button data-view="side">Side</button>
  <button data-view="top">Top</button>
</div>
<div id="hud">
  <div class="label">seq</div><div class="value" id="seqValue">-</div>
  <div class="label">ground Z (ankle p5)</div><div class="value" id="groundValue">{ground_z:.3f} m</div>
  <div class="label">left foot above ground</div><div class="value" id="leftFoot">-</div>
  <div class="label">right foot above ground</div><div class="value" id="rightFoot">-</div>
</div>
<div id="plot"></div>
<script>
const frames = {json.dumps(frames, separators=(",", ":"))};
const footHeights = {json.dumps(foot_heights, separators=(",", ":"))};
const views = {json.dumps(views, separators=(",", ":"))};
const groundZ = {ground_z:.6f};
const center = {json.dumps([cx, cy, cz])};
let index = 0;
let playing = false;
let timer = null;

function formatHeight(value) {{
  return value === null || value === undefined ? "n/a" : `${{(value * 1000).toFixed(0)}} mm`;
}}

function updateHud(i) {{
  const item = footHeights[i];
  document.getElementById("seqValue").textContent = String(item.seq);
  document.getElementById("groundValue").textContent = groundZ.toFixed(3) + " m";
  document.getElementById("leftFoot").textContent = formatHeight(item.left_above_m);
  document.getElementById("rightFoot").textContent = formatHeight(item.right_above_m);
}}

function showFrame(i, {{animate = true}} = {{}}) {{
  index = Math.max(0, Math.min(frames.length - 1, i));
  updateHud(index);
  if (animate) {{
    return Plotly.animate("plot", [frames[index].name], {{
      mode: "immediate",
      frame: {{duration: 0, redraw: true}},
      transition: {{duration: 0}}
    }}).then(() => Plotly.relayout("plot", {{"sliders[0].active": index}}));
  }}
  return Plotly.relayout("plot", {{"sliders[0].active": index}});
}}

function stopPlayback() {{
  playing = false;
  if (timer !== null) {{
    clearInterval(timer);
    timer = null;
  }}
}}

function startPlayback() {{
  if (playing) return;
  playing = true;
  timer = setInterval(() => {{
    const next = index + 1;
    if (next >= frames.length) {{
      stopPlayback();
      return;
    }}
    showFrame(next);
  }}, 33);
}}

function setView(name) {{
  document.querySelectorAll("#toolbar [data-view]").forEach((button) => {{
    button.classList.toggle("active", button.dataset.view === name);
  }});
  const camera = Object.assign({{ center: {{x: center[0], y: center[1], z: center[2]}} }}, views[name]);
  return Plotly.relayout("plot", {{"scene.camera": camera}});
}}

const layout = {{
  margin: {{l: 0, r: 0, t: 30, b: 80}},
  paper_bgcolor: "#111",
  plot_bgcolor: "#111",
  font: {{color: "#eee"}},
  showlegend: false,
  scene: {{
    xaxis: {{title: "world X (m)", range: {json.dumps(ranges[0])}, showbackground: false}},
    yaxis: {{title: "world Y (m)", range: {json.dumps(ranges[1])}, showbackground: false}},
    zaxis: {{title: "world Z (m)", range: {json.dumps(ranges[2])}, showbackground: false}},
    aspectmode: "cube",
    camera: Object.assign({{ center: {{x: center[0], y: center[1], z: center[2]}} }}, views.iso)
  }},
  sliders: [{{
    active: 0,
    pad: {{t: 30}},
    currentvalue: {{prefix: "seq ", font: {{color: "#eee"}}}},
    steps: {json.dumps(slider_steps, separators=(",", ":"))}
  }}]
}};

Plotly.newPlot("plot", frames[0].data, layout, {{responsive: true, displayModeBar: false}})
  .then(() => Plotly.addFrames("plot", frames))
  .then(() => updateHud(0));

document.getElementById("play").onclick = () => startPlayback();
document.getElementById("pause").onclick = () => stopPlayback();
document.querySelectorAll("#toolbar [data-view]").forEach((button) => {{
  button.onclick = () => setView(button.dataset.view);
}});
document.getElementById("plot").on("plotly_sliderchange", (event) => {{
  stopPlayback();
  index = event.slider.active;
  updateHud(index);
}});
</script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.results)
    if args.max_frames is not None:
        records = records[: args.max_frames]
    config = load_json(args.config)
    report = load_json(args.manifest_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = args.output_dir / "four_view_reprojection.mp4"
    viewer = args.output_dir / "camera_rigid_axes_and_skeleton_3d.html"
    render_video(
        records,
        video_paths(report, args.video_root),
        load_models(config),
        video,
        bool(config.get("display_rotate_180", False)),
    )
    write_viewer(records, viewer)
    print(json.dumps({"frames": len(records), "video": str(video), "viewer": str(viewer)}))


if __name__ == "__main__":
    main()
