"""
Simple 3D skeleton player for exported mocap joint CSV.

Default input:
  C:\\Users\\hand\\Desktop\\Dataset\\0714\\002\\fbx_mocap_csv\\mocap_joints_long.csv

Run:
  C:\\Users\\hand\\miniconda3\\python.exe C:\\Users\\hand\\Desktop\\Dataset\\tools\\play_mocap_skeleton_3d.py

Useful options:
  --frame-step 3      Play every 3rd frame for faster preview.
  --speed 1.5         Playback speed multiplier.
  --trail 30          Show recent Hips trajectory.
  --save-mp4 out.mp4  Save animation instead of only showing window.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


DEFAULT_ROOT = Path(r"C:\Users\hand\Desktop\Dataset\0714\002\fbx_mocap_csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", default=str(DEFAULT_ROOT / "mocap_joints_long.csv"))
    parser.add_argument("--skeleton", default=str(DEFAULT_ROOT / "mocap_skeleton.csv"))
    parser.add_argument("--frame-step", type=int, default=2, help="Use every Nth frame for playback.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for quick preview. 0 = all.")
    parser.add_argument("--trail", type=int, default=60, help="Number of Hips points to keep as trajectory.")
    parser.add_argument("--include-end-bones", action="store_true", help="Include *_End bones.")
    parser.add_argument("--save-mp4", default="", help="Optional output mp4 path.")
    parser.add_argument("--fps", type=float, default=30.0, help="Display/save FPS.")
    parser.add_argument("--elev", type=float, default=16.0)
    parser.add_argument("--azim", type=float, default=-72.0)
    parser.add_argument("--dry-run", action="store_true", help="Load data and print summary, then exit.")
    return parser.parse_args()


def include_bone(name: str, include_end_bones: bool) -> bool:
    if name == "Actor_1":
        return False
    if not include_end_bones and name.endswith("_End"):
        return False
    return True


def load_skeleton(path: Path, include_end_bones: bool) -> tuple[list[str], list[tuple[str, str]]]:
    bones: list[str] = []
    parents: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            parent = row.get("parent_name", "")
            if include_bone(name, include_end_bones):
                bones.append(name)
                parents[name] = parent

    bone_set = set(bones)
    edges = []
    for bone in bones:
        parent = parents.get(bone, "")
        if parent in bone_set:
            edges.append((parent, bone))
    return bones, edges


def load_joint_frames(
    path: Path,
    keep_bones: set[str],
    frame_step: int,
    max_frames: int,
) -> tuple[list[int], list[float], list[dict[str, tuple[float, float, float]]]]:
    raw_frames: dict[int, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    times: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            bone = row["bone"]
            if bone not in keep_bones:
                continue
            frame = int(row["frame_index"])
            if frame % frame_step != 0:
                continue
            raw_frames[frame][bone] = (
                float(row["world_x"]),
                float(row["world_y"]),
                float(row["world_z"]),
            )
            times[frame] = float(row["time_sec"])

    frame_ids = sorted(raw_frames)
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    frame_times = [times[i] for i in frame_ids]
    frames = [raw_frames[i] for i in frame_ids]
    return frame_ids, frame_times, frames


def axis_limits(frames: list[dict[str, tuple[float, float, float]]]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    xs, ys, zs = [], [], []
    for frame in frames:
        for x, y, z in frame.values():
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        raise RuntimeError("No joint points loaded.")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    cz = 0.5 * (min_z + max_z)
    radius = 0.55 * max(max_x - min_x, max_y - min_y, max_z - min_z)
    radius = max(radius, 1.0)
    return (cx - radius, cx + radius), (cy - radius, cy + radius), (cz - radius, cz + radius)


def main() -> None:
    args = parse_args()
    joints_path = Path(args.joints)
    skeleton_path = Path(args.skeleton)
    frame_step = max(1, args.frame_step)

    bones, edges = load_skeleton(skeleton_path, args.include_end_bones)
    frame_ids, frame_times, frames = load_joint_frames(
        joints_path, set(bones), frame_step, args.max_frames
    )
    if not frames:
        raise RuntimeError("No frames loaded. Check input CSV paths.")

    xlim, ylim, zlim = axis_limits(frames)
    if args.dry_run:
        print(f"joints_csv={joints_path}")
        print(f"skeleton_csv={skeleton_path}")
        print(f"frames_loaded={len(frames)}")
        print(f"first_frame={frame_ids[0]} time={frame_times[0]:.6f}")
        print(f"last_frame={frame_ids[-1]} time={frame_times[-1]:.6f}")
        print(f"bones={len(bones)} edges={len(edges)}")
        print(f"xlim={xlim}")
        print(f"ylim={ylim}")
        print(f"zlim={zlim}")
        return

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Mocap Skeleton")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(elev=args.elev, azim=args.azim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    scatter = ax.scatter([], [], [], s=24, c="#f97316", depthshade=True)
    lines = [ax.plot([], [], [], lw=2.0, c="#2563eb")[0] for _ in edges]
    trail_line = ax.plot([], [], [], lw=1.5, c="#16a34a", alpha=0.8)[0]
    text = ax.text2D(0.03, 0.96, "", transform=ax.transAxes)

    hips_history: list[tuple[float, float, float]] = []

    def update(i: int):
        frame = frames[i]
        points = [frame[b] for b in bones if b in frame]
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]
            scatter._offsets3d = (xs, ys, zs)

        for line, (parent, child) in zip(lines, edges):
            if parent in frame and child in frame:
                p0 = frame[parent]
                p1 = frame[child]
                line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
                line.set_3d_properties([p0[2], p1[2]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        if "Hips" in frame and args.trail > 0:
            hips_history.append(frame["Hips"])
            del hips_history[:-args.trail]
            trail_line.set_data([p[0] for p in hips_history], [p[1] for p in hips_history])
            trail_line.set_3d_properties([p[2] for p in hips_history])

        text.set_text(
            f"frame={frame_ids[i]}  time={frame_times[i]:.3f}s  "
            f"shown={i + 1}/{len(frames)}  bones={len(points)}"
        )
        return [scatter, trail_line, text, *lines]

    interval_ms = 1000.0 / max(1e-6, args.fps * max(1e-6, args.speed))
    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=False)

    if args.save_mp4:
        out = Path(args.save_mp4)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=args.fps, bitrate=3000)
        anim.save(str(out), writer=writer)
        print(f"saved_mp4={out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
