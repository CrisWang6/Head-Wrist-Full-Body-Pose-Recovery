#!/usr/bin/env python3
"""Render one five-point-optimized CH07 skeleton variant into both head views."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from process_external_stereo_to_head import NAMES, Omni, draw_pose, load_json, writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--comparison-json", type=Path)
    p.add_argument("--ch07-csv", type=Path,
                   help="Direct optimized CH07 skeleton; preferred for full-dataset rendering.")
    p.add_argument("--head-intrinsics", type=Path, required=True)
    p.add_argument("--head-rigid", type=Path, required=True)
    p.add_argument("--head-a", type=Path, required=True)
    p.add_argument("--head-d", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variant", choices=("after", "adaptive"), default="adaptive")
    return p.parse_args()


def main() -> None:
    a = parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    if not a.comparison_json and not a.ch07_csv:
        raise ValueError("Provide --comparison-json or --ch07-csv")
    data = load_json(a.comparison_json) if a.comparison_json else None
    intr = load_json(a.head_intrinsics)
    rigid = load_json(a.head_rigid)
    right_name = "CAM_D" if "CAM_D" in intr["cameras"] else "CAM_C"
    cams = {"A": Omni(intr, "CAM_A"), "D": Omni(intr, right_name)}
    stereo = np.asarray(intr["stereo_extrinsics"]["T_CAM_D_CAM_A"], np.float64)
    r_da = stereo[:3, :3]

    # Same validated z+90-equivalent basis used by five-point optimization:
    # calibrated -CAM_A X baseline maps to physical -head-rigid Y.
    r_ra = np.asarray([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    c_a = np.asarray(rigid["cameras"]["left"]["p_rigid_camera_mm"], np.float64) / 1000.0
    c_d = np.asarray(rigid["cameras"]["right"]["p_rigid_camera_mm"], np.float64) / 1000.0
    c2r_a = np.eye(4); c2r_a[:3, :3] = r_ra; c2r_a[:3, 3] = c_a
    c2r_d = np.eye(4); c2r_d[:3, :3] = r_ra @ r_da.T; c2r_d[:3, 3] = c_d
    r2c = {"A": np.linalg.inv(c2r_a), "D": np.linalg.inv(c2r_d)}

    projected = {"A": {}, "D": {}}
    output_rows = []
    source_frames=[]
    if data is not None:
        source_frames=[(int(frame["sequence"]),zip(data["joint_names"],frame[a.variant])) for frame in data["frames"]]
    else:
        by_seq={}
        with a.ch07_csv.open("r",encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                by_seq.setdefault(int(row["sequence"]),[]).append((row["joint"],[float(row["x_m"]),float(row["y_m"]),float(row["z_m"])]))
        source_frames=sorted(by_seq.items())
    for seq, named_values in source_frames:
        for name, value in named_values:
            if value is None or (name != "nose" and name not in NAMES[5:]):
                continue
            point = np.asarray(value, np.float64)
            row = {"sequence": seq, "joint": name,
                   "ch07_x_m": point[0], "ch07_y_m": point[1], "ch07_z_m": point[2]}
            for cam in "AD":
                uv = cams[cam].project((r2c[cam] @ np.r_[point, 1.0])[:3])
                if uv is not None:
                    projected[cam].setdefault(seq, {})[name] = np.asarray(uv)
                    row[f"head_{cam}_u_px"], row[f"head_{cam}_v_px"] = map(float, uv)
                else:
                    row[f"head_{cam}_u_px"] = row[f"head_{cam}_v_px"] = float("nan")
            output_rows.append(row)

    csv_path = a.output_dir / f"fivepoint_{a.variant}_head_2d.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0])); w.writeheader(); w.writerows(output_rows)

    paths = {}
    for cam, video in (("A", a.head_a), ("D", a.head_d)):
        cap = cv2.VideoCapture(str(video))
        path = a.output_dir / f"head_CAM_{cam}_fivepoint_{a.variant}_clean.mp4"
        out = writer(path, (1920, 1200), 50); seq = 0
        while True:
            ok, image = cap.read()
            if not ok: break
            draw_pose(image, projected[cam].get(seq, {}), (0, 255, 255),
                      f"five-point {a.variant} -> HEAD_{cam} seq={seq}", body_only=True)
            out.write(image); seq += 1
        cap.release(); out.release(); paths[cam] = path

    ca, cd = cv2.VideoCapture(str(paths["A"])), cv2.VideoCapture(str(paths["D"]))
    stereo_path = a.output_dir / f"head_stereo_fivepoint_{a.variant}_clean.mp4"
    out = writer(stereo_path, (1920, 600), 50); frames = 0
    while True:
        oka, ia = ca.read(); okd, id_ = cd.read()
        if not oka or not okd: break
        out.write(np.hstack([cv2.resize(ia, (960, 600)), cv2.resize(id_, (960, 600))])); frames += 1
    ca.release(); cd.release(); out.release()
    report = {"variant": a.variant, "frames": frames, "fps": 50,
              "head_axis_basis": "xy_swap + z_flip (z+90 equivalent)",
              "comparison_source": str(a.comparison_json) if a.comparison_json else None,
              "ch07_source":str(a.ch07_csv) if a.ch07_csv else None,"csv": str(csv_path)}
    (a.output_dir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
