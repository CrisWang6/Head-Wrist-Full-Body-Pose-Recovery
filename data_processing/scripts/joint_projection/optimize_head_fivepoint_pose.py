#!/usr/bin/env python3
"""Fit one global CH07 SE(3) correction from head-stereo nose/shoulders/elbows.

The external stereo skeleton remains rigid: no joint or bone length is edited.
RTMW supplies a fixed stereo nose anchor; the previously generated head-view
YOLO pose candidates supply left/right shoulder and elbow observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation

from process_external_stereo_to_head import NAMES, Omni, closest_rays, draw_pose, load_json, qrot, writer


TARGET_JOINTS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
SEGMENTS = (
    ("left_elbow", "left_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
)
BONES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--head-a-candidates", type=Path, required=True)
    p.add_argument("--head-d-candidates", type=Path, required=True)
    p.add_argument("--head-a-nose-csv", type=Path, required=True)
    p.add_argument("--head-d-nose-csv", type=Path, required=True)
    p.add_argument("--head-intrinsics", type=Path, required=True)
    p.add_argument("--head-rigid", type=Path, required=True)
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--world-csv", type=Path, required=True)
    p.add_argument("--head-a", type=Path, required=True)
    p.add_argument("--head-d", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ch07-event-offset", type=int, default=71)
    p.add_argument("--confidence", type=float, default=.05)
    p.add_argument("--max-ray-gap-mm", type=float, default=45.0)
    p.add_argument("--position-weight", type=float, default=1.0)
    p.add_argument("--direction-length-m", type=float, default=.22)
    p.add_argument("--rotation-prior-m-per-rad", type=float, default=.025)
    p.add_argument("--translation-prior", type=float, default=.20)
    p.add_argument("--head-axis-basis", choices=("auto", "file", "xy_swap"), default="auto",
                   help="Resolve the calibrated stereo X baseline against the head-rigid Y baseline.")
    return p.parse_args()


def dominant_candidates(path: Path) -> dict[int, dict[str, tuple[np.ndarray, float]]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        people = row.get("candidates", [])
        if not people:
            continue
        person = max(people, key=lambda q: (q["box_xyxy"][2]-q["box_xyxy"][0]) *
                     (q["box_xyxy"][3]-q["box_xyxy"][1]) * (.7+.3*float(q["box_confidence"])))
        result[int(row["frame_index"])] = {
            name: (np.asarray(person["keypoints"][name][:2], np.float64),
                   float(person["keypoints"][name][2])) for name in TARGET_JOINTS
        }
    return result


def smooth_observations(values: dict[int, dict[str, tuple[np.ndarray, float]]], threshold: float
                        ) -> dict[int, dict[str, tuple[np.ndarray, float]]]:
    """Five-frame confidence-weighted median, without filling missing detections."""
    out: dict[int, dict[str, tuple[np.ndarray, float]]] = {}
    last = max(values) if values else -1
    for frame in range(last+1):
        for name in TARGET_JOINTS:
            current = values.get(frame, {}).get(name)
            if current is None or current[1] < threshold:
                continue
            nearby = [values[k][name][0] for k in range(max(0, frame-2), min(last, frame+2)+1)
                      if k in values and name in values[k] and values[k][name][1] >= threshold]
            point = np.median(np.asarray(nearby), axis=0) if nearby else current[0]
            out.setdefault(frame, {})[name] = (point, current[1])
    return out


def fixed_nose(path: Path) -> np.ndarray:
    points = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                p = [float(row["face_nose_u_px"]), float(row["face_nose_v_px"])]
            except (KeyError, ValueError):
                continue
            if np.isfinite(p).all(): points.append(p)
    values = np.asarray(points, np.float64)
    center = np.median(values, axis=0)
    radii = np.linalg.norm(values-center, axis=1)
    med = np.median(radii); mad = np.median(np.abs(radii-med))
    return np.median(values[radii <= med + max(4.0, 4.5*1.4826*mad)], axis=0)


def load_world(path: Path) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            result.setdefault(int(row["sequence"]), {})[row["joint"]] = np.asarray(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], np.float64)
    return result


def unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v/n if n > 1e-7 else None


def main() -> None:
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    intr = load_json(a.head_intrinsics)
    right_name = "CAM_D" if "CAM_D" in intr["cameras"] else "CAM_C"
    cams = {"A": Omni(intr, "CAM_A"), "D": Omni(intr, right_name)}
    stereo = np.asarray(intr["stereo_extrinsics"]["T_CAM_D_CAM_A"], np.float64)
    r_da, t_da = stereo[:3, :3], stereo[:3, 3]
    origin_d_a = -r_da.T @ t_da

    rigid = load_json(a.head_rigid)
    file_c2r = {}
    for cam, side in (("A", "left"), ("D", "right")):
        c2r = np.asarray(rigid["cameras"][side]["T_rigid_camera"], np.float64)
        c2r[:3, 3] /= 1000.0
        file_c2r[cam] = c2r

    # Kalibr observes the D optical center mainly along -CAM_A X, while the
    # physical/theoretical head rig places D mainly along -rigid Y.  The old
    # file rotation (x/y unchanged, z reflected) cannot map those baselines
    # and makes the pose optimizer absorb an artificial ~90/180 degree turn.
    # The established head projection convention swaps x/y and flips z; use
    # that proper rotation when the file baseline is inconsistent.
    file_delta = file_c2r["D"][:3, 3] - file_c2r["A"][:3, 3]
    predicted_delta = file_c2r["A"][:3, :3] @ origin_d_a
    baseline_cos = float(np.dot(file_delta, predicted_delta) /
                         max(np.linalg.norm(file_delta)*np.linalg.norm(predicted_delta), 1e-12))
    basis_mode = a.head_axis_basis
    if basis_mode == "auto":
        basis_mode = "xy_swap" if baseline_cos < .70 else "file"
    if basis_mode == "xy_swap":
        r_ra = np.asarray([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        c2r_a = np.eye(4); c2r_a[:3, :3] = r_ra; c2r_a[:3, 3] = file_c2r["A"][:3, 3]
        c2r_d = np.eye(4); c2r_d[:3, :3] = r_ra @ r_da.T
        c2r_d[:3, 3] = file_c2r["D"][:3, 3]
        chosen = {"A": c2r_a, "D": c2r_d}
    else:
        chosen = file_c2r
    transforms = {cam: {"c2r": c2r, "r2c": np.linalg.inv(c2r)}
                  for cam, c2r in chosen.items()}

    detections = {
        "A": smooth_observations(dominant_candidates(a.head_a_candidates), a.confidence),
        "D": smooth_observations(dominant_candidates(a.head_d_candidates), a.confidence),
    }
    raw_target: dict[int, dict[str, tuple[np.ndarray, float]]] = {}
    raw_target_meta = []
    for seq in sorted(set(detections["A"]) & set(detections["D"])):
        for name in TARGET_JOINTS:
            if name not in detections["A"][seq] or name not in detections["D"][seq]: continue
            uv_a, conf_a = detections["A"][seq][name]; uv_d, conf_d = detections["D"][seq][name]
            ans = closest_rays(np.zeros(3), cams["A"].ray(uv_a), origin_d_a,
                               r_da.T @ cams["D"].ray(uv_d))
            if ans is None or ans[2] <= .04 or ans[3] <= .04: continue
            point_a, gap, depth_a, depth_d = ans
            if gap*1000.0 > a.max_ray_gap_mm or not .04 < depth_a < 2.0 or not .04 < depth_d < 2.0: continue
            point_rigid = (transforms["A"]["c2r"] @ np.r_[point_a, 1.0])[:3]
            quality = float(min(conf_a, conf_d) * np.exp(-(gap*1000.0)/30.0))
            raw_target.setdefault(seq, {})[name] = (point_rigid, quality)
            raw_target_meta.append({"sequence":seq, "joint":name, "head_A_u":uv_a[0], "head_A_v":uv_a[1],
                                "head_D_u":uv_d[0], "head_D_v":uv_d[1], "confidence_A":conf_a,
                                "confidence_D":conf_d, "ray_gap_mm":gap*1000.0,
                                "selection_quality":quality,
                                "target_ch07_x_m":point_rigid[0], "target_ch07_y_m":point_rigid[1],
                                "target_ch07_z_m":point_rigid[2]})

    # Do not require all four body joints.  Keep the most reliable combination
    # per frame, with a small bonus for subsets that form the requested segments.
    target: dict[int, dict[str, np.ndarray]] = {}
    selected_pairs = set()
    frame_quality = {}
    import itertools
    for seq, candidates in raw_target.items():
        names = list(candidates)
        best_names, best_score = (), -np.inf
        for size in range(1, min(3, len(names))+1):
            for subset in itertools.combinations(names, size):
                formed = sum(begin in subset and end in subset for begin, end in SEGMENTS)
                score = sum(candidates[n][1] for n in subset) + .08*formed
                if score > best_score:
                    best_names, best_score = subset, score
        target[seq] = {name:candidates[name][0] for name in best_names}
        frame_quality[seq] = best_score
        selected_pairs.update((seq, name) for name in best_names)
    target_meta = [row | {"selected_for_optimization":int((row["sequence"],row["joint"]) in selected_pairs)}
                   for row in raw_target_meta]

    fixed_uv = {"A": fixed_nose(a.head_a_nose_csv), "D": fixed_nose(a.head_d_nose_csv)}
    nose_ans = closest_rays(np.zeros(3), cams["A"].ray(fixed_uv["A"]), origin_d_a,
                            r_da.T @ cams["D"].ray(fixed_uv["D"]))
    if nose_ans is None: raise RuntimeError("Nose stereo triangulation failed")
    nose_target = (transforms["A"]["c2r"] @ np.r_[nose_ans[0], 1.0])[:3]

    world = load_world(a.world_csv)
    with a.aligned.open("r", encoding="utf-8-sig", newline="") as f: aligned = list(csv.DictReader(f))
    baseline: dict[int, dict[str, np.ndarray]] = {}
    poses = {}
    # Recover the already approved nose-only translation from the head nose anchor.
    nose_gt_ch07 = np.asarray([0.0, -.015, -.125])
    nose_translation = nose_target - nose_gt_ch07
    for seq, joints in world.items():
        idx = seq+a.ch07_event_offset
        if not 0 <= idx < len(aligned): continue
        row = aligned[idx]
        rot = qrot([float(row[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
        trans = np.asarray([float(row[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        poses[seq] = (rot, trans)
        baseline[seq] = {name: rot.T@(point-trans)+nose_translation for name, point in joints.items()}

    all_position_obs = []
    all_direction_obs = []
    for seq, targets in target.items():
        sources = baseline.get(seq, {})
        for name, dst in targets.items():
            if name in sources: all_position_obs.append((seq, name, sources[name], dst))
        for begin, end in SEGMENTS:
            if begin in targets and end in targets and begin in sources and end in sources:
                src_dir = unit(sources[end]-sources[begin]); dst_dir = unit(targets[end]-targets[begin])
                if src_dir is not None and dst_dir is not None:
                    all_direction_obs.append((seq, f"{begin}->{end}", src_dir, dst_dir))

    # A sparse, high-quality and temporally distributed set estimates the
    # stable global offset; it is intentionally not fitted from every frame.
    eligible = [seq for seq in target if len(target[seq]) >= 2 and
                any(b in target[seq] and e in target[seq] for b,e in SEGMENTS)]
    selected_global_sequences=[]
    for seq in sorted(eligible,key=lambda s:frame_quality[s],reverse=True):
        if all(abs(seq-old)>=5 for old in selected_global_sequences):
            selected_global_sequences.append(seq)
        if len(selected_global_sequences)>=24: break
    selected_global_sequences=set(selected_global_sequences)
    position_obs=[x for x in all_position_obs if x[0] in selected_global_sequences]
    direction_obs=[x for x in all_direction_obs if x[0] in selected_global_sequences]

    # The nose fixes the pivot/translation without overwhelming the many moving-joint samples.
    nose_sources = [j["nose"] for j in baseline.values() if "nose" in j]
    nose_source = np.median(np.asarray(nose_sources), axis=0)

    def residual(x: np.ndarray, details: bool=False):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        # The approved RTMW stereo nose is the exact translation pivot.  This
        # prevents noisier shoulder/elbow stereo depths from moving the head.
        t = nose_target - R@nose_source
        values = []
        for _, _, src, dst in position_obs: values.extend((R@src+t-dst)*a.position_weight)
        for _, _, src, dst in direction_obs: values.extend((R@src-dst)*a.direction_length_m)
        values.extend(x[:3]*a.rotation_prior_m_per_rad)
        return np.asarray(values)

    fit = least_squares(residual, np.zeros(3), loss="soft_l1", f_scale=.025, max_nfev=2000)
    fit_R = Rotation.from_rotvec(fit.x[:3]).as_matrix(); fit_t = nose_target-fit_R@nose_source
    optimized = {seq:{name:fit_R@p+fit_t for name,p in joints.items()} for seq,joints in baseline.items()}

    # Per-frame adaptive alternative.  It still rotates about the fixed RTMW
    # nose, so only upper-body orientation is adaptive.  Raw frame solutions
    # are strongly de-jittered with a median + zero-phase Savitzky-Golay pass.
    seqs = sorted(baseline)
    raw_rotvecs = np.full((len(seqs), 3), np.nan, np.float64)
    for si, seq in enumerate(seqs):
        frame_targets = target.get(seq, {})
        frame_sources = baseline[seq]
        pos = [(frame_sources[n], p) for n,p in frame_targets.items() if n in frame_sources]
        dirs = []
        for begin,end in SEGMENTS:
            if begin in frame_targets and end in frame_targets and begin in frame_sources and end in frame_sources:
                sd=unit(frame_sources[end]-frame_sources[begin]); td=unit(frame_targets[end]-frame_targets[begin])
                if sd is not None and td is not None: dirs.append((sd,td))
        if len(pos) < 2 and not dirs: continue
        def frame_residual(x):
            R=Rotation.from_rotvec(x).as_matrix(); t=nose_target-R@nose_source
            vals=[]
            for src,dst in pos: vals.extend(R@src+t-dst)
            for src,dst in dirs: vals.extend((R@src-dst)*a.direction_length_m)
            # Weakly prefer the global result when the current frame is sparse.
            vals.extend((x-fit.x[:3])*(.035 if len(pos)>=3 else .060))
            return np.asarray(vals)
        local=least_squares(frame_residual,fit.x[:3],loss="soft_l1",f_scale=.02,max_nfev=300)
        if np.linalg.norm(local.x-fit.x[:3]) <= np.deg2rad(30): raw_rotvecs[si]=local.x
    valid=np.all(np.isfinite(raw_rotvecs),axis=1); axis=np.arange(len(seqs))
    adaptive_rotvecs=np.empty_like(raw_rotvecs)
    for d in range(3):
        filled=np.interp(axis,axis[valid],raw_rotvecs[valid,d]) if valid.sum()>=2 else np.full(len(seqs),fit.x[d])
        median=medfilt(filled,kernel_size=9)
        adaptive_rotvecs[:,d]=savgol_filter(median,31,2,mode="interp") if len(seqs)>=31 else median
    adaptive={}
    for si,seq in enumerate(seqs):
        R=Rotation.from_rotvec(adaptive_rotvecs[si]).as_matrix(); t=nose_target-R@nose_source
        adaptive[seq]={name:R@p+t for name,p in baseline[seq].items()}

    def errors(points):
        vals=[]
        for seq,name,_,dst in all_position_obs:
            if name in points.get(seq,{}): vals.append(np.linalg.norm(points[seq][name]-dst)*1000)
        return np.asarray(vals)
    before_err, after_err = errors(baseline), errors(optimized)
    dir_before=[]; dir_after=[]
    for seq,_,src,dst in all_direction_obs:
        dir_before.append(np.degrees(np.arccos(np.clip(np.dot(src,dst),-1,1))))
        dir_after.append(np.degrees(np.arccos(np.clip(np.dot(fit_R@src,dst),-1,1))))

    # Export both CH07 comparison data and optimized world skeleton.
    viewer_frames=[]; world_rows=[]; adaptive_world_rows=[]
    for seq in sorted(baseline):
        raw=baseline[seq]; opt=optimized[seq]; rot,trans=poses[seq]
        viewer_frames.append({"sequence":seq,
            "before":[raw[n].round(5).tolist() if n in raw else None for n in NAMES],
            "after":[opt[n].round(5).tolist() if n in opt else None for n in NAMES],
            "adaptive":[adaptive[seq][n].round(5).tolist() if n in adaptive[seq] else None for n in NAMES],
            "targets":{n:p.round(5).tolist() for n,p in target.get(seq,{}).items()},
            "valid_gt":len(target.get(seq,{}))})
        for name,p in opt.items():
            pw=rot@p+trans
            world_rows.append({"sequence":seq,"joint":name,"x_m":pw[0],"y_m":pw[1],"z_m":pw[2]})
        for name,p in adaptive[seq].items():
            pw=rot@p+trans
            adaptive_world_rows.append({"sequence":seq,"joint":name,"x_m":pw[0],"y_m":pw[1],"z_m":pw[2]})
    viewer_data={"joint_names":list(NAMES),"bones":[[NAMES.index(a),NAMES.index(b)] for a,b in BONES],
                 "target_names":list(TARGET_JOINTS),"frames":viewer_frames}
    (a.output_dir/"fivepoint_pose_comparison.json").write_text(json.dumps(viewer_data,separators=(",",":")),encoding="utf-8")
    with (a.output_dir/"fivepoint_pose_optimized_world.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(world_rows[0]));w.writeheader();w.writerows(world_rows)
    with (a.output_dir/"fivepoint_pose_adaptive_world.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(adaptive_world_rows[0]));w.writeheader();w.writerows(adaptive_world_rows)
    with (a.output_dir/"head_shoulder_elbow_stereo_gt.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(target_meta[0]));w.writeheader();w.writerows(target_meta)

    # Project before/after and render both head views.
    projected={"A":{},"D":{}}; projected_before={"A":{},"D":{}}; projected_adaptive={"A":{},"D":{}}
    for seq,joints in optimized.items():
        for cam in "AD":
            for name,p in joints.items():
                uv=cams[cam].project((transforms[cam]["r2c"]@np.r_[p,1])[:3])
                if uv is not None: projected[cam].setdefault(seq,{})[name]=np.asarray(uv)
            for name,p in baseline[seq].items():
                uv=cams[cam].project((transforms[cam]["r2c"]@np.r_[p,1])[:3])
                if uv is not None: projected_before[cam].setdefault(seq,{})[name]=np.asarray(uv)
            for name,p in adaptive[seq].items():
                uv=cams[cam].project((transforms[cam]["r2c"]@np.r_[p,1])[:3])
                if uv is not None: projected_adaptive[cam].setdefault(seq,{})[name]=np.asarray(uv)

    for cam,video in (("A",a.head_a),("D",a.head_d)):
        cap=cv2.VideoCapture(str(video)); out=writer(a.output_dir/f"head_CAM_{cam}_fivepoint_pose_projection.mp4",(1920,1200),50)
        seq=0
        while True:
            ok,img=cap.read()
            if not ok:break
            draw_pose(img,projected_before[cam].get(seq,{}),(255,255,0),"",body_only=True)
            draw_pose(img,projected[cam].get(seq,{}),(0,255,255),
                      f"five-point SE3 -> HEAD_{cam} seq={seq}",body_only=True)
            draw_pose(img,projected_adaptive[cam].get(seq,{}),(0,128,255),"",body_only=True)
            for name,(uv,score) in detections[cam].get(seq,{}).items():
                cv2.circle(img,tuple(np.rint(uv).astype(int)),7,(255,0,255),-1,cv2.LINE_AA)
            cv2.circle(img,tuple(np.rint(fixed_uv[cam]).astype(int)),8,(0,255,0),-1,cv2.LINE_AA)
            cv2.putText(img,"cyan=before  yellow=global  orange=adaptive  magenta=head YOLO GT  green=RTMW nose",(24,76),
                        cv2.FONT_HERSHEY_SIMPLEX,.68,(0,255,255),2,cv2.LINE_AA)
            out.write(img);seq+=1
        cap.release();out.release()
    ca=cv2.VideoCapture(str(a.output_dir/"head_CAM_A_fivepoint_pose_projection.mp4"))
    cd=cv2.VideoCapture(str(a.output_dir/"head_CAM_D_fivepoint_pose_projection.mp4"))
    out=writer(a.output_dir/"head_stereo_fivepoint_pose_projection.mp4",(1920,600),50)
    while True:
        oka,ia=ca.read();okd,id_=cd.read()
        if not oka or not okd:break
        out.write(np.hstack([cv2.resize(ia,(960,600)),cv2.resize(id_,(960,600))]))
    ca.release();cd.release();out.release()

    adaptive_err=errors(adaptive)
    raw_jitter=np.degrees(np.linalg.norm(np.diff(raw_rotvecs[valid],axis=0),axis=1)) if valid.sum()>1 else np.asarray([])
    smooth_jitter=np.degrees(np.linalg.norm(np.diff(adaptive_rotvecs,axis=0),axis=1))
    report={"method":"global robust CH07 SE(3) plus temporally smoothed per-frame adaptive alternative",
            "head_pose_model":"YOLO11x-pose (existing head-view candidates)",
            "nose_model":"RTMW WholeBody face landmark 53, robust fixed A/D point",
            "confidence_threshold":a.confidence,"max_ray_gap_mm":a.max_ray_gap_mm,
            "selection_policy":"nose always + best confidence/geometry combination, maximum 3 shoulder/elbow joints per frame",
            "triangulated_joint_counts":{n:sum(n in q for q in target.values()) for n in TARGET_JOINTS},
            "global_fit_sequences":sorted(selected_global_sequences),
            "position_observations_all":len(all_position_obs),"direction_observations_all":len(all_direction_obs),
            "position_observations_global_fit":len(position_obs),"direction_observations_global_fit":len(direction_obs),
            "rotation_vector_rad":fit.x[:3].tolist(),"rotation_angle_deg":float(np.linalg.norm(fit.x[:3])*180/np.pi),
            "translation_m":fit_t.tolist(),"nose_only_translation_m":nose_translation.tolist(),
            "joint_target_error_mm":{"before_median":float(np.median(before_err)),"before_p90":float(np.percentile(before_err,90)),
                                     "global_median":float(np.median(after_err)),"global_p90":float(np.percentile(after_err,90)),
                                     "adaptive_median":float(np.median(adaptive_err)),"adaptive_p90":float(np.percentile(adaptive_err,90))},
            "segment_direction_error_deg":{"before_median":float(np.median(dir_before)),"before_p90":float(np.percentile(dir_before,90)),
                                           "after_median":float(np.median(dir_after)),"after_p90":float(np.percentile(dir_after,90))},
            "adaptive_rotation":{"solved_frames":int(valid.sum()),"raw_step_median_deg":float(np.median(raw_jitter)) if len(raw_jitter) else None,
                                   "smoothed_step_median_deg":float(np.median(smooth_jitter)),
                                   "smoothed_step_p90_deg":float(np.percentile(smooth_jitter,90))},
            "head_axis_basis":basis_mode,"file_stereo_baseline_cosine":baseline_cos,
            "policy":"global and adaptive are rigid corrections around the fixed RTMW nose; no bone length or per-joint deformation"}
    (a.output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__": main()
