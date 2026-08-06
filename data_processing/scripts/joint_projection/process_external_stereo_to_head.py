#!/usr/bin/env python3
"""Select stereo poses, triangulate in CH01, and project through CH07 to module01."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
EDGES = ((0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--left-candidates", type=Path, required=True)
    p.add_argument("--right-candidates", type=Path, required=True)
    p.add_argument("--head-a-mp4", type=Path, required=True)
    p.add_argument("--head-d-mp4", type=Path, required=True)
    p.add_argument("--head-a-map", type=Path, required=True)
    p.add_argument("--head-d-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mocap-tools", type=Path, default=Path("/home/gaoweijian/tools"))
    p.add_argument("--refinement", type=Path)
    p.add_argument("--external-stereo-yaml", type=Path)
    p.add_argument("--external-a-mp4", type=Path)
    p.add_argument("--external-d-mp4", type=Path)
    p.add_argument("--skip-videos", action="store_true")
    p.add_argument("--skip-external-video", action="store_true")
    p.add_argument("--skip-head-videos", action="store_true")
    p.add_argument("--kp-conf", type=float, default=.16)
    p.add_argument("--ch07-event-offset", type=int, default=0,
                   help="Use CH07 pose from event sequence s + this offset; camera frames stay at s")
    p.add_argument("--ch07-aligned", type=Path,
                   help="Optional longer aligned table used only as CH07 temporal context")
    return p.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> dict[int, dict]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[int(row["frame_index"])] = row
    return out


def qrot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


class Omni:
    def __init__(self, doc: dict, camera: str):
        c = doc["cameras"][camera]
        self.xi, self.fx, self.fy, self.cx, self.cy = map(float, c["intrinsics"])
        self.d = np.asarray(c["distortion_coeffs"], np.float64)
        self.k = np.asarray(c["K"], np.float64)

    def ray(self, uv) -> np.ndarray:
        pt = np.asarray(uv, np.float64).reshape(1, 1, 2)
        xy = cv2.undistortPoints(pt, self.k, self.d).reshape(2)
        r2 = float(xy @ xy)
        root = max(0.0, 1.0 + (1.0-self.xi*self.xi)*r2)
        lam = (self.xi + math.sqrt(root)) / (1.0+r2)
        ray = np.array([lam*xy[0], lam*xy[1], lam-self.xi])
        return ray / max(np.linalg.norm(ray), 1e-12)

    def project(self, p) -> tuple[float, float] | None:
        x, y, z = map(float, p)
        den = z + self.xi * math.sqrt(x*x+y*y+z*z)
        if den <= 1e-9:
            return None
        xn, yn = x/den, y/den
        k1,k2,p1,p2 = self.d
        r2 = xn*xn+yn*yn
        radial = 1+k1*r2+k2*r2*r2
        xd = xn*radial + 2*p1*xn*yn + p2*(r2+2*xn*xn)
        yd = yn*radial + p1*(r2+2*yn*yn) + 2*p2*xn*yn
        return self.fx*xd+self.cx, self.fy*yd+self.cy


def closest_rays(o1,d1,o2,d2):
    w=o1-o2; a=float(d1@d1); b=float(d1@d2); c=float(d2@d2)
    d=float(d1@w); e=float(d2@w); den=a*c-b*b
    if abs(den)<1e-9: return None
    s=(b*e-c*d)/den; t=(a*e-b*d)/den
    p1=o1+s*d1; p2=o2+t*d2
    return (p1+p2)/2, float(np.linalg.norm(p1-p2)), s, t


def ext_frame_meta(path: Path, camera: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows=[r for r in csv.DictReader(f) if r["camera"]==camera]
    rows.sort(key=lambda r:int(r["frame_index"]))
    return rows


def head_frame_meta(path: Path, camera: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows=[r for r in csv.DictReader(f) if r["module"]=="1" and r["camera"]==camera]
    return rows


def pair_candidates(left, right, cams, ext_doc, prev_center, kp_conf):
    best=None
    for li,l in enumerate(left):
        for ri,r in enumerate(right):
            pts={}; gaps=[]; qualities=[]
            for name in NAMES:
                kl=l["keypoints"][name]; kr=r["keypoints"][name]
                if kl[2]<kp_conf or kr[2]<kp_conf: continue
                dl=ext_doc["Rleft"] @ cams["left"].ray(kl[:2])
                dr=ext_doc["Rright"] @ cams["right"].ray(kr[:2])
                ans=closest_rays(ext_doc["oleft"],dl,ext_doc["oright"],dr)
                if ans is None: continue
                p,g,s,t=ans
                if s<=0 or t<=0 or g>.35: continue
                pl=ext_doc["R_ch01_left"].T@(p-ext_doc["oleft"])
                pr=ext_doc["R_right_left"]@pl+ext_doc["t_right_left"]
                ul=cams["left"].project(pl); ur=cams["right"].project(pr)
                reproj_l=float(np.linalg.norm(np.asarray(ul)-np.asarray(kl[:2]))) if ul else math.inf
                reproj_r=float(np.linalg.norm(np.asarray(ur)-np.asarray(kr[:2]))) if ur else math.inf
                if max(reproj_l,reproj_r)>45.0: continue
                pts[name]={"xyz":p, "ray_gap_m":g, "confidence":math.sqrt(kl[2]*kr[2]),
                           "left_uv":kl[:2], "right_uv":kr[:2],
                           "left_reprojection_error_px":reproj_l,
                           "right_reprojection_error_px":reproj_r}
                gaps.append(g); qualities.append(math.sqrt(kl[2]*kr[2]))
            if len(pts)<4: continue
            center=np.mean([v["xyz"] for v in pts.values()],axis=0)
            temporal=0 if prev_center is None else min(float(np.linalg.norm(center-prev_center)),2.0)
            gap=float(np.median(gaps)); quality=float(np.mean(qualities))
            area_l=max(0,(l["box_xyxy"][2]-l["box_xyxy"][0])*(l["box_xyxy"][3]-l["box_xyxy"][1]))
            area_r=max(0,(r["box_xyxy"][2]-r["box_xyxy"][0])*(r["box_xyxy"][3]-r["box_xyxy"][1]))
            score=len(pts)*2.0 + quality*4.0 + math.log1p(math.sqrt(area_l*area_r))/5.0 - gap*35.0 - temporal*3.0
            if best is None or score>best[0]: best=(score,li,ri,pts,center,gap)
    return best


FILTER_2D = {
    # Offline zero-phase filtering: torso anchors are deliberately much stronger.
    "left_shoulder": (5, .12), "right_shoulder": (5, .12),
    "left_hip": (5, .12), "right_hip": (5, .12),
    "left_elbow": (4, .16), "right_elbow": (4, .16),
    "left_knee": (3, .22), "right_knee": (3, .22),
    "left_wrist": (3, .22), "right_wrist": (3, .22),
    "left_ankle": (2, .34), "right_ankle": (2, .34),
}


def zero_phase_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    forward = values.copy()
    for i in range(1, len(values)):
        forward[i] = alpha * values[i] + (1.0-alpha) * forward[i-1]
    backward = values.copy()
    for i in range(len(values)-2, -1, -1):
        backward[i] = alpha * values[i] + (1.0-alpha) * backward[i+1]
    return (forward + backward) * .5


def robust_filter_2d(values: np.ndarray, radius: int, alpha: float) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    median = np.empty_like(values)
    for i in range(len(values)):
        median[i] = np.median(values[max(0, i-radius):min(len(values), i+radius+1)], axis=0)
    residual = np.linalg.norm(values-median, axis=1)
    scale = max(float(np.median(residual))*3.5, 4.0)
    weight = np.minimum(1.0, scale/np.maximum(residual, 1e-9))[:, None]
    cleaned = median + (values-median)*weight
    return zero_phase_ema(cleaned, alpha)


def filter_selected_2d_and_retriangulate(records, cams, ext_doc):
    stats = {}
    for name in NAMES:
        radius, alpha = FILTER_2D.get(name, (2, .27))
        for side in ("left", "right"):
            samples = [(i, int(r["sequence"]), np.asarray(r["joints"][name][f"{side}_uv"], np.float64))
                       for i, r in enumerate(records) if name in r["joints"]]
            filtered = {}
            raw_steps, filtered_steps = [], []
            start = 0
            while start < len(samples):
                end = start + 1
                while end < len(samples) and samples[end][1]-samples[end-1][1] <= 3:
                    end += 1
                segment = samples[start:end]
                values = np.asarray([item[2] for item in segment])
                smooth = robust_filter_2d(values, radius, alpha)
                for (record_i, _, raw_uv), uv in zip(segment, smooth):
                    joint = records[record_i]["joints"][name]
                    joint[f"{side}_uv_raw"] = raw_uv.copy()
                    joint[f"{side}_uv"] = uv.copy()
                    filtered[record_i] = uv
                if len(values) > 1:
                    raw_steps.extend(np.linalg.norm(np.diff(values, axis=0), axis=1))
                    filtered_steps.extend(np.linalg.norm(np.diff(smooth, axis=0), axis=1))
                start = end
            stats[f"{side}_{name}"] = {
                "samples": len(samples), "median_step_raw_px": float(np.median(raw_steps)) if raw_steps else 0.0,
                "median_step_filtered_px": float(np.median(filtered_steps)) if filtered_steps else 0.0,
                "median_radius_frames": radius, "zero_phase_ema_alpha": alpha,
            }

    # Discard the old depths and triangulate again using only the filtered 2D rays.
    for record in records:
        for name in list(record["joints"]):
            joint = record["joints"][name]
            left_uv = np.asarray(joint["left_uv"], np.float64)
            right_uv = np.asarray(joint["right_uv"], np.float64)
            dl = ext_doc["Rleft"] @ cams["left"].ray(left_uv)
            dr = ext_doc["Rright"] @ cams["right"].ray(right_uv)
            answer = closest_rays(ext_doc["oleft"], dl, ext_doc["oright"], dr)
            if answer is None:
                del record["joints"][name]
                continue
            point, gap, left_depth, right_depth = answer
            if left_depth <= 0 or right_depth <= 0 or gap > .35:
                del record["joints"][name]
                continue
            point_left = ext_doc["R_ch01_left"].T @ (point-ext_doc["oleft"])
            point_right = ext_doc["R_right_left"] @ point_left + ext_doc["t_right_left"]
            reproj_left = cams["left"].project(point_left)
            reproj_right = cams["right"].project(point_right)
            if reproj_left is None or reproj_right is None:
                del record["joints"][name]
                continue
            joint["xyz"] = point
            joint["ray_gap_m"] = float(gap)
            joint["left_reprojection_error_px"] = float(np.linalg.norm(np.asarray(reproj_left)-left_uv))
            joint["right_reprojection_error_px"] = float(np.linalg.norm(np.asarray(reproj_right)-right_uv))
    return stats


def smooth_tracks(records: list[dict]) -> None:
    # Robust five-frame temporal median, only across adjacent trigger sequences.
    for j,name in enumerate(NAMES):
        raw=[r["joints"].get(name,{}).get("xyz") for r in records]
        for i,r in enumerate(records):
            if raw[i] is None: continue
            vals=[]
            for k in range(max(0,i-2),min(len(records),i+3)):
                if raw[k] is not None and abs(records[k]["sequence"]-r["sequence"])<=2:
                    vals.append(raw[k])
            if vals: r["joints"][name]["xyz_smooth"]=np.median(np.asarray(vals),axis=0)


def draw_pose(img, uv, color=(0,255,255), title="", body_only=False):
    for a,b in EDGES:
        if body_only and (a < 5 or b < 5):
            continue
        pa=uv.get(NAMES[a]); pb=uv.get(NAMES[b])
        if pa is not None and pb is not None:
            cv2.line(img,tuple(np.rint(pa).astype(int)),tuple(np.rint(pb).astype(int)),color,3,cv2.LINE_AA)
    for name,p in uv.items():
        if not body_only or name == "nose" or name in NAMES[5:]:
            cv2.circle(img,tuple(np.rint(p).astype(int)),5,(0,0,255),-1,cv2.LINE_AA)
    if title: cv2.putText(img,title,(24,42),cv2.FONT_HERSHEY_SIMPLEX,.8,color,2,cv2.LINE_AA)


def writer(path: Path, size, fps=50):
    return cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,size)


def main() -> None:
    a=parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    candidate_stem = a.left_candidates.stem.lower()
    pose_model = ("RTMPose-X (Body7/COCO 17 keypoints)" if "rtmpose" in candidate_stem else
                  "YOLO11x-pose (COCO 17 keypoints)" if "_x" in candidate_stem else
                  "YOLO11n-pose (COCO 17 keypoints)")
    ext_intr=load_json(a.calib/"handle_ac_intrinsics_kalibr_omni_1920x1200_20260729.json")
    head_intr=load_json(a.calib/"head_intrinsics_kalibr_omni_1920x1200.json")
    ext_rigid=load_json(a.calib/"external_stereo_rigid_k_extrinsics.json")
    head_rigid=load_json(a.calib/"head_stereo_rigid_extrinsics.json")
    ext_right_name = "CAM_D" if "CAM_D" in ext_intr["cameras"] else "CAM_C"
    head_right_name = "CAM_D" if "CAM_D" in head_intr["cameras"] else "CAM_C"
    cams={"left":Omni(ext_intr,"CAM_A"),"right":Omni(ext_intr,ext_right_name)}
    hcams={"A":Omni(head_intr,"CAM_A"),"D":Omni(head_intr,head_right_name)}
    r_ch01_left=np.asarray(ext_rigid["cameras"]["left"]["R_rigid_camera"],np.float64)
    o_ch01_left=np.asarray(ext_rigid["cameras"]["left"]["p_rigid_camera_mm"],np.float64)/1000
    if a.external_stereo_yaml:
        stereo_doc=yaml.safe_load(a.external_stereo_yaml.read_text(encoding="utf-8"))
        t_right_left=np.asarray(stereo_doc["cam1"]["T_cn_cnm1"],np.float64)
        r_rl=t_right_left[:3,:3]; trl=t_right_left[:3,3]
        # Kalibr: p_right = R_rl p_left + t_rl.  Express both rays in
        # the left camera, then map them through the measured CH01 mount.
        ext_geom={
            "Rleft":r_ch01_left,
            "Rright":r_ch01_left@r_rl.T,
            "oleft":o_ch01_left,
            "oright":o_ch01_left+r_ch01_left@(-r_rl.T@trl),
            "R_ch01_left":r_ch01_left,
            "R_right_left":r_rl,
            "t_right_left":trl,
        }
        stereo_source=str(a.external_stereo_yaml)
    else:
        ext_geom={
            "Rleft":r_ch01_left,
            "Rright":np.asarray(ext_rigid["cameras"]["right"]["R_rigid_camera"]),
            "oleft":o_ch01_left,
            "oright":np.asarray(ext_rigid["cameras"]["right"]["p_rigid_camera_mm"])/1000,
            "R_ch01_left":r_ch01_left,
            "R_right_left":np.eye(3),
            "t_right_left":np.array([.210,0,0]),
        }
        stereo_source="theoretical rigid measurements"

    left=load_jsonl(a.left_candidates); right=load_jsonl(a.right_candidates)
    lm=ext_frame_meta(a.input/"external_timestamps.csv","CAM_A")
    rm=ext_frame_meta(a.input/"external_timestamps.csv","CAM_D")
    lf={int(r["sequence"]):int(r["frame_index"]) for r in lm}
    rf={int(r["sequence"]):int(r["frame_index"]) for r in rm}
    records=[]; prev=None
    for seq in sorted(set(lf)&set(rf)):
        lc=left.get(lf[seq],{}).get("candidates",[]); rc=right.get(rf[seq],{}).get("candidates",[])
        chosen=pair_candidates(lc,rc,cams,ext_geom,prev,a.kp_conf)
        if chosen is None: continue
        score,li,ri,pts,prev,gap=chosen
        records.append({"sequence":seq,"left_frame":lf[seq],"right_frame":rf[seq],
                        "left_candidate":li,"right_candidate":ri,"pair_score":score,
                        "median_ray_gap_m":gap,"joints":pts})
    filter_2d_stats = filter_selected_2d_and_retriangulate(records, cams, ext_geom)

    # Load CH01 from ABX2 and CH07/timing from the aligned table.
    sys.path.insert(0,str(a.mocap_tools))
    from export_abx2_mocap_rigid_csv import read_abx2_header,pwr_map_for_sensors,extract_ch3_rigids
    abx=a.input/"001.abx2"; info,cfg=read_abx2_header(abx)
    ch01=extract_ch3_rigids(abx,pwr_map_for_sensors(cfg,(301,)),60.0)[301]
    with (a.input/"aligned_50hz.csv").open("r",encoding="utf-8-sig",newline="") as f:
        aligned=list(csv.DictReader(f))
    # The aligned file was reindexed; recover original trigger sequence by exact external timestamp.
    ts_to_seq={int(r["exposure_end_device_timestamp_us"]):int(r["sequence"]) for r in lm}
    aligned_by_seq={}
    for row in aligned:
        try:
            raw=ts_to_seq.get(int(float(row["external_CAM_A_exposure_end_device_timestamp_us"])))
        except (TypeError,ValueError):
            raw=None
        if raw is not None: aligned_by_seq[raw]=row
    if a.ch07_aligned:
        with a.ch07_aligned.open("r",encoding="utf-8-sig",newline="") as f:
            ch07_rows=list(csv.DictReader(f))
        ch07_by_seq={i:row for i,row in enumerate(ch07_rows)}
    else:
        ch07_by_seq=aligned_by_seq

    head_T={}
    for key,side in (("A","left"),("D","right")):
        mat=np.asarray(head_rigid["cameras"][side]["T_camera_rigid"],np.float64)
        mat[:3,3]/=1000.0; head_T[key]=mat
    refinement=load_json(a.refinement) if a.refinement else None
    if refinement:
        refined_world_R=np.asarray(refinement["R_world_ch01"],np.float64)
        refined_world_t=np.asarray(refinement["t_world_ch01_m"],np.float64)
        refined_ch07_R=np.asarray(refinement["R_ch07_axis_correction"],np.float64)
        refined_ch07_t=np.asarray(refinement["t_ch07_axis_correction_m"],np.float64)
    projections={"A":{},"D":{}}
    rows3=[]
    for rec in records:
        seq=rec["sequence"]; ar=aligned_by_seq.get(seq)
        if ar is None: continue
        mi=int(float(ar["mocap_frame_index"])); c1=ch01[mi]
        R01=qrot([c1["qw"],c1["qx"],c1["qy"],c1["qz"]]); t01=np.array([c1["x"],c1["y"],c1["z"]])
        ch07_ar=ch07_by_seq.get(seq+a.ch07_event_offset)
        if ch07_ar is None: continue
        R07=qrot([float(ch07_ar[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
        t07=np.array([float(ch07_ar[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        for name,j in rec["joints"].items():
            p01=np.asarray(j.get("xyz_smooth",j["xyz"]))
            if refinement:
                # The supplied mechanical frames contain a fixed axis/order
                # ambiguity.  A multi-frame CH07 head anchor resolves CH01 ->
                # world; the remaining module01 correction is Z +90 degrees
                # plus a shared translation offset.
                pw=refined_world_R@p01+refined_world_t
                p07=refined_ch07_R@(R07.T@(pw-t07))+refined_ch07_t
            else:
                pw=R01@p01+t01; p07=R07.T@(pw-t07)
            j["xyz_ch01"]=p01; j["xyz_world"]=pw; j["xyz_ch07"]=p07
            if name == "nose" or name in NAMES[5:]:
                for key in ("A","D"):
                    pc=(head_T[key]@np.r_[p07,1])[:3]; uv=hcams[key].project(pc)
                    if uv is not None and -200<=uv[0]<2120 and -200<=uv[1]<1400:
                        projections[key].setdefault(seq,{})[name]=uv
            rows3.append({"sequence":seq,"mocap_frame_index":mi,"joint":name,
                          "confidence":j["confidence"],"ray_gap_m":j["ray_gap_m"],
                          **{f"ch01_{x}_m":p01[i] for i,x in enumerate("xyz")},
                          **{f"world_{x}_m":pw[i] for i,x in enumerate("xyz")},
                          **{f"ch07_{x}_m":p07[i] for i,x in enumerate("xyz")},
                          **{f"head_A_{x}":projections["A"].get(seq,{}).get(name,(math.nan,math.nan))[i] for i,x in enumerate(("u_px","v_px"))},
                          **{f"head_D_{x}":projections["D"].get(seq,{}).get(name,(math.nan,math.nan))[i] for i,x in enumerate(("u_px","v_px"))}})

    fields=list(rows3[0]) if rows3 else ["sequence"]
    with (a.output/"stereo_3d_and_head_2d.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows3)
    rows2=[]
    for r in records:
        for name,j in r["joints"].items():
            rows2.append({"sequence":r["sequence"],"left_frame_index":r["left_frame"],
                          "right_frame_index":r["right_frame"],"joint":name,
                          "left_u_raw_px":j["left_uv_raw"][0],"left_v_raw_px":j["left_uv_raw"][1],
                          "right_u_raw_px":j["right_uv_raw"][0],"right_v_raw_px":j["right_uv_raw"][1],
                          "left_u_px":j["left_uv"][0],"left_v_px":j["left_uv"][1],
                          "right_u_px":j["right_uv"][0],"right_v_px":j["right_uv"][1],
                          "stereo_confidence":j["confidence"],
                          "left_reprojection_error_px":j["left_reprojection_error_px"],
                          "right_reprojection_error_px":j["right_reprojection_error_px"]})
    if rows2:
        with (a.output/"external_stereo_2d_keypoints.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows2[0])); w.writeheader(); w.writerows(rows2)
    serial=[]
    for r in records:
        serial.append({k:v for k,v in r.items() if k!="joints"} | {"joints":{n:{kk:(vv.tolist() if isinstance(vv,np.ndarray) else vv) for kk,vv in j.items()} for n,j in r["joints"].items()}})
    (a.output/"stereo_selection_and_3d.json").write_text(json.dumps(serial,ensure_ascii=False),encoding="utf-8")

    if a.skip_videos:
        report={"external_common_trigger_frames":len(set(lf)&set(rf)),"stereo_pose_frames":len(records),
                "joint_rows":len(rows3),"external_stereo_source":stereo_source,
                "median_ray_gap_mm":float(np.median([j["ray_gap_m"] for r in records for j in r["joints"].values()])*1000),
                "pre_triangulation_2d_filter":filter_2d_stats,
                "post_triangulation_3d_filter":"disabled",
                "coordinate_chain":"external cameras -> CH01 -> world -> CH07 -> module01 cameras",
                "model":pose_model,"alignment":"shared hardware-trigger sequence"}
        (a.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False)); return

    # Separate upright external-camera reviews: orange raw 2D, green filtered 2D.
    if not a.skip_external_video:
        left_path=a.external_a_mp4 or a.input/"left_CAM_A_1920x1200_50fps.mjpeg"
        right_path=a.external_d_mp4 or a.input/"right_CAM_D_1920x1200_50fps.mjpeg"
        srcL=cv2.VideoCapture(str(left_path)); srcR=cv2.VideoCapture(str(right_path))
        outL=writer(a.output/"external_CAM_A_2d_raw_vs_filtered.mp4",(1920,1200))
        outR=writer(a.output/"external_CAM_D_2d_raw_vs_filtered.mp4",(1920,1200))
        iL=iR=-1; imL=imR=None
        for rec in records:
            while iL < rec["left_frame"]:
                okL,imL=srcL.read(); iL+=1
                if not okL: break
            while iR < rec["right_frame"]:
                okR,imR=srcR.read(); iR+=1
                if not okR: break
            if imL is None or imR is None: break
            imL=cv2.rotate(imL,cv2.ROTATE_180); imR=cv2.rotate(imR,cv2.ROTATE_180)
            rawL={n:(1920-j["left_uv_raw"][0],1200-j["left_uv_raw"][1]) for n,j in rec["joints"].items()}
            rawR={n:(1920-j["right_uv_raw"][0],1200-j["right_uv_raw"][1]) for n,j in rec["joints"].items()}
            filteredL={n:(1920-j["left_uv"][0],1200-j["left_uv"][1]) for n,j in rec["joints"].items()}
            filteredR={n:(1920-j["right_uv"][0],1200-j["right_uv"][1]) for n,j in rec["joints"].items()}
            draw_pose(imL,rawL,(0,128,255)); draw_pose(imR,rawR,(0,128,255))
            draw_pose(imL,filteredL,(0,255,0),f"CAM_A green=filtered orange=raw seq={rec['sequence']}")
            draw_pose(imR,filteredR,(0,255,0),f"CAM_D green=filtered orange=raw seq={rec['sequence']}")
            outL.write(imL); outR.write(imR)
        srcL.release();srcR.release();outL.release();outR.release()

    # Projected videos use decoded-to-capture maps, then timestamp row -> trigger seq.
    for key,mp4,map_path,camname in (() if a.skip_head_videos else (("A",a.head_a_mp4,a.head_a_map,"CAM_A"),("D",a.head_d_mp4,a.head_d_map,"CAM_D"))):
        meta=head_frame_meta(a.input/"head_timestamps.csv",camname)
        cap=cv2.VideoCapture(str(mp4)); decoded_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # ffmpeg error concealment produced one MP4 frame per recorded H.265
        # access unit for this capture.  When counts match, MP4 frame i maps
        # directly to timestamp row i.  The packet-decoder map intentionally
        # omits concealed frames and would accumulate a false temporal lead.
        mapping=list(range(len(meta))) if decoded_count==len(meta) else load_json(map_path)["decoded_to_capture_index"]
        out=writer(a.output/f"head_CAM_{key}_projected_pose.mp4",(1920,1200)); di=0
        while True:
            ok,img=cap.read()
            if not ok: break
            ci=mapping[di] if di<len(mapping) else di
            if ci<len(meta):
                seq=int(meta[ci]["seq"]); draw_pose(img,projections[key].get(seq,{}),(255,255,0),f"external stereo 3D -> head {key} seq={seq}",body_only=True)
            out.write(img); di+=1
        cap.release();out.release()

    report={"external_common_trigger_frames":len(set(lf)&set(rf)),"stereo_pose_frames":len(records),
            "joint_rows":len(rows3),"coordinate_chain":"external cameras -> CH01 -> world -> CH07 -> module01 cameras",
            "external_stereo_source":stereo_source,
            "median_ray_gap_mm":float(np.median([j["ray_gap_m"] for r in records for j in r["joints"].values()])*1000),
            "pre_triangulation_2d_filter":filter_2d_stats,
            "post_triangulation_3d_filter":"disabled",
            "model":pose_model,"alignment":"shared hardware-trigger sequence; aligned row recovered by exact external exposure timestamp",
            "coordinate_refinement":str(a.refinement) if a.refinement else None,
            "ch07_event_offset_frames":a.ch07_event_offset,
            "outputs":["external_CAM_A_2d_raw_vs_filtered.mp4","external_CAM_D_2d_raw_vs_filtered.mp4","head_CAM_A_projected_pose.mp4","head_CAM_D_projected_pose.mp4","external_stereo_2d_keypoints.csv","stereo_3d_and_head_2d.csv"]}
    (a.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__": main()
