#!/usr/bin/env python3
"""Strong zero-phase CH07 filtering for shoulders/hips with fixed torso lengths."""

from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from process_external_stereo_to_head import NAMES, Omni, draw_pose, load_json, qrot, writer


TORSO=("left_shoulder","right_shoulder","left_hip","right_hip")
EDGES=(
    ("left_shoulder","right_shoulder","shoulder_width"),
    ("left_hip","right_hip","hip_width"),
    ("left_shoulder","left_hip","left_torso"),
    ("right_shoulder","right_hip","right_torso"),
    ("left_shoulder","right_hip","diag_lr"),
    ("right_shoulder","left_hip","diag_rl"),
)


def arguments():
    p=argparse.ArgumentParser();p.add_argument("--world-csv",type=Path,required=True)
    p.add_argument("--aligned",type=Path,required=True);p.add_argument("--head-intrinsics",type=Path,required=True)
    p.add_argument("--head-rigid",type=Path,required=True);p.add_argument("--head-a",type=Path,required=True)
    p.add_argument("--head-d",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--ch07-event-offset",type=int,default=71);p.add_argument("--window",type=int,default=31)
    p.add_argument("--median-window",type=int,default=7)
    p.add_argument("--profile",choices=("strong","weak"),default="strong",
                   help="weak follows the raw trajectory more closely and relaxes torso-length constraints")
    p.add_argument("--filtered-only",action="store_true")
    p.add_argument("--skip-videos",action="store_true",
                   help="Write the filtered world CSV/report without rendering head views.")
    return p.parse_args()


def read_world(path):
    frames={}
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f): frames.setdefault(int(r["sequence"]),{})[r["joint"]]=np.array([float(r["x_m"]),float(r["y_m"]),float(r["z_m"])])
    return frames


def stats(values):
    a=np.asarray(values,float)
    return {"median":float(np.median(a)),"std":float(np.std(a)),"p05":float(np.percentile(a,5)),"p95":float(np.percentile(a,95)),"range":float(np.ptp(a))}


def main():
    a=arguments();a.output_dir.mkdir(parents=True,exist_ok=True)
    world=read_world(a.world_csv)
    with a.aligned.open("r",encoding="utf-8-sig",newline="") as f: aligned=list(csv.DictReader(f))
    seqs=sorted(world); poses={}; ch07={}
    for seq in seqs:
        row=aligned[seq+a.ch07_event_offset]
        R=qrot([float(row[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"]);t=np.array([float(row[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        poses[seq]=(R,t);ch07[seq]={n:R.T@(p-t) for n,p in world[seq].items()}
    raw=np.asarray([[ch07[s][n] for n in TORSO] for s in seqs])
    temporal=raw.copy();window=min(a.window,len(seqs) if len(seqs)%2 else len(seqs)-1);window=max(5,window|1)
    median_window=max(1,a.median_window|1)
    for j in range(4):
        for d in range(3):
            clean=median_filter(raw[:,j,d],size=median_window,mode="nearest")
            temporal[:,j,d]=savgol_filter(clean,window,2,mode="interp")
    targets={label:float(np.median([np.linalg.norm(ch07[s][u]-ch07[s][v]) for s in seqs])) for u,v,label in EDGES}
    solved=np.empty_like(raw)
    for i,seq in enumerate(seqs):
        x0=temporal[i].reshape(-1)
        def residual(x):
            p={n:x[k*3:k*3+3] for k,n in enumerate(TORSO)};res=[]
            if a.profile == "weak":
                # Prefer the current raw pose so fast motion is not attenuated by
                # the temporal reference. Constraints remain frame-local.
                temporal_sigma,raw_sigma,constraint_scale=.015,.008,3.0
            else:
                temporal_sigma,raw_sigma,constraint_scale=.006,.025,1.0
            res.extend(((x.reshape(4,3)-temporal[i])/temporal_sigma).reshape(-1))
            res.extend(((x.reshape(4,3)-raw[i])/raw_sigma).reshape(-1))
            for u,v,label in EDGES:
                sigma=(.0015 if label in ("shoulder_width","hip_width","left_torso","right_torso") else .003)*constraint_scale
                res.append((np.linalg.norm(p[u]-p[v])-targets[label])/sigma)
            return np.asarray(res)
        solved[i]=least_squares(residual,x0,loss="soft_l1",f_scale=2,max_nfev=120).x.reshape(4,3)

    filtered={s:{n:p.copy() for n,p in ch07[s].items()} for s in seqs}
    for i,seq in enumerate(seqs):
        deltas={n:solved[i,j]-ch07[seq][n] for j,n in enumerate(TORSO)}
        for j,n in enumerate(TORSO): filtered[seq][n]=solved[i,j]
        # Move the attached limb as one unit so upper-arm/thigh geometry is not broken.
        for anchor,followers in (("left_shoulder",("left_elbow","left_wrist")),("right_shoulder",("right_elbow","right_wrist")),
                                 ("left_hip",("left_knee","left_ankle")),("right_hip",("right_knee","right_ankle"))):
            for n in followers:
                if n in filtered[seq]: filtered[seq][n]+=deltas[anchor]

    rows=[]
    for seq in seqs:
        R,t=poses[seq]
        for n,p in filtered[seq].items():
            pw=R@p+t;rows.append({"sequence":seq,"joint":n,"x_m":pw[0],"y_m":pw[1],"z_m":pw[2]})
    profile=a.profile
    with (a.output_dir/f"torso4_{profile}3d_filtered_world.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def length_report(frames):
        return {label:stats([np.linalg.norm(frames[s][u]-frames[s][v])*1000 for s in seqs]) for u,v,label in EDGES[:4]}
    def acceleration(points):
        values=np.linalg.norm(np.diff(points,n=2,axis=0),axis=2)*1000
        return {n:{"median_mm_per_frame2":float(np.median(values[:,j])),"p90_mm_per_frame2":float(np.percentile(values[:,j],90))} for j,n in enumerate(TORSO)}

    if a.skip_videos:
        report={"method":f"CH07 median({median_window})+centered Savitzky-Golay({window}) + {profile} six-edge torso constraint",
                "filter_profile":profile,"median_window":median_window,"savgol_window":window,
                "aligned_table":str(a.aligned),"ch07_event_offset_frames":a.ch07_event_offset,
                "phase_policy":"centered offline filter; no temporal phase delay","length_targets_mm":{k:v*1000 for k,v in targets.items()},
                "render_policy":"skipped; CSV only","length_before_mm":length_report(ch07),"length_after_mm":length_report(filtered),
                "trajectory_acceleration_before":acceleration(raw),"trajectory_acceleration_after":acceleration(solved),
                "limb_policy":"shoulder delta propagated to elbow/wrist; hip delta propagated to knee/ankle; nose unchanged"}
        (a.output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False)); return

    intr=load_json(a.head_intrinsics);right="CAM_D" if "CAM_D" in intr["cameras"] else "CAM_C"
    cams={"A":Omni(intr,"CAM_A"),"D":Omni(intr,right)};rigid=load_json(a.head_rigid);T={}
    for cam,side in (("A","left"),("D","right")):
        c2r=np.asarray(rigid["cameras"][side]["T_rigid_camera"],float);c2r[:3,3]/=1000;T[cam]=np.linalg.inv(c2r)
    projected={"A":{},"D":{}};baseline={"A":{},"D":{}}
    for seq in seqs:
        for cam in "AD":
            for name,p in ch07[seq].items():
                uv=cams[cam].project((T[cam]@np.r_[p,1])[:3])
                if uv is not None: baseline[cam].setdefault(seq,{})[name]=np.asarray(uv)
            for name,p in filtered[seq].items():
                uv=cams[cam].project((T[cam]@np.r_[p,1])[:3])
                if uv is not None: projected[cam].setdefault(seq,{})[name]=np.asarray(uv)
    for cam,video in (("A",a.head_a),("D",a.head_d)):
        cap=cv2.VideoCapture(str(video));out=writer(a.output_dir/f"head_CAM_{cam}_torso4_{profile}3d_filter.mp4",(1920,1200),50);seq=0
        while True:
            ok,img=cap.read()
            if not ok:break
            if not a.filtered_only:
                draw_pose(img,baseline[cam].get(seq,{}),(255,255,0),"",body_only=True)
            draw_pose(img,projected[cam].get(seq,{}),(0,255,255),f"torso4 {profile} 3D filter -> HEAD_{cam} seq={seq}",body_only=True)
            label=("yellow=filtered 3D skeleton only" if a.filtered_only else f"cyan=before  yellow={profile} centered 3D torso filter")
            cv2.putText(img,label,(24,76),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,255,255),2,cv2.LINE_AA)
            out.write(img);seq+=1
        cap.release();out.release()
    ca=cv2.VideoCapture(str(a.output_dir/f"head_CAM_A_torso4_{profile}3d_filter.mp4"));cd=cv2.VideoCapture(str(a.output_dir/f"head_CAM_D_torso4_{profile}3d_filter.mp4"))
    out=writer(a.output_dir/f"head_stereo_torso4_{profile}3d_filter.mp4",(1920,600),50)
    while True:
        oka,ia=ca.read();okd,id_=cd.read()
        if not oka or not okd:break
        out.write(np.hstack([cv2.resize(ia,(960,600)),cv2.resize(id_,(960,600))]))
    ca.release();cd.release();out.release()
    report={"method":f"CH07 median({median_window})+centered Savitzky-Golay({window}) + {profile} six-edge torso constraint",
            "filter_profile":profile,"median_window":median_window,"savgol_window":window,
            "aligned_table":str(a.aligned),"ch07_event_offset_frames":a.ch07_event_offset,
            "phase_policy":"centered offline filter; no temporal phase delay","length_targets_mm":{k:v*1000 for k,v in targets.items()},
            "render_policy":"filtered 3D skeleton only" if a.filtered_only else "before and filtered overlay",
            "length_before_mm":length_report(ch07),"length_after_mm":length_report(filtered),
            "trajectory_acceleration_before":acceleration(raw),"trajectory_acceleration_after":acceleration(solved),
            "limb_policy":"shoulder delta propagated to elbow/wrist; hip delta propagated to knee/ankle; nose unchanged"}
    (a.output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":main()
