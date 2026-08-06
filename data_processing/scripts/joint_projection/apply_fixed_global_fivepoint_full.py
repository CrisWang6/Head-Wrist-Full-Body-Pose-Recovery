#!/usr/bin/env python3
"""Apply an approved fixed five-point CH07 correction to a full world skeleton."""

from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from process_external_stereo_to_head import qrot


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--world-csv",type=Path,required=True)
    p.add_argument("--aligned",type=Path,required=True)
    p.add_argument("--fit-report",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--ch07-event-offset",type=int,default=0)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    frames={}
    with a.world_csv.open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            frames.setdefault(int(r["sequence"]),{})[r["joint"]]=np.asarray(
                [float(r["x_m"]),float(r["y_m"]),float(r["z_m"])])
    with a.aligned.open("r",encoding="utf-8-sig",newline="") as f: aligned=list(csv.DictReader(f))
    fit=json.loads(a.fit_report.read_text(encoding="utf-8"))
    correction_R=Rotation.from_rotvec(np.asarray(fit["rotation_vector_rad"])).as_matrix()
    correction_t=np.asarray(fit["translation_m"],float)
    nose_translation=np.asarray(fit["nose_only_translation_m"],float)
    world_rows=[];ch07_rows=[]
    for seq,joints in sorted(frames.items()):
        idx=seq+a.ch07_event_offset
        if not 0<=idx<len(aligned): continue
        row=aligned[idx]
        body_R=qrot([float(row[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
        body_t=np.asarray([float(row[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        for name,p_world in joints.items():
            p_ch07=body_R.T@(p_world-body_t)+nose_translation
            p_opt=correction_R@p_ch07+correction_t
            p_out=body_R@p_opt+body_t
            ch07_rows.append({"sequence":seq,"joint":name,"x_m":p_opt[0],"y_m":p_opt[1],"z_m":p_opt[2]})
            world_rows.append({"sequence":seq,"joint":name,"x_m":p_out[0],"y_m":p_out[1],"z_m":p_out[2]})
    for name,rows in (("global_fivepoint_ch07.csv",ch07_rows),("global_fivepoint_world.csv",world_rows)):
        with (a.output_dir/name).open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={"frames":len(frames),"ch07_event_offset_frames":a.ch07_event_offset,
            "fit_report":str(a.fit_report),"rotation_angle_deg":float(np.linalg.norm(fit["rotation_vector_rad"])*180/np.pi),
            "rotation_vector_rad":fit["rotation_vector_rad"],"translation_m":fit["translation_m"],
            "nose_only_translation_m":fit["nose_only_translation_m"],
            "policy":"one fixed global five-point transform learned from approved first 344 strict frames"}
    (a.output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__": main()
