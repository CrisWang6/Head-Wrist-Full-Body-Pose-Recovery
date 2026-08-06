#!/usr/bin/env python3
"""Render a CH07 skeleton as synchronized front and side orthographic views."""

from __future__ import annotations

import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np

NAMES=("nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
       "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
       "left_knee","right_knee","left_ankle","right_ankle")
EDGES=(("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
       ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),("left_shoulder","left_hip"),
       ("right_shoulder","right_hip"),("left_hip","right_hip"),("left_hip","left_knee"),
       ("left_knee","left_ankle"),("right_hip","right_knee"),("right_knee","right_ankle"))

def main():
    p=argparse.ArgumentParser();p.add_argument("--ch07-csv",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fps",type=float,default=50);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    frames={}
    with a.ch07_csv.open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):frames.setdefault(int(r["sequence"]),{})[r["joint"]]=np.asarray([float(r["x_m"]),float(r["y_m"]),float(r["z_m"])])
    seqs=sorted(frames);noses=np.asarray([frames[s]["nose"] for s in seqs if "nose" in frames[s]])
    origin=np.median(noses,axis=0);scale=350.0;W,H=1920,720;panel=W//2
    out=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*"mp4v"),a.fps,(W,H))
    def project(point,side):
        horizontal=(point[0]-origin[0]) if side else -(point[1]-origin[1])
        vertical=origin[2]-point[2]
        return int(panel/2+horizontal*scale),int(90+vertical*scale)
    for seq in seqs:
        im=np.full((H,W,3),(13,18,27),np.uint8);joints=frames[seq]
        cv2.line(im,(panel,0),(panel,H),(58,68,84),2)
        for pi,(title,side) in enumerate((("FRONT (Y / Z)",False),("SIDE (X / Z)",True))):
            off=pi*panel
            cv2.putText(im,title,(off+32,44),cv2.FONT_HERSHEY_SIMPLEX,1.0,(235,240,248),2,cv2.LINE_AA)
            cv2.putText(im,f"global five-point | seq={seq}",(off+32,76),cv2.FONT_HERSHEY_SIMPLEX,.65,(150,170,195),2,cv2.LINE_AA)
            for u,v in EDGES:
                if u in joints and v in joints:
                    pu=project(joints[u],side);pv=project(joints[v],side)
                    cv2.line(im,(pu[0]+off,pu[1]),(pv[0]+off,pv[1]),(0,220,255),5,cv2.LINE_AA)
            for name,p3 in joints.items():
                if name=="nose" or name in NAMES[5:]:
                    q=project(p3,side);cv2.circle(im,(q[0]+off,q[1]),7,(40,80,255),-1,cv2.LINE_AA)
        out.write(im)
    out.release();print(json.dumps({"frames":len(seqs),"fps":a.fps,"output":str(a.output),"views":["front Y-Z","side X-Z"]}))

if __name__=="__main__":main()
