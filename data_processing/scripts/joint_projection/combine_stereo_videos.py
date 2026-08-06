#!/usr/bin/env python3
"""Resize two synchronized 1920x1200 streams and place them side by side."""
import argparse,json
from pathlib import Path
import cv2,numpy as np

p=argparse.ArgumentParser();p.add_argument("--left",type=Path,required=True);p.add_argument("--right",type=Path,required=True)
p.add_argument("--output",type=Path,required=True);p.add_argument("--fps",type=float,default=50);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
ca,cd=cv2.VideoCapture(str(a.left)),cv2.VideoCapture(str(a.right));out=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*"mp4v"),a.fps,(1920,600));frames=0
while True:
    oka,ia=ca.read();okd,id_=cd.read()
    if not oka or not okd:break
    out.write(np.hstack([cv2.resize(ia,(960,600)),cv2.resize(id_,(960,600))]));frames+=1
ca.release();cd.release();out.release();print(json.dumps({"frames":frames,"fps":a.fps,"output":str(a.output)}))
