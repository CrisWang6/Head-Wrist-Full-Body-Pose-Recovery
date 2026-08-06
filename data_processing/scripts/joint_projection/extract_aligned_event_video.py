#!/usr/bin/env python3
"""Extract selected source-frame indices into a compact event video."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import cv2

p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--aligned',type=Path,required=True);p.add_argument('--index-field',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=10.0);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
with a.aligned.open('r',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
t0=float(rows[0]['module01_CAM_A_device_ts_ms']);rows=[r for r in rows if float(r['module01_CAM_A_device_ts_ms'])<t0+a.seconds*1000.0];targets=[int(r[a.index_field]) for r in rows]
cap=cv2.VideoCapture(str(a.video));writer=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),50,(1920,1200));current=-1;frame=None
for target in targets:
 while current<target:
  ok,frame=cap.read();current+=1
  if not ok:raise RuntimeError(f'decode stopped at source frame {current}, target={target}')
 writer.write(frame)
cap.release();writer.release();print({'output':str(a.output),'frames':len(targets),'first_source':targets[0],'last_source':targets[-1]})
