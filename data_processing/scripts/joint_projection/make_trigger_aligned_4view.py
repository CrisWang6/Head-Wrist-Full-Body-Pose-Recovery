#!/usr/bin/env python3
"""Combine external stereo and both head views using exact trigger sequences."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np

p=argparse.ArgumentParser();p.add_argument('--external',type=Path,required=True);p.add_argument('--head-a',type=Path,required=True);p.add_argument('--head-d',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--map-a',type=Path,required=True);p.add_argument('--map-d',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
records=json.loads(a.selection.read_text(encoding='utf-8'))
ma=json.loads(a.map_a.read_text(encoding='utf-8'));md=json.loads(a.map_d.read_text(encoding='utf-8'))
ia={int(s):i for i,s in enumerate(ma['decoded_to_trigger_sequence'])};idd={int(s):i for i,s in enumerate(md['decoded_to_trigger_sequence'])}
ce=cv2.VideoCapture(str(a.external));ca=cv2.VideoCapture(str(a.head_a));cd=cv2.VideoCapture(str(a.head_d));a.output.parent.mkdir(parents=True,exist_ok=True)
out=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),50,(1920,1200));pa=pd=-1;fa=fd=None;written=0
for rec in records:
    ok,ext=ce.read()
    if not ok:break
    seq=int(rec['sequence']);ta=ia.get(seq);td=idd.get(seq)
    if ta is None or td is None:continue
    while pa<ta:
        oka,fa=ca.read();pa+=1
        if not oka:break
    while pd<td:
        okd,fd=cd.read();pd+=1
        if not okd:break
    if fa is None or fd is None:break
    head=np.hstack([cv2.resize(fa,(960,600)),cv2.resize(fd,(960,600))])
    out.write(np.vstack([cv2.resize(ext,(1920,600)),head]));written+=1
ce.release();ca.release();cd.release();out.release();print(json.dumps({'frames':written,'fps':50,'output':str(a.output)}))
