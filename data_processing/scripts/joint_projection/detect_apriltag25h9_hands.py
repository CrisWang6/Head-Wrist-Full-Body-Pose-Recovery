#!/usr/bin/env python3
"""Detect tag25h9 hand markers and retain their image centers and quality."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2
from pupil_apriltags import Detector

p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int,required=True);p.add_argument('--family',default='tag16h5');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
detector=Detector(families=a.family,nthreads=a.threads,quad_decimate=1.0,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25)
cap=cv2.VideoCapture(str(a.video));a.output.parent.mkdir(parents=True,exist_ok=True)
detected_frames=total=0;tag_ids={}
with a.output.open('w',encoding='utf-8') as f:
    for frame_index in range(a.max_frames):
        ok,frame=cap.read()
        if not ok:break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);items=[]
        for d in detector.detect(gray,estimate_tag_pose=False):
            item={'tag_id':int(d.tag_id),'center':[float(d.center[0]),float(d.center[1])],'corners':[[float(x),float(y)] for x,y in d.corners],'decision_margin':float(d.decision_margin),'hamming':int(d.hamming)}
            items.append(item);tag_ids[str(d.tag_id)]=tag_ids.get(str(d.tag_id),0)+1;total+=1
        if items:detected_frames+=1
        f.write(json.dumps({'frame_index':frame_index,'detections':items},separators=(',',':'))+'\n')
cap.release()
summary={'family':a.family,'requested_frames':a.max_frames,'processed_frames':frame_index+1,'detected_frames':detected_frames,'total_detections':total,'tag_id_counts':tag_ids}
a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
