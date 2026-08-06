#!/usr/bin/env python3
"""Projection-guided MediaPipe hand landmark detection for fisheye head views."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np,mediapipe as mp

p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--camera',choices=['CAM_A','CAM_D'],required=True);p.add_argument('--projection-csv',type=Path,required=True);p.add_argument('--timestamps',type=Path,required=True);p.add_argument('--strict-sequences',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fraction',type=float,default=1/3);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
with a.timestamps.open('r',encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['module']=='1' and r['camera']==a.camera]
seq_to_frame={int(r['seq']):i for i,r in enumerate(rows)};cutoff=int(len(rows)*a.fraction)
strict=json.loads(a.strict_sequences.read_text(encoding='utf-8'))['kept_sequences'];selected=[int(s) for s in strict if int(s) in seq_to_frame and seq_to_frame[int(s)]<cutoff]
prefix='head_A' if a.camera=='CAM_A' else 'head_D';proj={}
with a.projection_csv.open('r',encoding='utf-8-sig',newline='') as f:
 for r in csv.DictReader(f):
  try:uv=[float(r[f'{prefix}_u_px']),float(r[f'{prefix}_v_px'])]
  except (ValueError,TypeError):continue
  if np.all(np.isfinite(uv)):proj.setdefault(int(r['sequence']),{})[r['joint']]=np.asarray(uv,np.float32)
BaseOptions=mp.tasks.BaseOptions;VisionRunningMode=mp.tasks.vision.RunningMode;HandLandmarker=mp.tasks.vision.HandLandmarker;Options=mp.tasks.vision.HandLandmarkerOptions
options=Options(base_options=BaseOptions(model_asset_path=str(a.model)),running_mode=VisionRunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.20,min_hand_presence_confidence=.20,min_tracking_confidence=.20)
cap=cv2.VideoCapture(str(a.video));current=-1;frame=None;detected=0;per_side={'left':0,'right':0}
with HandLandmarker.create_from_options(options) as model,a.output.open('w',encoding='utf-8') as out:
 for seq in selected:
  target=seq_to_frame[seq]
  while current<target:
   ok,frame=cap.read();current+=1
   if not ok:raise RuntimeError(f'decode ended at {current}')
  hands={};pj=proj.get(seq,{})
  for side in ('left','right'):
   wrist=pj.get(f'{side}_wrist');elbow=pj.get(f'{side}_elbow')
   if wrist is None:continue
   direction=np.zeros(2,np.float32) if elbow is None else wrist-elbow
   center=wrist+.18*direction;half=int(np.clip(180+.25*np.linalg.norm(direction),180,300))
   x0=max(0,int(center[0]-half));y0=max(0,int(center[1]-half));x1=min(frame.shape[1],int(center[0]+half));y1=min(frame.shape[0],int(center[1]+half))
   if x1-x0<80 or y1-y0<80:continue
   crop=frame[y0:y1,x0:x1];rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB);result=model.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
   candidates=[]
   for hi,landmarks in enumerate(result.hand_landmarks):
    xy=np.asarray([[lm.x*(x1-x0)+x0,lm.y*(y1-y0)+y0] for lm in landmarks],np.float32)
    palm=np.mean(xy[[0,5,9,13,17]],axis=0);distance=float(np.linalg.norm(palm-wrist));score=float(result.handedness[hi][0].score) if hi<len(result.handedness) and result.handedness[hi] else 0.0
    candidates.append((distance-score*30,xy,score,palm))
   if candidates:
    _,xy,score,palm=min(candidates,key=lambda z:z[0])
    if float(np.linalg.norm(palm-wrist))<=300:
     hands[side]={'wrist':[float(x) for x in xy[0]],'palm_center':[float(x) for x in palm],'landmarks':[[float(x),float(y)] for x,y in xy],'confidence':score,'roi':[x0,y0,x1,y1]};detected+=1;per_side[side]+=1
  out.write(json.dumps({'sequence':seq,'frame_index':target,'hands':hands},separators=(',',':'))+'\n')
cap.release();summary={'model':'MediaPipe Hand Landmarker float16','camera':a.camera,'processed_strict_frames':len(selected),'detected_hands':detected,'per_side':per_side,'cutoff':cutoff,'method':'crop guided by projected elbow-wrist; 21 landmarks; endpoint uses landmark 0 wrist'};a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
