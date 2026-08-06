#!/usr/bin/env python3
"""Export only trigger frames present and reliable in all four camera views."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np

p=argparse.ArgumentParser();p.add_argument('--external',type=Path,required=True);p.add_argument('--head-a',type=Path,required=True);p.add_argument('--head-d',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--head-timestamps',type=Path,required=True);p.add_argument('--packet-map-a',type=Path);p.add_argument('--packet-map-d',type=Path);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
records=json.loads(a.selection.read_text(encoding='utf-8'))
with a.head_timestamps.open('r',encoding='utf-8-sig',newline='') as f:ts=list(csv.DictReader(f))
idx={}
for cam in ('CAM_A','CAM_D'):
    camera_rows=[r for r in ts if r['module']=='1' and r['camera']==cam]
    idx[cam]={int(r['seq']):i for i,r in enumerate(camera_rows)}
def decoded_sequences(path):
    if path is None:return None
    obj=json.loads(path.read_text(encoding='utf-8'))
    values=obj.get('decoded_to_trigger_sequence',obj)
    return {int(v) for v in values if v is not None}
decoded_a=decoded_sequences(a.packet_map_a);decoded_d=decoded_sequences(a.packet_map_d)

required={'left_shoulder','right_shoulder','left_elbow','right_elbow','left_hip','right_hip','left_knee','right_knee'}
def reliable(rec):
    joints=rec['joints']; names=set(joints); body_count=sum(n not in {'nose','left_eye','right_eye','left_ear','right_ear'} for n in names)
    if body_count<10 or len(required&names)<7:return False
    gaps=[float(v['ray_gap_m']) for v in joints.values()]
    reps=[max(float(v.get('left_reprojection_error_px',999)),float(v.get('right_reprojection_error_px',999))) for v in joints.values()]
    return float(np.median(gaps))<=.020 and float(np.percentile(reps,90))<=12.0

ce=cv2.VideoCapture(str(a.external));ca=cv2.VideoCapture(str(a.head_a));cd=cv2.VideoCapture(str(a.head_d))
wa=cv2.VideoWriter(str(a.output_dir/'strict_head_CAM_A_projected.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50,(1920,1200));wd=cv2.VideoWriter(str(a.output_dir/'strict_head_CAM_D_projected.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50,(1920,1200));we=cv2.VideoWriter(str(a.output_dir/'strict_external_stereo_2d.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50,(1920,600));wh=cv2.VideoWriter(str(a.output_dir/'strict_head_stereo_projected.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50,(1920,600));w4=cv2.VideoWriter(str(a.output_dir/'strict_trigger_aligned_4view.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50,(1920,1200))
pa=pd=-1;fa=fd=None;kept=[];reasons={'quality':0,'head_missing':0,'head_undecodable':0}
for rec in records:
    oke,ext=ce.read()
    if not oke:break
    seq=int(rec['sequence']);ta=idx['CAM_A'].get(seq);td=idx['CAM_D'].get(seq)
    if ta is None or td is None:
        reasons['head_missing']+=1;continue
    if (decoded_a is not None and seq not in decoded_a) or (decoded_d is not None and seq not in decoded_d):
        reasons['head_undecodable']+=1;continue
    if not reliable(rec):
        reasons['quality']+=1;continue
    while pa<ta:
        oka,fa=ca.read();pa+=1
        if not oka:break
    while pd<td:
        okd,fd=cd.read();pd+=1
        if not okd:break
    if fa is None or fd is None:break
    head=np.hstack([cv2.resize(fa,(960,600)),cv2.resize(fd,(960,600))])
    we.write(ext);wa.write(fa);wd.write(fd);wh.write(head);w4.write(np.vstack([ext,head]));kept.append(seq)
for x in (ce,ca,cd):x.release()
for x in (wa,wd,we,wh,w4):x.release()
(a.output_dir/'strict_kept_trigger_sequences.json').write_text(json.dumps({'kept_count':len(kept),'kept_sequences':kept,'removed':reasons},indent=2),encoding='utf-8')
print(json.dumps({'kept':len(kept),'input_records':len(records),'removed':reasons}))
