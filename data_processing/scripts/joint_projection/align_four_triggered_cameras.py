#!/usr/bin/env python3
"""Align head A/D and external A/D by recovered 50 Hz trigger ordinal."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--head-timestamps',type=Path,required=True);p.add_argument('--external-timestamps',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
def read(path):
 with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
head=read(a.head_timestamps);ext=read(a.external_timestamps)
def recover(rows,camera,time_field,filter_fn,index_field,time_scale=1.0):
 rs=[r for r in rows if filter_fn(r) and r['camera']==camera];times=np.asarray([float(r[time_field])*time_scale for r in rs],float)
 diffs=np.diff(times);steps=np.maximum(1,np.rint(diffs/20000.0).astype(int));period=float(np.median(diffs/steps));origin=float([r for r in rows if filter_fn(r) and r['camera']=='CAM_A'][0][time_field])*time_scale
 out={};res=[]
 for local_index,(r,t) in enumerate(zip(rs,times)):
  ordinal=int(round((t-origin)/period));residual=float(t-(origin+ordinal*period));idx=int(r[index_field]) if index_field in r else local_index
  item=dict(r);item['_index']=idx;item['_residual_us']=residual
  if ordinal not in out or abs(residual)<abs(out[ordinal]['_residual_us']):out[ordinal]=item
  res.append(residual)
 return out,period,{'rows':len(rs),'median_abs_residual_us':float(np.median(np.abs(res))),'p99_abs_residual_us':float(np.percentile(np.abs(res),99))}
h={};e={};reports={}
for cam in ('CAM_A','CAM_D'):
 h[cam],hp,hr=recover(head,cam,'exposure_start_ts_ms',lambda r:r['module']=='1','__local__',1000.0);reports['head_'+cam]=hr
 e[cam],ep,er=recover(ext,cam,'exposure_start_device_timestamp_us',lambda r:r.get('jpeg_valid','1')=='1','frame_index');reports['external_'+cam]=er
common=sorted(set(h['CAM_A'])&set(h['CAM_D'])&set(e['CAM_A'])&set(e['CAM_D']))
fields=['aligned_index','trigger_ordinal','time_s','head_CAM_A_frame','head_CAM_D_frame','external_CAM_A_frame','external_CAM_D_frame','head_CAM_A_seq','head_CAM_D_seq','external_CAM_A_sequence','external_CAM_D_sequence']
with a.output.open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for i,o in enumerate(common):w.writerow({'aligned_index':i,'trigger_ordinal':o,'time_s':o/50.0,'head_CAM_A_frame':h['CAM_A'][o]['_index'],'head_CAM_D_frame':h['CAM_D'][o]['_index'],'external_CAM_A_frame':e['CAM_A'][o]['_index'],'external_CAM_D_frame':e['CAM_D'][o]['_index'],'head_CAM_A_seq':h['CAM_A'][o]['seq'],'head_CAM_D_seq':h['CAM_D'][o]['seq'],'external_CAM_A_sequence':e['CAM_A'][o]['sequence'],'external_CAM_D_sequence':e['CAM_D'][o]['sequence']})
summary={'schema':'hearwristcam.four_trigger_alignment.v1','method':'independent device-clock origin at first CAM_A exposure start; robust recovered trigger period; four-camera ordinal intersection','head_period_us':hp,'external_period_us':ep,'common_frames':len(common),'first_ordinal':common[0],'last_ordinal':common[-1],'duration_span_s':(common[-1]-common[0])/50.0,'first_10s_common_frames':sum(o<common[0]+500 for o in common),'reports':reports}
a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
