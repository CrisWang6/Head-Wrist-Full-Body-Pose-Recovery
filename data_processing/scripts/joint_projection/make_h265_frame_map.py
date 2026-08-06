#!/usr/bin/env python3
"""Create decoded-frame to capture-index mapping from elementary H.265 PTS."""
from __future__ import annotations
import argparse, json, re, subprocess
from pathlib import Path

PATTERN=re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)")
p=argparse.ArgumentParser(); p.add_argument("source",type=Path); p.add_argument("output",type=Path); a=p.parse_args()
proc=subprocess.Popen(["ffmpeg","-hide_banner","-i",str(a.source),"-vf","showinfo","-an","-f","null","-"],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,errors="replace")
pts=[]
for line in proc.stderr:
    m=PATTERN.search(line)
    if m: pts.append(int(m.group(2)))
if proc.wait()!=0 or not pts: raise SystemExit("ffmpeg decoding failed")
steps=[b-c for c,b in zip(pts,pts[1:]) if b>c]; nominal=min(steps)
mapping=[int(round((v-pts[0])/nominal)) for v in pts]
doc={"source":str(a.source),"decoded_frames":len(pts),"first_pts":pts[0],"nominal_pts_step":nominal,"last_capture_index":mapping[-1],"missing_from_pts_gaps":mapping[-1]+1-len(mapping),"decoded_to_capture_index":mapping}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in doc.items() if k!="decoded_to_capture_index"}))
