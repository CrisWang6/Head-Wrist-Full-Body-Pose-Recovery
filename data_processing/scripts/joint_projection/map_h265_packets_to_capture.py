#!/usr/bin/env python3
"""Decode timestamp-sized H.265 access units and retain their capture row index."""
from __future__ import annotations
import argparse,csv,json
from fractions import Fraction
from pathlib import Path
import av

p=argparse.ArgumentParser(); p.add_argument("source",type=Path); p.add_argument("timestamps",type=Path); p.add_argument("camera"); p.add_argument("output",type=Path); a=p.parse_args()
with a.timestamps.open("r",encoding="utf-8-sig",newline="") as f:
    rows=[r for r in csv.DictReader(f) if r["module"]=="1" and r["camera"]==a.camera]
codec=av.CodecContext.create("hevc","r"); decoded=[]
with a.source.open("rb") as f:
    for capture_index,row in enumerate(rows):
        data=f.read(int(row["bytes"]))
        if len(data)!=int(row["bytes"]): raise RuntimeError("truncated source")
        packet=av.Packet(data); packet.pts=capture_index; packet.dts=capture_index; packet.time_base=Fraction(1,50)
        for frame in codec.decode(packet): decoded.append(int(frame.pts))
    for frame in codec.decode(None): decoded.append(int(frame.pts))
if len(set(decoded))!=len(decoded): raise RuntimeError("non-unique decoded capture indices")
doc={"source":str(a.source),"camera":a.camera,"capture_rows":len(rows),"decoded_frames":len(decoded),"missing_frames":len(rows)-len(decoded),"decoded_to_capture_index":decoded,
     "decoded_to_trigger_sequence":[int(rows[i]["seq"]) for i in decoded]}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in doc.items() if not isinstance(v,list)}))
