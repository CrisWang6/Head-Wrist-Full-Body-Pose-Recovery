#!/usr/bin/env python3
"""Extract a compact aligned video using source-frame indices from a strict table."""

import argparse, csv
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser();p.add_argument("--video",type=Path,required=True);p.add_argument("--aligned",type=Path,required=True)
    p.add_argument("--column",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--limit",type=int,default=344)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.aligned.open("r",encoding="utf-8-sig",newline="") as f: indices=[int(r[a.column]) for r in list(csv.DictReader(f))[:a.limit]]
    wanted={frame:i for i,frame in enumerate(indices)};cap=cv2.VideoCapture(str(a.video))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*"mp4v"),50,(width,height));written=0;source=0
    maximum=max(indices)
    while source<=maximum:
        ok,image=cap.read()
        if not ok:break
        if source in wanted:out.write(image);written+=1
        source+=1
    cap.release();out.release()
    if written!=len(indices):raise RuntimeError(f"wrote {written}/{len(indices)} frames; decoded through {source-1}, need {maximum}")
    print({"output":str(a.output),"frames":written,"first_source":indices[0],"last_source":indices[-1]})


if __name__=="__main__":main()
