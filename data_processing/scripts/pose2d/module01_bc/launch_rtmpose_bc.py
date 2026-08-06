import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def decoded_count(showinfo_path):
    pattern = re.compile(r"n:\s*(\d+)")
    last = -1
    for line in Path(showinfo_path).read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            last = max(last, int(match.group(1)))
    if last < 0:
        raise RuntimeError(f"No decoded frames found in {showinfo_path}")
    return last + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    python = Path(os.environ.get("SAPIENS_PYTHON", sys.executable))
    worker = Path(__file__).resolve().with_name("rtmpose_video_worker.py")
    parts = root / "output" / "rtmpose_parts"
    parts.mkdir(parents=True, exist_ok=True)

    library_paths = os.environ.get("SAPIENS_LIBRARY_PATHS", "")
    jobs = []
    for camera, gpu in (("B", "0"), ("C", "1")):
        video = root / "input" / f"module01_D45D2E00_CAM_{camera}.h265"
        count = decoded_count(root / "output" / f"cam_{camera.lower()}_showinfo.log")
        midpoint = count // 2
        for part, start, end in (
            (0, 0, midpoint - 1),
            (1, midpoint, count - 1),
        ):
            output = parts / f"cam_{camera.lower()}_part_{part}.jsonl"
            log = parts / f"cam_{camera.lower()}_part_{part}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            if library_paths:
                env["LD_LIBRARY_PATH"] = library_paths
            stream = log.open("w")
            process = subprocess.Popen(
                [
                    str(python),
                    str(worker),
                    "--video",
                    str(video),
                    "--start",
                    str(start),
                    "--end",
                    str(end),
                    "--output",
                    str(output),
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            jobs.append((camera, part, process, stream))
            print(
                f"started CAM_{camera} part {part}: frames {start}-{end}, pid={process.pid}",
                flush=True,
            )

    failed = []
    for camera, part, process, stream in jobs:
        return_code = process.wait()
        stream.close()
        if return_code:
            failed.append({"camera": camera, "part": part, "return_code": return_code})
    (parts / "launcher_result.json").write_text(
        json.dumps({"failed": failed}, indent=2)
    )
    if failed:
        print(json.dumps(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
