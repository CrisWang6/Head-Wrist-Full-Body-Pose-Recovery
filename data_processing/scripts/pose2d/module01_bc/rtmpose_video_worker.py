import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from rtmlib import Body


WIDTH = 1920
HEIGHT = 1200
FRAME_BYTES = WIDTH * HEIGHT * 3
TARGETS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
}


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = Body(mode="performance", backend="onnxruntime", device="cuda")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-err_detect",
        "ignore_err",
        "-i",
        args.video,
        "-vf",
        f"select=between(n\\,{args.start}\\,{args.end})",
        "-vsync",
        "0",
        "-frames:v",
        str(args.end - args.start + 1),
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    decoder = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    inference_s = 0.0
    started = time.time()

    with output_path.open("w", encoding="utf-8") as output:
        while True:
            raw = read_exact(decoder.stdout, FRAME_BYTES)
            if not raw:
                break
            if len(raw) != FRAME_BYTES:
                raise RuntimeError(f"Partial raw frame: {len(raw)} bytes")
            image = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            start = time.perf_counter()
            keypoints, scores = model(image)
            inference_s += time.perf_counter() - start
            decoded_index = args.start + count
            record = {
                "decoded_frame_index": decoded_index,
                "status": "no_person",
            }
            if len(keypoints):
                person = int(np.argmax(np.mean(scores[:, :17], axis=1)))
                record["status"] = "ok"
                for name, index in TARGETS.items():
                    x, y = keypoints[person, index]
                    record[f"{name}_x"] = float(x)
                    record[f"{name}_y"] = float(y)
            output.write(json.dumps(record) + "\n")
            count += 1
            if count % 200 == 0:
                print(
                    f"frames={count} decoded={decoded_index} "
                    f"wall_fps={count / max(time.time() - started, 1e-6):.2f}",
                    flush=True,
                )

    return_code = decoder.wait()
    wall_s = time.time() - started
    summary = {
        "start": args.start,
        "end": args.end,
        "frames": count,
        "ffmpeg_return_code": return_code,
        "wall_s": wall_s,
        "inference_s": inference_s,
        "fps_wall": count / max(wall_s, 1e-6),
    }
    Path(f"{args.output}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
