import argparse
import json
import random
import subprocess
from pathlib import Path

import cv2
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
COLORS = {
    "left_shoulder": (0, 220, 255),
    "left_elbow": (0, 220, 255),
    "right_shoulder": (255, 120, 0),
    "right_elbow": (255, 120, 0),
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
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260723)
    edges = np.linspace(0, args.frame_count, args.samples + 1, dtype=int)
    selected = sorted(
        rng.randrange(int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1])))
        for i in range(args.samples)
    )
    wanted = set(selected)

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
        "-vsync",
        "0",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    records = []
    decoded_index = 0
    while True:
        raw = read_exact(process.stdout, FRAME_BYTES)
        if not raw:
            break
        if decoded_index not in wanted:
            decoded_index += 1
            continue
        image = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3).copy()
        keypoints, scores = model(image)
        record = {"decoded_frame_index": decoded_index, "status": "no_person"}
        if len(keypoints):
            person = int(np.argmax(np.mean(scores[:, :17], axis=1)))
            points = {}
            for name, keypoint_index in TARGETS.items():
                x, y = keypoints[person, keypoint_index]
                score = float(scores[person, keypoint_index])
                points[name] = (int(round(x)), int(round(y)))
                cv2.circle(image, points[name], 11, COLORS[name], -1, cv2.LINE_AA)
                cv2.putText(
                    image,
                    f"{name} {score:.2f}",
                    (points[name][0] + 14, points[name][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    COLORS[name],
                    2,
                    cv2.LINE_AA,
                )
            cv2.line(image, points["left_shoulder"], points["left_elbow"], COLORS["left_shoulder"], 5)
            cv2.line(image, points["right_shoulder"], points["right_elbow"], COLORS["right_shoulder"], 5)
            cv2.line(image, points["left_shoulder"], points["right_shoulder"], (80, 255, 80), 4)
            record.update(status="ok", scores={
                name: float(scores[person, index]) for name, index in TARGETS.items()
            })
        cv2.putText(
            image,
            f"RTMPose-X decoded frame {decoded_index}",
            (35, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / f"review_{decoded_index:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append(record)
        print(f"saved review frame {decoded_index}", flush=True)
        decoded_index += 1
    process.wait()
    (output_dir / "review_manifest.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
