import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Body


TARGETS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
}


def load_sapiens(path):
    data = json.loads(Path(path).read_text())
    result = {}
    for frame in data["frames"]:
        if not frame["instances"]:
            continue
        instance = frame["instances"][0]
        result[frame["image_name"]] = {
            name: (
                float(instance["keypoints"][index][0]),
                float(instance["keypoints"][index][1]),
                float(instance["keypoint_scores"][index]),
            )
            for name, index in TARGETS.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--sapiens-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.images).glob("*CAM_C*.jpg"))
    sapiens = load_sapiens(args.sapiens_json)
    model = Body(mode="performance", backend="onnxruntime", device="cuda")

    results = []
    elapsed = []
    for path in paths:
        image = cv2.imread(str(path))
        start = time.perf_counter()
        keypoints, scores = model(image)
        seconds = time.perf_counter() - start
        elapsed.append(seconds)

        row = {"image_name": path.name, "seconds": seconds, "joints": {}}
        if len(keypoints):
            person = int(np.argmax(np.mean(scores[:, :17], axis=1)))
            for name, index in TARGETS.items():
                x, y = keypoints[person, index]
                score = scores[person, index]
                joint = {"x": float(x), "y": float(y), "score": float(score)}
                if path.name in sapiens:
                    sx, sy, ss = sapiens[path.name][name]
                    joint.update(
                        sapiens_x=sx,
                        sapiens_y=sy,
                        sapiens_score=ss,
                        pixel_difference=float(np.hypot(x - sx, y - sy)),
                    )
                row["joints"][name] = joint
            row["status"] = "ok"
        else:
            row["status"] = "no_person"
        results.append(row)
        print(f"{path.name}: {row['status']} {seconds:.3f}s", flush=True)

    compared = [
        joint["pixel_difference"]
        for row in results
        for joint in row["joints"].values()
        if "pixel_difference" in joint
    ]
    payload = {
        "model": "RTMPose-X 384x288 + YOLOX-X",
        "images": len(paths),
        "elapsed_total_s": float(sum(elapsed)),
        "mean_s_per_frame": float(np.mean(elapsed)),
        "steady_mean_s_per_frame": float(np.mean(elapsed[1:] or elapsed)),
        "steady_fps": float(1.0 / np.mean(elapsed[1:] or elapsed)),
        "mean_difference_from_sapiens_px": float(np.mean(compared)) if compared else None,
        "median_difference_from_sapiens_px": float(np.median(compared)) if compared else None,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
