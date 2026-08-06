import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
from sapiens.pose.models import init_model


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


def prepare(image, model):
    data = model.pipeline(
        {
            "img": image,
            "bbox": np.array([[0, 0, WIDTH - 1, HEIGHT - 1]], dtype=np.float32),
            "bbox_score": np.ones(1, dtype=np.float32),
        }
    )
    return model.data_preprocessor(data)


def infer_batch(items, model):
    inputs = torch.cat([item["prepared"]["inputs"] for item in items], dim=0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(inputs)
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            flipped = model(inputs.flip(-1)).flip(-1)
            flipped = flipped[:, model.pose_metainfo["flip_indices"]]
            pred = (pred + flipped) / 2.0
    pred = pred.float().cpu().numpy()

    records = []
    for index, item in enumerate(items):
        keypoints, scores = model.codec.decode(pred[index])
        meta = item["prepared"]["data_samples"]["meta"]
        keypoints = (
            keypoints / meta["input_size"] * meta["bbox_scale"]
            + meta["bbox_center"]
            - 0.5 * meta["bbox_scale"]
        )[0]
        scores = scores[0]
        record = {
            "decoded_frame_index": item["decoded_frame_index"],
            "status": "ok",
        }
        for name, keypoint_index in TARGETS.items():
            record[f"{name}_x"] = float(keypoints[keypoint_index, 0])
            record[f"{name}_y"] = float(keypoints[keypoint_index, 1])
            record[f"{name}_score"] = float(scores[keypoint_index])
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device="cuda:0")
    model.pose_metainfo = parse_pose_metainfo(
        dict(from_file=str(Path(args.config).parents[2] / "_base_" / "keypoints308.py"))
    )
    codec_config = dict(model.cfg.codec)
    codec_config.pop("type")
    model.codec = UDPHeatmap(**codec_config)

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
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    inference_total = 0.0
    count = 0
    batch = []

    with output_path.open("w", encoding="utf-8") as output:
        while True:
            raw = read_exact(process.stdout, FRAME_BYTES)
            if not raw:
                break
            if len(raw) != FRAME_BYTES:
                raise RuntimeError(f"Partial raw frame: {len(raw)} bytes")
            image = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            decoded_index = args.start + count
            batch.append(
                {
                    "decoded_frame_index": decoded_index,
                    "prepared": prepare(image, model),
                }
            )
            count += 1
            if len(batch) < args.batch_size:
                continue

            torch.cuda.synchronize()
            start = time.perf_counter()
            records = infer_batch(batch, model)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            inference_total += elapsed
            per_frame = elapsed / len(records)
            for record in records:
                record["inference_s"] = per_frame
                output.write(json.dumps(record) + "\n")
            output.flush()
            batch.clear()
            if count % 100 == 0:
                print(
                    f"frames={count} last_decoded={decoded_index} "
                    f"fps={count / max(time.time() - started_at, 1e-6):.2f}",
                    flush=True,
                )

        if batch:
            torch.cuda.synchronize()
            start = time.perf_counter()
            records = infer_batch(batch, model)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            inference_total += elapsed
            for record in records:
                record["inference_s"] = elapsed / len(records)
                output.write(json.dumps(record) + "\n")

    return_code = process.wait()
    summary = {
        "start": args.start,
        "end": args.end,
        "frames": count,
        "ffmpeg_return_code": return_code,
        "wall_s": time.time() - started_at,
        "inference_s": inference_total,
        "fps_wall": count / max(time.time() - started_at, 1e-6),
    }
    Path(f"{args.output}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
