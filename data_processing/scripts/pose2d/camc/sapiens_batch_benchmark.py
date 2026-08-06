import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
from sapiens.pose.models import init_model


TARGETS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
}


def prepare(image, model):
    height, width = image.shape[:2]
    data = model.pipeline(
        {
            "img": image,
            "bbox": np.array([[0, 0, width - 1, height - 1]], dtype=np.float32),
            "bbox_score": np.ones(1, dtype=np.float32),
        }
    )
    return model.data_preprocessor(data)


def infer(items, model):
    inputs = torch.cat([item["inputs"] for item in items], dim=0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(inputs)
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            flipped = model(inputs.flip(-1)).flip(-1)
            flipped = flipped[:, model.pose_metainfo["flip_indices"]]
            pred = (pred + flipped) / 2.0
    pred = pred.float().cpu().numpy()
    output = []
    for index, item in enumerate(items):
        keypoints, scores = model.codec.decode(pred[index])
        meta = item["data_samples"]["meta"]
        keypoints = (
            keypoints / meta["input_size"] * meta["bbox_scale"]
            + meta["bbox_center"]
            - 0.5 * meta["bbox_scale"]
        )[0]
        scores = scores[0]
        output.append(
            {
                name: [float(keypoints[k, 0]), float(keypoints[k, 1]), float(scores[k])]
                for name, k in TARGETS.items()
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device="cuda:0")
    model.pose_metainfo = parse_pose_metainfo(
        dict(from_file=str(Path(args.config).parents[2] / "_base_" / "keypoints308.py"))
    )
    codec_config = dict(model.cfg.codec)
    codec_config.pop("type")
    model.codec = UDPHeatmap(**codec_config)

    image_paths = sorted(Path(args.images).glob("*CAM_C*.jpg"))
    images = [cv2.imread(str(path)) for path in image_paths]
    prepared = [prepare(image, model) for image in images]

    records = []
    for batch_size in [int(value) for value in args.batch_sizes.split(",")]:
        if batch_size > len(prepared):
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        infer(prepared[:1], model)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = []
        for offset in range(0, len(prepared), batch_size):
            output.extend(infer(prepared[offset : offset + batch_size], model))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        record = {
            "batch_size": batch_size,
            "images": len(prepared),
            "elapsed_s": elapsed,
            "fps": len(prepared) / elapsed,
            "seconds_per_video_minute_at_30fps": 1800.0 / (len(prepared) / elapsed),
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "sample": output[0],
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    Path(args.output).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
