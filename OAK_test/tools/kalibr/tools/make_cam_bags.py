#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import rosbag
import rospy
from sensor_msgs.msg import Image


CAMS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")


def image_msg(path, stamp, frame_id):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("failed to read {}".format(path))

    msg = Image()
    msg.header.stamp = rospy.Time.from_sec(stamp)
    msg.header.frame_id = frame_id
    msg.height, msg.width = image.shape[:2]
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.step = msg.width
    msg.data = image.tobytes()
    return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    samples = sorted((dataset / "images").glob("sample_*"))
    if not samples:
        raise RuntimeError("no sample folders under {}".format(dataset / "images"))

    for cam in CAMS:
        bag_path = out / "{}.bag".format(cam)
        written = 0
        with rosbag.Bag(str(bag_path), "w") as bag:
            for index, sample in enumerate(samples):
                meta_path = sample / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    cam_meta = meta.get("cameras", {}).get(cam, {})
                    if not cam_meta.get("board_detected", False):
                        continue
                path = sample / "{}.png".format(cam)
                if not path.exists():
                    continue
                stamp = index / args.fps
                msg = image_msg(path, stamp, cam)
                bag.write("/{}/image_raw".format(cam), msg, msg.header.stamp)
                written += 1
        print("{}: wrote {} frames to {}".format(cam, written, bag_path))


if __name__ == "__main__":
    main()
