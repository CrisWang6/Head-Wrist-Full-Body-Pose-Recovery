#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import rosbag
import rospy
from sensor_msgs.msg import Image


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
    parser.add_argument("--cams", nargs="+", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    samples = sorted((dataset / "images").glob("sample_*"))
    if not samples:
        raise RuntimeError("no sample folders under {}".format(dataset / "images"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = {cam: 0 for cam in args.cams}
    with rosbag.Bag(str(out), "w") as bag:
        for index, sample in enumerate(samples):
            stamp = index / args.fps
            for cam in args.cams:
                path = sample / "{}.png".format(cam)
                if not path.exists():
                    continue
                msg = image_msg(path, stamp, cam)
                bag.write("/{}/image_raw".format(cam), msg, msg.header.stamp)
                written[cam] += 1

    for cam, count in written.items():
        print("{}: wrote {} frames".format(cam, count))
    print("bag: {}".format(out))


if __name__ == "__main__":
    main()
