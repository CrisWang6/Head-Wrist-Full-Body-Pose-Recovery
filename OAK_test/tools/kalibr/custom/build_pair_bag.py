#!/usr/bin/env python
import argparse
import os

import cv2
import cv_bridge
import rosbag
import rospy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-root', required=True)
    parser.add_argument('--cameras', nargs=2, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    bridge = cv_bridge.CvBridge()
    with rosbag.Bag(args.output, 'w') as bag:
        for output_index, sample_index in enumerate(range(args.start, args.end + 1)):
            stamp = rospy.Time.from_sec(1.0 + output_index * 0.2)
            for camera in args.cameras:
                path = os.path.join(
                    args.images_root,
                    'sample_{0:06d}'.format(sample_index),
                    camera + '.png')
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError('Could not read image: ' + path)
                if image.shape[:2] != (1200, 1920):
                    raise RuntimeError('Unexpected image size for {0}: {1}'.format(path, image.shape))
                message = bridge.cv2_to_imgmsg(image, encoding='bgr8')
                message.header.stamp = stamp
                message.header.frame_id = camera
                bag.write('/{0}/image_raw'.format(camera), message, stamp)
    print('Wrote {0} synchronized pairs to {1}'.format(args.end - args.start + 1, args.output))


if __name__ == '__main__':
    main()
