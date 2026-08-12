"""Shared 0806 training pipeline constants (15 delivery joints, 480x300 input / 120x75 heatmaps)."""

from __future__ import annotations

from delivery_keypoints import DELIVERY_JOINTS

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1200
HEATMAP_WIDTH = 120
HEATMAP_HEIGHT = 75
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 300

LABEL_NPZ_NAME = "heatmap_labels_120x75.npz"
POSE3D_NPZ_NAME = "pose3d_nose_pre_limb_15j.npz"
JOINT_RADIUS_CONFIG_NAME = "joint_radius_px_120x75_delivery15.json"
RADIUS_VIDEO_TO_HEATMAP_STRIDE = VIDEO_WIDTH / HEATMAP_WIDTH
DEFAULT_JOINT_RADIUS_HEATMAP_PX = 10.0 / RADIUS_VIDEO_TO_HEATMAP_STRIDE

LIMB_ORDER = ("ankle", "wrist", "wu")
JOINT_NAMES = tuple(DELIVERY_JOINTS)
JOINT_COUNT = len(JOINT_NAMES)

COORDINATE_CONVENTION = "p_nose = p_world_m - p_nose_world_m (translation only, meters)"
