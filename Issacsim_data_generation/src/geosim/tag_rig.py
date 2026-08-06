from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from geosim.config import TagRigConfig
from geosim.linalg import rotx


@dataclass(frozen=True)
class TagCorner:
    key: str
    point_wrist: np.ndarray


@dataclass(frozen=True)
class WristTagRig:
    corners: tuple[TagCorner, ...]

    @property
    def model_points(self) -> dict[str, np.ndarray]:
        return {corner.key: corner.point_wrist for corner in self.corners}

    def world_points(self, wrist_pos: np.ndarray, wrist_rot: np.ndarray) -> dict[str, np.ndarray]:
        return {corner.key: wrist_pos + wrist_rot @ corner.point_wrist for corner in self.corners}


def make_wrist_tag_rig(config: TagRigConfig) -> WristTagRig:
    """Create two coplanar wrist tags perpendicular to the forearm axis.

    The wrist frame +x axis is expected to follow the forearm from elbow to
    wrist. Both square tags therefore live in the local yz plane. The second tag
    is rotated in that plane by the configured angle, so adjacent tag edges are
    angled without sharing a common edge.
    """
    side = float(config.tag_size_m)
    half = side * 0.5
    angle = math.radians(config.dihedral_angle_deg)
    hinge_offset = np.asarray(config.wrist_to_hinge_offset_m, dtype=float)
    gap = side * 0.1

    corners: list[TagCorner] = []
    local_corners = [
        np.array([0.0, -half, -half]),
        np.array([0.0, half, -half]),
        np.array([0.0, half, half]),
        np.array([0.0, -half, half]),
    ]
    tag_layout = (
        ("tag0", np.array([0.0, -(half + gap * 0.5), 0.0]), 0.0),
        ("tag1", np.array([0.0, half * math.sqrt(2.0) + gap * 0.5, 0.0]), angle),
    )
    for tag_name, center, tag_angle in tag_layout:
        rot = rotx(tag_angle)
        for idx, point in enumerate(local_corners):
            corners.append(TagCorner(f"{tag_name}_c{idx}", hinge_offset + center + rot @ point))
    return WristTagRig(tuple(corners))
