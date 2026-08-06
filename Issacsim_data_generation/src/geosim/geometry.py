from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ray:
    camera_name: str
    origin: np.ndarray
    direction: np.ndarray


def triangulate_rays(rays: list[Ray]) -> np.ndarray:
    if len(rays) < 2:
        raise ValueError("At least two rays are required for triangulation.")
    a = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    eye = np.eye(3)
    for ray in rays:
        d = ray.direction / np.linalg.norm(ray.direction)
        projector = eye - np.outer(d, d)
        a += projector
        b += projector @ ray.origin
    return np.linalg.solve(a, b)
