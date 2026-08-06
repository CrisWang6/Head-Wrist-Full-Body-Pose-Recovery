"""Manifest-driven dataset for head CAM_B/C heatmap refinement."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


REQUIRED_COLUMNS = (
    "image_b",
    "image_c",
    "gt_heatmap_b",
    "gt_heatmap_c",
)


def _load_heatmap(path: Path) -> torch.Tensor:
    array = np.load(path)
    if isinstance(array, np.lib.npyio.NpzFile):
        if "heatmap" not in array:
            raise KeyError(f"{path} must contain a 'heatmap' array")
        array = array["heatmap"]
    tensor = torch.from_numpy(np.asarray(array)).float()
    if tensor.ndim != 3:
        raise ValueError(f"{path} must have shape [J,H,W]")
    return tensor.clamp(0.0, 1.0)


def _placeholder_initial_heatmap(
    ground_truth: torch.Tensor,
    max_shift_px: int,
    noise_std: float,
) -> torch.Tensor:
    shift_y = random.randint(-max_shift_px, max_shift_px)
    shift_x = random.randint(-max_shift_px, max_shift_px)
    shifted = torch.roll(ground_truth, shifts=(shift_y, shift_x), dims=(-2, -1))
    if shift_y > 0:
        shifted[..., :shift_y, :] = 0
    elif shift_y < 0:
        shifted[..., shift_y:, :] = 0
    if shift_x > 0:
        shifted[..., :, :shift_x] = 0
    elif shift_x < 0:
        shifted[..., :, shift_x:] = 0
    return (shifted + torch.randn_like(shifted) * noise_std).clamp(0.0, 1.0)


class HeadStereoHeatmapDataset(Dataset):
    """Load CAM_B/C images and heatmaps listed in a CSV manifest."""

    def __init__(
        self,
        manifest: str | Path,
        image_size: tuple[int, int] = (256, 256),
        placeholder_max_shift_px: int = 5,
        placeholder_noise_std: float = 0.03,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.image_size = image_size
        self.placeholder_max_shift_px = placeholder_max_shift_px
        self.placeholder_noise_std = placeholder_noise_std
        with self.manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"manifest is missing columns: {sorted(missing)}")
            self.rows = list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _load_image(self, value: str) -> torch.Tensor:
        image = Image.open(self._resolve(value)).convert("RGB")
        image = image.resize(self.image_size[::-1], Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return (tensor - mean) / std

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        images = torch.stack(
            (self._load_image(row["image_b"]), self._load_image(row["image_c"])),
            dim=0,
        )
        ground_truth = torch.stack(
            (
                _load_heatmap(self._resolve(row["gt_heatmap_b"])),
                _load_heatmap(self._resolve(row["gt_heatmap_c"])),
            ),
            dim=0,
        )

        initial_paths = (row.get("initial_heatmap_b", ""), row.get("initial_heatmap_c", ""))
        if all(initial_paths):
            initial = torch.stack(
                tuple(_load_heatmap(self._resolve(path)) for path in initial_paths), dim=0
            )
        else:
            initial = torch.stack(
                tuple(
                    _placeholder_initial_heatmap(
                        ground_truth[view],
                        self.placeholder_max_shift_px,
                        self.placeholder_noise_std,
                    )
                    for view in range(2)
                ),
                dim=0,
            )

        if initial.shape != ground_truth.shape:
            initial = F.interpolate(
                initial, size=ground_truth.shape[-2:], mode="bilinear", align_corners=False
            )
        return {
            "images": images,
            "initial_heatmaps": initial,
            "gt_heatmaps": ground_truth,
            "sample_id": row.get("sample_id", str(index)),
        }
