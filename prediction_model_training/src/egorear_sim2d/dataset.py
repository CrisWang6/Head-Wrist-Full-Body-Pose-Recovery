from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from egorear_sim2d.labels import DEFAULT_JOINT_HEATMAP_RADIUS_PX, generate_heatmaps, resolve_joint_radii_px

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class LabelRecord:
    path: Path
    frame_count: int


def discover_label_files(label_root: Path) -> list[Path]:
    label_root = label_root.expanduser()
    if label_root.is_file():
        return [label_root]
    return sorted(label_root.rglob("heatmap_labels_*.npz"))


class MultiViewHeatmapDataset:
    """EgoRear-style dataset returning one frame with all camera views.

    The item format is:
      img: [V, 3, image_h, image_w]
      head_gt_heatmap: [V, J, heatmap_h, heatmap_w]
      wrist_gt_heatmap: [V, 7, heatmap_h, heatmap_w]
      camera_is_head: [V]
      camera_is_wrist: [V]
    """

    def __init__(
        self,
        label_files: list[Path],
        *,
        frame_root: Path | None = None,
        render_root: Path | None = None,
        image_size: tuple[int, int] = (456, 256),
        visible_only_loss: bool = False,
        joint_radius_px: dict[str, float] | None = None,
        default_joint_radius_px: float = 10.0,
    ):
        if not label_files:
            raise FileNotFoundError("No heatmap label files were provided.")
        self.label_files = [Path(path).expanduser() for path in label_files]
        self.frame_root = Path(frame_root).expanduser().resolve() if frame_root is not None else None
        self.render_root = Path(render_root).expanduser().resolve() if render_root is not None else None
        self.image_size = tuple(int(v) for v in image_size)
        self.visible_only_loss = bool(visible_only_loss)
        self.joint_radius_px = (
            dict(joint_radius_px) if joint_radius_px is not None else dict(DEFAULT_JOINT_HEATMAP_RADIUS_PX)
        )
        self.default_joint_radius_px = float(default_joint_radius_px)
        self.records: list[LabelRecord] = []
        self.index: list[tuple[int, int]] = []
        for label_idx, label_path in enumerate(self.label_files):
            data = np.load(label_path, allow_pickle=True)
            frame_count = int(np.asarray(data["keypoints"]).shape[0])
            self.records.append(LabelRecord(label_path, frame_count))
            self.index.extend((label_idx, frame_idx) for frame_idx in range(frame_count))
        self._cache: dict[int, dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, object]:
        label_idx, frame_idx = self.index[int(idx)]
        data = self._load_label(label_idx)
        images = self._load_multiview_images(data, frame_idx)
        head_keypoints = data["head_keypoints"][frame_idx]
        head_visible = data["head_visible"][frame_idx]
        head_joint_mask = data["head_joint_mask"]
        wrist_keypoints = data["wrist_keypoints"][frame_idx]
        wrist_visible = data["wrist_visible"][frame_idx]
        wrist_joint_mask = data["wrist_joint_mask"]
        head_radii = resolve_joint_radii_px(
            data["head_joint_names"],
            self.joint_radius_px,
            default_radius_px=self.default_joint_radius_px,
        )
        wrist_radii = resolve_joint_radii_px(
            data["wrist_joint_names"],
            self.joint_radius_px,
            default_radius_px=self.default_joint_radius_px,
        )
        head_heatmaps = generate_heatmaps(
            head_keypoints,
            head_visible,
            video_size=data["video_size"],
            heatmap_size=data["heatmap_size"],
            joint_radii_px=head_radii,
        )
        wrist_heatmaps = generate_heatmaps(
            wrist_keypoints,
            wrist_visible,
            video_size=data["video_size"],
            heatmap_size=data["heatmap_size"],
            joint_radii_px=wrist_radii,
        )
        head_loss_mask = head_visible if self.visible_only_loss else head_joint_mask
        wrist_loss_mask = wrist_visible if self.visible_only_loss else wrist_joint_mask
        return {
            "img": images.astype(np.float32),
            "head_gt_heatmap": head_heatmaps.astype(np.float32),
            "head_joint_mask": head_joint_mask.astype(np.float32),
            "head_visible": head_visible.astype(np.float32),
            "head_loss_mask": head_loss_mask.astype(np.float32),
            "wrist_gt_heatmap": wrist_heatmaps.astype(np.float32),
            "wrist_joint_mask": wrist_joint_mask.astype(np.float32),
            "wrist_visible": wrist_visible.astype(np.float32),
            "wrist_loss_mask": wrist_loss_mask.astype(np.float32),
            "camera_is_head": data["camera_is_head"].astype(np.float32),
            "camera_is_wrist": data["camera_is_wrist"].astype(np.float32),
            "label_path": str(data["label_path"]),
            "frame_idx": np.asarray(data["frame_indices"][frame_idx], dtype=np.int64),
            "camera_names": data["camera_names"],
            "head_joint_names": data["head_joint_names"],
        }

    def _load_label(self, label_idx: int) -> dict[str, object]:
        if label_idx in self._cache:
            return self._cache[label_idx]
        label_path = self.label_files[label_idx]
        data = np.load(label_path, allow_pickle=True)
        loaded = {
            "label_path": label_path,
            "head_keypoints": np.asarray(data["head_keypoints"], dtype=np.float32),
            "head_visible": np.asarray(data["head_visible"], dtype=bool),
            "head_joint_mask": np.asarray(data["head_joint_mask"], dtype=bool),
            "wrist_keypoints": np.asarray(data["wrist_keypoints"], dtype=np.float32),
            "wrist_visible": np.asarray(data["wrist_visible"], dtype=bool),
            "wrist_joint_mask": np.asarray(data["wrist_joint_mask"], dtype=bool),
            "camera_is_head": np.asarray(data["camera_is_head"], dtype=bool),
            "camera_is_wrist": np.asarray(data["camera_is_wrist"], dtype=bool),
            "camera_names": [str(name) for name in data["camera_names"]],
            "head_joint_names": [
                str(name)
                for name in (
                    data["head_camera_joints"]
                    if "head_camera_joints" in data.files
                    else data["head_joint_names"]
                )
            ],
            "wrist_joint_names": [
                str(name)
                for name in (
                    data["wrist_camera_joints"]
                    if "wrist_camera_joints" in data.files
                    else data["wrist_joint_names"]
                )
            ],
            "video_paths": [str(path) for path in data["video_paths"]],
            "video_size": tuple(int(v) for v in data["video_size"]),
            "heatmap_size": tuple(int(v) for v in data["heatmap_size"]),
            "sigma": float(np.asarray(data["sigma"]).reshape(-1)[0]),
            "source_render_dir": str(data["source_render_dir"][0]) if "source_render_dir" in data.files else "",
            "frame_indices": (
                np.asarray(data["frame_indices"], dtype=np.int64)
                if "frame_indices" in data.files
                else np.arange(int(data["head_keypoints"].shape[0]), dtype=np.int64)
            ),
            "image_paths": (
                np.asarray(data["image_paths"]).astype(str)
                if "image_paths" in data.files
                else None
            ),
        }
        self._cache[label_idx] = loaded
        return loaded

    def _load_multiview_images(self, data: dict[str, object], frame_idx: int) -> np.ndarray:
        width, height = self.image_size
        frames = []
        image_paths = data.get("image_paths")
        if image_paths is not None:
            for image_path in image_paths[frame_idx]:
                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Could not read real frame image: {image_path}")
                frames.append(self._preprocess_frame(frame, width, height))
            return np.stack(frames, axis=0)

        frame_paths = self._frame_paths_for(data, frame_idx)
        if frame_paths is not None:
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Could not read frame image: {frame_path}")
                frames.append(self._preprocess_frame(frame, width, height))
            return np.stack(frames, axis=0)

        for video_path in data["video_paths"]:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
            frames.append(self._preprocess_frame(frame, width, height))
        return np.stack(frames, axis=0)

    def _frame_paths_for(self, data: dict[str, object], frame_idx: int) -> list[Path] | None:
        if self.frame_root is None:
            return None
        source_render_dir = str(data.get("source_render_dir", ""))
        if not source_render_dir:
            return None
        source_path = Path(source_render_dir).expanduser().resolve()
        if self.render_root is not None:
            try:
                rel = source_path.relative_to(self.render_root)
            except ValueError:
                rel = Path(source_path.name)
        else:
            rel = Path(source_path.name)
        sample_dir = self.frame_root / rel
        frame_paths = [sample_dir / str(camera_name) / f"{int(frame_idx):06d}.jpg" for camera_name in data["camera_names"]]
        return frame_paths if all(path.exists() for path in frame_paths) else None

    @staticmethod
    def _preprocess_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
        return frame.transpose(2, 0, 1)


def torch_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    import torch

    keys = (
        "img",
        "head_gt_heatmap",
        "head_joint_mask",
        "head_visible",
        "head_loss_mask",
        "wrist_gt_heatmap",
        "wrist_joint_mask",
        "wrist_visible",
        "wrist_loss_mask",
        "camera_is_head",
        "camera_is_wrist",
        "frame_idx",
    )
    output: dict[str, object] = {}
    for key in keys:
        output[key] = torch.as_tensor(np.stack([item[key] for item in batch], axis=0))
    output["label_path"] = [str(item["label_path"]) for item in batch]
    output["camera_names"] = batch[0]["camera_names"]
    return output
