from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from egorear_sim2d.model import EgoRearStage1HeatmapNet


def heatmap_anchors(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return peak coordinates in grid_sample convention and peak values."""
    batch, views, joints, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, views, joints, height * width)
    confidence, index = flat.max(dim=-1)
    y = torch.div(index, width, rounding_mode="floor")
    x = index.remainder(width)
    x = (x.float() + 0.5) / float(width) * 2.0 - 1.0
    y = (y.float() + 0.5) / float(height) * 2.0 - 1.0
    return torch.stack((x, y), dim=-1), confidence


class DeformableJointSampler(nn.Module):
    """EgoRear-style feature sampling around each view's proposal anchors."""

    def __init__(self, feature_channels: int, query_dim: int, points: int = 8):
        super().__init__()
        self.points = int(points)
        self.value_projection = nn.Conv2d(feature_channels, query_dim, kernel_size=1)
        self.offsets = nn.Linear(query_dim, self.points * 2)
        self.weights = nn.Linear(query_dim, self.points)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def forward(
        self,
        query: torch.Tensor,
        feature: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        batch, joints, _ = query.shape
        value = self.value_projection(feature)
        offsets = torch.tanh(self.offsets(query)).reshape(batch, joints, self.points, 2)
        grid = (anchor.unsqueeze(2) + offsets * 0.15).clamp(-1.0, 1.0)
        sampled = F.grid_sample(
            value,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        weights = self.weights(query).softmax(dim=-1).unsqueeze(-1)
        return (sampled * weights).sum(dim=2)


class HeadBCHeatmapRefinementNet(nn.Module):
    """Refine CAM_B/C heatmaps using symmetric multi-view feature fusion."""

    def __init__(
        self,
        stage1: EgoRearStage1HeatmapNet,
        *,
        num_joints: int = 12,
        heatmap_size: tuple[int, int] = (114, 64),
        base_channels: int = 64,
        query_dim: int = 256,
        attention_heads: int = 4,
        sampling_points: int = 8,
        freeze_stage1: bool = True,
    ):
        super().__init__()
        self.stage1 = stage1
        self.num_joints = int(num_joints)
        self.heatmap_size = tuple(int(v) for v in heatmap_size)
        self.freeze_stage1 = bool(freeze_stage1)
        feature_channels = int(base_channels) * 4
        heatmap_elements = self.heatmap_size[0] * self.heatmap_size[1]

        self.joint_embedding = nn.Embedding(self.num_joints, query_dim)
        self.heatmap_query = nn.Sequential(
            nn.Linear(heatmap_elements, query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim, query_dim),
        )
        self.global_query = nn.Linear(feature_channels, query_dim)
        self.confidence_query = nn.Linear(1, query_dim)
        self.sampler = DeformableJointSampler(
            feature_channels, query_dim, points=sampling_points
        )
        self.view_fusion = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim, query_dim),
        )
        self.cross_view_norm = nn.LayerNorm(query_dim)
        self.joint_attention = nn.MultiheadAttention(
            query_dim, attention_heads, batch_first=True
        )
        self.joint_norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim * 2, query_dim),
        )
        self.ffn_norm = nn.LayerNorm(query_dim)
        self.pixel_projection = nn.Conv2d(feature_channels, query_dim, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(0.05))

        if self.freeze_stage1:
            self.stage1.requires_grad_(False)
            self.stage1.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_stage1:
            self.stage1.eval()
        return self

    def stage1_forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 5 or images.shape[1] != 2:
            raise ValueError(f"Expected CAM_B/C images [B,2,3,H,W], got {tuple(images.shape)}")
        batch, views = images.shape[:2]
        flat = images.reshape(batch * views, *images.shape[2:])

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            features = self.stage1.head_branch.forward_features(flat)
            heatmaps = self.stage1.head_branch.head(features)
            return (
                heatmaps.reshape(batch, views, *heatmaps.shape[1:]),
                features.reshape(batch, views, *features.shape[1:]),
            )

        if self.freeze_stage1:
            with torch.no_grad():
                return run()
        return run()

    def forward(
        self,
        images: torch.Tensor,
        proposal_heatmaps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        initial_stage1, features = self.stage1_forward(images)
        proposal = initial_stage1 if proposal_heatmaps is None else proposal_heatmaps
        if proposal.shape != initial_stage1.shape:
            raise ValueError(
                f"Proposal shape {tuple(proposal.shape)} != stage-1 shape {tuple(initial_stage1.shape)}"
            )
        if proposal.shape[2] != self.num_joints:
            raise ValueError("Proposal joint count does not match the refiner")

        batch, views, joints, height, width = proposal.shape
        if (width, height) != self.heatmap_size:
            raise ValueError(
                f"Expected heatmaps {self.heatmap_size}, got {(width, height)}"
            )
        anchors, confidence = heatmap_anchors(proposal.detach())
        base = self.joint_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        refined_views = []
        joint_tokens = []
        pixel_features = []

        for target_view in range(2):
            query = base + self.heatmap_query(proposal[:, target_view].flatten(-2))
            pooled = features[:, target_view].mean(dim=(-2, -1))
            query = query + self.global_query(pooled).unsqueeze(1)
            query = query + self.confidence_query(confidence[:, target_view].unsqueeze(-1))
            sampled = [
                self.sampler(query, features[:, source_view], anchors[:, source_view])
                for source_view in range(2)
            ]
            query = self.cross_view_norm(query + self.view_fusion(torch.cat(sampled, dim=-1)))
            attended, _ = self.joint_attention(query, query, query, need_weights=False)
            query = self.joint_norm(query + attended)
            query = self.ffn_norm(query + self.ffn(query))
            joint_tokens.append(query)

            pixels = self.pixel_projection(features[:, target_view])
            pixels = F.interpolate(
                pixels, size=(height, width), mode="bilinear", align_corners=False
            )
            pixel_features.append(pixels)
            residual = torch.einsum("bjd,bdhw->bjhw", query, pixels)
            residual = residual / float(query.shape[-1]) ** 0.5
            refined_views.append(
                proposal[:, target_view] + self.residual_scale * residual
            )

        refined = torch.stack(refined_views, dim=1)
        tokens = torch.stack(joint_tokens, dim=1)
        pixels = torch.stack(pixel_features, dim=1)
        joint_probability = torch.softmax(
            refined.flatten(-2) * 10.0, dim=-1
        ).reshape_as(refined)
        joint_context = torch.einsum(
            "bvjhw,bvjd->bvdhw", joint_probability, tokens
        )
        return {
            "initial_stage1": initial_stage1,
            "proposal": proposal,
            "refined": refined,
            "joint_tokens": tokens,
            "refined_features": pixels + self.residual_scale * joint_context,
        }


def load_stage1_model(
    checkpoint_path: str | Path,
    *,
    base_channels: int = 64,
    num_head_heatmaps: int | None = None,
) -> EgoRearStage1HeatmapNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if num_head_heatmaps is None:
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        num_head_heatmaps = int(config.get("num_head_heatmaps", 0))
        if num_head_heatmaps <= 0:
            output_weight = state.get("head_branch.head.3.weight")
            if output_weight is None:
                raise ValueError(
                    "Cannot infer stage-1 joint count; pass num_head_heatmaps explicitly"
                )
            num_head_heatmaps = int(output_weight.shape[0])
    model = EgoRearStage1HeatmapNet(
        num_head_heatmaps=int(num_head_heatmaps),
        base_channels=base_channels,
    )
    model.load_state_dict(state, strict=True)
    return model


def refiner_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    target = model.module if hasattr(model, "module") else model
    return {
        key: value
        for key, value in target.state_dict().items()
        if not key.startswith("stage1.")
    }


def load_refiner_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    target = model.module if hasattr(model, "module") else model
    result = target.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys if not key.startswith("stage1.")]
    if unexpected or missing:
        raise RuntimeError(f"Refiner state mismatch: missing={missing}, unexpected={unexpected}")
