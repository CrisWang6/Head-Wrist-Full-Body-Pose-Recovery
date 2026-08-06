"""Two-view heatmap refinement inspired by EgoRear's MVF module."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _anchors_from_heatmaps(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized peak coordinates and peak confidence for [B,V,J,H,W]."""
    b, v, j, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, v, j, h * w)
    confidence, index = flat.max(dim=-1)
    y = torch.div(index, w, rounding_mode="floor")
    x = index.remainder(w)
    x = (x.float() + 0.5) / w * 2.0 - 1.0
    y = (y.float() + 0.5) / h * 2.0 - 1.0
    return torch.stack((x, y), dim=-1), confidence


class SharedImageEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = images.shape
        features = self.net(images.reshape(b * v, c, h, w))
        return features.reshape(b, v, *features.shape[1:])


class DeformableAnchorSampler(nn.Module):
    """Sample each view around its own first-stage joint anchors."""

    def __init__(self, query_dim: int, feature_dim: int, points: int = 8) -> None:
        super().__init__()
        self.points = points
        self.offsets = nn.Linear(query_dim, points * 2)
        self.weights = nn.Linear(query_dim, points)
        self.value = nn.Conv2d(feature_dim, query_dim, 1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def forward(
        self,
        query: torch.Tensor,
        feature: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        b, joints, _ = query.shape
        value = self.value(feature)
        offsets = torch.tanh(self.offsets(query)).reshape(b, joints, self.points, 2)
        offsets = offsets * 0.15
        grid = (anchors.unsqueeze(2) + offsets).clamp(-1.0, 1.0)
        sampled = F.grid_sample(
            value,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.permute(0, 2, 3, 1)
        weights = self.weights(query).softmax(dim=-1).unsqueeze(-1)
        return (sampled * weights).sum(dim=2)


class StereoHeatmapRefiner(nn.Module):
    """Refine a pair of first-stage heatmaps with symmetric multi-view fusion.

    Inputs:
        images: [B, 2, 3, image_h, image_w], ImageNet-normalized RGB.
        initial_heatmaps: [B, 2, J, heatmap_h, heatmap_w], values in [0, 1].
    Output:
        refined heatmaps with the same shape as initial_heatmaps.
    """

    def __init__(
        self,
        num_joints: int,
        feature_dim: int = 128,
        query_dim: int = 256,
        attention_heads: int = 4,
        sampling_points: int = 8,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.encoder = SharedImageEncoder(feature_dim)
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(num_joints, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        self.joint_embedding = nn.Embedding(num_joints, query_dim)
        self.confidence_embedding = nn.Linear(1, query_dim)
        self.anchor_sampler = DeformableAnchorSampler(
            query_dim, feature_dim, sampling_points
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
        self.pixel_projection = nn.Conv2d(feature_dim, query_dim, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        images: torch.Tensor,
        initial_heatmaps: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != 2:
            raise ValueError("images must have shape [B,2,3,H,W]")
        if initial_heatmaps.ndim != 5 or initial_heatmaps.shape[1] != 2:
            raise ValueError("initial_heatmaps must have shape [B,2,J,H,W]")
        if initial_heatmaps.shape[2] != self.num_joints:
            raise ValueError("initial heatmap joint count does not match the model")

        b, views, joints, hm_h, hm_w = initial_heatmaps.shape
        image_features = self.encoder(images)
        feat_h, feat_w = image_features.shape[-2:]
        heatmap_features = self.heatmap_encoder(
            initial_heatmaps.reshape(b * views, joints, hm_h, hm_w)
        )
        heatmap_features = F.interpolate(
            heatmap_features,
            size=(feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, views, -1, feat_h, feat_w)
        features = image_features + heatmap_features

        anchors, confidence = _anchors_from_heatmaps(initial_heatmaps.detach())
        base_query = self.joint_embedding.weight.unsqueeze(0).expand(b, -1, -1)
        refined_views = []
        eps = 1e-4

        for target_view in range(2):
            query = base_query + self.confidence_embedding(
                confidence[:, target_view].unsqueeze(-1)
            )
            sampled_per_view = [
                self.anchor_sampler(query, features[:, source_view], anchors[:, source_view])
                for source_view in range(2)
            ]
            fused = self.view_fusion(torch.cat(sampled_per_view, dim=-1))
            query = self.cross_view_norm(query + fused)
            attended, _ = self.joint_attention(query, query, query, need_weights=False)
            query = self.joint_norm(query + attended)
            query = self.ffn_norm(query + self.ffn(query))

            pixels = self.pixel_projection(features[:, target_view])
            pixels = F.interpolate(
                pixels, size=(hm_h, hm_w), mode="bilinear", align_corners=False
            )
            residual = torch.einsum("bjd,bdhw->bjhw", query, pixels)
            residual = residual / (query.shape[-1] ** 0.5)
            initial = initial_heatmaps[:, target_view].clamp(eps, 1.0 - eps)
            initial_logits = torch.logit(initial)
            refined = torch.sigmoid(initial_logits + self.residual_scale * residual)
            refined_views.append(refined)

        return torch.stack(refined_views, dim=1)
