from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EgoRearPose3DNet(nn.Module):
    """EgoRear-style stereo lifting from RGB, refined heatmaps and features.

    The design follows the official third stage: an MLP pose proposal is refined
    by joint-wise transformer layers using spatial evidence from both views.
    """

    def __init__(
        self,
        *,
        num_joints: int = 12,
        refined_feature_channels: int = 256,
        hidden_dim: int = 128,
        transformer_layers: int = 3,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(self.num_joints, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.refined_feature_projection = nn.Sequential(
            nn.Conv2d(refined_feature_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        fused_channels = 64 + 32 + hidden_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pose_proposal = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, self.num_joints * 3),
        )
        self.joint_embedding = nn.Embedding(self.num_joints, hidden_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.joint_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.pose_offset = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def _heatmap_statistics(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views, joints, height, width = heatmaps.shape
        flat = heatmaps.flatten(-2)
        probability = torch.softmax(flat * 10.0, dim=-1)
        x_axis = torch.linspace(-1.0, 1.0, width, device=heatmaps.device, dtype=heatmaps.dtype)
        y_axis = torch.linspace(-1.0, 1.0, height, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
        x = (probability * grid_x.flatten()).sum(-1)
        y = (probability * grid_y.flatten()).sum(-1)
        confidence = flat.max(-1).values
        return torch.stack((x, y), dim=-1), confidence

    def forward(
        self,
        rgb_images: torch.Tensor,
        refined_heatmaps: torch.Tensor,
        refined_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if rgb_images.ndim != 5 or rgb_images.shape[1] != 2:
            raise ValueError(f"Expected RGB [B,2,3,H,W], got {tuple(rgb_images.shape)}")
        batch, views = rgb_images.shape[:2]
        rgb = self.rgb_encoder(rgb_images.flatten(0, 1))
        rgb = F.interpolate(
            rgb, size=refined_heatmaps.shape[-2:], mode="bilinear", align_corners=False
        )
        heatmap_feature = self.heatmap_encoder(refined_heatmaps.flatten(0, 1))
        refined = self.refined_feature_projection(refined_features.flatten(0, 1))
        refined = F.interpolate(
            refined, size=refined_heatmaps.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = self.fusion(torch.cat((rgb, heatmap_feature, refined), dim=1))
        fused = fused.reshape(batch, views, *fused.shape[1:])

        pooled = fused.mean(dim=(-2, -1)).flatten(1)
        proposal = self.pose_proposal(pooled).reshape(batch, self.num_joints, 3)
        anchors, confidence = self._heatmap_statistics(refined_heatmaps)
        sampled_views = []
        for view in range(2):
            grid = anchors[:, view].unsqueeze(2)
            sampled = F.grid_sample(
                fused[:, view],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(-1).transpose(1, 2)
            sampled_views.append(sampled)
        stereo_joint_feature = torch.cat(sampled_views, dim=-1)
        heatmap_state = torch.cat(
            (anchors[:, 0], confidence[:, 0, :, None], anchors[:, 1], confidence[:, 1, :, None]),
            dim=-1,
        )
        query = self.query_projection(
            torch.cat((stereo_joint_feature, proposal, heatmap_state), dim=-1)
        )
        query = query + self.joint_embedding.weight.unsqueeze(0)
        query = self.joint_transformer(query)
        refined_pose = proposal + self.pose_offset(query)
        return {"proposal": proposal, "pose3d": refined_pose}


class EgoRearStage3Pipeline(nn.Module):
    """Freeze the trained two-stage 2D stack and train only the 3D lifter."""

    def __init__(self, stage2: nn.Module, pose3d: EgoRearPose3DNet):
        super().__init__()
        self.stage2 = stage2
        self.pose3d = pose3d
        self.stage2.requires_grad_(False)
        self.stage2.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage2.eval()
        return self

    def forward(self, rgb_images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            refinement = self.stage2(rgb_images)
        output = self.pose3d(
            rgb_images,
            refinement["refined"],
            refinement["refined_features"],
        )
        output["refined_heatmaps"] = refinement["refined"]
        return output
