from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetHeatmapBranch(nn.Module):
    """ResNet-18 style heatmap branch with output stride 4."""

    def __init__(self, out_channels: int, base_channels: int = 64):
        super().__init__()
        c = int(base_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(c, c, blocks=2, stride=1)
        self.layer2 = self._make_layer(c, c * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(c * 2, c * 4, blocks=2, stride=1)
        self.layer4 = self._make_layer(c * 4, c * 4, blocks=2, stride=1)
        self.head = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, int(out_channels), kernel_size=1),
        )

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, int(blocks)):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class EgoRearStage1HeatmapNet(nn.Module):
    """Stage-1 heatmap estimator with separate head-camera and wrist-camera weights."""

    def __init__(
        self,
        *,
        num_head_heatmaps: int = 16,
        num_wrist_heatmaps: int = 7,
        base_channels: int = 64,
    ):
        super().__init__()
        self.num_head_heatmaps = int(num_head_heatmaps)
        self.num_wrist_heatmaps = int(num_wrist_heatmaps)
        self.head_branch = ResNetHeatmapBranch(self.num_head_heatmaps, base_channels=base_channels)
        self.wrist_branch = ResNetHeatmapBranch(self.num_wrist_heatmaps, base_channels=base_channels)

    def forward(self, img: torch.Tensor, branch: str = "all") -> dict[str, torch.Tensor]:
        if img.ndim != 5:
            raise ValueError(f"Expected [B,V,3,H,W], got {tuple(img.shape)}")
        if branch not in {"all", "head", "wrist"}:
            raise ValueError(f"Unknown branch {branch!r}")
        batch, views = img.shape[:2]
        x = img.reshape(batch * views, *img.shape[2:])
        out: dict[str, torch.Tensor] = {}
        if branch in {"all", "head"}:
            head = self.head_branch(x)
            out["head"] = head.reshape(batch, views, *head.shape[1:])
        if branch in {"all", "wrist"}:
            wrist = self.wrist_branch(x)
            out["wrist"] = wrist.reshape(batch, views, *wrist.shape[1:])
        return out
