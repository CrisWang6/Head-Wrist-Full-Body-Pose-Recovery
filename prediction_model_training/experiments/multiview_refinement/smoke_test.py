"""Shape and gradient smoke test; no dataset or stage-one model is required."""

import torch

from model import StereoHeatmapRefiner


def main() -> None:
    torch.manual_seed(7)
    model = StereoHeatmapRefiner(num_joints=16, feature_dim=32, query_dim=64)
    images = torch.randn(2, 2, 3, 256, 256)
    initial = torch.rand(2, 2, 16, 64, 64)
    refined = model(images, initial)
    assert refined.shape == initial.shape
    assert torch.isfinite(refined).all()
    refined.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    print(f"smoke test passed: {tuple(refined.shape)}")


if __name__ == "__main__":
    main()
