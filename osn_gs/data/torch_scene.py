from __future__ import annotations

"""Torch training scene helpers."""

from dataclasses import dataclass
from typing import Any

from osn_gs.render.torch_fallback import TorchCamera
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchImageBatch:
    """Batch passed from the scene sampler to the trainer."""

    cameras: list[TorchCamera]
    images: Any


@dataclass
class TorchScene:
    """Minimal scene protocol required by the torch trainer."""

    initial_points: Any
    initial_colors: Any
    cameras: list[TorchCamera]
    images: Any
    device: str
    extent: float = 1.0

    def sample_views(self, iteration: int, batch_size: int = 1) -> TorchImageBatch:
        """Sample a deterministic batch of views for training."""

        torch = require_torch()
        count = len(self.cameras)
        if count == 0:
            raise ValueError("TorchScene requires at least one camera.")
        indices = [(iteration + offset) % count for offset in range(batch_size)]
        image_indices = torch.as_tensor(indices, dtype=torch.long, device=self.images.device)
        images = self.images[image_indices]
        if images.device.type != self.device:
            images = images.to(device=self.device, dtype=torch.float32, non_blocking=self.images.device.type == "cpu")
        return TorchImageBatch(cameras=[self.cameras[idx] for idx in indices], images=images)
