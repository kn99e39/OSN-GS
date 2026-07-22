from __future__ import annotations

"""Torch training scene helpers."""

from dataclasses import dataclass, field
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
    view_sampling_seed: int = 0
    _view_permutations: dict[int, tuple[int, ...]] = field(default_factory=dict, init=False, repr=False)

    def _view_indices(self, iteration: int, batch_size: int) -> list[int]:
        """Return reproducible epoch-shuffled indices without replacement."""

        torch = require_torch()
        count = len(self.cameras)
        if count == 0:
            raise ValueError("TorchScene requires at least one camera.")
        batch_size = max(1, int(batch_size))
        first_position = max(0, int(iteration) - 1) * batch_size
        indices: list[int] = []
        for position in range(first_position, first_position + batch_size):
            epoch, offset = divmod(position, count)
            permutation = self._view_permutations.get(epoch)
            if permutation is None:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(self.view_sampling_seed) + epoch)
                permutation = tuple(int(value) for value in torch.randperm(count, generator=generator).tolist())
                self._view_permutations[epoch] = permutation
                if len(self._view_permutations) > 2:
                    oldest = min(self._view_permutations)
                    self._view_permutations.pop(oldest, None)
            indices.append(permutation[offset])
        return indices

    def sample_views(self, iteration: int, batch_size: int = 1) -> TorchImageBatch:
        """Sample an epoch-shuffled batch with deterministic seed replay."""

        torch = require_torch()
        indices = self._view_indices(iteration, batch_size)
        if isinstance(self.images, (list, tuple)):
            selected = [self.images[idx] for idx in indices]
            images = torch.stack(selected, dim=0)
        else:
            image_indices = torch.as_tensor(indices, dtype=torch.long, device=self.images.device)
            images = self.images[image_indices]
        if images.device.type != self.device:
            images = images.to(device=self.device, dtype=torch.float32, non_blocking=images.device.type == "cpu")
        return TorchImageBatch(cameras=[self.cameras[idx] for idx in indices], images=images)
