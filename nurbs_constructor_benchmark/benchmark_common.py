"""Shared duck-typed scoring helpers reused by the Phase 3/4 benchmark reports.

``metrics.py``'s ``ground_truth_metrics``/``patch_union_metrics``/
``support_boundary_conformality`` were written against the real
``TorchPipelineState``/``TorchGaussianModel`` classes, but only read
``state.surface_patches`` and ``state.model.cluster_ids``. ``PseudoState``/
``PseudoModel`` are the minimal stand-ins that let every constructor variant
(legacy, Stage 1, Phase 3, Phase 4, ...) share the exact same scoring code
instead of each report re-implementing its own metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PseudoModel:
    cluster_ids: Any


@dataclass
class PseudoState:
    """Duck-typed stand-in for a ``TorchPipelineState`` (only the fields the
    reused ``metrics.py`` functions actually read)."""

    surface_patches: list
    model: PseudoModel


def uv_support_payload(surface: Any) -> dict[str, Any] | None:
    mask = getattr(surface, "uv_support_mask", None)
    if mask is None:
        return None
    return {"resolution": [int(mask.shape[0]), int(mask.shape[1])], "mask": mask.detach().cpu().bool().tolist()}
