from __future__ import annotations

import sys
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from intrinsic_chart_atlas_gate_replay import (  # noqa: E402
    CATEGORY_DOMAIN_INVALID,
    CATEGORY_INSUFFICIENT_PATCH_SUPPORT,
    MIN_PATCH_SUPPORT,
    _fit_and_classify,
)


def test_insufficient_patch_support_never_conflated_with_domain_invalid():
    # A tiny but domain-VALID chart (well below the fixed 6x6 grid's own
    # 36-control-point count) must be reported as
    # CHART_DOMAIN_VALID_BUT_INSUFFICIENT_PATCH_SUPPORT, never as
    # PARAMETER_DOMAIN_INVALID -- these are semantically different: one
    # means the domain itself is broken, the other means the domain is
    # fine but there isn't enough evidence for the FIXED patch model.
    small_count = MIN_PATCH_SUPPORT - 5
    assert small_count > 0
    points = torch.randn(small_count, 3)
    uv = torch.rand(small_count, 2)
    record = _fit_and_classify(points, uv, torch.randn(5, 3), 1.0)
    assert record["category"] == CATEGORY_INSUFFICIENT_PATCH_SUPPORT
    assert record["category"] != CATEGORY_DOMAIN_INVALID


def test_sufficient_support_reaches_actual_fit_attempt():
    # With enough points (>= MIN_PATCH_SUPPORT), the classifier must at
    # least ATTEMPT a real fit rather than short-circuiting to the
    # insufficient-support category.
    count = MIN_PATCH_SUPPORT + 20
    coords = torch.linspace(-1.0, 1.0, count)
    points = torch.stack([coords, coords, torch.zeros(count)], dim=1)
    uv = torch.stack([(coords + 1.0) / 2.0, (coords + 1.0) / 2.0], dim=1)
    record = _fit_and_classify(points, uv, torch.randn(5, 3), 1.0)
    assert record["category"] != CATEGORY_INSUFFICIENT_PATCH_SUPPORT
