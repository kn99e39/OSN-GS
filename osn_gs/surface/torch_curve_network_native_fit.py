from __future__ import annotations

"""Worklog 97 -- curve-network-native NURBS fitting and paired-comparison
diagnostics against the existing PCA-UV point-based path.

Fixed and unmodified in this batch: NURBS degree (2) and control-grid
resolution (6x6), the Worklog 95/96 latent-surface support, seed/curve
tracing, continuous-support validation, curve-family/block construction,
and the held-out evaluation convention already established in
``scripts/devtools/latent_surface_curve_network_prototype_replay.py``.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_curve_network_uv_parameterization import CurveNetworkUV, build_curve_network_uv
from osn_gs.surface.torch_latent_surface_curve_families import CurveNetworkBlock
from osn_gs.surface.torch_nurbs import (
    TorchNURBSSurface,
    fit_torch_visible_surface_from_uv,
    fit_torch_visible_surface_lsq,
)
from osn_gs.utils.torch_ops import require_torch

DEGREE_U = 2
DEGREE_V = 2
RESOLUTION_U = 6
RESOLUTION_V = 6


@dataclass(frozen=True)
class ResidualStats:
    mean: float | None
    median: float | None
    p95: float | None
    max: float | None
    count: int


def _residual_stats(surface: TorchNURBSSurface, points: Any, uv: Any) -> ResidualStats:
    torch = require_torch()
    if points is None or int(points.shape[0]) == 0:
        return ResidualStats(None, None, None, None, 0)
    predicted = surface.evaluate(uv)
    error = (predicted - points).norm(dim=1)
    values = error.detach().cpu()
    if values.numel() == 0:
        return ResidualStats(None, None, None, None, 0)
    return ResidualStats(
        float(values.mean().item()), float(values.median().item()),
        float(torch.quantile(values, 0.95).item()), float(values.max().item()), int(values.numel()),
    )


@dataclass(frozen=True)
class CurveNetworkNativeFitResult:
    valid_parameterization: bool
    invalid_reason: str | None
    surface: TorchNURBSSurface | None
    curve_network_uv: CurveNetworkUV
    overall_residual: ResidualStats
    trace_family_residual: ResidualStats  # "U-family residual" (curves at fixed u)
    rung_family_residual: ResidualStats  # "V-family / correspondence residual" (curves at fixed v)


def fit_curve_network_native(block: CurveNetworkBlock) -> CurveNetworkNativeFitResult:
    """Curve-network-native fit: parameterization comes ENTIRELY from
    :func:`build_curve_network_uv` (chord-length along curves, reconciled
    across the U/V families) -- never PCA. Fails closed (no surface) if
    the network's own correspondence is inconsistent; never repaired by
    falling back to a PCA-UV fit."""

    curve_network_uv = build_curve_network_uv(block)
    if not curve_network_uv.valid:
        empty = ResidualStats(None, None, None, None, 0)
        return CurveNetworkNativeFitResult(False, curve_network_uv.invalid_reason, None, curve_network_uv, empty, empty, empty)

    surface = fit_torch_visible_surface_from_uv(
        curve_network_uv.points, curve_network_uv.uv,
        resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V, degree_u=DEGREE_U, degree_v=DEGREE_V,
    )
    overall = _residual_stats(surface, curve_network_uv.points, curve_network_uv.uv)

    torch = require_torch()
    provenance = curve_network_uv.provenance
    trace_mask = torch.tensor([tag.startswith("trace_family:") for tag in provenance], dtype=torch.bool)
    rung_mask = torch.tensor([tag.startswith("rung_family:") for tag in provenance], dtype=torch.bool)
    trace_points = curve_network_uv.points[trace_mask] if bool(trace_mask.any()) else curve_network_uv.points[:0]
    trace_uv = curve_network_uv.uv[trace_mask] if bool(trace_mask.any()) else curve_network_uv.uv[:0]
    rung_points = curve_network_uv.points[rung_mask] if bool(rung_mask.any()) else curve_network_uv.points[:0]
    rung_uv = curve_network_uv.uv[rung_mask] if bool(rung_mask.any()) else curve_network_uv.uv[:0]

    trace_residual = _residual_stats(surface, trace_points, trace_uv)
    rung_residual = _residual_stats(surface, rung_points, rung_uv)
    return CurveNetworkNativeFitResult(True, None, surface, curve_network_uv, overall, trace_residual, rung_residual)


@dataclass(frozen=True)
class PcaUvFitResult:
    surface: TorchNURBSSurface | None
    fit_error: str | None
    overall_residual: ResidualStats


def fit_pca_uv(block: CurveNetworkBlock) -> PcaUvFitResult:
    """Baseline A: identical curve-network samples, existing PCA-UV point
    fit path (:func:`fit_torch_visible_surface_lsq`, unchanged), same
    6x6/degree-2 capacity."""

    if block.all_points is None or int(block.all_points.shape[0]) < 4:
        empty = ResidualStats(None, None, None, None, 0)
        return PcaUvFitResult(None, "insufficient_points", empty)
    try:
        surface, uv = fit_torch_visible_surface_lsq(
            block.all_points, resolution_u=RESOLUTION_U, resolution_v=RESOLUTION_V,
            degree_u=DEGREE_U, degree_v=DEGREE_V,
        )
    except Exception as exc:  # noqa: BLE001
        empty = ResidualStats(None, None, None, None, 0)
        return PcaUvFitResult(None, f"{type(exc).__name__}: {exc}", empty)
    residual = _residual_stats(surface, block.all_points, uv)
    return PcaUvFitResult(surface, None, residual)
