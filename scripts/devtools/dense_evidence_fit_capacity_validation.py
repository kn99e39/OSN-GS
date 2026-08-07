"""Worklog 68: Dense-Evidence Fit Capacity and Fidelity Calibration.

Diagnoses WHY worklog 67 reclassified 20/21 patches `extrapolative` once
region-owned full evidence replaced representative-only support: fitting
CAPACITY insufficiency, DENSITY DEPENDENCE of the dense-nearest-neighbor
normalization metric itself, or non-uniform evidence WEIGHTING. Region
formation, representative topology, chart boundary, and ownership gating
are never touched -- this operates entirely on the boundary+evidence that
worklog 67 already recovered (`bundle.region_owned_full_evidence_fits`,
production, unmodified).

Judgment (per patch, applied literally per the task's own rule):
  - raw AND held-out error both drop as grid resolution rises, geometry
    stays safe -> "capacity_insufficient", minimum sufficient resolution noted.
  - raw error is similar across resolutions but only the dense-NN-normalized
    result changes/fails -> "metric_density_dependent".
  - only TRAINING error drops at higher resolution while held-out error does
    not improve (or folding/degenerate cells increase) -> "overfitting",
    resolution increase not adopted.
  - density-compensated weighting alone improves things (grid unchanged)
    -> "weighting_problem".
  - none of the above cleanly applies -> "inconclusive".

Global orientation reversal (single arbitrary reference direction, never a
defect by itself) and LOCAL orientation folding (adjacent grid samples
disagreeing, a real geometric red flag) are reported as SEPARATE fields
(`torch_local_orientation_folding.py` vs the existing global
`compute_orientation_consistency`) and never conflated.

`validate_simple_closed_loop` is reused unmodified and reported strictly as
a BOUNDARY-LOOP simple-polygon check -- it is not, and is not described as,
a surface self-intersection check (nothing in this pipeline checks that).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

import osn_gs.core.torch_pipeline  # noqa: F401 -- resolve osn_gs's own circular-import order first
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, pca_parameterize_points
from osn_gs.surface.torch_parametric_diagnostics import compute_orientation_consistency, compute_parametric_jacobian_metrics

import baseline_ply_replay_analysis as baseline_ply_analysis  # noqa: E402

GRID_RESOLUTIONS = (6, 8, 10)
WEIGHTING_SCHEMES = ("uniform", "density_compensated")
HOLDOUT_CHECKER_K = 4  # PCA-uv checkerboard cell count per axis for the deterministic spatial split
SAMPLE_RESOLUTION = 24
DENSITY_KNN = 8


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "max": float(values.max()),
    }


def _median_nn_spacing(points: torch.Tensor) -> float:
    n = int(points.shape[0])
    if n < 2:
        return 1e-6
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    value = float(d.min(dim=1).values.median())
    return value if value > 0 else 1e-6


def _deterministic_spatial_holdout_split(evidence: torch.Tensor, k: int = HOLDOUT_CHECKER_K) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA-uv checkerboard split -- deterministic, spatially interleaved
    (not clustered), reproducible given the same evidence set. Reuses the
    existing `pca_parameterize_points` (unmodified) as the deterministic
    spatial layout."""

    if int(evidence.shape[0]) < 8:
        # too few points for a meaningful held-out set; everything trains.
        return evidence, evidence[:0]
    uv = pca_parameterize_points(evidence)
    cell_u = (uv[:, 0] * k).clamp(0, k - 1e-6).floor().long()
    cell_v = (uv[:, 1] * k).clamp(0, k - 1e-6).floor().long()
    holdout_mask = ((cell_u + cell_v) % 2) == 0
    if int(holdout_mask.sum()) == 0 or int((~holdout_mask).sum()) == 0:
        return evidence, evidence[:0]
    return evidence[~holdout_mask], evidence[holdout_mask]


def _density_compensated_weights(points: torch.Tensor, k: int = DENSITY_KNN) -> torch.Tensor:
    """Inverse local-density weight (bigger spacing to nearest neighbors ->
    higher weight), mean-normalized to 1 so the overall LSQ regularization
    scale stays comparable to the uniform-weight fit."""

    n = int(points.shape[0])
    if n < 2:
        return torch.ones((n,), dtype=points.dtype, device=points.device)
    neighbors = min(k, n - 1)
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    local_spacing = d.topk(neighbors, dim=1, largest=False).values.mean(dim=1)
    weight = local_spacing.clamp_min(1e-9)
    return weight / weight.mean().clamp_min(1e-9)


def _sample_surface_with_normals(surface, resolution: int = SAMPLE_RESOLUTION):
    device = surface.control_grid.device
    dtype = surface.control_grid.dtype
    grid = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    su, sv = torch.meshgrid(grid, grid, indexing="ij")
    uv = torch.stack((su.reshape(-1), sv.reshape(-1)), dim=1)
    points, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
    normals = torch.cross(deriv_u, deriv_v, dim=1)
    return points, deriv_u, deriv_v, normals


def _surface_area(sample_points: torch.Tensor, resolution: int) -> float:
    pts = sample_points.reshape(resolution, resolution, 3)
    p00, p01, p10, p11 = pts[:-1, :-1], pts[:-1, 1:], pts[1:, :-1], pts[1:, 1:]
    a1 = 0.5 * torch.cross(p01 - p00, p10 - p00, dim=-1).norm(dim=-1)
    a2 = 0.5 * torch.cross(p11 - p01, p10 - p01, dim=-1).norm(dim=-1)
    return float((a1 + a2).sum())


def _residual_decomposition(evidence: torch.Tensor, sample_points: torch.Tensor, sample_normals: torch.Tensor) -> dict:
    """Per-evidence-point offset from its nearest sample, split into the
    normal component (along that sample's local unit normal) and the
    tangent component (orthogonal remainder)."""

    if int(evidence.shape[0]) == 0:
        return {"normal_abs": np.array([]), "tangent": np.array([]), "raw_point_to_surface": np.array([])}
    d = torch.cdist(evidence, sample_points)
    nearest_idx = d.argmin(dim=1)
    raw_distance = d.min(dim=1).values
    nearest_sample = sample_points[nearest_idx]
    unit_normal = sample_normals[nearest_idx] / sample_normals[nearest_idx].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    offset = evidence - nearest_sample
    normal_component = (offset * unit_normal).sum(dim=-1)
    tangent_component = (offset - normal_component.unsqueeze(-1) * unit_normal).norm(dim=-1)
    return {
        "normal_abs": normal_component.abs().detach().cpu().numpy(),
        "tangent": tangent_component.detach().cpu().numpy(),
        "raw_point_to_surface": raw_distance.detach().cpu().numpy(),
    }


def _robust_normal_noise_scale(evidence: torch.Tensor, reference_surface) -> float:
    """Median absolute deviation of the normal-direction residual against a
    FIXED reference fit (base uniform 6x6), used only as an independent
    normalization yardstick -- never re-derived per resolution/weighting
    under test, to avoid circularity."""

    sample_points, _, _, normals = _sample_surface_with_normals(reference_surface)
    decomposition = _residual_decomposition(evidence, sample_points, normals)
    normal_abs = decomposition["normal_abs"]
    if normal_abs.size < 2:
        return 1e-6
    median = np.median(normal_abs)
    mad = np.median(np.abs(normal_abs - median)) * 1.4826
    return float(mad) if mad > 0 else 1e-6


def _raw_and_normalized_errors(
    evidence: torch.Tensor, surface, scale_dense_nn: float, scale_representative: float,
    scale_normal_noise: float, scale_patch_diameter: float,
) -> dict:
    sample_points, _, _, normals = _sample_surface_with_normals(surface)
    fwd_raw = torch.cdist(evidence, sample_points).min(dim=1).values.detach().cpu().numpy() if int(evidence.shape[0]) else np.array([])
    bwd_raw = torch.cdist(sample_points, evidence).min(dim=1).values.detach().cpu().numpy() if int(evidence.shape[0]) else np.array([])
    decomposition = _residual_decomposition(evidence, sample_points, normals)

    def _norm(values, scale):
        return _percentiles(values / max(scale, 1e-9))

    return {
        "point_to_surface_raw": _percentiles(fwd_raw),
        "surface_to_evidence_raw": _percentiles(bwd_raw),
        "point_to_surface_norm_dense_nn": _norm(fwd_raw, scale_dense_nn),
        "surface_to_evidence_norm_dense_nn": _norm(bwd_raw, scale_dense_nn),
        "point_to_surface_norm_representative": _norm(fwd_raw, scale_representative),
        "surface_to_evidence_norm_representative": _norm(bwd_raw, scale_representative),
        "point_to_surface_norm_normal_noise": _norm(fwd_raw, scale_normal_noise),
        "surface_to_evidence_norm_normal_noise": _norm(bwd_raw, scale_normal_noise),
        "point_to_surface_norm_patch_diameter": _norm(fwd_raw, scale_patch_diameter),
        "surface_to_evidence_norm_patch_diameter": _norm(bwd_raw, scale_patch_diameter),
        "normal_direction_residual": _percentiles(decomposition["normal_abs"]),
        "tangent_direction_residual": _percentiles(decomposition["tangent"]),
    }


# A single, pre-declared "meaningfully changed" bound applied uniformly to
# every drop-ratio comparison below -- never adjusted per patch/result.
MEANINGFUL_DROP_RATIO = 0.10

# Absolute floor for "local folding got worse" -- a bare epsilon comparison
# on `local_fold_fraction` would flag almost any resolution change (a single
# extra folded cell out of 500+ adjacent pairs moves the fraction by a tiny
# amount that is still > any epsilon). At least 1% of ALL adjacent sample
# pairs must newly disagree -- a genuinely visible fraction of the sampled
# grid, not single-cell noise -- to count as a real geometric degradation.
FOLD_INCREASE_ABS_THRESHOLD = 0.01


def _drop_ratio(values: list[float | None]) -> float:
    """Fractional decrease from the lowest-resolution value to the highest,
    0.0 if not enough data or the series does not monotonically improve."""

    vals = [v for v in values if v is not None]
    if len(vals) < 2 or vals[0] <= 1e-12:
        return 0.0
    if not all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])):
        return 0.0  # not monotonically non-increasing -- no clean "drop" to report
    return (vals[0] - vals[-1]) / vals[0]


def _relative_spread(values: list[float | None]) -> float:
    """(max - min) / min across the series, direction-agnostic -- used to
    detect INSTABILITY of a metric across resolutions, not a monotonic
    trend (unlike `_drop_ratio`)."""

    vals = [v for v in values if v is not None]
    if len(vals) < 2 or min(vals) <= 1e-12:
        return 0.0
    return (max(vals) - min(vals)) / min(vals)


def _classify_patch(sweep: dict) -> tuple[str, list[str]]:
    """Literal decision procedure for the task's four judgment rules, checked
    in this fixed order (a geometry red flag always overrides an apparent
    improvement; weighting_problem/metric_density_dependent are only
    evaluated once neither capacity nor overfitting cleanly applies)."""

    reasons = []

    def series(scheme: str, split: str, key: str) -> list[float | None]:
        return [sweep[r][scheme][split][key]["p95"] for r in GRID_RESOLUTIONS]

    train_raw = series("uniform", "train", "surface_to_evidence_raw")
    held_out_raw = series("uniform", "holdout", "surface_to_evidence_raw")
    degenerate = [sweep[r]["uniform"]["jacobian_near_degenerate_count"] for r in GRID_RESOLUTIONS]
    fold_fraction = [sweep[r]["uniform"]["local_fold_fraction"] for r in GRID_RESOLUTIONS]

    train_drop = _drop_ratio(train_raw)
    held_out_drop = _drop_ratio(held_out_raw) if any(v is not None for v in held_out_raw) else None
    degenerate_increases = any(d > degenerate[0] for d in degenerate[1:])
    fold_increases = any(f > fold_fraction[0] + FOLD_INCREASE_ABS_THRESHOLD for f in fold_fraction[1:])

    if degenerate_increases or fold_increases:
        reasons.append(f"geometry_degrades_with_resolution degenerate={degenerate} fold_fraction={fold_fraction}")
        return "overfitting", reasons
    if train_drop >= MEANINGFUL_DROP_RATIO and (held_out_drop is None or held_out_drop < MEANINGFUL_DROP_RATIO):
        reasons.append(f"train_p95_drop_ratio={train_drop:.2f} held_out_p95_drop_ratio={held_out_drop}")
        return "overfitting", reasons
    if train_drop >= MEANINGFUL_DROP_RATIO and held_out_drop is not None and held_out_drop >= MEANINGFUL_DROP_RATIO:
        reasons.append(f"train_p95_drop_ratio={train_drop:.2f} held_out_p95_drop_ratio={held_out_drop:.2f} geometry_safe")
        return "capacity_insufficient", reasons

    dense_norm = series("uniform", "train", "surface_to_evidence_norm_dense_nn")
    rep_norm = series("uniform", "train", "surface_to_evidence_norm_representative")
    dense_norm_spread = _relative_spread(dense_norm)
    rep_norm_spread = _relative_spread(rep_norm)
    if train_drop < MEANINGFUL_DROP_RATIO and dense_norm_spread >= MEANINGFUL_DROP_RATIO and dense_norm_spread > rep_norm_spread + MEANINGFUL_DROP_RATIO:
        reasons.append(
            f"raw_p95_stable(drop={train_drop:.2f}) but_dense_nn_normalized_swings(spread={dense_norm_spread:.2f})_vs_representative_normalized(spread={rep_norm_spread:.2f})"
        )
        return "metric_density_dependent", reasons

    weighted_train_p95 = sweep[GRID_RESOLUTIONS[0]]["density_compensated"]["train"]["surface_to_evidence_raw"]["p95"]
    uniform_train_p95 = sweep[GRID_RESOLUTIONS[0]]["uniform"]["train"]["surface_to_evidence_raw"]["p95"]
    if weighted_train_p95 is not None and uniform_train_p95 is not None and uniform_train_p95 > 1e-12:
        weighting_drop = (uniform_train_p95 - weighted_train_p95) / uniform_train_p95
        if weighting_drop >= MEANINGFUL_DROP_RATIO:
            reasons.append(f"density_compensated_weighting_reduces_base_resolution_p95_by={weighting_drop:.2f}")
            return "weighting_problem", reasons

    reasons.append(f"no_criterion_cleanly_matched train_drop={train_drop:.2f} held_out_drop={held_out_drop} dense_norm_spread={dense_norm_spread:.2f}")
    return "inconclusive", reasons


def analyze_patch(chart_type: str, region_id: int, boundary_points: torch.Tensor, full_evidence: torch.Tensor) -> dict:
    train_evidence, holdout_evidence = _deterministic_spatial_holdout_split(full_evidence)
    scale_dense_nn = _median_nn_spacing(train_evidence) if int(train_evidence.shape[0]) >= 2 else _median_nn_spacing(boundary_points)
    scale_representative = _median_nn_spacing(boundary_points)
    scale_patch_diameter = float(torch.cdist(boundary_points, boundary_points).max()) if int(boundary_points.shape[0]) >= 2 else 1e-6

    base_observed = torch.cat((boundary_points, train_evidence), dim=0) if int(train_evidence.shape[0]) else boundary_points
    reference_surface, _ = fit_torch_visible_surface_lsq(base_observed, resolution_u=GRID_RESOLUTIONS[0], resolution_v=GRID_RESOLUTIONS[0], degree_u=2, degree_v=2)
    scale_normal_noise = _robust_normal_noise_scale(train_evidence, reference_surface) if int(train_evidence.shape[0]) >= 2 else 1e-6

    sweep: dict = {}
    for resolution in GRID_RESOLUTIONS:
        sweep[resolution] = {}
        for scheme in WEIGHTING_SCHEMES:
            if scheme == "uniform" or int(train_evidence.shape[0]) < 2:
                weights = None
            else:
                evidence_weights = _density_compensated_weights(train_evidence)
                boundary_weights = torch.ones((int(boundary_points.shape[0]),), dtype=evidence_weights.dtype, device=evidence_weights.device)
                weights = torch.cat((boundary_weights, evidence_weights))
            observed = torch.cat((boundary_points, train_evidence), dim=0) if int(train_evidence.shape[0]) else boundary_points
            try:
                surface, _ = fit_torch_visible_surface_lsq(
                    observed, resolution_u=resolution, resolution_v=resolution, degree_u=2, degree_v=2, point_weights=weights,
                )
            except Exception as exc:  # noqa: BLE001
                sweep[resolution][scheme] = {"fit_failed": type(exc).__name__}
                continue

            sample_points, deriv_u, deriv_v, normals = _sample_surface_with_normals(surface)
            if not torch.isfinite(sample_points).all():
                sweep[resolution][scheme] = {"fit_failed": "non_finite_evaluate"}
                continue
            jacobian = compute_parametric_jacobian_metrics(deriv_u, deriv_v, scale=scale_dense_nn)
            local_fold = compute_local_orientation_folding(normals, SAMPLE_RESOLUTION)
            global_orientation = compute_orientation_consistency(normals)

            train_metrics = _raw_and_normalized_errors(train_evidence, surface, scale_dense_nn, scale_representative, scale_normal_noise, scale_patch_diameter)
            holdout_metrics = _raw_and_normalized_errors(holdout_evidence, surface, scale_dense_nn, scale_representative, scale_normal_noise, scale_patch_diameter)

            sweep[resolution][scheme] = {
                "train": train_metrics,
                "holdout": holdout_metrics,
                "jacobian_near_degenerate_count": jacobian["near_degenerate_count"],
                "jacobian_condition_p95": jacobian["jacobian_condition_p95"],
                "jacobian_max_condition": jacobian["max_jacobian_condition"],
                "local_fold_count": local_fold["local_fold_count"],
                "local_fold_fraction": local_fold["local_fold_fraction"],
                "global_orientation_flip_count": global_orientation["orientation_flip_count"],
                "global_orientation_valid_sample_count": global_orientation["valid_sample_count"],
                "patch_area": _surface_area(sample_points, SAMPLE_RESOLUTION),
            }

    classification, reasons = _classify_patch(sweep) if all(
        "fit_failed" not in sweep[r][s] for r in GRID_RESOLUTIONS for s in WEIGHTING_SCHEMES
    ) else ("fit_failed", ["at_least_one_grid/weighting_combination_failed_to_fit"])

    return {
        "chart_type": chart_type, "source_region_id": region_id,
        "train_evidence_count": int(train_evidence.shape[0]), "holdout_evidence_count": int(holdout_evidence.shape[0]),
        "scale_dense_nn": scale_dense_nn, "scale_representative": scale_representative,
        "scale_normal_noise": scale_normal_noise, "scale_patch_diameter": scale_patch_diameter,
        "sweep": sweep, "classification": classification, "classification_reasons": reasons,
    }


def analyze_condition(model, cap: int, device: str, label: str) -> dict:
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    stable_ids = list(range(int(model.get_xyz.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            model.get_xyz.detach(), covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    construction = bundle.construction
    items = [("physical", item) for item in construction.materialized_visible_nurbs_surfaces if item.surface is not None]
    items += [("parametric", item) for item in construction.materialized_parametric_chart_surfaces if item.surface is not None]

    patches = []
    for chart_type, item in items:
        key = (chart_type, item.input.source_region_id)
        fit = bundle.region_owned_full_evidence_fits.get(key)
        if fit is None or fit.state != "materialized":
            continue  # only worklog 67's full-evidence-materialized patches are in scope for this round
        boundary_points = item.input.ordered_boundary_points
        if fit.full_evidence_stable_ids:
            evidence_index = torch.tensor(list(fit.full_evidence_stable_ids), dtype=torch.long, device=boundary_points.device)
            full_evidence = model.get_xyz.detach()[evidence_index]
        else:
            full_evidence = boundary_points[:0]
        print(f"  patch {chart_type}/{item.input.source_region_id}: evidence={int(full_evidence.shape[0])} ...", flush=True)
        patches.append(analyze_patch(chart_type, item.input.source_region_id, boundary_points, full_evidence))

    return {"label": label, "patch_count": len(patches), "patches": patches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"))
    parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline"))
    parser.add_argument("--iterations", nargs="+", type=int, default=[2900, 3100])
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val68/dense_evidence_fit_capacity_report.json"))
    args = parser.parse_args()
    device = "cuda"

    from osn_gs.gaussian.torch_model import TorchGaussianModel

    def load_ckpt(ckpt_dir: Path):
        payload = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
        raw = payload["model_raw"]
        rest_dim = int(raw["features_rest"].shape[-2])
        degree = 0
        while (degree + 1) ** 2 - 1 < rest_dim:
            degree += 1
        model = TorchGaussianModel(sh_degree=degree, device=device)
        model.replace_tensors(
            xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
            opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
            uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
            surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
            surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"),
            stable_gaussian_ids=raw.get("stable_gaussian_ids"),
        )
        return model

    report: dict = {"baseline_compatible": {}, "baseline": {}}
    for it in args.iterations:
        ckpt = args.baseline_compatible_run_dir / str(it)
        if (ckpt / "checkpoint.pt").exists():
            print(f"analyzing baseline_compatible@{it} ...", flush=True)
            model = load_ckpt(ckpt)
            report["baseline_compatible"][str(it)] = analyze_condition(model, args.cap, device, f"baseline_compatible@{it}")
        ply = args.baseline_run_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if ply.exists():
            print(f"analyzing baseline@{it} ...", flush=True)
            model = baseline_ply_analysis.load_baseline_ply_as_model(ply, device)
            report["baseline"][str(it)] = analyze_condition(model, args.cap, device, f"baseline@{it}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
