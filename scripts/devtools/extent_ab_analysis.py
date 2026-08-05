"""Worklog 62: scene-extent-basis A/B analysis.

Loads two OSN-GS checkpoints trained identically except for
`--position_lr_extent_mode` (scene=A, point-cloud-based; calibration=B,
camera-based/Graphdeco-compatible) and reports every metric the causal
verification protocol asked for: extent-derived thresholds, Gaussian/ADC
counts, scale & anisotropy percentiles, min-scale collapse, screen-space
footprint, held-out PSNR/SSIM/LPIPS, reliability-gate stage counts, region
count/size histogram, and physical/parametric chart materialization counts.

Does not change any visible-surface algorithm or threshold -- read-only
analysis over already-trained checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.data.colmap_scene import load_colmap_scene_with_eval_split
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer

MIN_SCALE_COLLAPSE_ABS = 1e-4  # world units; documented, not a production threshold


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _read_metrics_txt(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def _sh_degree_from_checkpoint(raw: dict) -> int:
    rest_dim = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest_dim:
        degree += 1
    return degree


def analyze(run_dir: Path, iteration: int, cap: int, source_path: Path, llffhold: int, device: str = "cuda") -> dict:
    ckpt_dir = run_dir / str(iteration)
    payload = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
    raw = payload["model_raw"]
    model = TorchGaussianModel(sh_degree=_sh_degree_from_checkpoint(raw), device=device)
    model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=raw.get("stable_gaussian_ids"),
    )

    with torch.no_grad():
        scale = model.get_scaling.detach().cpu().numpy()
        sorted_scale = np.sort(scale, axis=1)
        s_min, s_mid, s_max = sorted_scale[:, 0], sorted_scale[:, 1], sorted_scale[:, 2]
        anisotropy = s_max / np.clip(s_min, 1e-12, None)
        min_scale_collapse = int((s_min < MIN_SCALE_COLLAPSE_ABS).sum())
        max_radii2d = model.max_radii2D.detach().cpu().numpy()
        opacity = torch.sigmoid(model.get_opacity.detach()).reshape(-1).cpu().numpy()

    metrics_txt = _read_metrics_txt(ckpt_dir / "metrics.txt")
    held_out_path = run_dir / "held_out_eval.json"
    held_out = json.loads(held_out_path.read_text(encoding="utf-8")) if held_out_path.exists() else None

    # Reliability / region-formation / chart materialization -- reuses the
    # unmodified production path (worklog 54-61), representative cap=2048.
    stable_ids = list(range(int(model.get_xyz.shape[0])))
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    with torch.no_grad():
        covariance = None
        from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            model.get_xyz.detach(), covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    s = bundle.construction.diagnostic_summary
    region_sizes = [len(r.member_ids) for r in bundle.construction.surface_regions.regions]
    singleton_count = sum(1 for size in region_sizes if size <= 1)
    small_region_count = sum(1 for size in region_sizes if size <= 3)

    # LPIPS on the exact same held-out split used for the trainer's own PSNR/SSIM.
    lpips_mean = None
    try:
        import lpips
        eval_split = load_colmap_scene_with_eval_split(
            source_path, device=device, llffhold=llffhold,
        )
        rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))
        lp = lpips.LPIPS(net="alex").to(device)
        scores = []
        with torch.no_grad():
            for camera, target in zip(eval_split.test_cameras, eval_split.test_images):
                render_pkg = rasterizer.render(camera, model)
                rendered = render_pkg["render"].clamp(0.0, 1.0)
                gt = target.to(device=device, dtype=torch.float32)
                scores.append(float(lp(rendered.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()))
        lpips_mean = float(np.mean(scores)) if scores else None
    except Exception as exc:  # noqa: BLE001 - LPIPS is supplementary, never block the rest of the report
        lpips_mean = f"unavailable:{type(exc).__name__}:{exc}"

    return {
        "run_dir": str(run_dir),
        "iteration": iteration,
        "extent": {
            "scene_extent_point_cloud": metrics_txt.get("scene_extent_point_cloud"),
            "calibration_extent_camera": metrics_txt.get("calibration_extent_camera"),
            "position_lr_extent_used": metrics_txt.get("position_lr_extent_used"),
            "position_lr_extent_mode": metrics_txt.get("position_lr_extent_mode"),
            "adc_dense_extent_threshold": metrics_txt.get("adc_dense_extent_threshold"),
            "adc_world_size_prune_threshold": metrics_txt.get("adc_world_size_prune_threshold"),
        },
        "adc_cumulative": {
            "cloned": metrics_txt.get("cumulative_adc_cloned"),
            "split": metrics_txt.get("cumulative_adc_split"),
            "pruned": metrics_txt.get("cumulative_adc_pruned"),
            "opacity_reset_count": metrics_txt.get("cumulative_adc_opacity_reset_count"),
        },
        "gaussian_count": int(model.get_xyz.shape[0]),
        "scale_s_min": _percentiles(s_min),
        "scale_s_mid": _percentiles(s_mid),
        "scale_s_max": _percentiles(s_max),
        "anisotropy": _percentiles(anisotropy),
        "min_scale_collapse_count": min_scale_collapse,
        "min_scale_collapse_threshold": MIN_SCALE_COLLAPSE_ABS,
        "screen_space_max_radii2d": _percentiles(max_radii2d[max_radii2d > 0]),
        "opacity_mean": float(opacity.mean()),
        "held_out_psnr_ssim": held_out,
        "held_out_lpips_mean": lpips_mean,
        "reliability": {
            "region_count": s["region_count"],
            "reliable_count": s["reliable_count"],
            "intrinsic_reliable_count": s["intrinsic_reliable_count"],
            "reliability_failure_stage": s["reliability_failure_stage"],
        },
        "region_size_histogram": {
            "count": len(region_sizes),
            "singleton_count": singleton_count,
            "small_le3_count": small_region_count,
            "median_size": float(np.median(region_sizes)) if region_sizes else None,
            "max_size": max(region_sizes) if region_sizes else None,
        },
        "physical_chart": {
            "eligible_closed_count": s["region_boundary_eligible_closed_count"],
            "materialized_count": s["materialized_surface_count"],
        },
        "parametric_chart": {
            "eligible_count": s["parametric_chart_eligible_count"],
            "materialized_count": s["parametric_chart_materialized_surface_count"],
            "insufficient_topology_count": s["parametric_chart_insufficient_topology_count"],
            "open_or_branching_count": s["parametric_chart_open_or_branching_count"],
            "self_intersecting_count": s["parametric_chart_self_intersecting_count"],
        },
        "combined_materialized_surface_count": (
            s["materialized_surface_count"] + s["parametric_chart_materialized_surface_count"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=Path, default=Path("DATASET"))
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    report = {}
    for run_dir in args.runs:
        for it in args.iterations:
            key = f"{run_dir.name}_{it}"
            report[key] = analyze(run_dir, it, args.cap, args.source_path, args.llffhold)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
