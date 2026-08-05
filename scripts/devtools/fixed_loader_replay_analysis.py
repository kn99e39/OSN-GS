"""Worklog 63: fixed-loader (worklog 62) 3k production replay analysis.

For a given OSN-GS checkpoint (before/after the FoV+resize loader fix),
reports every metric the causal-verification protocol asked for: Gaussian
count, cumulative clone/split/prune, screen-size prune count (from the
training log), s_min/s_mid/s_max + anisotropy percentiles, min-scale
collapse ratio, held-out PSNR/SSIM/LPIPS, reliability/region-size
distribution, and physical/parametric visible-NURBS materialization counts.

Read-only analysis over already-trained checkpoints -- no algorithm or
threshold changed here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

import osn_gs.core.torch_pipeline  # noqa: F401 -- see lockstep_parity_harness.py's own note
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.data.colmap_scene import load_colmap_scene_with_eval_split
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.eval.held_out_metrics import evaluate_held_out_cameras

MIN_SCALE_COLLAPSE_ABS = 1e-4


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _sh_degree_from_checkpoint(raw: dict) -> int:
    rest_dim = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest_dim:
        degree += 1
    return degree


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


def _log_adc_line(log_path: Path, iteration: int) -> dict | None:
    if not log_path.exists():
        return None
    pattern = re.compile(rf"OSN-GS ADC: iteration={iteration} ([^\r\n]*)")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = pattern.search(text)
    if not match:
        return None
    fields = {}
    for token in match.group(1).split():
        if "=" in token:
            key, _, value = token.partition("=")
            try:
                fields[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                fields[key] = value
    return fields


def analyze(
    ckpt_dir: Path, log_path: Path | None, iteration: int, cap: int, source_path: Path, llffhold: int,
    device: str = "cuda",
) -> dict:
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

    metrics_txt = _read_metrics_txt(ckpt_dir / "metrics.txt")
    adc_line = _log_adc_line(log_path, iteration) if log_path else None

    # PSNR/SSIM/LPIPS computed directly (not relying on train.py's own
    # held_out_eval.json, which is only written for the true FINAL iteration).
    eval_split = load_colmap_scene_with_eval_split(source_path, device=device, llffhold=llffhold)
    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))
    held_out = evaluate_held_out_cameras(
        rasterizer, model, eval_split.test_cameras, eval_split.test_images, device=device,
    )
    lpips_mean = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex").to(device)
        scores = []
        with torch.no_grad():
            for camera, target in zip(eval_split.test_cameras, eval_split.test_images):
                render_pkg = rasterizer.render(camera, model)
                rendered = render_pkg["render"].clamp(0.0, 1.0)
                gt = target.to(device=device, dtype=torch.float32)
                scores.append(float(lp(rendered.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()))
        lpips_mean = float(np.mean(scores)) if scores else None
    except Exception as exc:  # noqa: BLE001
        lpips_mean = f"unavailable:{type(exc).__name__}:{exc}"

    # Visible-NURBS reconstruction (worklog 54-61, unmodified) -- only run at
    # cap-bounded representative sets, same as every prior real-checkpoint trace.
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    stable_ids = list(range(int(model.get_xyz.shape[0])))
    with torch.no_grad():
        from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            model.get_xyz.detach(), covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    s = bundle.construction.diagnostic_summary
    region_sizes = [len(r.member_ids) for r in bundle.construction.surface_regions.regions]

    return {
        "checkpoint_dir": str(ckpt_dir),
        "iteration": iteration,
        "gaussian_count": int(model.get_xyz.shape[0]),
        "cumulative_adc": {
            "cloned": metrics_txt.get("cumulative_adc_cloned"),
            "split": metrics_txt.get("cumulative_adc_split"),
            "pruned": metrics_txt.get("cumulative_adc_pruned"),
            "opacity_reset_count": metrics_txt.get("cumulative_adc_opacity_reset_count"),
        },
        "this_step_adc_event": adc_line,
        "extent": {
            "scene_extent_point_cloud": metrics_txt.get("scene_extent_point_cloud"),
            "calibration_extent_camera": metrics_txt.get("calibration_extent_camera"),
        },
        "scale_s_min": _percentiles(s_min), "scale_s_mid": _percentiles(s_mid), "scale_s_max": _percentiles(s_max),
        "anisotropy": _percentiles(anisotropy),
        "min_scale_collapse_count": min_scale_collapse,
        "min_scale_collapse_ratio": min_scale_collapse / max(len(s_min), 1),
        "psnr_mean": held_out["psnr_mean"], "ssim_mean": held_out["ssim_mean"], "lpips_mean": lpips_mean,
        "reliability": {
            "region_count": s["region_count"], "reliable_count": s["reliable_count"],
            "intrinsic_reliable_count": s["intrinsic_reliable_count"],
        },
        "region_size_histogram": {
            "count": len(region_sizes),
            "singleton_count": sum(1 for x in region_sizes if x <= 1),
            "small_le3_count": sum(1 for x in region_sizes if x <= 3),
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
        },
        "combined_materialized_surface_count": (
            s["materialized_surface_count"] + s["parametric_chart_materialized_surface_count"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--log_path", type=Path, default=None)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--source_path", type=Path, default=Path("DATASET"))
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()

    report = {}
    for it in args.iterations:
        ckpt_dir = args.run_dir / str(it)
        if not (ckpt_dir / "checkpoint.pt").exists():
            report[str(it)] = {"error": f"no checkpoint at {ckpt_dir}"}
            continue
        report[str(it)] = analyze(ckpt_dir, args.log_path, it, args.cap, args.source_path, args.llffhold)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
