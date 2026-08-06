"""Single-GPU, cache-reusing replacement candidate for Worklog 65 analysis.

Semantic scope matches gaussian_init_mode_3k_validation_analysis.py.  It avoids
reloading scene/rasterizer/LPIPS for every model and renders each held-out
camera once per model for PSNR, SSIM, and LPIPS together.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

import gaussian_init_mode_3k_validation_analysis as legacy
import fixed_loader_replay_analysis as osn_ckpt
import baseline_ply_replay_analysis as baseline_ply
import osn_gs.core.torch_pipeline  # noqa: F401
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.data.colmap_scene import load_colmap_scene_with_eval_split
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.losses.torch_losses import ssim as compute_ssim
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.utils.torch_ops import psnr_from_mse

MIN_SCALE_COLLAPSE_ABS = legacy.MIN_SCALE_COLLAPSE_ABS
BASELINE_ROOT = legacy.BASELINE_ROOT


@dataclass
class _AnalysisContext:
    eval_split: Any
    rasterizer: Any
    lpips_model: Any | None
    test_targets: tuple[Any, ...]
    device: str
    background: Any


def _make_context(source_path: Path, llffhold: int, device: str) -> _AnalysisContext:
    eval_split = load_colmap_scene_with_eval_split(source_path, device=device, llffhold=llffhold)
    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception:  # match legacy per-model fallback below without failing the analysis
        lpips_model = None
    targets = tuple(image.to(device=device, dtype=torch.float32) for image in eval_split.test_images)
    return _AnalysisContext(eval_split, rasterizer, lpips_model, targets, device, torch.zeros((3,), dtype=torch.float32, device=device))


def _render_quality_metrics(context: _AnalysisContext, model: Any) -> tuple[float, float, float | str | None]:
    psnrs: list[float] = []
    ssims: list[float] = []
    lpips_scores: list[float] = []
    try:
        with torch.no_grad():
            for camera, target in zip(context.eval_split.test_cameras, context.test_targets):
                rendered = context.rasterizer.render(camera, model, context.background)["render"].to(context.device, torch.float32).clamp(0.0, 1.0)
                mse = float(torch.nn.functional.mse_loss(rendered, target).detach().cpu())
                psnrs.append(psnr_from_mse(mse))
                ssims.append(float(compute_ssim(rendered, target).detach().cpu()))
                if context.lpips_model is not None:
                    lpips_scores.append(float(context.lpips_model(rendered.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()))
        finite = [value for value in psnrs if value != float("inf")]
        return (sum(finite) / len(finite) if finite else float("inf"), sum(ssims) / len(ssims) if ssims else float("nan"),
                float(np.mean(lpips_scores)) if context.lpips_model is not None else "unavailable:LPIPSInitializationError")
    except Exception as exc:  # preserve legacy behavior: quality remains available only if this block succeeds
        return float("nan"), float("nan"), f"unavailable:{type(exc).__name__}:{exc}"


def _radius_percentiles(context: _AnalysisContext, model: Any, sample: int = 8) -> dict:
    cameras = context.eval_split.train_scene.cameras
    chosen = cameras[::max(1, len(cameras) // sample)][:sample]
    values = []
    with torch.no_grad():
        for camera in chosen:
            radii = context.rasterizer.render(camera, model).get("radii")
            if radii is not None:
                raw = radii.detach().float().reshape(-1).cpu().numpy()
                values.append(raw[raw > 0])
    return legacy._percentiles(np.concatenate(values) if values else np.array([]))


def _model_metrics(model: Any, iteration: int, cap: int, context: _AnalysisContext, label: str) -> dict:
    with torch.no_grad():
        scale = model.get_scaling.detach().cpu().numpy()
        ordered = np.sort(scale, axis=1)
        s_min, s_mid, s_max = ordered[:, 0], ordered[:, 1], ordered[:, 2]
        anisotropy = s_max / np.clip(s_min, 1e-12, None)
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
    # Preserve the legacy order: held-out metrics, then radius, then canonical construction.
    psnr, ssim, lpips_mean = _render_quality_metrics(context, model)
    radius = _radius_percentiles(context, model)
    with torch.no_grad():
        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=context.device)
        bundle = pipeline._construct_canonical_with_full_evidence(
            model.get_xyz.detach(), covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1),
            list(range(int(model.get_xyz.shape[0]))),
        )
    summary = bundle.construction.diagnostic_summary
    region_sizes = [len(region.member_ids) for region in bundle.construction.surface_regions.regions]
    return {
        "label": label, "iteration": iteration, "gaussian_count": int(model.get_xyz.shape[0]),
        "scale_s_min": legacy._percentiles(s_min), "scale_s_mid": legacy._percentiles(s_mid),
        "scale_s_max": legacy._percentiles(s_max), "anisotropy": legacy._percentiles(anisotropy),
        "min_scale_collapse_count": int((s_min < MIN_SCALE_COLLAPSE_ABS).sum()),
        "min_scale_collapse_ratio": int((s_min < MIN_SCALE_COLLAPSE_ABS).sum()) / max(len(s_min), 1),
        "projected_radius": radius, "psnr_mean": psnr, "ssim_mean": ssim, "lpips_mean": lpips_mean,
        "reliability": {"region_count": summary["region_count"], "reliable_count": summary["reliable_count"], "intrinsic_reliable_count": summary["intrinsic_reliable_count"]},
        "region_size_histogram": {"count": len(region_sizes), "singleton_count": sum(x <= 1 for x in region_sizes), "small_le3_count": sum(x <= 3 for x in region_sizes), "median_size": float(np.median(region_sizes)) if region_sizes else None, "max_size": max(region_sizes) if region_sizes else None},
        "physical_chart": {"eligible_closed_count": summary["region_boundary_eligible_closed_count"], "materialized_count": summary["materialized_surface_count"]},
        "parametric_chart": {"eligible_count": summary["parametric_chart_eligible_count"], "materialized_count": summary["parametric_chart_materialized_surface_count"]},
        "combined_materialized_surface_count": summary["materialized_surface_count"] + summary["parametric_chart_materialized_surface_count"],
    }


def _load_osn_model(checkpoint: Path, device: str) -> TorchGaussianModel:
    raw = torch.load(checkpoint / "checkpoint.pt", map_location=device, weights_only=False)["model_raw"]
    model = TorchGaussianModel(sh_degree=osn_ckpt._sh_degree_from_checkpoint(raw), device=device)
    model.replace_tensors(xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"], opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"], uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"], surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"], surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"), stable_gaussian_ids=raw.get("stable_gaussian_ids"))
    return model


def _with_osn_metadata(result: dict, checkpoint: Path, log_path: Path, iteration: int) -> dict:
    result["checkpoint_dir"] = str(checkpoint)
    metrics = osn_ckpt._read_metrics_txt(checkpoint / "metrics.txt")
    result["cumulative_adc"] = {"cloned": metrics.get("cumulative_adc_cloned"), "split": metrics.get("cumulative_adc_split"), "pruned": metrics.get("cumulative_adc_pruned"), "opacity_reset_count": metrics.get("cumulative_adc_opacity_reset_count")}
    result["this_step_adc_event"] = osn_ckpt._log_adc_line(log_path, iteration)
    result["extent"] = {"scene_extent_point_cloud": metrics.get("scene_extent_point_cloud"), "calibration_extent_camera": metrics.get("calibration_extent_camera")}
    result.pop("label", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=Path, default=Path("DATASET")); parser.add_argument("--llffhold", type=int, default=8); parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--covariance_knn_run_dir", type=Path, default=Path("output/extent_ab/val64/covariance_knn")); parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible")); parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline")); parser.add_argument("--iterations", nargs="+", type=int, default=[600, 2900, 3000, 3100]); parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val64/full_report_optimized.json")); parser.add_argument("--skip_init", action="store_true"); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); context = _make_context(args.source_path, args.llffhold, args.device); report: dict = {"covariance_knn": {}, "baseline_compatible": {}, "baseline": {}}
    if not args.skip_init:
        for label, mode in (("covariance_knn", "covariance_knn"), ("baseline_compatible", "baseline_compatible")):
            split = context.eval_split; pipeline = TorchOSNGSPipeline(TorchPipelineConfig(gaussian_initialization_mode=mode, canonical_construction_max_points=args.cap), device=args.device); state = pipeline.initialize(split.train_scene.initial_points, split.train_scene.initial_colors); report[label]["0"] = _model_metrics(state.model, 0, args.cap, context, f"osn_init[{mode}]"); report[label]["0"].update({"cumulative_adc": {"cloned": 0, "split": 0, "pruned": 0, "opacity_reset_count": 0}, "this_step_adc_event": None})
        sys.path.insert(0, str(BASELINE_ROOT)); from scene import Scene, GaussianModel; from utils.general_utils import safe_state; from argparse import ArgumentParser as BP; from arguments import ModelParams, OptimizationParams, PipelineParams
        p = BP(); lp = ModelParams(p); op = OptimizationParams(p); PipelineParams(p); p.add_argument('--debug_from', type=int, default=-1); p.add_argument('--detect_anomaly', action='store_true', default=False); p.add_argument('--test_iterations', nargs='+', type=int, default=[]); p.add_argument('--save_iterations', nargs='+', type=int, default=[]); p.add_argument('--quiet', action='store_true'); p.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[]); p.add_argument('--start_checkpoint', type=str, default=None); parsed = p.parse_args(['-s', str(args.source_path), '-m', 'output/extent_ab/val64/_iter0_scratch', '--eval']); safe_state(False); dataset = lp.extract(parsed); op.extract(parsed); base = GaussianModel(dataset.sh_degree, 'default'); Scene(dataset, base); n = int(base.get_xyz.shape[0]); model = TorchGaussianModel(sh_degree=dataset.sh_degree, device=args.device); model.replace_tensors(xyz=base._xyz.detach().clone(), features_dc=base._features_dc.detach().clone(), features_rest=base._features_rest.detach().clone(), opacity=base._opacity.detach().clone(), scaling=base._scaling.detach().clone(), rotation=base._rotation.detach().clone(), uncertain_confidence=torch.full((n,1),12.0,dtype=torch.float32,device=args.device), uncertain_mask=torch.zeros((n,),dtype=torch.bool,device=args.device), surface_uv=torch.zeros((n,2),dtype=torch.float32,device=args.device), cluster_ids=torch.full((n,),-1,dtype=torch.long,device=args.device), stable_gaussian_ids=torch.arange(n,dtype=torch.long,device=args.device)); report["baseline"]["0"] = _model_metrics(model, 0, args.cap, context, "baseline_init"); report["baseline"]["0"].update({"cumulative_adc": {"cloned": 0, "split": 0, "pruned": 0, "opacity_reset_count": 0}, "this_step_adc_event": None})
    for iteration in args.iterations:
        for label, run_dir in (("covariance_knn", args.covariance_knn_run_dir), ("baseline_compatible", args.baseline_compatible_run_dir)):
            checkpoint = run_dir / str(iteration)
            if (checkpoint / "checkpoint.pt").exists(): report[label][str(iteration)] = _with_osn_metadata(_model_metrics(_load_osn_model(checkpoint, args.device), iteration, args.cap, context, label), checkpoint, run_dir.with_suffix('.log'), iteration)
        ply = args.baseline_run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if ply.exists():
            result = _model_metrics(baseline_ply.load_baseline_ply_as_model(ply, args.device), iteration, args.cap, context, "baseline"); result["ply_path"] = str(ply); result.pop("label", None); result["region_size_histogram"].pop("median_size", None); result["region_size_histogram"].pop("max_size", None); report["baseline"][str(iteration)] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle: json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)

if __name__ == "__main__": main()