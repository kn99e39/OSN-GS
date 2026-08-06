"""Worklog 65 (Baseline-Compatible 3k Production Validation): full metric
battery for the 3-way comparison (covariance_knn OSN-GS / baseline_compatible
OSN-GS / Graphdeco baseline) at iteration 0/600/2900/3000/3100.

Reuses `fixed_loader_replay_analysis.analyze()` (OSN-GS checkpoints) and
`baseline_ply_replay_analysis.analyze()` (baseline PLYs) unmodified for
checkpointed iterations, and adds:
  - projected screen-space radius percentiles (not previously reported)
  - iteration-0 analysis (no checkpoint file exists yet -- built directly
    from a fresh `pipeline.initialize()` / baseline `create_from_pcd()`)

Read-only analysis. No algorithm or threshold is changed anywhere in this
file or the modules it imports.
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
from osn_gs.data.colmap_scene import load_colmap_scene_with_eval_split
from osn_gs.eval.held_out_metrics import evaluate_held_out_cameras
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation

import fixed_loader_replay_analysis as osn_ckpt_analysis  # noqa: E402
import baseline_ply_replay_analysis as baseline_ply_analysis  # noqa: E402

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "gaussian-splatting"
MIN_SCALE_COLLAPSE_ABS = 1e-4


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _radius_percentiles(rasterizer, model, cameras, device: str, sample: int = 8) -> dict:
    """Projected screen-space radius distribution over a fixed camera sample."""

    step = max(1, len(cameras) // sample)
    chosen = cameras[::step][:sample]
    radii_all = []
    with torch.no_grad():
        for camera in chosen:
            render_pkg = rasterizer.render(camera, model)
            radii = render_pkg.get("radii")
            if radii is None:
                continue
            radii = radii.detach().float().reshape(-1).cpu().numpy()
            radii_all.append(radii[radii > 0])
    if not radii_all:
        return _percentiles(np.array([]))
    return _percentiles(np.concatenate(radii_all))


def _full_battery_from_model(
    model, iteration: int, cap: int, source_path: Path, llffhold: int, device: str, label: str,
) -> dict:
    """Compute the same metric battery as the checkpoint/ply analyzers, but
    directly from an in-memory model (used for the iteration-0 conditions,
    which have no checkpoint/PLY file on disk)."""

    with torch.no_grad():
        scale = model.get_scaling.detach().cpu().numpy()
        sorted_scale = np.sort(scale, axis=1)
        s_min, s_mid, s_max = sorted_scale[:, 0], sorted_scale[:, 1], sorted_scale[:, 2]
        anisotropy = s_max / np.clip(s_min, 1e-12, None)
        min_scale_collapse = int((s_min < MIN_SCALE_COLLAPSE_ABS).sum())

    eval_split = load_colmap_scene_with_eval_split(source_path, device=device, llffhold=llffhold)
    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))
    held_out = evaluate_held_out_cameras(rasterizer, model, eval_split.test_cameras, eval_split.test_images, device=device)
    lpips_mean = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex").to(device)
        scores = []
        with torch.no_grad():
            for camera, target in zip(eval_split.test_cameras, eval_split.test_images):
                rendered = rasterizer.render(camera, model)["render"].clamp(0.0, 1.0)
                gt = target.to(device=device, dtype=torch.float32)
                scores.append(float(lp(rendered.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()))
        lpips_mean = float(np.mean(scores)) if scores else None
    except Exception as exc:  # noqa: BLE001
        lpips_mean = f"unavailable:{type(exc).__name__}:{exc}"

    radius = _radius_percentiles(rasterizer, model, eval_split.train_scene.cameras, device)

    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    stable_ids = list(range(int(model.get_xyz.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            model.get_xyz.detach(), covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    s = bundle.construction.diagnostic_summary
    region_sizes = [len(r.member_ids) for r in bundle.construction.surface_regions.regions]

    return {
        "label": label, "iteration": iteration,
        "gaussian_count": int(model.get_xyz.shape[0]),
        "cumulative_adc": {"cloned": 0, "split": 0, "pruned": 0, "opacity_reset_count": 0},
        "this_step_adc_event": None,
        "scale_s_min": _percentiles(s_min), "scale_s_mid": _percentiles(s_mid), "scale_s_max": _percentiles(s_max),
        "anisotropy": _percentiles(anisotropy),
        "min_scale_collapse_count": min_scale_collapse,
        "min_scale_collapse_ratio": min_scale_collapse / max(len(s_min), 1),
        "projected_radius": radius,
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


def analyze_osn_checkpoint_with_radius(ckpt_dir: Path, log_path: Path, iteration: int, cap: int, source_path: Path, llffhold: int, device: str) -> dict:
    result = osn_ckpt_analysis.analyze(ckpt_dir, log_path, iteration, cap, source_path, llffhold, device)
    from osn_gs.gaussian.torch_model import TorchGaussianModel
    payload = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=False)
    raw = payload["model_raw"]
    model = TorchGaussianModel(sh_degree=osn_ckpt_analysis._sh_degree_from_checkpoint(raw), device=device)
    model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=raw.get("stable_gaussian_ids"),
    )
    eval_split = load_colmap_scene_with_eval_split(source_path, device=device, llffhold=llffhold)
    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))
    result["projected_radius"] = _radius_percentiles(rasterizer, model, eval_split.train_scene.cameras, device)
    return result


def analyze_baseline_ply_with_radius(ply_path: Path, iteration: int, cap: int, source_path: Path, llffhold: int, device: str) -> dict:
    result = baseline_ply_analysis.analyze(ply_path, iteration, cap, source_path, llffhold, device)
    model = baseline_ply_analysis.load_baseline_ply_as_model(ply_path, device)
    eval_split = load_colmap_scene_with_eval_split(source_path, device=device, llffhold=llffhold)
    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))
    result["projected_radius"] = _radius_percentiles(rasterizer, model, eval_split.train_scene.cameras, device)
    return result


def analyze_osn_init(mode: str, cap: int, source_path: Path, llffhold: int, device: str) -> dict:
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(gaussian_initialization_mode=mode, canonical_construction_max_points=cap), device=device)
    eval_split = load_colmap_scene_with_eval_split(source_path, device=device, llffhold=llffhold)
    state = pipeline.initialize(eval_split.train_scene.initial_points, eval_split.train_scene.initial_colors)
    return _full_battery_from_model(state.model, 0, cap, source_path, llffhold, device, f"osn_init[{mode}]")


def analyze_baseline_init(cap: int, source_path: Path, llffhold: int, device: str) -> dict:
    sys.path.insert(0, str(BASELINE_ROOT))
    from scene import Scene, GaussianModel
    from utils.general_utils import safe_state
    from argparse import ArgumentParser as BaselineArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams

    baseline_parser = BaselineArgumentParser()
    lp = ModelParams(baseline_parser)
    op = OptimizationParams(baseline_parser)
    PipelineParams(baseline_parser)
    baseline_parser.add_argument('--debug_from', type=int, default=-1)
    baseline_parser.add_argument('--detect_anomaly', action='store_true', default=False)
    baseline_parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    baseline_parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    baseline_parser.add_argument("--quiet", action="store_true")
    baseline_parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    baseline_parser.add_argument("--start_checkpoint", type=str, default=None)
    parsed = baseline_parser.parse_args([
        "-s", str(source_path), "-m", "output/extent_ab/val64/_iter0_scratch", "--eval",
    ])
    safe_state(False)
    dataset = lp.extract(parsed)
    op.extract(parsed)

    baseline_model = GaussianModel(dataset.sh_degree, "default")
    Scene(dataset, baseline_model)

    from osn_gs.gaussian.torch_model import TorchGaussianModel
    n = int(baseline_model.get_xyz.shape[0])
    model = TorchGaussianModel(sh_degree=dataset.sh_degree, device=device)
    model.replace_tensors(
        xyz=baseline_model._xyz.detach().clone(),
        features_dc=baseline_model._features_dc.detach().clone(),
        features_rest=baseline_model._features_rest.detach().clone(),
        opacity=baseline_model._opacity.detach().clone(),
        scaling=baseline_model._scaling.detach().clone(),
        rotation=baseline_model._rotation.detach().clone(),
        uncertain_confidence=torch.full((n, 1), 12.0, dtype=torch.float32, device=device),
        uncertain_mask=torch.zeros((n,), dtype=torch.bool, device=device),
        surface_uv=torch.zeros((n, 2), dtype=torch.float32, device=device),
        cluster_ids=torch.full((n,), -1, dtype=torch.long, device=device),
        stable_gaussian_ids=torch.arange(n, dtype=torch.long, device=device),
    )
    return _full_battery_from_model(model, 0, cap, source_path, llffhold, device, "baseline_init")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=Path, default=Path("DATASET"))
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--covariance_knn_run_dir", type=Path, default=Path("output/extent_ab/val64/covariance_knn"))
    parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"))
    parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline"))
    parser.add_argument("--iterations", nargs="+", type=int, default=[600, 2900, 3000, 3100])
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val64/full_report.json"))
    parser.add_argument("--skip_init", action="store_true")
    args = parser.parse_args()
    device = "cuda"

    report: dict = {"covariance_knn": {}, "baseline_compatible": {}, "baseline": {}}

    if not args.skip_init:
        print("analyzing iteration 0 ...", flush=True)
        report["covariance_knn"]["0"] = analyze_osn_init("covariance_knn", args.cap, args.source_path, args.llffhold, device)
        report["baseline_compatible"]["0"] = analyze_osn_init("baseline_compatible", args.cap, args.source_path, args.llffhold, device)
        report["baseline"]["0"] = analyze_baseline_init(args.cap, args.source_path, args.llffhold, device)

    for it in args.iterations:
        print(f"analyzing iteration {it} ...", flush=True)
        ckpt = args.covariance_knn_run_dir / str(it)
        if (ckpt / "checkpoint.pt").exists():
            report["covariance_knn"][str(it)] = analyze_osn_checkpoint_with_radius(
                ckpt, args.covariance_knn_run_dir.with_suffix(".log"), it, args.cap, args.source_path, args.llffhold, device,
            )
        ckpt = args.baseline_compatible_run_dir / str(it)
        if (ckpt / "checkpoint.pt").exists():
            report["baseline_compatible"][str(it)] = analyze_osn_checkpoint_with_radius(
                ckpt, args.baseline_compatible_run_dir.with_suffix(".log"), it, args.cap, args.source_path, args.llffhold, device,
            )
        ply = args.baseline_run_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if ply.exists():
            report["baseline"][str(it)] = analyze_baseline_ply_with_radius(ply, it, args.cap, args.source_path, args.llffhold, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
