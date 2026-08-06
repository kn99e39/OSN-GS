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
import concurrent.futures
import time
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


def _run_analysis_task(task: dict) -> tuple[str, str, dict, float]:
    """Pickle-safe worker: exactly one independent condition/iteration."""
    cpu_threads = task.get("cpu_threads")
    if cpu_threads is not None:
        torch.set_num_threads(int(cpu_threads))
    started = time.monotonic()
    kind = task["kind"]
    if kind == "osn_init":
        result = analyze_osn_init(task["mode"], task["cap"], task["source_path"], task["llffhold"], task["device"])
    elif kind == "baseline_init":
        result = analyze_baseline_init(task["cap"], task["source_path"], task["llffhold"], task["device"])
    elif kind == "osn_checkpoint":
        result = analyze_osn_checkpoint_with_radius(
            task["checkpoint_dir"], task["log_path"], task["iteration"], task["cap"],
            task["source_path"], task["llffhold"], task["device"],
        )
    elif kind == "baseline_ply":
        result = analyze_baseline_ply_with_radius(
            task["ply_path"], task["iteration"], task["cap"], task["source_path"],
            task["llffhold"], task["device"],
        )
    else:
        raise ValueError(f"unknown analysis task kind: {kind}")
    return task["label"], str(task["iteration"]), result, time.monotonic() - started


def _append_progress(path: Path, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=Path, default=Path("DATASET"))
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--covariance_knn_run_dir", type=Path, default=Path("output/extent_ab/val64/covariance_knn"))
    parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"))
    parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline"))
    parser.add_argument("--iterations", nargs="+", type=int, default=[600, 2900, 3000, 3100])
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val64/full_report_parallel.json"))
    parser.add_argument("--skip_init", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Independent analysis worker processes.")
    parser.add_argument("--cpu_threads_per_worker", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_shared_cuda", action="store_true", help="Required before workers > 1 share one CUDA device.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.device.startswith("cuda") and args.workers > 1 and not args.allow_shared_cuda:
        parser.error("CUDA workers share one device; pass --allow_shared_cuda only when GPU memory headroom is sufficient")

    common = {"cap": args.cap, "source_path": args.source_path, "llffhold": args.llffhold, "device": args.device, "cpu_threads": args.cpu_threads_per_worker}
    tasks: list[dict] = []
    if not args.skip_init:
        tasks.extend([
            {**common, "kind": "osn_init", "label": "covariance_knn", "iteration": 0, "mode": "covariance_knn"},
            {**common, "kind": "osn_init", "label": "baseline_compatible", "iteration": 0, "mode": "baseline_compatible"},
            {**common, "kind": "baseline_init", "label": "baseline", "iteration": 0},
        ])
    for iteration in args.iterations:
        for label, run_dir in (("covariance_knn", args.covariance_knn_run_dir), ("baseline_compatible", args.baseline_compatible_run_dir)):
            checkpoint = run_dir / str(iteration)
            if (checkpoint / "checkpoint.pt").exists():
                tasks.append({**common, "kind": "osn_checkpoint", "label": label, "iteration": iteration, "checkpoint_dir": checkpoint, "log_path": run_dir.with_suffix(".log")})
        ply = args.baseline_run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if ply.exists():
            tasks.append({**common, "kind": "baseline_ply", "label": "baseline", "iteration": iteration, "ply_path": ply})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.out.with_suffix(args.out.suffix + ".progress.jsonl")
    if progress_path.exists():
        progress_path.unlink()
    report: dict = {"covariance_knn": {}, "baseline_compatible": {}, "baseline": {}}
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_analysis_task, task): task for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            task = futures[future]
            label, iteration, result, task_seconds = future.result()
            report[label][iteration] = result
            elapsed = time.monotonic() - started
            _append_progress(progress_path, {
                "completed": completed, "total": len(tasks), "label": label, "iteration": iteration,
                "task_seconds": task_seconds, "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (elapsed / completed) * (len(tasks) - completed),
            })
            print(f"completed {completed}/{len(tasks)}: {label} iteration {iteration} ({task_seconds:.1f}s)", flush=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()