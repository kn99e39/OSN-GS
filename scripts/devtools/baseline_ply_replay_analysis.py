"""Worklog 63: same metric battery as fixed_loader_replay_analysis.py, but
for a Graphdeco baseline PLY snapshot (loaded into a TorchGaussianModel for
analysis/rendering only -- no OSN-GS-side training or optimizer state).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

import osn_gs.core.torch_pipeline  # noqa: F401
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.data.colmap_scene import load_colmap_scene_with_eval_split
from osn_gs.eval.held_out_metrics import evaluate_held_out_cameras
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer

MIN_SCALE_COLLAPSE_ABS = 1e-4


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def load_baseline_ply_as_model(path: Path, device: str) -> TorchGaussianModel:
    ply = PlyData.read(str(path))
    v = ply.elements[0]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    opacity_raw = np.asarray(v["opacity"])[..., None]
    f_dc = np.stack([np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])], axis=1)[:, None, :]
    extra_names = sorted(
        (p.name for p in v.properties if p.name.startswith("f_rest_")), key=lambda x: int(x.split("_")[-1]),
    )
    n = xyz.shape[0]
    f_rest_flat = np.stack([np.asarray(v[name]) for name in extra_names], axis=1) if extra_names else np.zeros((n, 0))
    rest_dim = len(extra_names) // 3
    sh_degree = 0
    while (sh_degree + 1) ** 2 - 1 < rest_dim:
        sh_degree += 1
    f_rest = f_rest_flat.reshape(n, 3, rest_dim).transpose(0, 2, 1) if rest_dim else np.zeros((n, rest_dim, 3))
    scale_names = sorted((p.name for p in v.properties if p.name.startswith("scale_")), key=lambda x: int(x.split("_")[-1]))
    scale_raw = np.stack([np.asarray(v[name]) for name in scale_names], axis=1)
    rot_names = sorted((p.name for p in v.properties if p.name.startswith("rot")), key=lambda x: int(x.split("_")[-1]))
    rot_raw = np.stack([np.asarray(v[name]) for name in rot_names], axis=1)

    model = TorchGaussianModel(sh_degree=sh_degree, device=device)
    model.replace_tensors(
        xyz=torch.as_tensor(xyz, dtype=torch.float32, device=device),
        features_dc=torch.as_tensor(f_dc, dtype=torch.float32, device=device),
        features_rest=torch.as_tensor(f_rest, dtype=torch.float32, device=device),
        opacity=torch.as_tensor(opacity_raw, dtype=torch.float32, device=device),
        scaling=torch.as_tensor(scale_raw, dtype=torch.float32, device=device),
        rotation=torch.as_tensor(rot_raw, dtype=torch.float32, device=device),
        uncertain_confidence=torch.full((n, 1), 12.0, dtype=torch.float32, device=device),
        uncertain_mask=torch.zeros((n,), dtype=torch.bool, device=device),
        surface_uv=torch.zeros((n, 2), dtype=torch.float32, device=device),
        cluster_ids=torch.full((n,), -1, dtype=torch.long, device=device),
        stable_gaussian_ids=torch.arange(n, dtype=torch.long, device=device),
    )
    return model


def analyze(ply_path: Path, iteration: int, cap: int, source_path: Path, llffhold: int, device: str = "cuda") -> dict:
    model = load_baseline_ply_as_model(ply_path, device)

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
        "ply_path": str(ply_path), "iteration": iteration,
        "gaussian_count": int(model.get_xyz.shape[0]),
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
    parser.add_argument("--run_dir", type=Path, required=True, help="gaussian-splatting output dir with point_cloud/iteration_N/point_cloud.ply")
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--source_path", type=Path, default=Path("DATASET"))
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--cap", type=int, default=2048)
    args = parser.parse_args()

    report = {}
    for it in args.iterations:
        ply_path = args.run_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if not ply_path.exists():
            report[str(it)] = {"error": f"missing {ply_path}"}
            continue
        report[str(it)] = analyze(ply_path, it, args.cap, args.source_path, args.llffhold)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
