"""Worklog 66: 2-worker parallel driver for visible_patch_fidelity_validation.py.

Profiling found the per-condition bottleneck is entirely inside the existing
production `_construct_canonical_with_full_evidence` full-cloud pass (~5-6
minutes regardless of resulting patch count) -- GPU utilization during a
single sequential run stayed around 10%, so the 6 independent conditions
(covariance_knn/baseline_compatible/baseline x 2900/3100) are latency- not
throughput-bound and parallelize cleanly across processes without changing
any single condition's own computation or result.

Runs each condition in its own spawned process (default on Windows already;
each gets its own CUDA context). 2 workers by default: one condition's peak
GPU memory was ~6.9GB, so 2 concurrent fit safely under a 16GB card with
headroom; 3 would risk OOM on the same hardware this was profiled on --
raise `--workers` only if you have confirmed more headroom.

Each worker writes its own viz shard (avoids pickling large CUDA/CPU tensors
across the process boundary); the main process only collects the small
JSON-safe summaries and merges them into one report, matching the serial
script's `--out` output exactly.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

DEVTOOLS_DIR = Path(__file__).resolve().parent
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))


def _worker(task: dict) -> dict:
    """Runs in its own spawned process -- imports must happen here, not at
    module import time, so each worker gets an independent CUDA context."""

    import torch
    import osn_gs.core.torch_pipeline  # noqa: F401 -- resolve osn_gs's own circular-import order first
    import baseline_ply_replay_analysis as baseline_ply_analysis
    from visible_patch_fidelity_validation import _load_osn_checkpoint, analyze_condition

    label, kind, path, iteration, cap, viz_shard_path = (
        task["label"], task["kind"], Path(task["path"]), task["iteration"], task["cap"], Path(task["viz_shard_path"]),
    )
    device = "cuda"
    if kind == "osn_checkpoint":
        model = _load_osn_checkpoint(path, device)
    elif kind == "baseline_ply":
        model = baseline_ply_analysis.load_baseline_ply_as_model(path, device)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    result = analyze_condition(model, cap, None, device, label)
    patches_raw = result.pop("_patches_raw")

    cpu_patches = []
    for p in patches_raw:
        cpu_patches.append({
            "chart_type": p["chart_type"], "source_region_id": p["source_region_id"],
            "classification": p["classification"], "classification_reasons": p["classification_reasons"],
            "sample_points": p["_sample_points"].detach().cpu(),
            "boundary_points": p["_boundary_points"].detach().cpu(),
            "interior_points": p["_interior_points"].detach().cpu() if p["_interior_points"] is not None else None,
            "point_to_surface_distance_normalized": p["point_to_surface_distance_normalized"],
            "surface_to_evidence_distance_normalized": p["surface_to_evidence_distance_normalized"],
        })
    viz_shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_patches, viz_shard_path)

    return {"cond_label": task["cond_label"], "iteration": str(iteration), "summary": result, "viz_shard_path": str(viz_shard_path)}


def _build_tasks(args: argparse.Namespace) -> list[dict]:
    tasks = []
    for it in args.iterations:
        for cond, run_dir in (
            ("baseline_compatible", args.baseline_compatible_run_dir),
            ("covariance_knn", args.covariance_knn_run_dir),
        ):
            ckpt = run_dir / str(it)
            if not (ckpt / "checkpoint.pt").exists():
                continue
            key = f"{cond}@{it}"
            tasks.append({
                "label": key, "kind": "osn_checkpoint", "path": str(ckpt), "iteration": it, "cap": args.cap,
                "cond_label": cond, "viz_shard_path": str(args.viz_shard_dir / f"{cond}_{it}.pt"),
            })
        ply = args.baseline_run_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if ply.exists():
            key = f"baseline@{it}"
            tasks.append({
                "label": key, "kind": "baseline_ply", "path": str(ply), "iteration": it, "cap": args.cap,
                "cond_label": "baseline", "viz_shard_path": str(args.viz_shard_dir / f"baseline_{it}.pt"),
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=2, help="Concurrent conditions. 2 is the profiled-safe default for a 16GB card (~6.9GB/condition); raise only with confirmed headroom.")
    parser.add_argument("--baseline_compatible_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline_compatible"))
    parser.add_argument("--covariance_knn_run_dir", type=Path, default=Path("output/extent_ab/val64/covariance_knn"))
    parser.add_argument("--baseline_run_dir", type=Path, default=Path("output/extent_ab/val64/baseline"))
    parser.add_argument("--iterations", nargs="+", type=int, default=[2900, 3100])
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val66/patch_fidelity_report.json"))
    parser.add_argument("--viz_shard_dir", type=Path, default=Path("output/extent_ab/val66/viz_shards"))
    args = parser.parse_args()

    tasks = _build_tasks(args)
    print(f"dispatching {len(tasks)} conditions across {args.workers} workers", flush=True)

    ctx = mp.get_context("spawn")
    report: dict[str, dict[str, Any]] = {}
    with ctx.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_worker, tasks):
            print(f"finished {result['cond_label']}@{result['iteration']}", flush=True)
            report.setdefault(result["cond_label"], {})[result["iteration"]] = result["summary"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)
    print(f"per-condition viz shards written under {args.viz_shard_dir}", flush=True)


if __name__ == "__main__":
    main()
