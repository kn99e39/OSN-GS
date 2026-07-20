"""Stage 1 ablation runner (migration plan §3 / user requirements §3).

Runs the constructor benchmark over the required config matrix on the required
scenes, then writes a per-run/per-scene summary table (JSON + Markdown) so the
boundary-first vs. voxel-per-patch comparison is one file.

Usage (repo root):
    python scripts/stage1_ablation.py [--output nurbs_constructor_benchmark/results/stage1_ablation]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nurbs_constructor_benchmark.runner import main as run_benchmark  # noqa: E402

SCENES = [
    "plane", "sine", "planar_hole", "crease", "density_gradient",
    "elongated_plane", "mild_curved_sheet", "close_parallel_sheets",
]

# name -> (extra CLI args, export renderer output?)
RUNS: dict[str, tuple[list[str], bool]] = {
    # 1. boundary_first constructor (default; the legacy path is no longer
    #    selectable in this benchmark, see nurbs_constructor_benchmark/runner.py)
    "boundary_first": ([], True),
    # 2. voxel-per-patch, voxel support mask OFF (untrimmed charts)
    "stage1_mask_off": (["--constructor", "voxel_patch_stage1", "--stage1-support", "none"], False),
    # 3a. voxel-per-patch, exact plane-AABB voxel support mask (polygon only)
    "stage1_mask_on": (["--constructor", "voxel_patch_stage1", "--stage1-support", "voxel"], True),
    # 3b. Stage 1-F: polygon + density-refined boundary (default mode)
    "stage1_density": (["--constructor", "voxel_patch_stage1", "--stage1-support", "voxel_density"], True),
    # 4. voxel_min_gaussian_count sweep (mask on)
    "stage1_min5": (["--constructor", "voxel_patch_stage1", "--voxel-min-count", "5"], False),
    "stage1_min20": (["--constructor", "voxel_patch_stage1", "--voxel-min-count", "20"], False),
    "stage1_min40": (["--constructor", "voxel_patch_stage1", "--voxel-min-count", "40"], False),
    # 5. voxel_max_gaussian_count / max_depth sweep (mask on)
    "stage1_max50": (["--constructor", "voxel_patch_stage1", "--voxel-max-count", "50"], True),
    "stage1_max100": (["--constructor", "voxel_patch_stage1", "--voxel-max-count", "100"], False),
    "stage1_max300": (["--constructor", "voxel_patch_stage1", "--voxel-max-count", "300"], False),
    "stage1_max50_depth2": (["--constructor", "voxel_patch_stage1", "--voxel-max-count", "50", "--voxel-max-depth", "2"], False),
}

SUMMARY_COLUMNS = [
    ("patches", lambda r: r["patches"]),
    ("controls", lambda r: r["control_points"]),
    ("chamfer", lambda r: round(r["ground_truth"]["chamfer_rms"], 6)),
    ("acc_rms", lambda r: round(r["ground_truth"]["accuracy_rms"], 6)),
    ("union_iou", lambda r: round(r["patch_union"]["union_iou"], 3)),
    ("holes", lambda r: r["patch_union"]["union_hole_count"]),
    ("gt_holes", lambda r: r["patch_union"]["union_gt_hole_count"]),
    ("hole_iou", lambda r: round(r["patch_union"]["union_hole_iou"], 3)),
    ("false_fill", lambda r: round(r["patch_union"]["union_false_fill_ratio"], 3)),
    ("tiny_false", lambda r: r["patch_union"]["union_tiny_false_hole_count"]),
    ("seam_comp", lambda r: r["patch_union"]["union_seam_component_count"]),
    ("overlap", lambda r: round(r["patch_union"]["union_patch_overlap_ratio"], 3)),
    ("gap", lambda r: round(r["patch_union"]["union_interpatch_gap_ratio"], 3)),
    ("conform", lambda r: round(r["support_conformality"]["support_conformality_ratio"], 3)),
    ("underdet", lambda r: (r.get("stage1") or {}).get("underdetermined_patch_count", 0)),
    ("time_s", lambda r: round(r["construction_seconds"], 2)),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("nurbs_constructor_benchmark/results/stage1_ablation"),
    )
    parser.add_argument("--points", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary: dict[str, dict[str, dict]] = {}
    for run_name, (extra_args, export_renderer) in RUNS.items():
        run_dir = args.output / run_name
        argv = [
            "--scenes", *SCENES,
            "--output", str(run_dir),
            "--points", str(args.points),
            "--seed", str(args.seed),
            *([] if export_renderer else ["--skip-renderer-export"]),
            *extra_args,
        ]
        print(f"\n=== ablation run: {run_name} ===", flush=True)
        start = time.perf_counter()
        run_benchmark(argv)
        print(f"=== {run_name} finished in {time.perf_counter() - start:.1f}s ===", flush=True)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        summary[run_name] = {result["scene"]: result for result in report["results"]}

    table_rows: list[str] = []
    header = "| run | scene | " + " | ".join(name for name, _ in SUMMARY_COLUMNS) + " |"
    table_rows.append(header)
    table_rows.append("|" + "---|" * (2 + len(SUMMARY_COLUMNS)))
    summary_compact: dict[str, dict[str, dict]] = {}
    for run_name, scenes in summary.items():
        summary_compact[run_name] = {}
        for scene_name in SCENES:
            result = scenes[scene_name]
            values = {name: extract(result) for name, extract in SUMMARY_COLUMNS}
            summary_compact[run_name][scene_name] = values
            table_rows.append(
                f"| {run_name} | {scene_name} | "
                + " | ".join(str(values[name]) for name, _ in SUMMARY_COLUMNS)
                + " |"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps({"points": args.points, "seed": args.seed, "runs": summary_compact}, indent=2),
        encoding="utf-8",
    )
    (args.output / "summary.md").write_text("\n".join(table_rows) + "\n", encoding="utf-8")
    print(f"\nablation summary: {args.output / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
