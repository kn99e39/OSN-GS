"""Independent Performance Track for the frozen Worklog 119 chart corpus.

The script never rewrites the WL119 scientific report.  It consumes the
explicit corpus artifact emitted by ``visible_nurbs_geometry_uv_control_correction.py``
and records one deterministic plan/backend per official run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osn_gs.surface.torch_nurbs_performance_batch import (  # noqa: E402
    DeterministicChartBatchConfig,
    batched_result_scalar_metrics,
    execute_chart_plan,
    fit_serial_chart_reference,
    plan_chart_corpus,
)


ATOL = 1e-6
RTOL = 1e-5
OFFICIAL_CONFIG = DeterministicChartBatchConfig(
    bucket_upper_bounds=(64, 128, 256, 512, 1024, 2048, 4096),
    max_batch_charts=256,
    max_padded_points=65536,
)

TENSOR_FIELDS = (
    "control_grid_a", "control_grid_b", "uv_footpoint", "uv_geo_b",
    "fitted_a_at_footpoint", "normals_a", "residual_g_a", "residual_g_b",
    "residual_c_a", "residual_c_b",
)


def _tensor_fields(result: Any) -> dict[str, torch.Tensor]:
    return {
        "control_grid_a": result.surface_a.control_grid,
        "control_grid_b": result.surface_b.control_grid,
        "uv_footpoint": result.uv_footpoint,
        "uv_geo_b": result.uv_geo_b,
        "fitted_a_at_footpoint": result.fitted_a_at_footpoint,
        "normals_a": result.normals_a,
        "residual_g_a": result.residual_g_a,
        "residual_g_b": result.residual_g_b,
        "residual_c_a": result.residual_c_a,
        "residual_c_b": result.residual_c_b,
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": int(array.size), "min": float(array.min()),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_corpus(path: Path, device: torch.device) -> tuple[dict[str, Any], list[Any], list[Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "wl119-performance-chart-corpus-v1":
        raise ValueError(f"Unsupported corpus schema: {payload.get('schema')!r}")
    charts = payload["charts"]
    expected_ids = list(range(len(charts)))
    actual_ids = [int(chart["chart_id"]) for chart in charts]
    if actual_ids != expected_ids:
        raise ValueError("Corpus chart IDs/order are not exact contiguous identity")
    points = [chart["world_points"].to(device) for chart in charts]
    uv = [chart["camera_uv"].to(device) for chart in charts]
    for index, chart in enumerate(charts):
        count = int(chart["pixel_count"])
        fields = (points[index], uv[index], chart["representative_id"], chart["pixel_row"], chart["pixel_col"])
        if any(int(value.shape[0]) != count for value in fields):
            raise ValueError(f"Corpus membership/order field length mismatch at chart {index}")
    return payload, points, uv


def _run_serial(points: list[Any], uv: list[Any], device: torch.device) -> tuple[list[Any], float]:
    _sync(device)
    started = time.perf_counter()
    results = [fit_serial_chart_reference(index, p, u) for index, (p, u) in enumerate(zip(points, uv))]
    _sync(device)
    return results, time.perf_counter() - started


def _run_candidate(plan: Any, points: list[Any], uv: list[Any], device: torch.device) -> tuple[list[Any], float]:
    _sync(device)
    started = time.perf_counter()
    results = execute_chart_plan(plan, points, uv)
    _sync(device)
    return results, time.perf_counter() - started


def _research_signature(metrics: list[dict[str, float]]) -> dict[str, Any]:
    def values(field: str) -> np.ndarray:
        return np.asarray([row[field] for row in metrics], dtype=np.float64)

    g_a, g_b = values("residual_g_arm_a_median"), values("residual_g_arm_b_median")
    c_a, c_b = values("residual_c_arm_a_median"), values("residual_c_arm_b_median")
    return {
        "metric_g_aggregate_arm_a_better": bool(np.median(g_a) < np.median(g_b)),
        "metric_c_aggregate_arm_b_better": bool(np.median(c_b) < np.median(c_a)),
        "metric_g_per_chart_arm_a_better_count": int((g_a < g_b).sum()),
        "metric_g_per_chart_tie_count": int((g_a == g_b).sum()),
        "metric_c_per_chart_arm_b_better_count": int((c_b < c_a).sum()),
        "metric_c_per_chart_tie_count": int((c_a == c_b).sum()),
        "metric_g_arm_a_chart_median": float(np.median(g_a)),
        "metric_g_arm_b_chart_median": float(np.median(g_b)),
        "metric_c_arm_a_chart_median": float(np.median(c_a)),
        "metric_c_arm_b_chart_median": float(np.median(c_b)),
    }


def _compare(reference: list[Any], candidate: list[Any], pathological: tuple[int, ...]) -> dict[str, Any]:
    if [result.chart_index for result in reference] != [result.chart_index for result in candidate]:
        raise AssertionError("Candidate changed chart ID/order identity")
    per_chart: list[dict[str, Any]] = []
    all_close = True
    nan_inf_parity = True
    for ref, actual in zip(reference, candidate):
        field_records: dict[str, Any] = {}
        chart_close = True
        chart_nan_inf = True
        for field in TENSOR_FIELDS:
            expected = _tensor_fields(ref)[field]
            observed = _tensor_fields(actual)[field]
            nan_equal = bool(torch.equal(torch.isnan(expected), torch.isnan(observed)))
            posinf_equal = bool(torch.equal(torch.isposinf(expected), torch.isposinf(observed)))
            neginf_equal = bool(torch.equal(torch.isneginf(expected), torch.isneginf(observed)))
            finite = torch.isfinite(expected) & torch.isfinite(observed)
            difference = (observed[finite] - expected[finite]).abs()
            tolerance = ATOL + RTOL * expected[finite].abs()
            close = bool((difference <= tolerance).all()) if difference.numel() else True
            max_abs = float(difference.max()) if difference.numel() else 0.0
            relative = difference / expected[finite].abs().clamp_min(1e-30)
            field_records[field] = {
                "close": close, "max_abs": max_abs,
                "max_relative": float(relative.max()) if relative.numel() else 0.0,
                "mismatch_count": int((difference > tolerance).sum()) if difference.numel() else 0,
                "nan_parity": nan_equal, "posinf_parity": posinf_equal, "neginf_parity": neginf_equal,
            }
            chart_close = chart_close and close
            chart_nan_inf = chart_nan_inf and nan_equal and posinf_equal and neginf_equal
        per_chart.append({
            "chart_id": ref.chart_index, "all_fields_close": chart_close,
            "nan_inf_parity": chart_nan_inf, "fields": field_records,
        })
        all_close = all_close and chart_close
        nan_inf_parity = nan_inf_parity and chart_nan_inf

    reference_metrics = [batched_result_scalar_metrics(result) for result in reference]
    candidate_metrics = [batched_result_scalar_metrics(result) for result in candidate]
    scalar_delta = {
        field: _distribution([
            abs(actual[field] - expected[field])
            for expected, actual in zip(reference_metrics, candidate_metrics)
        ])
        for field in reference_metrics[0]
    }
    ranked = sorted(
        per_chart,
        key=lambda row: max(record["max_abs"] for record in row["fields"].values()),
        reverse=True,
    )
    reference_signature = _research_signature(reference_metrics)
    candidate_signature = _research_signature(candidate_metrics)
    conclusion_keys = (
        "metric_g_aggregate_arm_a_better", "metric_c_aggregate_arm_b_better",
        "metric_g_per_chart_arm_a_better_count", "metric_g_per_chart_tie_count",
        "metric_c_per_chart_arm_b_better_count", "metric_c_per_chart_tie_count",
    )
    conclusion_invariant = all(reference_signature[key] == candidate_signature[key] for key in conclusion_keys)
    failed_chart_ids = [row["chart_id"] for row in per_chart if not row["all_fields_close"]]
    nan_inf_failed_chart_ids = [row["chart_id"] for row in per_chart if not row["nan_inf_parity"]]
    field_failed_chart_counts = {
        field: sum(not row["fields"][field]["close"] for row in per_chart)
        for field in TENSOR_FIELDS
    }
    conclusion_flip_chart_ids = []
    for index, (expected, actual) in enumerate(zip(reference_metrics, candidate_metrics)):
        expected_relations = (
            expected["residual_g_arm_a_median"] < expected["residual_g_arm_b_median"],
            expected["residual_g_arm_a_median"] == expected["residual_g_arm_b_median"],
            expected["residual_c_arm_b_median"] < expected["residual_c_arm_a_median"],
            expected["residual_c_arm_b_median"] == expected["residual_c_arm_a_median"],
        )
        actual_relations = (
            actual["residual_g_arm_a_median"] < actual["residual_g_arm_b_median"],
            actual["residual_g_arm_a_median"] == actual["residual_g_arm_b_median"],
            actual["residual_c_arm_b_median"] < actual["residual_c_arm_a_median"],
            actual["residual_c_arm_b_median"] == actual["residual_c_arm_a_median"],
        )
        if expected_relations != actual_relations:
            conclusion_flip_chart_ids.append(index)
    pathological_rows = [per_chart[index] for index in pathological]
    return {
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "all_continuous_fields_close": all_close,
        "nan_inf_parity": nan_inf_parity,
        "research_conclusion_invariant": conclusion_invariant,
        "failed_chart_count": len(failed_chart_ids),
        "failed_chart_ids": failed_chart_ids,
        "nan_inf_failed_chart_ids": nan_inf_failed_chart_ids,
        "field_failed_chart_counts": field_failed_chart_counts,
        "conclusion_flip_chart_ids": conclusion_flip_chart_ids,
        "reference_research_signature": reference_signature,
        "candidate_research_signature": candidate_signature,
        "scalar_absolute_delta_distributions": scalar_delta,
        "worst_case_charts": ranked[:20],
        "pathological_chart_validation": pathological_rows,
        "pathological_all_close": all(row["all_fields_close"] for row in pathological_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--execution", choices=("serial-reference", "deterministic-batched", "validate"),
        required=True,
    )
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    if device.type != "cuda" and arguments.execution != "serial-reference":
        raise ValueError("Official Performance Track batching/validation requires CUDA")
    payload, points, uv = _load_corpus(arguments.corpus, device)
    plan, eligibility = plan_chart_corpus(points, uv, OFFICIAL_CONFIG)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    reference = candidate = None
    reference_seconds = candidate_seconds = None
    if arguments.execution in ("serial-reference", "validate"):
        reference, reference_seconds = _run_serial(points, uv, device)
    if arguments.execution in ("deterministic-batched", "validate"):
        candidate, candidate_seconds = _run_candidate(plan, points, uv, device)

    identity = {
        "chart_ids_exact": [int(chart["chart_id"]) for chart in payload["charts"]] == list(range(payload["chart_count"])),
        "component_ids": [int(chart["component_id"]) for chart in payload["charts"]],
        "pixel_counts": [int(chart["pixel_count"]) for chart in payload["charts"]],
        "output_order_exact": all(
            results is None or [result.chart_index for result in results] == list(range(payload["chart_count"]))
            for results in (reference, candidate)
        ),
    }
    report: dict[str, Any] = {
        "track": "WL119 Performance Track (independent of Main Architecture Track)",
        "scientific_result_policy": "WL119 scientific report is immutable and is not reinterpreted here",
        "corpus": str(arguments.corpus), "corpus_source": payload["source"],
        "execution": arguments.execution, "device": str(device),
        "torch_version": torch.__version__,
        "official_backend": "torch-reference-terminal-projection-batch-v1",
        "runtime_dependent_splitting": False, "silent_backend_fallback": False,
        "plan": plan.to_record(),
        "eligibility": [asdict(record) for record in eligibility],
        "identity": identity,
        "runtime_seconds": {
            "serial_reference": reference_seconds,
            "deterministic_batched": candidate_seconds,
        },
        "throughput_charts_per_second": {
            "serial_reference": payload["chart_count"] / reference_seconds if reference_seconds else None,
            "deterministic_batched": payload["chart_count"] / candidate_seconds if candidate_seconds else None,
        },
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    if reference_seconds and candidate_seconds:
        report["speedup"] = reference_seconds / candidate_seconds
    if reference is not None and candidate is not None:
        report["equivalence"] = _compare(
            reference, candidate, plan.pathological_chart_indices
        )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(arguments.out), "plan": plan.digest_sha256,
        "runtime_seconds": report["runtime_seconds"],
        "speedup": report.get("speedup"),
        "equivalence": None if "equivalence" not in report else {
            key: report["equivalence"][key] for key in (
                "all_continuous_fields_close", "nan_inf_parity",
                "research_conclusion_invariant", "pathological_all_close",
            )
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
