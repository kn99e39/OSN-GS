from __future__ import annotations

"""Worklog 169: attribute W168 strict premature blocker counterexamples.

The audit is read-only with respect to W168.  It replays the exact fixtures,
joins every W168 premature ray to its frozen first-hit triangle, evaluates
analytic residuals at the hit and all triangle vertices, and recomputes the
selected triangle/analytic intersection with Decimal arithmetic.  No epsilon,
geometry correction, filtering, or blocker-semantic change is introduced.
"""

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo import worklog_167_raw_zero_set_ray_blocker_audit as w167  # noqa: E402
from devtools.demo import worklog_168_raw_zero_set_first_hit_positive_occlusion_evidence_audit as w168  # noqa: E402


DEFAULT_W168_ROOT = REPO_ROOT / "output/168_raw_zero_set_first_hit_positive_per_view_occlusion_evidence_audit"
DEFAULT_OUT = REPO_ROOT / "output/169_w168_strict_premature_zero_set_blocker_counterexample_attribution"

ATTR_GEOMETRIC = "GEOMETRIC_FRONT_BIAS"
ATTR_NUMERICAL = "NUMERICAL_ORDERING_ONLY"
ATTR_MIXED = "MIXED"
ATTR_NONE = "NO_PREMATURE_RAYS"
ATTR_UNRESOLVED = "HIGH_PRECISION_REFERENCE_UNRESOLVED"
DECIMAL_PRECISION = 80


def _write_json(path: Path, value: Any) -> None:
    w167._write_json(path, value)


def _decimal(value: float | np.floating) -> Decimal:
    return Decimal.from_float(float(value))


def _dvec(values: np.ndarray) -> tuple[Decimal, Decimal, Decimal]:
    array = np.asarray(values, dtype=np.float64).reshape(3)
    return _decimal(array[0]), _decimal(array[1]), _decimal(array[2])


def _dsub(a: tuple[Decimal, Decimal, Decimal], b: tuple[Decimal, Decimal, Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _dadd(a: tuple[Decimal, Decimal, Decimal], b: tuple[Decimal, Decimal, Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _dscale(a: tuple[Decimal, Decimal, Decimal], scale: Decimal) -> tuple[Decimal, Decimal, Decimal]:
    return a[0] * scale, a[1] * scale, a[2] * scale


def _ddot(a: tuple[Decimal, Decimal, Decimal], b: tuple[Decimal, Decimal, Decimal]) -> Decimal:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _dcross(a: tuple[Decimal, Decimal, Decimal], b: tuple[Decimal, Decimal, Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def high_precision_triangle_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    triangle: np.ndarray,
) -> dict[str, Any]:
    """Re-evaluate one frozen triangle using exact float inputs at 80 digits."""

    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        o = _dvec(origin)
        d = _dvec(direction)
        v0, v1, v2 = (_dvec(row) for row in np.asarray(triangle, dtype=np.float64))
        edge1 = _dsub(v1, v0)
        edge2 = _dsub(v2, v0)
        pvec = _dcross(d, edge2)
        determinant = _ddot(edge1, pvec)
        if determinant == 0:
            return {"valid": False, "reason": "EXACT_PARALLEL_OR_DEGENERATE", "t": None, "u": None, "v": None}
        inverse = Decimal(1) / determinant
        tvec = _dsub(o, v0)
        u = _ddot(tvec, pvec) * inverse
        qvec = _dcross(tvec, edge1)
        v = _ddot(d, qvec) * inverse
        t = _ddot(edge2, qvec) * inverse
        valid = t > 0 and u >= 0 and v >= 0 and (u + v) <= 1
        return {
            "valid": bool(valid),
            "reason": "VALID" if valid else "OUTSIDE_STRICT_FROZEN_TRIANGLE",
            "t": t,
            "u": u,
            "v": v,
            "determinant": determinant,
        }


def high_precision_analytic_intersection(
    surface: w167.AnalyticSurface,
    origin: np.ndarray,
    direction: np.ndarray,
) -> Decimal | None:
    """Use Decimal plane/quadratic arithmetic on the frozen analytic inputs."""

    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        o = _dvec(origin)
        d = _dvec(direction)
        center = _dvec(surface.center)
        if surface.kind == "plane_rectangle":
            if surface.normal is None:
                raise ValueError("plane surface has no normal")
            normal = _dvec(surface.normal)
            denominator = _ddot(d, normal)
            if denominator == 0:
                return None
            t = _ddot(_dsub(center, o), normal) / denominator
            return t if t > 0 else None
        if surface.kind == "sphere":
            if surface.radius is None:
                raise ValueError("sphere surface has no radius")
            offset = _dsub(o, center)
            a = _ddot(d, d)
            b = Decimal(2) * _ddot(d, offset)
            radius = _decimal(surface.radius)
            c = _ddot(offset, offset) - radius * radius
            discriminant = b * b - Decimal(4) * a * c
            if discriminant < 0 or a == 0:
                return None
            root = discriminant.sqrt()
            near = (-b - root) / (Decimal(2) * a)
            far = (-b + root) / (Decimal(2) * a)
            if near > 0:
                return near
            return far if far > 0 else None
        raise ValueError(surface.kind)


def analytic_surface_residual(surface: w167.AnalyticSurface, points: np.ndarray) -> np.ndarray:
    """Signed exact-equation residual in float64 for reporting."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if surface.kind == "plane_rectangle":
        if surface.normal is None:
            raise ValueError("plane surface has no normal")
        return (points - surface.center) @ surface.normal
    if surface.kind == "sphere":
        if surface.radius is None:
            raise ValueError("sphere surface has no radius")
        return np.linalg.norm(points - surface.center, axis=1) - surface.radius
    raise ValueError(surface.kind)


def high_precision_surface_residual(
    surface: w167.AnalyticSurface,
    point: tuple[Decimal, Decimal, Decimal],
) -> Decimal:
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        center = _dvec(surface.center)
        relative = _dsub(point, center)
        if surface.kind == "plane_rectangle":
            if surface.normal is None:
                raise ValueError("plane surface has no normal")
            return _ddot(relative, _dvec(surface.normal))
        if surface.kind == "sphere":
            if surface.radius is None:
                raise ValueError("sphere surface has no radius")
            return _ddot(relative, relative).sqrt() - _decimal(surface.radius)
        raise ValueError(surface.kind)


def _distribution(values: np.ndarray) -> dict[str, Any]:
    return w167._distribution(np.asarray(values, dtype=np.float64))


def classify_attribution(high_precision_depth_deltas: Iterable[Decimal | None]) -> dict[str, Any]:
    geometric = 0
    numerical = 0
    equal = 0
    reversed_count = 0
    unresolved = 0
    for delta in high_precision_depth_deltas:
        if delta is None:
            unresolved += 1
        elif delta < 0:
            geometric += 1
        else:
            numerical += 1
            equal += int(delta == 0)
            reversed_count += int(delta > 0)
    if unresolved:
        verdict = ATTR_UNRESOLVED
    elif geometric and numerical:
        verdict = ATTR_MIXED
    elif geometric:
        verdict = ATTR_GEOMETRIC
    elif numerical:
        verdict = ATTR_NUMERICAL
    else:
        verdict = ATTR_NONE
    return {
        "verdict": verdict,
        "geometric_front_bias_count": geometric,
        "numerical_ordering_only_count": numerical,
        "higher_precision_exact_equal_count": equal,
        "higher_precision_reversed_count": reversed_count,
        "higher_precision_unresolved_count": unresolved,
        "premature_sign_stable_count": geometric,
    }


def _surface_for_premature_ray(fixture: dict[str, Any], ray_index: int) -> w167.AnalyticSurface:
    analytic = fixture["analytic"]
    if fixture["name"] == "layered_two_sheet":
        return analytic["surfaces"][0]
    surface_index = int(analytic["first_surface_index"][ray_index])
    if surface_index < 0:
        raise ValueError("premature ray has no analytic surface")
    return analytic["surfaces"][surface_index]


def _float_array_exact_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(np.array_equal(left.view(np.uint8), right.view(np.uint8)))


def reproduce_w168_fixture(fixture: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    with np.load(artifact_path, allow_pickle=False) as frozen:
        replay_masks = w168._interval_masks(
            fixture["analytic"]["first_depth"],
            fixture["analytic"]["first_hit"],
            fixture["zero_hit"].depth,
            fixture["zero_hit"].status,
        )
        checks = {
            "analytic_first_depth_bitwise": _float_array_exact_equal(fixture["analytic"]["first_depth"], frozen["analytic_first_depth"]),
            "analytic_hit_exact": bool(np.array_equal(fixture["analytic"]["first_hit"], frozen["analytic_hit"])),
            "zero_set_first_hit_depth_bitwise": _float_array_exact_equal(fixture["zero_hit"].depth, frozen["zero_set_first_hit_depth"]),
            "zero_set_status_exact": bool(np.array_equal(fixture["zero_hit"].status.astype(str), frozen["zero_set_status"])),
            "triangle_id_exact": bool(np.array_equal(fixture["zero_hit"].triangle_id, frozen["triangle_id"])),
            "component_id_exact": bool(np.array_equal(fixture["zero_hit"].component_id, frozen["component_id"])),
            "premature_mask_exact": bool(np.array_equal(replay_masks["premature"], frozen["premature"])),
        }
        premature_count = int(np.count_nonzero(frozen["premature"]))
    return {
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": w167._sha256_file(artifact_path),
        "premature_count": premature_count,
        "checks": checks,
        "all_exact": all(checks.values()),
    }


def _decimal_string(value: Decimal | None) -> str:
    return "UNRESOLVED" if value is None else format(value, ".70E")


def attribute_fixture(fixture: dict[str, Any], w168_artifact: Path, out_root: Path) -> dict[str, Any]:
    name = fixture["name"]
    with np.load(w168_artifact, allow_pickle=False) as frozen:
        premature = np.asarray(frozen["premature"], dtype=bool)
        ray_indices = np.flatnonzero(premature).astype(np.int64)
        frozen_z_s = np.asarray(frozen["zero_set_first_hit_depth"], dtype=np.float64)
        frozen_z_star = np.asarray(frozen["analytic_first_depth"], dtype=np.float64)
        triangle_ids = np.asarray(frozen["triangle_id"], dtype=np.int64)
        component_ids = np.asarray(frozen["component_id"], dtype=np.int64)
        interior = np.asarray(frozen["interior"], dtype=bool)
        boundary = np.asarray(frozen["silhouette_or_support_boundary"], dtype=bool)

    n = len(ray_indices)
    hit_residual = np.zeros((n,), dtype=np.float64)
    vertex_residual = np.zeros((n, 3), dtype=np.float64)
    hp_hit_residual = np.full((n,), np.nan, dtype=np.float64)
    hp_vertex_residual = np.full((n, 3), np.nan, dtype=np.float64)
    hp_triangle_depth = np.full((n,), np.nan, dtype=np.float64)
    hp_analytic_depth = np.full((n,), np.nan, dtype=np.float64)
    hp_delta = np.full((n,), np.nan, dtype=np.float64)
    hp_triangle_depth_text: list[str] = []
    hp_analytic_depth_text: list[str] = []
    hp_delta_text: list[str] = []
    hp_hit_residual_text: list[str] = []
    hp_vertex_residual_text: list[list[str]] = []
    hp_reference_valid = np.zeros((n,), dtype=bool)
    decimal_deltas: list[Decimal | None] = []

    rays = fixture["rays"]
    vertices = fixture["vertices"]
    faces = fixture["faces"]
    for output_index, ray_index in enumerate(ray_indices.tolist()):
        triangle_id = int(triangle_ids[ray_index])
        if triangle_id < 0:
            decimal_deltas.append(None)
            hp_triangle_depth_text.append("UNRESOLVED")
            hp_analytic_depth_text.append("UNRESOLVED")
            hp_delta_text.append("UNRESOLVED")
            hp_hit_residual_text.append("UNRESOLVED")
            hp_vertex_residual_text.append(["UNRESOLVED"] * 3)
            continue
        triangle = vertices[faces[triangle_id]]
        surface = _surface_for_premature_ray(fixture, ray_index)
        hit_residual[output_index] = analytic_surface_residual(surface, fixture["zero_hit"].world_xyz[ray_index])[0]
        vertex_residual[output_index] = analytic_surface_residual(surface, triangle)
        high_triangle = high_precision_triangle_intersection(rays.origins[ray_index], rays.directions[ray_index], triangle)
        high_analytic = high_precision_analytic_intersection(surface, rays.origins[ray_index], rays.directions[ray_index])
        if not high_triangle["valid"] or high_analytic is None:
            decimal_deltas.append(None)
            hp_triangle_depth_text.append(_decimal_string(high_triangle.get("t")))
            hp_analytic_depth_text.append(_decimal_string(high_analytic))
            hp_delta_text.append("UNRESOLVED")
            hp_hit_residual_text.append("UNRESOLVED")
            hp_vertex_residual_text.append(["UNRESOLVED"] * 3)
            continue
        scale = _decimal(rays.camera_depth_scale[ray_index])
        triangle_depth = high_triangle["t"] * scale
        analytic_depth = high_analytic * scale
        delta = triangle_depth - analytic_depth
        high_point = _dadd(_dvec(rays.origins[ray_index]), _dscale(_dvec(rays.directions[ray_index]), high_triangle["t"]))
        residual = high_precision_surface_residual(surface, high_point)
        vertex_residual_decimal = [high_precision_surface_residual(surface, _dvec(vertex)) for vertex in triangle]
        hp_reference_valid[output_index] = True
        hp_triangle_depth[output_index] = float(triangle_depth)
        hp_analytic_depth[output_index] = float(analytic_depth)
        hp_delta[output_index] = float(delta)
        hp_hit_residual[output_index] = float(residual)
        hp_vertex_residual[output_index] = [float(value) for value in vertex_residual_decimal]
        hp_triangle_depth_text.append(_decimal_string(triangle_depth))
        hp_analytic_depth_text.append(_decimal_string(analytic_depth))
        hp_delta_text.append(_decimal_string(delta))
        hp_hit_residual_text.append(_decimal_string(residual))
        hp_vertex_residual_text.append([_decimal_string(value) for value in vertex_residual_decimal])
        decimal_deltas.append(delta)

    attribution = classify_attribution(decimal_deltas)
    float_interval = frozen_z_star[ray_indices] - frozen_z_s[ray_indices]
    hp_premature = hp_reference_valid & (hp_delta < 0.0)
    hp_interval = -hp_delta[hp_premature]
    fixture_root = out_root / "fixtures" / name
    fixture_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fixture_root / "premature_attribution.npz",
        ray_index=ray_indices,
        triangle_id=triangle_ids[ray_indices],
        component_id=component_ids[ray_indices],
        interior=interior[ray_indices],
        silhouette_or_support_boundary=boundary[ray_indices],
        float64_zero_set_depth=frozen_z_s[ray_indices],
        float64_analytic_depth=frozen_z_star[ray_indices],
        float64_depth_delta=frozen_z_s[ray_indices] - frozen_z_star[ray_indices],
        float64_premature_interval_length=float_interval,
        hit_point_analytic_residual=hit_residual,
        triangle_vertex_analytic_residual=vertex_residual,
        high_precision_reference_valid=hp_reference_valid,
        high_precision_triangle_depth=hp_triangle_depth,
        high_precision_analytic_depth=hp_analytic_depth,
        high_precision_depth_delta=hp_delta,
        high_precision_hit_point_analytic_residual=hp_hit_residual,
        high_precision_triangle_vertex_analytic_residual=hp_vertex_residual,
        high_precision_triangle_depth_decimal=np.asarray(hp_triangle_depth_text),
        high_precision_analytic_depth_decimal=np.asarray(hp_analytic_depth_text),
        high_precision_depth_delta_decimal=np.asarray(hp_delta_text),
        high_precision_hit_residual_decimal=np.asarray(hp_hit_residual_text),
        high_precision_triangle_vertex_residual_decimal=np.asarray(hp_vertex_residual_text),
        attribution=np.asarray([
            ATTR_UNRESOLVED if delta is None else (ATTR_GEOMETRIC if delta < 0 else ATTR_NUMERICAL)
            for delta in decimal_deltas
        ]),
    )
    component_count = Counter(int(value) for value in component_ids[ray_indices].tolist())
    report = {
        "fixture": name,
        "premature_ray_count": n,
        "prevalence": {
            "premature_ray_count": n,
            "both_hit_ray_count": int(fixture["intervals"]["both_hit_rays"]),
            "fraction_of_both_hit_rays": float(n / max(int(fixture["intervals"]["both_hit_rays"]), 1)),
        },
        "float64_premature_interval_length": _distribution(float_interval),
        "float64_premature_interval_length_over_h": _distribution(float_interval / w167.SYNTHETIC_H),
        "hit_point_analytic_residual": _distribution(hit_residual),
        "hit_point_analytic_residual_over_h": _distribution(hit_residual / w167.SYNTHETIC_H),
        "triangle_vertex_analytic_residual": _distribution(vertex_residual.reshape(-1)),
        "triangle_vertex_analytic_residual_over_h": _distribution(vertex_residual.reshape(-1) / w167.SYNTHETIC_H),
        "triangle_vertex_front_residual_count": int(np.count_nonzero(vertex_residual < 0.0)),
        "triangle_vertex_on_surface_residual_count": int(np.count_nonzero(vertex_residual == 0.0)),
        "triangle_vertex_back_residual_count": int(np.count_nonzero(vertex_residual > 0.0)),
        "high_precision_hit_point_analytic_residual": _distribution(hp_hit_residual[hp_reference_valid]),
        "high_precision_hit_point_analytic_residual_over_h": _distribution(hp_hit_residual[hp_reference_valid] / w167.SYNTHETIC_H),
        "high_precision_triangle_vertex_analytic_residual": _distribution(hp_vertex_residual[hp_reference_valid].reshape(-1)),
        "high_precision_triangle_vertex_analytic_residual_over_h": _distribution(hp_vertex_residual[hp_reference_valid].reshape(-1) / w167.SYNTHETIC_H),
        "high_precision_triangle_vertex_front_count": int(np.count_nonzero(hp_vertex_residual[hp_reference_valid] < 0.0)),
        "high_precision_triangle_vertex_on_surface_count": int(np.count_nonzero(hp_vertex_residual[hp_reference_valid] == 0.0)),
        "high_precision_triangle_vertex_back_count": int(np.count_nonzero(hp_vertex_residual[hp_reference_valid] > 0.0)),
        "high_precision_depth_delta": _distribution(hp_delta[hp_reference_valid]),
        "high_precision_depth_delta_over_h": _distribution(hp_delta[hp_reference_valid] / w167.SYNTHETIC_H),
        "higher_precision_remaining_premature_interval_length": _distribution(hp_interval),
        "higher_precision_remaining_premature_interval_length_over_h": _distribution(hp_interval / w167.SYNTHETIC_H),
        "attribution": attribution,
        "interior_attribution": classify_attribution([delta for delta, keep in zip(decimal_deltas, interior[ray_indices].tolist()) if keep]),
        "boundary_attribution": classify_attribution([delta for delta, keep in zip(decimal_deltas, boundary[ray_indices].tolist()) if keep]),
        "first_hit_triangle_count": len(set(int(value) for value in triangle_ids[ray_indices].tolist())),
        "first_hit_component_count": len(component_count),
        "first_hit_component_provenance": [{"component_id": key, "premature_hit_count": value} for key, value in sorted(component_count.items())],
        "raw_artifact": str((fixture_root / "premature_attribution.npz").resolve()),
        "semantic_epsilon": None,
    }
    _write_json(fixture_root / "fixture_report.json", report)
    fixture_root.joinpath("README.md").write_text(
        f"# W169 `{name}` Premature Attribution\n\n"
        f"이 directory는 W168 `{name}`의 strict `z_s < z*` ray {n}개를 exact first-hit triangle ID로 join한 diagnostic이다. `premature_attribution.npz`에는 hit XYZ residual, 세 triangle vertex residual, float64 depth ordering, 80-digit Decimal reference depth/sign을 ray별로 보존한다.\n\n"
        "Signed residual semantics: plane/layered front surface에서는 exact plane equation `(p-c)·n`; sphere에서는 radial residual `||p-c||-r`이다. depth attribution은 high-precision `z_s-z*<0`이면 GEOMETRIC_FRONT_BIAS, 0 이상이면 NUMERICAL_ORDERING_ONLY이며 tolerance를 사용하지 않는다.\n\n"
        "공통 조건: W168 fixture, historical SparseProjectiveTSDF/extraction, raw geometry, W167 first-hit triangle, strict semantics, 모든 component를 변경 없이 사용한다. 이 output은 correction이나 trusted subset이 아니며 production blocker semantics를 바꾸지 않는다.\n",
        encoding="utf-8",
    )
    return report


def _prepare_output(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fixtures = out / "fixtures"
    if fixtures.exists():
        shutil.rmtree(fixtures)
    for name in ("README.md", "worklog_169_report.json"):
        path = out / name
        if path.exists():
            path.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    _prepare_output(args.out)
    w168_report_path = args.w168_root / "worklog_168_report.json"
    immutable_paths = [w168_report_path] + [args.w168_root / "synthetic" / name / "fixture_audit.npz" for name in w168.FIXTURE_NAMES]
    hashes_before = {str(path.resolve()): w167._sha256_file(path) for path in immutable_paths}
    fixtures_root = args.out / "fixtures"
    fixtures_root.mkdir(parents=True, exist_ok=True)
    fixtures_root.joinpath("README.md").write_text(
        "# W169 Fixture Attribution\n\n"
        "각 하위 directory는 W168 fixture 하나의 premature ray만 포함하며, exact selected triangle과 analytic surface residual, 80-digit Decimal ordering reference를 저장한다. W168 artifact는 읽기 전용이며 재생 결과와 bitwise/exact join을 먼저 검증한다.\n\n"
        "상태 정의: GEOMETRIC_FRONT_BIAS는 high-precision에서도 `z_s-z*<0`; NUMERICAL_ORDERING_ONLY는 high-precision에서 0 또는 양수; MIXED는 두 population이 모두 존재함을 뜻한다. semantic epsilon, threshold, component filtering은 없다.\n",
        encoding="utf-8",
    )

    reproduction: dict[str, Any] = {}
    attribution: dict[str, Any] = {}
    for name in w168.FIXTURE_NAMES:
        print(f"[worklog 169] replaying and attributing {name}", flush=True)
        fixture = w168._build_fixture(name)
        artifact = args.w168_root / "synthetic" / name / "fixture_audit.npz"
        reproduction[name] = reproduce_w168_fixture(fixture, artifact)
        if not reproduction[name]["all_exact"]:
            raise RuntimeError(f"W168 reproduction failed for {name}")
        attribution[name] = attribute_fixture(fixture, artifact, args.out)

    hashes_after = {str(path.resolve()): w167._sha256_file(path) for path in immutable_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("W168 artifact hash changed during read-only W169 audit")
    total_premature = sum(int(value["premature_ray_count"]) for value in attribution.values())
    total_stable = sum(int(value["attribution"]["premature_sign_stable_count"]) for value in attribution.values())
    total_numerical = sum(int(value["attribution"]["numerical_ordering_only_count"]) for value in attribution.values())
    total_equal = sum(int(value["attribution"]["higher_precision_exact_equal_count"]) for value in attribution.values())
    total_reversed = sum(int(value["attribution"]["higher_precision_reversed_count"]) for value in attribution.values())
    global_unresolved = sum(int(value["attribution"]["higher_precision_unresolved_count"]) for value in attribution.values())
    if global_unresolved:
        global_verdict = ATTR_UNRESOLVED
    elif total_stable and total_numerical:
        global_verdict = ATTR_MIXED
    elif total_stable:
        global_verdict = ATTR_GEOMETRIC
    elif total_numerical:
        global_verdict = ATTR_NUMERICAL
    else:
        global_verdict = ATTR_NONE
    global_attribution = {
        "verdict": global_verdict,
        "geometric_front_bias_count": total_stable,
        "numerical_ordering_only_count": total_numerical,
        "higher_precision_exact_equal_count": total_equal,
        "higher_precision_reversed_count": total_reversed,
        "higher_precision_unresolved_count": global_unresolved,
        "premature_sign_stable_count": total_stable,
    }
    report = {
        "status": "COMPLETE_WL169_W168_STRICT_PREMATURE_ZERO_SET_BLOCKER_COUNTEREXAMPLE_ATTRIBUTION",
        "worklog": 169,
        "1_intent_alignment": {
            "status": "PASS",
            "question": "Are W168 strict premature counterexamples intrinsic geometric front-bias or finite-precision comparison artifacts?",
            "diagnostic_only": True,
            "correction_implemented": False,
            "epsilon_or_tolerance": None,
            "w168_artifacts_modified": False,
        },
        "2_w168_reproduction": {
            "all_fixtures_exact": all(value["all_exact"] for value in reproduction.values()),
            "per_fixture": reproduction,
            "immutable_sha256_before": hashes_before,
            "immutable_sha256_after": hashes_after,
            "hashes_unchanged": hashes_before == hashes_after,
        },
        "3_premature_magnitude_accounting": {
            "total_premature_rays": total_premature,
            "per_fixture": {
                name: {
                    "prevalence": value["prevalence"],
                    "world_units": value["float64_premature_interval_length"],
                    "normalized_by_h": value["float64_premature_interval_length_over_h"],
                    "higher_precision_remaining_count": value["attribution"]["premature_sign_stable_count"],
                    "higher_precision_remaining_world_units": value["higher_precision_remaining_premature_interval_length"],
                    "higher_precision_remaining_over_h": value["higher_precision_remaining_premature_interval_length_over_h"],
                }
                for name, value in attribution.items()
            },
            "count_vs_magnitude": "prevalence and displacement magnitude are reported separately; no count is treated as a large geometric failure without its world/h magnitude",
        },
        "4_hit_point_analytic_residuals": {
            name: {
                "float64_world_units": value["hit_point_analytic_residual"],
                "float64_over_h": value["hit_point_analytic_residual_over_h"],
                "high_precision_world_units": value["high_precision_hit_point_analytic_residual"],
                "high_precision_over_h": value["high_precision_hit_point_analytic_residual_over_h"],
            }
            for name, value in attribution.items()
        },
        "5_hit_triangle_vertex_residuals": {
            name: {
                "world_units": value["triangle_vertex_analytic_residual"],
                "normalized_by_h": value["triangle_vertex_analytic_residual_over_h"],
                "front_count": value["triangle_vertex_front_residual_count"],
                "on_surface_count": value["triangle_vertex_on_surface_residual_count"],
                "back_count": value["triangle_vertex_back_residual_count"],
                "high_precision_world_units": value["high_precision_triangle_vertex_analytic_residual"],
                "high_precision_normalized_by_h": value["high_precision_triangle_vertex_analytic_residual_over_h"],
                "high_precision_front_count": value["high_precision_triangle_vertex_front_count"],
                "high_precision_on_surface_count": value["high_precision_triangle_vertex_on_surface_count"],
                "high_precision_back_count": value["high_precision_triangle_vertex_back_count"],
                "first_hit_triangle_count": value["first_hit_triangle_count"],
            }
            for name, value in attribution.items()
        },
        "6_high_precision_ordering_check": {
            "reference": "Decimal.from_float exact frozen float inputs, 80-digit Moller-Trumbore and analytic plane/quadratic arithmetic on W168-premature selected triangles only",
            "platform_longdouble_is_not_higher_precision": np.finfo(np.longdouble).nmant == np.finfo(np.float64).nmant,
            "per_fixture": {name: {"depth_delta": value["high_precision_depth_delta"], "depth_delta_over_h": value["high_precision_depth_delta_over_h"], **value["attribution"]} for name, value in attribution.items()},
            "total_premature_sign_stable": total_stable,
            "total_numerical_ordering_only": total_numerical,
            "unresolved": global_unresolved,
        },
        "7_geometry_vs_numerical_attribution": {
            "verdict": global_attribution["verdict"],
            "global_counts": global_attribution,
            "per_fixture": {name: value["attribution"] for name, value in attribution.items()},
            "classification_rule": "no epsilon: high-precision z_s-z*<0 is GEOMETRIC_FRONT_BIAS; exact zero or positive is NUMERICAL_ORDERING_ONLY",
        },
        "8_architecture_interpretation": {
            "w168_result_preserved": True,
            "blocker_semantics_modified": False,
            "automatic_correction_proposed": False,
            "interpretation": "filled after exact high-precision accounting; count prevalence and geometric magnitude remain separate",
        },
        "9_retained_rejected_open": {
            "retained": [
                "W168 fixtures/result and strict z_s<z_q semantics",
                "historical SparseProjectiveTSDF/extraction/raw zero-set geometry",
                "W167 ray-triangle contract and all components",
                "per-view diagnostic scope",
            ],
            "rejected": [
                "epsilon/tolerance or offset",
                "mesh dilation/erosion, iso-value or h/mu change",
                "component filtering or trusted subset",
                "blocker semantic modification, NURBS, global aggregation",
            ],
            "open": ["any future architectural response requires a separate authorized batch"],
        },
        "outputs": {
            "root": str(args.out.resolve()),
            "fixture_reports": {name: value["raw_artifact"] for name, value in attribution.items()},
            "visualizations_generated": False,
            "ppm_count": 0,
            "w153_replay_cache_copied_to_temp": False,
        },
        "runtime_seconds": time.time() - started,
    }
    # State the answer directly without changing the measured classification.
    if global_attribution["verdict"] == ATTR_MIXED:
        answer = "Both intrinsic geometric front-bias and finite-precision ordering artifacts are present."
    elif global_attribution["verdict"] == ATTR_GEOMETRIC:
        answer = "The strict premature population is stable under high precision and is intrinsic geometric front-bias of the frozen zero-set."
    elif global_attribution["verdict"] == ATTR_NUMERICAL:
        answer = "Higher precision removes or reverses every apparent strict premature ordering; the population is numerical ordering only."
    elif global_attribution["verdict"] == ATTR_NONE:
        answer = "No premature population was present."
    else:
        answer = "The high-precision selected-triangle reference is unresolved for at least one premature ray."
    report["7_geometry_vs_numerical_attribution"]["answer"] = answer
    report["8_architecture_interpretation"]["interpretation"] = (
        answer
        + " W168 remains the frozen strict-contract result; this diagnostic does not add an epsilon or authorize a correction."
    )
    _write_json(args.out / "worklog_169_report.json", report)
    args.out.joinpath("README.md").write_text(
        "# Worklog 169 — W168 Strict Premature Counterexample Attribution\n\n"
        "이 output은 W168 strict premature ray를 geometry와 finite-precision ordering으로 attribution하는 read-only diagnostic이다. W168 fixture/NPZ hash를 실행 전후 대조하고, selected first-hit triangle의 hit point와 세 vertex analytic residual 및 80-digit Decimal intersection sign을 저장한다.\n\n"
        f"Attribution verdict: `{global_attribution['verdict']}`. {answer}\n\n"
        "`fixtures/<fixture>/premature_attribution.npz`가 ray-level authoritative artifact다. visualization은 새로 생성하지 않았으며 W168 geometry, strict semantics, 모든 component, Candidate-B와 production behavior는 변경하지 않았다.\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w168-root", type=Path, default=DEFAULT_W168_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run(args)
    print(json.dumps({"status": report["status"], "verdict": report["7_geometry_vs_numerical_attribution"]["verdict"], "answer": report["7_geometry_vs_numerical_attribution"]["answer"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
