from __future__ import annotations

"""W172 diagnostic largest-component control; never a production selection rule."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from devtools.demo import worklog_171_zero_set_derived_visible_structural_nurbs_audit as w171

CASES = ("real_tabletop_coherent", "real_curved_vase_coherent")
OUT = ROOT / "output/172_fragmentation_gate_vs_structural_fit_audit"
FAMILIES = ("complete_support_overview", "component_boundary", "nurbs_fit_common_world", "residual_review")
ROW_FIELDS = ("source_cell_keys", "cell_indices", "world_xyz", "normals", "corner_values", "corner_support_count")


def frozen_manifest() -> dict[str, str]:
    paths = [Path(w171.__file__), ROOT / "osn_gs/surface/torch_gaussian_region_owned_tsdf.py",
             ROOT / "osn_gs/surface/torch_nurbs.py"]
    paths += sorted(p for p in w171.DEFAULT_OUT.rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def row_hash(samples) -> str:
    digest = hashlib.sha256()
    for key in ROW_FIELDS:
        array = getattr(samples, key).numpy()
        digest.update(key.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def largest_component(samples, components):
    """Maximum cardinality; equal sizes break by minimum frozen cell key, then region."""
    if not components:
        raise ValueError("no native component")
    return min(components, key=lambda c: (-int(c.sample_indices.numel()),
               int(samples.source_cell_keys[c.sample_indices].min()), int(c.region_id)))


def selected_runtime(complete, component):
    import torch

    indices = torch.sort(component.sample_indices).values
    arrays = {key: (getattr(complete.samples, key)[indices].clone()
                   if getattr(complete.samples, key).shape[0] else getattr(complete.samples, key).clone())
              for key in ROW_FIELDS}
    samples = w171.TSDFVisibleSurfaceSamples(**arrays, h=complete.samples.h, stats=complete.samples.stats)
    regions = (complete.region_ids[indices.numpy()].copy() if complete.region_ids is not None
               else np.full(len(indices), component.region_id, dtype=np.int64))
    runtime = w171.CaseRuntime(
        name=complete.name, kind="real_diagnostic_single_component", samples=samples,
        selection={"rule": "diagnostic-only maximum native component cardinality; minimum cell key tie-break",
                   "original_component_id": int(component.component_id), "prior_region_id": int(component.region_id),
                   "selected_support_hash_before_analysis": w171._sha256_arrays(
                       samples.source_cell_keys.numpy(), samples.world_xyz.numpy())},
        context_xyz=np.empty((0, 3), dtype=np.float32), region_ids=regions)
    return runtime, indices.numpy()


def diagnose(complete, baseline):
    before = row_hash(complete.samples)
    region_before = w171._sha256_arrays(complete.region_ids)
    reproduced = w171.analyze_case(complete)
    if w171._jsonable(reproduced) != baseline:
        raise AssertionError(f"W171 complete-support baseline mismatch: {complete.name}")
    component = largest_component(complete.samples, complete.components)
    diagnostic, indices = selected_runtime(complete, component)
    selected_before = row_hash(diagnostic.samples)
    result = w171.analyze_case(diagnostic)
    if before != row_hash(complete.samples) or selected_before != row_hash(diagnostic.samples):
        raise AssertionError("zero-set support changed")
    if region_before != w171._sha256_arrays(complete.region_ids):
        raise AssertionError("complete region organization changed")
    for key in ROW_FIELDS:
        original = getattr(complete.samples, key)
        expected = original[indices] if original.shape[0] else original
        if not np.array_equal(expected.numpy(), getattr(diagnostic.samples, key).numpy()):
            raise AssertionError(f"selected support row changed: {key}")
    row = {
        "case": complete.name, "baseline": baseline, "baseline_reproduced_exactly": True,
        "complete_support_count": len(complete.samples.source_cell_keys),
        "complete_component_count": len(complete.components),
        "diagnostic_component_id": int(component.component_id),
        "diagnostic_region_id": int(component.region_id),
        "diagnostic_component_count": len(indices),
        "diagnostic_component_fraction": len(indices) / len(complete.samples.source_cell_keys),
        "excluded_component_count": len(complete.components) - 1,
        "excluded_support_count": len(complete.samples.source_cell_keys) - len(indices),
        "minimum_cell_key": int(diagnostic.samples.source_cell_keys.min()),
        "selected_cell_keys_sha256": w171._sha256_arrays(diagnostic.samples.source_cell_keys.numpy()),
        "complete_rows_hash_before_and_after": before,
        "selected_rows_hash_before_and_after": selected_before,
        "selected_rows_unchanged": True, "existing_region_organization_unchanged": True,
        "diagnostic": result,
        "boundary_closure_semantics": "W154 closed means exactly one traced loop; false does not assert every individual loop is open",
        "boundary_loop_vertex_counts": [len(loop) for loop in diagnostic.boundary.loops],
        "attribution": ("FRAGMENTATION_PRECONDITION_BLOCKED_W171_FIT_CONTROL_MATERIALIZED"
                        if result["result"] == w171.MATERIALIZED_REPRESENTATIVE else
                        "SINGLE_COMPONENT_CONTROL_STILL_FAILS_" + result["reason"]),
        "least_squares_attempted": bool(diagnostic.boundary.eligible and result["solvability"]["full_rank"]),
        "production_component_filtering": False,
    }
    return diagnostic, indices, row


def plot_overview(complete, diagnostic, indices, path):
    plt = w171._prepare_matplotlib()
    points = complete.samples.world_xyz.numpy()
    selected = diagnostic.samples.world_xyz.numpy()
    mask = np.ones(len(points), dtype=bool)
    mask[indices] = False
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    # Every complete-support row is drawn, including every excluded fragment.
    ax.scatter(*points[mask].T, s=2.0, color=w171.CONTEXT_RGB, alpha=.72,
               label=f"other components: {mask.sum():,} rows")
    ax.scatter(*selected.T, s=1.2, color=w171.SUPPORT_RGB, alpha=.55,
               label=f"diagnostic largest: {len(selected):,} rows")
    ax.set_title(f"{complete.name}\ncomplete zero-set support; all {len(points):,} rows shown")
    w171._label_common_world(ax)
    # Retain the original W171 overview extent, including prior-region context.
    extent = np.concatenate((points, complete.context_xyz))
    w171._equal_3d_axes(ax, extent)
    ax.legend(loc="upper right", fontsize=7)
    w171._save_figure(fig, path)
    plt.close(fig)


def plot_boundary(runtime, path):
    plt = w171._prepare_matplotlib()
    points = runtime.samples.world_xyz.numpy()
    shown, _ = w171._stable_plot_subset(points)
    boundary = runtime.boundary
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(121, projection="3d")
    chart = fig.add_subplot(122)
    ax.scatter(*shown.T, s=1.2, color=w171.SUPPORT_RGB, alpha=.55, label="selected zero-set support")
    origin = boundary.chart_origin.numpy()
    tu, tv = boundary.tangent_u.numpy(), boundary.tangent_v.numpy()
    projection = np.column_stack(((shown - origin) @ tu, (shown - origin) @ tv)) / runtime.samples.h
    chart.scatter(*projection.T, s=2, color=w171.DOMAIN_RGB, alpha=.55, label="support chart")
    extents = [points]
    for i, loop in enumerate(boundary.loops):
        line = loop.numpy()
        line = np.vstack((line, line[:1]))
        extents.append(line)
        ax.plot(*line.T, color=w171.BOUNDARY_RGB, linewidth=.65,
                label="ordered chart loops" if i == 0 else None)
        uv = np.column_stack(((line - origin) @ tu, (line - origin) @ tv)) / runtime.samples.h
        chart.plot(*uv.T, color=w171.BOUNDARY_RGB, linewidth=.7,
                   label="ordered chart loops" if i == 0 else None)
    w171._label_common_world(ax)
    w171._equal_3d_axes(ax, np.concatenate(extents))
    chart.set(xlabel="boundary chart u / h", ylabel="boundary chart v / h")
    chart.set_aspect("equal", adjustable="box")
    chart.grid(alpha=.2)
    ax.legend(fontsize=7)
    chart.legend(fontsize=7)
    fig.suptitle(f"{runtime.name}: loops={len(boundary.loops)}, closed={boundary.closed}, valid={boundary.eligible}\n"
                 f"{boundary.reason} — orange loops lie on the chart plane")
    w171._save_figure(fig, path)
    plt.close(fig)


def plot_fit_or_residual(runtime, path, *, residual=False):
    if runtime.representative.status == w171.MATERIALIZED_REPRESENTATIVE:
        plotter = w171._plot_residual_review if residual else w171._plot_nurbs_fit_common_world
        plotter(runtime, path)
        return
    plt = w171._prepare_matplotlib()
    shown, _ = w171._stable_plot_subset(runtime.samples.world_xyz.numpy())
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*shown.T, s=1.2, color=w171.CONTEXT_RGB if residual else w171.SUPPORT_RGB,
               alpha=.55, label="selected zero-set support")
    w171._label_common_world(ax)
    w171._equal_3d_axes(ax, shown)
    family = "residual_review" if residual else "nurbs_fit_common_world"
    ax.set_title(f"{runtime.name} — {family}\nABSTAIN — {runtime.result['reason']}")
    ax.legend(loc="upper right", fontsize=7)
    fig.subplots_adjust(bottom=.17)
    fig.text(.5, .02, "Residual undefined: no fit was performed." if residual else
             "No NURBS or control net was created.", ha="center", color="#b91c1c")
    w171._save_figure(fig, path)
    plt.close(fig)


SHARED = (
    "입력은 W171에서 고정한 W154 zero-set sample의 world XYZ, normal, cell identity와 existing region이다. "
    "Cyan은 진단용 largest native component, gray는 overview의 나머지 complete-support component 또는 residual 미정의 support, "
    "orange는 W154 ordered chart boundary, light blue는 chart support, blue는 materialized NURBS, magenta는 control net이다. "
    "Viridis는 fitted UV에서의 Euclidean residual / h이며 큰 값이 큰 오차다. Red text는 fit/residual 부재 안내다. 색은 visibility나 신뢰 등급이 아니다.\n\n"
    "W171과 동일한 world 좌표, Matplotlib 기본 3D 시점, equal XYZ scale, white background, 150-dpi PNG 및 "
    "8x4 degree-2 W154 fit을 사용한다. Overview는 전체 row를 빠짐없이 그리며 다른 view의 stable display stride는 "
    "계산 support를 바꾸지 않는다. Boundary view는 world/chart 두 panel이다. 축 범위는 각 표시 geometry의 전체 범위로 정하며 "
    "overview는 기존 W171 context까지 포함한 범위를 유지한다. 순수 support plot이며 Gaussian rendering이 아니다.\n\n"
    "Largest component는 diagnostic control일 뿐 trusted surface나 production filtering 규칙이 아니다. "
    "Case 이름의 coherent/curved/vase는 기존 target 이름이며 physical sheet 정답을 보장하지 않는다. "
    "Orange loop는 support chart plane의 경계로 실제 3D surface edge와 같지 않다. "
    "Boundary가 invalid하면 NURBS/control net/residual은 생성되지 않으며 해당 PNG에 실패 상태를 표시한다. "
    "단일 시점의 겹침과 point 밀도로 작은 fragment가 가려질 수 있다. Occluded continuation은 평가하지 않는다.\n"
)


def write_readmes(out):
    descriptions = {
        "": "W172는 complete-support fragmentation gate와 single-component boundary/NURBS 계약의 실패 지점을 비교한다.",
        "review_views": "두 real coherent case에 대해 전체 support, 선택 component의 boundary, NURBS, residual을 비교한다.",
        "case_artifacts": "Case별 JSON과 selected_support.npz에 baseline 비교, 원래 row index 및 변경 없는 선택 support를 보존한다.",
        "review_views/complete_support_overview": "전체 W171 raw zero-set support에서 진단 component만 cyan으로 강조한다. Gray fragment는 삭제하거나 display subsample하지 않는다.",
        "review_views/component_boundary": "선택 component와 모든 ordered loop를 world 3D 및 local chart로 나란히 보여 준다. Multiple loop를 하나로 합치지 않는다.",
        "review_views/nurbs_fit_common_world": "선택 support와 fitted NURBS/control net을 같은 world frame에서 비교한다. 실패하면 support와 정확한 실패 reason을 표시한다.",
        "review_views/residual_review": "Fitted UV residual / h를 support point에 표시한다. Fit이 없으면 gray support와 residual 미정의를 표시한다.",
    }
    for name in CASES:
        descriptions[f"case_artifacts/{name}"] = f"{name}의 diagnostic-only component와 complete W171 baseline 비교 자료다. selected_support.npz의 complete_row_indices로 원래 support에 대응된다."
    for directory, meaning in descriptions.items():
        dest = out / directory
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text(f"# W172 {directory or '진단 산출물'}\n\n{meaning}\n\n{SHARED}", encoding="utf-8")


def validate_artifacts(out):
    from PIL import Image

    paths = [out / "review_views" / family / f"{case}.png" for family in FAMILIES for case in CASES]
    for path in paths:
        with Image.open(path) as img:
            assert img.format == "PNG" and min(img.size) > 500
            img.verify()
    for directory in [out] + sorted(p for p in out.rglob("*") if p.is_dir()):
        assert (directory / "README.md").read_text(encoding="utf-8")
    assert not list(out.rglob("*.ppm"))
    return {"png_count": len(paths), "readme_count": len(list(out.rglob("README.md"))), "complete": True}


def run(out=OUT):
    manifest_before = frozen_manifest()
    baseline = json.loads((w171.DEFAULT_OUT / "worklog_171_report.json").read_text(encoding="utf-8"))
    # Reuse the W171 loader exactly. The mixed case is loaded by that loader but
    # never analyzed, selected, fitted or used as a positive control in W172.
    complete_cases = w171._load_real_runtimes(w171.DEFAULT_W154, w171.DEFAULT_W145)
    write_readmes(out)
    rows = {}
    for name in CASES:
        print(f"[W172] diagnostic {name}", flush=True)
        complete = complete_cases[name]
        runtime, indices, row = diagnose(complete, baseline["cases"][name])
        rows[name] = row
        print(f"[W172] {len(indices)} rows: {row['diagnostic']['reason']}", flush=True)
        case_out = out / "case_artifacts" / name
        np.savez_compressed(case_out / "selected_support.npz", complete_row_indices=indices,
                            region_ids=runtime.region_ids,
                            **{key: getattr(runtime.samples, key).numpy() for key in ROW_FIELDS})
        if runtime.representative.status == w171.MATERIALIZED_REPRESENTATIVE:
            np.savez_compressed(case_out / "bounded_nurbs.npz",
                                control_grid=runtime.representative.surface.control_grid.numpy(),
                                weights=runtime.representative.surface.weights.numpy(), uv=runtime.representative.uv.numpy())
        boundary = runtime.boundary
        np.savez_compressed(case_out / "boundary_chart.npz",
                            chart_occupancy=boundary.chart_occupancy.numpy(),
                            chart_origin=boundary.chart_origin.numpy(), tangent_u=boundary.tangent_u.numpy(),
                            tangent_v=boundary.tangent_v.numpy(),
                            loop_offsets=np.cumsum([0] + [len(loop) for loop in boundary.loops]),
                            loop_world_xyz=(np.concatenate([loop.numpy() for loop in boundary.loops])
                                            if boundary.loops else np.empty((0, 3))))
        w171._write_json(case_out / "result.json", row)
        review = out / "review_views"
        plot_overview(complete, runtime, indices, review / FAMILIES[0] / f"{name}.png")
        plot_boundary(runtime, review / FAMILIES[1] / f"{name}.png")
        plot_fit_or_residual(runtime, review / FAMILIES[2] / f"{name}.png")
        plot_fit_or_residual(runtime, review / FAMILIES[3] / f"{name}.png", residual=True)
    manifest_after = frozen_manifest()
    if manifest_before != manifest_after:
        raise AssertionError("frozen W171 baseline or W154 fitter changed")
    report = {
        "batch": 172, "status": "COMPLETE", "diagnostic_only": True,
        "baseline_and_fitter_manifest": manifest_before, "baseline_files_unchanged": True,
        "fit_contract": w171.FIT_CONTRACT, "cases": rows,
        "selection_rule": "largest native component by count; minimum cell key then region ID tie-break",
        "visualization": validate_artifacts(out), "mixed_contact_fit_attempted": False,
        "production_changed": False, "largest_component_filtering_promoted": False,
        "geometry_repair_or_parameter_tuning": False, "occluded_continuation": False,
        "interpretation": "W171 stopped at complete-support fragmentation. A diagnostic largest component can still fail the unchanged boundary contract before LSQ. Boundary-stage failure does not establish insufficient NURBS approximation capacity.",
        "architecture_result": ("FRAGMENTATION_PRECONDITION_BLOCKED_W171"
                                if all(row["diagnostic"]["result"] == w171.MATERIALIZED_REPRESENTATIVE for row in rows.values())
                                else "SINGLE_COMPONENT_CONTROL_STILL_FAILS_PRESERVE_STRUCTURAL_CONTRACT_FAILURE"),
    }
    w171._write_json(out / "worklog_172_report.json", report)
    print(json.dumps({"architecture_result": report["architecture_result"], "visualization": report["visualization"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    run(parser.parse_args().out)
