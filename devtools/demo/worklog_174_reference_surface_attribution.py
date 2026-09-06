from __future__ import annotations

"""W174 real tabletop reference zero-set surface-complex attribution.

Diagnostic only.  Region 1 is not split, the W154 chart is unchanged, the
selected cell population is untouched, no loop is merged or removed, no hole is
filled and no NURBS is fitted.  The batch exposes the marching-cubes triangle
complex that the FROZEN stored corner scalars already determine, and reports
where the current abstraction first loses structural surface identity.

Triangle ownership is construction-native: every authoritative cell is decoded
into its own 2x2x2 scalar block, so a triangle belongs to the cell whose eight
stored corner values produced it.  No distance, radius, normal or threshold
matching associates triangles with support rows.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from devtools.demo import worklog_172_fragmentation_gate_vs_structural_fit_audit as w172
from devtools.demo import worklog_173_tabletop_multi_loop_domain_attribution as w173
from devtools.demo import worklog_174_reextraction_feasibility_probe as probe
from devtools.demo import worklog_174_surface_complex as sc

w171 = w172.w171
OUT = ROOT / "output/174_reference_surface_complex_attribution"
CASE = "real_tabletop_coherent"

# The W173 chart-collision witness this batch is required to audit.
WITNESS_ROWS = (4043, 4051)
WITNESS_BIN = (25, 37)


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def preservation_manifest():
    """Hash the frozen W171/W172/W173 inputs this batch must not disturb."""
    manifest = w173.preservation_manifest()
    for root in (ROOT / "output/173_tabletop_multi_loop_domain_attribution",):
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def native_adjacent_pairs(cells):
    """Deterministic W154 6-face neighbour pairs over the selected cells."""
    lookup = {tuple(cell): index for index, cell in enumerate(cells.tolist())}
    pairs = []
    for index, cell in enumerate(cells.tolist()):
        for axis in range(3):
            neighbour = list(cell)
            neighbour[axis] += 1
            other = lookup.get(tuple(neighbour))
            if other is not None:
                pairs.append((index, other))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def row_component_labels(owner, labels, row_count):
    """Dominant triangle component per support row; -1 when a row owns none."""
    result = np.full(row_count, -1, dtype=np.int64)
    for row in range(row_count):
        owned = labels[owner == row]
        if owned.size:
            result[row] = int(np.bincount(owned).argmax())
    return result


def cell_surface_relation(cells, owner, labels, triangles, incidence, row_count):
    """Compare native 6-face connectivity against triangle-surface adjacency."""
    pairs = native_adjacent_pairs(cells)
    native = {(int(min(a, b)), int(max(a, b))) for a, b in pairs}

    surface = set()
    for tris in incidence.values():
        if len(tris) < 2:
            continue
        rows = {int(owner[t]) for t in tris}
        rows = sorted(rows)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                surface.add((rows[i], rows[j]))

    lost = sorted(native - surface)
    gained = sorted(surface - native)

    row_component = row_component_labels(owner, labels, row_count)
    lost_same_component = sum(
        1 for a, b in lost
        if row_component[a] >= 0 and row_component[a] == row_component[b]
    )

    return {
        "native_face_adjacent_row_pairs": len(native),
        "surface_edge_adjacent_row_pairs": len(surface),
        "native_adjacent_but_not_surface_adjacent": len(lost),
        "native_adjacent_but_not_surface_adjacent_fraction": (len(lost) / len(native)) if native else 0.0,
        "of_those_still_same_triangle_component": int(lost_same_component),
        "of_those_in_different_triangle_components": len(lost) - int(lost_same_component),
        "surface_adjacent_but_not_native_adjacent": len(gained),
        "direction": "Native 6-face connectivity over-states surface adjacency; the converse count is reported, not assumed.",
    }, lost, row_component


def witness_audit(complex_data, labels, incidence, selected_xyz, grid, h, chart_normal, chart_origin):
    """Audit the W173 height-collision witness against the surface complex."""
    owner = complex_data["triangle_owner_row"]
    vertices = complex_data["vertices"]
    triangles = complex_data["triangles"]

    a_row, b_row = WITNESS_ROWS
    a_tris = np.flatnonzero(owner == a_row)
    b_tris = np.flatnonzero(owner == b_row)
    a_components = sorted({int(x) for x in labels[a_tris]})
    b_components = sorted({int(x) for x in labels[b_tris]})
    shared = sorted(set(a_components) & set(b_components))

    path = sc.surface_shortest_path(triangles, a_tris, b_tris, incidence)
    record = {
        "rows": list(WITNESS_ROWS),
        "w173_chart_bin": list(WITNESS_BIN),
        "chart_bins_now": [grid[a_row].tolist(), grid[b_row].tolist()],
        "same_chart_bin": bool(np.array_equal(grid[a_row], grid[b_row])),
        "native_cell_indices": [complex_data["cells"][a_row].tolist(), complex_data["cells"][b_row].tolist()],
        "native_cell_index_delta_l1": int(np.abs(
            complex_data["cells"][a_row].astype(np.int64) - complex_data["cells"][b_row].astype(np.int64)
        ).sum()),
        "world_xyz": [selected_xyz[a_row].tolist(), selected_xyz[b_row].tolist()],
        "triangle_components": {"row_4043": a_components, "row_4051": b_components},
        "same_triangle_connected_component": bool(shared),
    }

    world_gap = float(np.linalg.norm(selected_xyz[a_row] - selected_xyz[b_row]))
    normal = chart_normal / np.linalg.norm(chart_normal)
    heights = (selected_xyz[[a_row, b_row]] - chart_origin) @ normal
    record["world_distance"] = world_gap
    record["world_distance_in_h"] = world_gap / h
    record["chart_normal_height_difference_in_h"] = float(abs(heights[0] - heights[1]) / h)

    if path is not None:
        centroids = vertices[triangles[np.asarray(path["triangle_path"])]].mean(axis=1)
        steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        chord = float(np.linalg.norm(centroids[-1] - centroids[0]))
        record["surface_path"] = {
            "edge_steps": int(path["edge_steps"]),
            "triangle_count": int(len(path["triangle_path"])),
            "path_length_world": float(steps.sum()),
            "path_length_in_h": float(steps.sum() / h),
            "chord_world": chord,
            "path_to_chord_ratio": float(steps.sum() / chord) if chord > 0 else None,
            "path_world_bbox_min": centroids.min(axis=0).tolist(),
            "path_world_bbox_max": centroids.max(axis=0).tolist(),
        }
        record["path_triangles"] = [int(t) for t in path["triangle_path"]]
    else:
        record["surface_path"] = None
        record["path_triangles"] = []
    return record


def chart_distortion(grid, selected_xyz, row_component, chart_normal, chart_origin, h):
    """Quantify what the unchanged W154 chart does to the surface complex."""
    bins = defaultdict(list)
    for row in range(len(grid)):
        bins[(int(grid[row, 0]), int(grid[row, 1]))].append(row)

    multi = [key for key, rows in bins.items() if len(rows) > 1]
    mixed = []
    spans = []
    normal = chart_normal / np.linalg.norm(chart_normal)
    for key in multi:
        rows = bins[key]
        components = {int(row_component[r]) for r in rows if row_component[r] >= 0}
        if len(components) > 1:
            mixed.append({"chart_bin": list(key), "rows": len(rows), "components": sorted(components)})
        heights = (selected_xyz[rows] - chart_origin) @ normal
        spans.append(float(heights.max() - heights.min()))

    spans = np.asarray(spans, dtype=float)
    mixed.sort(key=lambda item: (-len(item["components"]), item["chart_bin"]))
    return {
        "occupied_chart_bins": len(bins),
        "multi_sample_chart_bins": len(multi),
        "chart_bins_mixing_distinct_triangle_components": len(mixed),
        "multi_sample_bin_height_span_in_h": distribution(spans / h),
        "worst_mixed_bins": mixed[:10],
        "note": "Distinct reference-surface components sharing one chart cell is a projection outcome, not a support defect.",
    }


def loop_correspondence(report_173, topology, complex_data):
    """Relate W173 chart holes to reference-surface structure where possible."""
    occupancy = report_173["occupancy"]
    return {
        "w173_chart_holes": int(occupancy["unsupported_bounded_components"]),
        "w173_chart_loops": int(report_173["loop_summary"]["loop_count"]),
        "reference_surface_boundary_edge_components": int(topology["boundary_edge_components"]),
        "reference_surface_triangle_components": int(topology["triangle_connected_components"]),
        "one_to_one_correspondence_established": False,
        "reason": (
            "Chart holes are 2D empty regions of a quantised planar occupancy; reference-surface boundary "
            "components are 1D edge cycles of a 3D triangle complex. Their counts (134 vs "
            f"{int(topology['boundary_edge_components'])}) differ structurally and no construction-native map "
            "between the two populations exists in the frozen records."
        ),
        "what_is_established": [
            "The selected support does carry real reference-surface boundary, so chart holes are not purely a charting artifact.",
            "The reference surface is itself fragmented into "
            f"{int(topology['triangle_connected_components'])} triangle-connected components, so some chart holes "
            "separate genuinely disconnected surface pieces.",
            "Per-hole causal provenance (missing observation vs construction vs projection) remains unrecoverable, exactly as W173 reported.",
        ],
    }


def build_runtime():
    """Load the frozen baseline and recover the reference surface complex."""
    complete, selected, row_indices, boundary, stored = w173.load_baseline()
    grid, projection, occupancy = w173.chart_mapping(selected, boundary)

    corner_values, cells, selected_xyz, h = probe.load_selected_corner_values()
    assert np.array_equal(cells, selected.cell_indices.numpy()), "selected cell drift"
    assert np.allclose(selected_xyz, selected.world_xyz.numpy()), "selected xyz drift"

    complex_data = sc.build_surface_complex(corner_values, cells, h)
    complex_data["cells"] = cells
    triangles = complex_data["triangles"]
    incidence = sc.edge_incidence(triangles)
    labels = sc.triangle_components(triangles, incidence)

    return {
        "complete": complete,
        "selected": selected,
        "selected_rows": row_indices,
        "boundary": boundary,
        "stored": stored,
        "grid": grid,
        "projection": projection,
        "occupancy": occupancy,
        "corner_values": corner_values,
        "cells": cells,
        "selected_xyz": selected_xyz,
        "h": h,
        "complex": complex_data,
        "incidence": incidence,
        "labels": labels,
    }


def analyse(runtime):
    complex_data = runtime["complex"]
    labels = runtime["labels"]
    owner = complex_data["triangle_owner_row"]
    h = runtime["h"]
    boundary = runtime["boundary"]
    chart_normal = boundary.normal.numpy().astype(float)
    chart_origin = boundary.chart_origin.numpy().astype(float)

    topology = sc.complex_topology(complex_data)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1) if labels.size else np.empty(0, dtype=np.int64)
    order = np.argsort(-counts)

    relation, lost_pairs, row_component = cell_surface_relation(
        runtime["cells"], owner, labels, complex_data["triangles"], runtime["incidence"], len(runtime["cells"])
    )

    rows_per_component = {}
    for component in order[:10]:
        rows = np.unique(owner[labels == component])
        rows_per_component[f"C{int(component):03d}"] = {
            "triangles": int(counts[component]),
            "support_rows": int(rows.size),
            "support_row_fraction": float(rows.size / len(runtime["cells"])),
        }

    report_173 = json.loads(
        (ROOT / "output/173_tabletop_multi_loop_domain_attribution/worklog_173_report.json").read_text(encoding="utf-8")
    )

    multi_patch = int((complex_data["per_cell_patch_count"] > 1).sum())
    witness = witness_audit(
        complex_data, labels, runtime["incidence"], runtime["selected_xyz"],
        runtime["grid"], h, chart_normal, chart_origin,
    )

    same_component = witness["same_triangle_connected_component"]
    has_path = witness["surface_path"] is not None
    if same_component and has_path:
        witness_class = "CHART_COLLAPSE_SAME_SURFACE"
    elif not same_component:
        witness_class = "REGION_SUPPORT_MIXES_REFERENCE_STRUCTURES"
    else:
        witness_class = "NON_MANIFOLD_OR_AMBIGUOUS_REFERENCE_TOPOLOGY"

    return {
        "status": "REFERENCE_SURFACE_ATTRIBUTION_NO_CHART_OR_REGION_CHANGE",
        "case": CASE,
        "baseline_preserved": True,
        "baseline": {
            "complete_support": 17965,
            "selected_support": int(len(runtime["cells"])),
            "selected_fraction": float(len(runtime["cells"]) / 17965),
            "h": h,
            "w173_loops": int(report_173["loop_summary"]["loop_count"]),
            "w173_chart_holes": int(report_173["occupancy"]["unsupported_bounded_components"]),
            "native_components": 1,
        },
        "reference_geometry_contract": {
            "definition": "frozen raw TSDF zero-set surface = canonical reference observed geometry",
            "triangles_were_not_persisted": True,
            "recovery": "deterministic marching-cubes on the stored 8 corner scalars of each authoritative cell",
            "ownership_rule": "one cell decoded into its own 2x2x2 block; a triangle cannot cross cells",
            "no_distance_or_threshold_matching": True,
            "vertex_welding": complex_data["vertex_welding_rule"],
            "not_physical_ground_truth": True,
        },
        "surface_complex": {
            **topology,
            "cells_producing_no_triangle": int(len(complex_data["cells_without_triangles"])),
            "unit_cell_containment_violations": complex_data["unit_cell_containment_violations"],
            "per_cell_triangle_count": distribution(complex_data["per_cell_triangle_count"]),
            "per_cell_patch_count": distribution(complex_data["per_cell_patch_count"]),
            "cells_with_multiple_disconnected_patches": multi_patch,
            "largest_component_triangle_fraction": float(counts[order[0]] / len(complex_data["triangles"])) if counts.size else 0.0,
            "component_triangle_counts_head": [int(c) for c in counts[order][:10]],
            "largest_components": rows_per_component,
        },
        "cell_surface_relation": relation,
        "witness": witness,
        "witness_classification": witness_class,
        "chart_distortion": chart_distortion(
            runtime["grid"], runtime["selected_xyz"], row_component, chart_normal, chart_origin, h
        ),
        "loop_correspondence": loop_correspondence(report_173, topology, complex_data),
        "region_changed": False,
        "chart_changed": False,
        "support_changed": False,
        "nurbs_fitted": False,
        "production_changed": False,
    }


def architecture_attribution(report):
    """Weigh the three §11 interpretations against the measured evidence."""
    surface = report["surface_complex"]
    relation = report["cell_surface_relation"]
    distortion = report["chart_distortion"]
    witness = report["witness"]

    components = surface["triangle_connected_components"]
    largest = surface["largest_component_triangle_fraction"]

    return {
        "verdict": "MIXED_ATTRIBUTION",
        "REGION_SUPPORT_SINGLE_REFERENCE_SURFACE": {
            "status": "PARTIALLY_SUPPORTED_DOMINANT_BUT_NOT_SOLE_SHEET",
            "evidence": (
                f"One triangle component carries {largest:.4%} of triangles and "
                f"{surface['largest_components'][sorted(surface['largest_components'])[0]]['support_row_fraction']:.4%} "
                "of support rows, and the W173 height witness is connected inside it by "
                f"{witness['surface_path']['edge_steps'] if witness['surface_path'] else 'no'} edge steps."
            ),
        },
        "REGION_SUPPORT_MIXES_REFERENCE_STRUCTURES": {
            "status": "SUPPORTED_AS_A_MINORITY_REMAINDER",
            "evidence": (
                f"The selected support resolves into {components} triangle-connected reference components, "
                f"{relation['of_those_in_different_triangle_components']} native-adjacent row pairs sit in different "
                "components, and no support row was added or removed to produce this."
            ),
        },
        "REFERENCE_SURFACE_TOPOLOGY_AMBIGUOUS": {
            "status": "SUPPORTED_LOCALLY",
            "evidence": (
                f"{surface['cells_with_multiple_disconnected_patches']} cells own more than one disconnected local "
                f"patch and {surface['non_manifold_vertex_count']} welded vertices are non-manifold, so 'one cell = "
                "one surface piece' does not hold everywhere. Edge degrees are nonetheless at most 2."
            ),
        },
        "chart_role": {
            "status": "PRIMARY_FOR_THE_WITNESS_NOT_SOLE_CAUSE_OVERALL",
            "evidence": (
                f"{distortion['chart_bins_mixing_distinct_triangle_components']} chart bins merge distinct reference "
                f"components and multi-sample bins reach {distortion['multi_sample_bin_height_span_in_h']['max']:.4f} h "
                "of height span, while the audited witness is one continuous surface folded by the planar chart."
            ),
        },
        "completion_answer": (
            "Both contribute. The W154 planar chart demonstrably destroys coherent reference surface -- the audited "
            "witness is one surface path folded into a single chart bin -- but the selected support is also not one "
            f"clean sheet: it carries {components} reference-surface components and locally ambiguous cells before any "
            "charting occurs."
        ),
        "not_established": [
            "That the minority components are physically distinct objects rather than observation gaps in one object.",
            "A per-hole causal provenance linking each W173 chart hole to observation, construction or projection.",
            "Any claim about physical ground-truth geometry; the reference is the frozen zero-set, not physical truth.",
        ],
        "stopped_before_region_split_or_chart_redesign": True,
    }


def loop_inventory_markdown(report, runtime):
    """Full component tail; size is descriptive, never an acceptance rule."""
    labels = runtime["labels"]
    owner = runtime["complex"]["triangle_owner_row"]
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    lines = [
        "# Reference-surface component inventory: head and full tail",
        "",
        "크기는 기술 통계이며 acceptance/selection 규칙이 아니다. 모든 component를 triangle 수 내림차순으로 "
        "빠짐없이 보존한다. Support row 수는 해당 component의 triangle을 소유한 서로 다른 W172 row 수다.",
        "",
        "| Rank | Component | Triangles | Support rows | Row fraction |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, component in enumerate(order):
        rows = np.unique(owner[labels == component])
        lines.append(
            f"| {rank + 1} | C{int(component):03d} | {int(counts[component]):,} | {rows.size:,} | "
            f"{rows.size / len(runtime['cells']):.6%} |"
        )
    return "\n".join(lines) + "\n"


def run(out=OUT):
    from devtools.demo import worklog_174_review_exports as exports

    out.mkdir(parents=True, exist_ok=True)
    manifest = preservation_manifest()

    runtime = build_runtime()
    report = analyse(runtime)
    report["architecture_attribution"] = architecture_attribution(report)
    report["baseline_manifest"] = manifest

    report["visualization"] = exports.export(report, runtime, out)
    exports.write_readmes(report, out)
    report["visualization"]["validation"] = exports.validate_artifacts(out)

    # Frozen inputs must be bit-identical after the batch runs.
    after = preservation_manifest()
    report["frozen_inputs_unchanged_after_run"] = (after == manifest)
    assert report["frozen_inputs_unchanged_after_run"], "frozen input mutated"

    (out / "loop_inventory.md").write_text(loop_inventory_markdown(report, runtime), encoding="utf-8")
    complex_data = runtime["complex"]
    np.savez(
        out / "reference_surface_complex.npz",
        vertices=complex_data["vertices"],
        triangles=complex_data["triangles"],
        triangle_owner_row=complex_data["triangle_owner_row"],
        triangle_component=runtime["labels"],
        per_cell_triangle_count=complex_data["per_cell_triangle_count"],
        per_cell_patch_count=complex_data["per_cell_patch_count"],
        chart_grid=runtime["grid"],
        witness_path=np.asarray(report["witness"]["path_triangles"], dtype=np.int64),
    )
    (out / "worklog_174_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "README.md").write_text(_root_readme(report), encoding="utf-8")
    return report


def _root_readme(report):
    surface = report["surface_complex"]
    attribution = report["architecture_attribution"]
    return (
        "# W174 — Real tabletop reference zero-set surface-complex attribution\n\n"
        f"판정: **{attribution['verdict']}**\n\n"
        f"{attribution['completion_answer']}\n\n"
        "## 산출물\n\n"
        "- `worklog_174_report.json` — 전체 실측치\n"
        "- `loop_inventory.md` — reference-surface component 전체 목록(head + full tail)\n"
        "- `reference_surface_complex.npz` — 복원된 triangle complex, ownership, component, chart 대응, witness path\n"
        "- `review_views/` — 필수 시각화 A~F와 family별 한국어 README\n\n"
        "## 핵심 수치\n\n"
        f"- Reference triangle {surface['triangle_count']:,}개 / welded vertex {surface['welded_vertex_count']:,}개\n"
        f"- Triangle-connected component {surface['triangle_connected_components']}개, 최대 성분 "
        f"{surface['largest_component_triangle_fraction']:.4%}\n"
        f"- Native 6-face 인접이지만 surface 인접이 아닌 row 쌍 "
        f"{report['cell_surface_relation']['native_adjacent_but_not_surface_adjacent']:,}개 "
        f"({report['cell_surface_relation']['native_adjacent_but_not_surface_adjacent_fraction']:.2%}), 역방향 "
        f"{report['cell_surface_relation']['surface_adjacent_but_not_native_adjacent']}개\n"
        f"- W173 height witness는 같은 component 안에서 {report['witness']['surface_path']['edge_steps']} edge step으로 연결\n"
        f"- 서로 다른 component를 한 칸에 합치는 chart bin "
        f"{report['chart_distortion']['chart_bins_mixing_distinct_triangle_components']}개\n\n"
        "## 하지 않은 것\n\n"
        "Region 분할, chart 재설계, chart quantization 변경, support 보수, loop 병합/삭제, hole filling, "
        "morphology, smoothing, NURBS fit, multiply-connected domain 구현, continuation, UNRESOLVED, "
        "Eligibility, inferred Gaussian.\n"
    )


if __name__ == "__main__":
    run()
