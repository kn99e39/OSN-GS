from __future__ import annotations

"""Static review exports for W174; all support, triangles and loops retained.

No component is hidden, no small fragment is filtered and no smoothing or gap
filling is applied for display.  Each family carries its own Korean README.
"""

from pathlib import Path

import numpy as np

from devtools.demo import worklog_174_reference_surface_attribution as w174
from devtools.demo import worklog_172_render_view_projection_review as projection

w171 = w174.w171

FAMILIES = (
    "A_reference_surface_complex_world",
    "B_surface_complex_components",
    "C_w173_height_witness",
    "D_world_vs_chart_surface",
    "E_loop_cause_review",
    "F_rgb_reference",
)


def _component_colors(labels, plt):
    """Deterministic per-component colors; the giant sheet stays neutral."""
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    rank = np.empty(len(counts), dtype=np.int64)
    rank[order] = np.arange(len(counts))
    cmap = plt.get_cmap("tab20")
    colors = np.empty((len(labels), 4))
    for index, label in enumerate(labels):
        if rank[label] == 0:
            colors[index] = (0.62, 0.66, 0.71, 0.55)  # dominant sheet
        else:
            colors[index] = cmap(int(rank[label]) % 20)
    return colors, rank


def export(report, runtime, out):
    plt = w171._prepare_matplotlib()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    complex_data = runtime["complex"]
    vertices = complex_data["vertices"]
    triangles = complex_data["triangles"]
    labels = runtime["labels"]
    xyz = runtime["selected_xyz"]
    grid = runtime["grid"]
    h = runtime["h"]
    witness = report["witness"]

    for family in FAMILIES:
        (out / "review_views" / family).mkdir(parents=True, exist_ok=True)

    reviewed = []

    def save(fig, family, name="tabletop.png"):
        target = out / "review_views" / family / name
        w171._save_figure(fig, target)
        plt.close(fig)
        reviewed.append(str(target.relative_to(out)))

    colors, rank = _component_colors(labels, plt)

    # A: the recovered surface itself, drawn as triangles so it reads as a sheet.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(vertices[triangles], alpha=0.75)
    mesh.set_facecolor((0.35, 0.55, 0.78, 0.75))
    mesh.set_edgecolor((0.15, 0.2, 0.3, 0.18))
    mesh.set_linewidth(0.08)
    ax.add_collection3d(mesh)
    ax.scatter(*xyz.T, s=0.6, c="#d1495b", alpha=0.35, label="15,189 W172 support samples")
    w171._equal_3d_axes(ax, vertices)
    w171._label_common_world(ax)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(
        "W174 A — reference zero-set surface recovered from frozen corner scalars\n"
        f"{len(triangles):,} triangles / {len(vertices):,} welded vertices; construction-native cell ownership"
    )
    save(fig, FAMILIES[0])

    # B: every triangle component, including the 21 single-triangle ones.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(vertices[triangles], alpha=0.9)
    mesh.set_facecolor(colors)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    w171._equal_3d_axes(ax, vertices)
    w171._label_common_world(ax)
    counts = np.bincount(labels)
    ax.set_title(
        f"W174 B — {int(labels.max()) + 1} triangle-connected reference components\n"
        f"grey: dominant sheet ({counts.max():,} tris, {counts.max() / len(triangles):.2%}); colors: all "
        f"{int(labels.max())} remaining components, none hidden"
    )
    save(fig, FAMILIES[1])

    # C: the witness -- why two points collapse into one chart bin.
    path = np.asarray(report["witness"]["path_triangles"], dtype=np.int64)
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(121, projection="3d")
    neighbourhood = np.unique(triangles[path].reshape(-1))
    local = np.linalg.norm(vertices[triangles].mean(axis=1) - vertices[neighbourhood].mean(axis=0), axis=1)
    context = np.flatnonzero(local < np.percentile(local, 12))
    mesh = Poly3DCollection(vertices[triangles[context]], alpha=0.35)
    mesh.set_facecolor((0.72, 0.76, 0.8, 0.35))
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    walk = Poly3DCollection(vertices[triangles[path]], alpha=0.95)
    walk.set_facecolor((0.85, 0.28, 0.36, 0.95))
    walk.set_edgecolor((0.3, 0.05, 0.1, 0.5))
    walk.set_linewidth(0.25)
    ax.add_collection3d(walk)
    a_row, b_row = w174.WITNESS_ROWS
    ax.scatter(*xyz[[a_row]].T, s=55, c="#111111", marker="o", label=f"row {a_row}")
    ax.scatter(*xyz[[b_row]].T, s=55, c="#111111", marker="^", label=f"row {b_row}")
    w171._equal_3d_axes(ax, vertices[triangles[context]].reshape(-1, 3))
    w171._label_common_world(ax)
    ax.legend(fontsize=7)
    sp = witness["surface_path"]
    ax.set_title(
        f"world — one continuous surface path\n{sp['edge_steps']} edge steps, "
        f"{sp['path_length_in_h']:.1f} h along surface vs {witness['world_distance_in_h']:.1f} h straight"
    )

    ax2 = fig.add_subplot(122)
    ax2.scatter(grid[:, 0], grid[:, 1], s=1.2, c="#c7ccd1", label="all 15,189 rows")
    owner = complex_data["triangle_owner_row"]
    path_rows = np.unique(owner[path])
    ax2.scatter(grid[path_rows, 0], grid[path_rows, 1], s=9, c="#d94f5c", label="surface-path rows")
    ax2.scatter(grid[[a_row, b_row], 0], grid[[a_row, b_row], 1], s=90, facecolors="none",
                edgecolors="#111111", linewidths=1.4, label=f"rows {a_row} / {b_row} — same bin")
    ax2.set(xlabel="chart u / h", ylabel="chart v / h")
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(fontsize=7)
    ax2.set_title(
        f"W154 chart — both rows land in bin {tuple(witness['chart_bins_now'][0])}\n"
        f"native cells are {witness['native_cell_index_delta_l1']} apart (L1)"
    )
    fig.suptitle("W174 C — the W173 height collision is a planar-chart fold of one reference surface", fontsize=11)
    save(fig, FAMILIES[2])

    # D: same neighbourhoods in world, continuous chart, quantised occupancy.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    row_component = np.full(len(xyz), -1, dtype=np.int64)
    for row in range(len(xyz)):
        owned = labels[owner == row]
        if owned.size:
            row_component[row] = int(np.bincount(owned).argmax())
    visible = row_component >= 0
    row_colors = np.array([colors[np.flatnonzero(labels == c)[0]] if c >= 0 else (0, 0, 0, 1)
                           for c in row_component])
    axes[0].scatter(xyz[visible, 1], xyz[visible, 2], s=1.4, c=row_colors[visible])
    axes[0].set(xlabel="world Y", ylabel="world Z", title="world (Y,Z) by reference component")
    axes[0].set_aspect("equal", adjustable="box")
    proj = runtime["projection"]
    axes[1].scatter(proj[visible, 0] / h, proj[visible, 1] / h, s=1.4, c=row_colors[visible])
    axes[1].set(xlabel="chart u / h", ylabel="chart v / h", title="continuous chart coordinates")
    axes[1].set_aspect("equal", adjustable="box")
    axes[2].imshow(runtime["occupancy"], origin="lower", cmap="Blues", interpolation="nearest")
    mixed = report["chart_distortion"]["worst_mixed_bins"]
    if mixed:
        mb = np.array([m["chart_bin"] for m in mixed])
        axes[2].scatter(mb[:, 0], mb[:, 1], s=42, facecolors="none", edgecolors="#d62828", linewidths=1.3,
                        label="worst component-mixing bins")
        axes[2].legend(fontsize=7)
    axes[2].set(xlabel="chart u index", ylabel="chart v index", title="quantised W154 occupancy")
    fig.suptitle(
        f"W174 D — {report['chart_distortion']['chart_bins_mixing_distinct_triangle_components']} of "
        f"{report['chart_distortion']['occupied_chart_bins']} occupied bins merge distinct reference components",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, FAMILIES[3])

    # E: chart holes against reference-surface boundary; no favourable selection.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(runtime["occupancy"], origin="lower", cmap="Greys", alpha=0.3, interpolation="nearest")
    boundary_rows = set()
    incidence = runtime["incidence"]
    for edge, tris in incidence.items():
        if len(tris) == 1:
            boundary_rows.add(int(owner[tris[0]]))
    boundary_rows = np.array(sorted(boundary_rows), dtype=np.int64)
    axes[0].scatter(grid[:, 0], grid[:, 1], s=1.0, c="#c7ccd1", label="all rows")
    axes[0].scatter(grid[boundary_rows, 0], grid[boundary_rows, 1], s=2.4, c="#e07a1f",
                    label=f"rows owning reference-surface boundary ({len(boundary_rows):,})")
    axes[0].set(xlabel="chart u index", ylabel="chart v index")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=7)
    axes[0].set_title("chart holes vs real surface boundary\n(no 1:1 map is claimed)")

    lc = report["loop_correspondence"]
    axes[1].bar(
        ["W173\nchart loops", "W173\nchart holes", "surface boundary\ncomponents", "triangle\ncomponents"],
        [lc["w173_chart_loops"], lc["w173_chart_holes"],
         lc["reference_surface_boundary_edge_components"], lc["reference_surface_triangle_components"]],
        color=["#4c78a8", "#4c78a8", "#e07a1f", "#7d4f9c"],
    )
    for index, value in enumerate([lc["w173_chart_loops"], lc["w173_chart_holes"],
                                   lc["reference_surface_boundary_edge_components"],
                                   lc["reference_surface_triangle_components"]]):
        axes[1].text(index, value, f"{value}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("count")
    axes[1].set_title("populations differ structurally\n2D chart regions vs 1D edge cycles vs 2D patches")
    fig.suptitle("W174 E — loop / reference-surface correspondence is NOT one-to-one", fontsize=11)
    fig.tight_layout()
    save(fig, FAMILIES[4])

    # F: RGB secondary context only; component identity carried over, no depth culling.
    from PIL import Image

    cameras, _ = w171.w155._build_named_cameras(w171.DEFAULT_DATASET, "images_8", "sparse/0", -1, 8, "cpu")
    camera_accounting = {}
    for camera_name, camera in cameras.items():
        with Image.open(w171.DEFAULT_DATASET / "images_8" / camera_name) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax in axes:
            ax.imshow(rgb)
            ax.set(xlim=(-0.5, camera.image_width - 0.5), ylim=(camera.image_height - 0.5, -0.5))
            ax.set_axis_off()
        axes[0].set_title("Original RGB photograph — secondary reference")
        pixels, valid = projection.project(xyz, camera)
        axes[1].scatter(*pixels[valid].T, s=0.6, c=row_colors[valid], alpha=0.45, linewidths=0)
        axes[1].set_title("Selected support colored by reference component; no depth culling")
        fig.suptitle(f"W174 {camera_name} — physical context only, not topology evidence")
        fig.tight_layout()
        save(fig, FAMILIES[5], Path(camera_name).stem + ".png")
        camera_accounting[camera_name] = {
            "support_projected": int(valid.sum()),
            "support_total": len(xyz),
            "depth_culling": False,
        }

    return {"png_count": len(reviewed), "png_paths": reviewed, "camera_accounting": camera_accounting}


def write_readmes(report, out):
    """One shared Korean README per family, each with its own analysis section."""
    surface = report["surface_complex"]
    relation = report["cell_surface_relation"]
    distortion = report["chart_distortion"]
    witness = report["witness"]
    path = witness["surface_path"]

    shared = (
        "입력은 W172 selected tabletop support 15,189 row와 그 frozen corner scalar에서 복원한 "
        f"reference zero-set triangle {surface['triangle_count']:,}개다. Triangle ownership은 "
        "construction-native다. 각 authoritative cell을 자기만의 2x2x2 block으로 marching-cubes에 넣으므로 "
        "triangle이 cell 경계를 넘을 수 없고, distance/radius/normal/threshold 매칭을 쓰지 않는다. "
        "Vertex welding은 global lattice-edge의 exact integer key이며 공간 tolerance 병합이 아니다.\n\n"
        "좌표계는 W171/W172/W173과 동일한 world frame이고 chart 좌표는 W154 chart를 그대로 쓴다. "
        "단위 h는 0.012105485424399376이다. 어떤 component도 숨기지 않았고 small fragment filtering, "
        "smoothing, gap filling, morphology를 적용하지 않았다.\n\n"
        "한계: reference는 frozen zero-set이며 physical ground truth가 아니다. "
        "Triangle은 저장된 것이 아니라 동일 corner scalar에서 결정론적으로 재계산한 것이다. "
        "RGB는 depth culling이 없어 image overlap을 topology 증거로 쓰지 않는다.\n"
    )

    analyses = {
        FAMILIES[0]: (
            f"복원된 표면은 triangle {surface['triangle_count']:,}개, welded vertex "
            f"{surface['welded_vertex_count']:,}개, edge {surface['edge_count']:,}개이며 Euler characteristic은 "
            f"{surface['euler_characteristic']}이다. Edge degree는 1이 {surface['edge_degree_histogram'].get('1', 0):,}개, "
            f"2가 {surface['edge_degree_histogram'].get('2', 0):,}개로 non-manifold edge는 "
            f"{surface['non_manifold_edge_count']}개다. 15,189 cell 전부가 triangle을 냈고 unit-cell 이탈은 0건이라 "
            "ownership이 exact함을 그림 자체로 확인할 수 있다. Euler/genus는 topology 집계일 뿐 물리적 의미로 읽지 않는다."
        ),
        FAMILIES[1]: (
            f"Triangle-connected component는 {surface['triangle_connected_components']}개다. 최대 성분이 "
            f"{surface['largest_component_triangle_fraction']:.4%}를 차지하지만 나머지 "
            f"{surface['triangle_connected_components'] - 1}개가 실재한다. 상위 성분 triangle 수는 "
            f"{surface['component_triangle_counts_head']}이고 tail에는 1~2 triangle짜리 조각도 있다. "
            "회색이 dominant sheet, 색상이 나머지 전부이며 크기로 숨긴 성분은 없다. "
            "성분이 여러 개라는 사실 자체가 곧 서로 다른 물리적 물체를 뜻하지는 않는다. "
            "관측 결손으로 한 물체가 끊어져 보이는 경우와 구분되지 않는다."
        ),
        FAMILIES[2]: (
            f"W173 witness인 row 4043/4051은 native cell 기준 L1 거리 {witness['native_cell_index_delta_l1']}만큼 "
            f"떨어져 있고 world 거리는 {witness['world_distance_in_h']:.2f} h인데 W154 chart에서는 같은 bin "
            f"{tuple(witness['chart_bins_now'][0])}에 들어간다. 그러나 두 row는 "
            f"**같은 triangle-connected component**에 속하며 {path['edge_steps']} edge step으로 이어진다. "
            f"표면을 따라간 길이는 {path['path_length_in_h']:.1f} h이고 직선 대비 비는 "
            f"{path['path_to_chord_ratio']:.2f}로, 상판에서 옆면으로 내려가는 연속 표면이다. "
            "따라서 이 witness는 support가 이질적 구조를 섞은 결과가 아니라 planar chart가 접은 결과다."
        ),
        FAMILIES[3]: (
            f"Occupied chart bin {distortion['occupied_chart_bins']:,}개 중 "
            f"{distortion['multi_sample_chart_bins']:,}개가 다중 샘플이고, 그중 "
            f"{distortion['chart_bins_mixing_distinct_triangle_components']}개는 서로 다른 reference component를 "
            f"한 칸에 합친다. Multi-sample bin의 height span은 median "
            f"{distortion['multi_sample_bin_height_span_in_h']['median']:.4f} h, p95 "
            f"{distortion['multi_sample_bin_height_span_in_h']['p95']:.4f} h, max "
            f"{distortion['multi_sample_bin_height_span_in_h']['max']:.4f} h다. "
            "왼쪽 world와 가운데 continuous chart를 비교하면 색이 섞이는 위치가 곧 quantization 이전에 이미 "
            "겹치는 위치임을 알 수 있다."
        ),
        FAMILIES[4]: (
            f"W173 chart loop {report['loop_correspondence']['w173_chart_loops']}개 및 chart hole "
            f"{report['loop_correspondence']['w173_chart_holes']}개와, reference surface의 boundary edge component "
            f"{report['loop_correspondence']['reference_surface_boundary_edge_components']}개, triangle component "
            f"{report['loop_correspondence']['reference_surface_triangle_components']}개는 서로 개수가 다르다. "
            "chart hole은 quantized planar occupancy의 2D 빈 영역이고 surface boundary component는 3D triangle "
            "complex의 1D edge cycle이므로 애초에 같은 종류의 대상이 아니다. 따라서 1:1 대응을 주장하지 않는다. "
            f"다만 selected support가 실제 surface boundary를 {surface['boundary_edge_count']:,} edge만큼 갖고 있으므로 "
            "chart hole 전부를 charting artifact로 돌릴 수도 없다. Per-hole 원인 규명은 여전히 불가능하다."
        ),
        FAMILIES[5]: (
            "원본 RGB에 selected support를 reference component 색으로 투영했다. Depth culling이 없어 "
            "가려진 점도 그려지므로 image overlap을 physical topology 증거로 쓰지 않는다. "
            "tabletop의 물리적 위치 확인용 secondary context로만 본다."
        ),
    }

    legends = {
        FAMILIES[0]: "파랑 면: 복원된 zero-set triangle. 빨강 점: 15,189 support sample.",
        FAMILIES[1]: "회색: 최대 triangle component. tab20 색: 나머지 모든 component(20색 반복, 의미 등급 아님).",
        FAMILIES[2]: "빨강 면: witness surface path의 triangle. 연회색 면: 주변 맥락. 검정 마커: row 4043(원)/4051(삼각형). 오른쪽 회색 점: 전체 row.",
        FAMILIES[3]: "점 색: 해당 row의 dominant reference component(B와 동일 규칙). 빨강 원: component를 섞는 상위 bin.",
        FAMILIES[4]: "주황 점: reference-surface boundary를 소유한 row. 회색 점: 나머지 row. 막대 색: 파랑=chart 계열, 주황=surface boundary, 보라=triangle component.",
        FAMILIES[5]: "점 색: reference component(B와 동일 규칙).",
    }

    root = out / "review_views"
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# W174 review views\n\n"
        "W174는 W172 selected tabletop support의 reference zero-set surface complex를 노출해, "
        "topology 실패가 support 자체에서 오는지 W154 planar chart에서 오는지 귀속한다. "
        "Region 분할, chart 변경, support 보수, loop 병합, hole filling, NURBS fit은 하지 않았다.\n\n"
        + shared
        + "\n## Family\n\n"
        + "\n".join(f"- `{family}/` — {analyses[family].split('.')[0]}." for family in FAMILIES)
        + "\n",
        encoding="utf-8",
    )
    for family in FAMILIES:
        (root / family).mkdir(parents=True, exist_ok=True)
        (root / family / "README.md").write_text(
            f"# {family}\n\n## 입력과 의미\n\n{shared}\n## 색 범례\n\n{legends[family]}\n\n"
            f"## 분석 및 평가\n\n{analyses[family]}\n",
            encoding="utf-8",
        )


def validate_artifacts(out):
    pngs = sorted(p for p in (out / "review_views").rglob("*.png"))
    readmes = sorted(p for p in (out / "review_views").rglob("README.md"))
    assert not list((out / "review_views").rglob("*.ppm")), "PPM must be converted"
    return {"png_count": len(pngs), "readme_count": len(readmes), "complete": len(pngs) >= 8 and len(readmes) == len(FAMILIES) + 1}
