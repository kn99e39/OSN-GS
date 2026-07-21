"""Boundary-First NURBS constructor: the ``osn-gs benchmark --constructor
boundary_first`` entry point.

Implements ``OSN_GS_Final_Boundary_First_NURBS_Direction.md`` Phases 1-4 as a
single, scoreable constructor selectable from the SAME unified benchmark CLI
(``nurbs_constructor_benchmark/runner.py``) as ``legacy``/``voxel_patch_stage1``,
instead of separate per-phase scripts: Stage 1 raw-count voxel hierarchy ->
Phase 1 surface components -> Phase 2 per-component boundary extraction ->
Phase 4 topology-routed chart generation (which already falls back to Phase
3's trimmed-rectangle baseline for every non-annulus topology, so there is no
separate Phase 3 codepath to run alongside it).

Returns a duck-typed pseudo state (``BoundaryFirstState``) compatible with
``runner.py``'s shared scoring body (``osn_gs.surface`` patches +
``model.cluster_ids``/``model.surface_uv``/``model.get_xyz``), so this
constructor is scored with the exact same metrics and reported in the exact
same ``report.json`` as ``legacy``/``voxel_patch_stage1``.

Does not touch ``osn_gs/core/torch_pipeline.py`` or the trainer: Gaussian
placement here is just the scene's own raw points (no covariance/opacity
fitting), since this constructor only answers "what NURBS charts fit these
points", the question the boundary-first phases were scoped to benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from osn_gs.surface.torch_annulus_chart import annulus_chart_payload, build_annulus_chart
from osn_gs.surface.torch_chart_topology import TOPOLOGY_ANNULUS, classify_boundary_result
from osn_gs.surface.torch_component_boundary import extract_component_boundary
from osn_gs.surface.torch_surface_components import build_surface_components
from osn_gs.surface.torch_trimmed_component_fitter import fit_trimmed_component
from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy, validate_hierarchy_conservation

from .benchmark_common import uv_support_payload
from .scenes import SyntheticGaussianScene


@dataclass
class _BoundaryFirstModel:
    get_xyz: Any
    cluster_ids: Any
    surface_uv: Any

    def __len__(self) -> int:
        return int(self.get_xyz.shape[0])


@dataclass
class BoundaryFirstState:
    """Duck-typed stand-in for a ``TorchPipelineState``, scored by the same
    ``runner.py`` body as ``legacy``/``voxel_patch_stage1``."""

    model: _BoundaryFirstModel
    surface_patches: list
    surface: Any = None  # no single combined surface; unassigned points score as zero residual
    voxel_hierarchy: Any = None
    per_component: list[dict[str, Any]] = field(default_factory=list)
    component_count: int = 0


def construct_boundary_first(
    scene: SyntheticGaussianScene,
    voxel_min_gaussian_count: int = 10,
    voxel_max_gaussian_count: int = 150,
    voxel_max_depth: int = 6,
    normal_threshold_degrees: float = 40.0,
    offset_threshold_ratio: float = 0.5,
    boundary_resolution: int = 64,
    density_threshold: float = 3.0,
    coarse_gap_closing_cells: int = 2,
    annulus_segments: int = 8,
    annulus_segment_placement: str = "uniform_angle",
    annulus_seam_phase_offset: float = 0.0,
    annulus_hermite_boundary_seed: bool = False,
    fallback_resolution_u: int = 12,
    fallback_resolution_v: int = 12,
    export_dir: Path | None = None,
) -> tuple[BoundaryFirstState, list[dict[str, Any]]]:
    """Build components -> boundaries -> topology-routed charts for one scene.

    ``density_threshold=3.0`` (not Phase 2's own default of 2.0): tuned
    against the ACTUAL rendered/trimmed surface, not just the boundary's own
    topology counts -- see ``docs/worklogs/35_phase3_trimmed_component_baseline.md``
    for the sweep. Returns ``(state, payload_patches)`` so the caller can
    write the renderer export using the same convention as the other
    constructors.
    """

    hierarchy = build_voxel_gaussian_hierarchy(
        scene.points,
        voxel_min_gaussian_count=voxel_min_gaussian_count,
        voxel_max_gaussian_count=voxel_max_gaussian_count,
        voxel_max_depth=voxel_max_depth,
    )
    validate_hierarchy_conservation(hierarchy)
    component_set = build_surface_components(
        hierarchy, scene.points,
        normal_threshold_degrees=normal_threshold_degrees,
        offset_threshold_ratio=offset_threshold_ratio,
    )
    if component_set.component_count() == 0:
        raise ValueError(f"{scene.name}: boundary-first produced zero components.")

    count = int(scene.points.shape[0])
    cluster_ids = torch.full((count,), -1, dtype=torch.long)
    surface_uv = torch.zeros((count, 2), dtype=scene.points.dtype)
    all_surfaces: list[Any] = []
    per_component: list[dict[str, Any]] = []
    payload_patches: list[dict[str, Any]] = []

    for component in component_set.components:
        boundary = extract_component_boundary(
            component, hierarchy, scene.points,
            resolution=boundary_resolution, density_threshold=density_threshold,
            coarse_gap_closing_cells=coarse_gap_closing_cells,
        )
        topology = classify_boundary_result(boundary)

        if topology == TOPOLOGY_ANNULUS:
            chart = build_annulus_chart(
                component, scene.points, boundary.frame, boundary.refined_mask,
                boundary.hole_loops[0].boundary_world_points, segments=annulus_segments,
                outer_boundary_world_points=boundary.outer_loops[0].boundary_world_points if boundary.outer_loops else None,
                segment_placement=annulus_segment_placement,
                seam_phase_offset=annulus_seam_phase_offset,
                hermite_boundary_seed=annulus_hermite_boundary_seed,
            )
            for sl in chart.slices:
                patch_id = len(all_surfaces)
                cluster_ids[sl.gaussian_indices] = patch_id
                surface_uv[sl.gaussian_indices] = sl.uv.to(surface_uv.dtype)
                all_surfaces.append(sl.surface)
                payload_patches.append(
                    {
                        "patch_id": patch_id,
                        "control_grid_shape": [int(x) for x in sl.surface.control_grid.shape],
                        "control_grid": sl.surface.control_grid.detach().cpu().tolist(),
                        "weights": sl.surface.weights.detach().cpu().tolist(),
                        "degree_u": int(sl.surface.degree_u),
                        "degree_v": int(sl.surface.degree_v),
                        "uv_support": uv_support_payload(sl.surface),
                        "fit_metrics": sl.fit_metrics,
                    }
                )
            seam_gaps = [s.mean_gap for s in chart.seams]
            per_component.append(
                {
                    "component_id": component.component_id,
                    "topology": topology,
                    "chart": "o_grid",
                    "segments": annulus_segments,
                    "mean_seam_gap": sum(seam_gaps) / len(seam_gaps) if seam_gaps else 0.0,
                    "max_seam_gap": max((s.max_gap for s in chart.seams), default=0.0),
                    "topology_checks": chart.topology_checks,
                    "chart_quality": chart.chart_quality,
                    "chart_export": annulus_chart_payload(chart) if export_dir is not None else None,
                }
            )
        else:
            fit = fit_trimmed_component(
                component, scene.points, boundary.frame, boundary.refined_mask,
                resolution_u=fallback_resolution_u, resolution_v=fallback_resolution_v,
            )
            patch_id = len(all_surfaces)
            cluster_ids[component.gaussian_indices] = patch_id
            surface_uv[component.gaussian_indices] = fit.uv.detach().to(surface_uv.dtype)
            all_surfaces.append(fit.surface)
            payload_patches.append(
                {
                    "patch_id": patch_id,
                    "control_grid_shape": [int(x) for x in fit.surface.control_grid.shape],
                    "control_grid": fit.surface.control_grid.detach().cpu().tolist(),
                    "weights": fit.surface.weights.detach().cpu().tolist(),
                    "degree_u": int(fit.surface.degree_u),
                    "degree_v": int(fit.surface.degree_v),
                    "uv_support": uv_support_payload(fit.surface),
                    "fit_metrics": fit.fit_metrics,
                }
            )
            per_component.append(
                {
                    "component_id": component.component_id,
                    "topology": topology,
                    "chart": "trimmed_rect_fallback",
                    "fit_metrics": fit.fit_metrics,
                }
            )

    state = BoundaryFirstState(
        model=_BoundaryFirstModel(get_xyz=scene.points, cluster_ids=cluster_ids, surface_uv=surface_uv),
        surface_patches=all_surfaces,
        voxel_hierarchy=hierarchy,
        per_component=per_component,
        component_count=component_set.component_count(),
    )
    return state, payload_patches


def write_point_cloud_ply(scene: SyntheticGaussianScene, path: Path) -> None:
    """Minimal renderer-compatible Gaussian PLY for a synthetic scene's raw points.

    Boundary-first only fits NURBS charts to the scene's existing points (no
    covariance/opacity optimization, unlike ``legacy``/``voxel_patch_stage1``
    which run the real trainer init step) -- opacity/scale/rotation are fixed
    placeholder values sufficient for the renderer to display the point cloud
    alongside the fitted NURBS geometry, per ``docs/RENDERER_INPUT_FORMAT.md``.
    """

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = scene.points.detach().cpu().numpy()
    # Renderer decodes color as clamp01(0.5 + 0.28209479177387814 * f_dc); invert that here.
    f_dc = (scene.colors.detach().cpu().numpy() - 0.5) / 0.28209479177387814
    count = xyz.shape[0]
    opacity = np.full((count, 1), 10.0, dtype=np.float32)  # sigmoid(10) ~= 1.0, fully opaque
    scale = np.full((count, 3), -4.6, dtype=np.float32)  # exp(-4.6) ~= 0.01, small isotropic dot
    rotation = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))  # identity quaternion
    columns = np.column_stack([xyz, f_dc, opacity, scale, rotation])
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header"
    )
    np.savetxt(path, columns, fmt=["%.9g"] * 14, header=header, comments="", encoding="utf-8")


def renderer_payload(scene_name: str, patches: list[dict[str, Any]]) -> dict[str, Any]:
    primary = patches[0]
    return {
        "type": "boundary_first_surface",
        "iteration": 0,
        "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
        "degree_u": primary["degree_u"],
        "degree_v": primary["degree_v"],
        "observed_v_max": 1.0,
        "control_grid_shape": primary["control_grid_shape"],
        "control_grid": primary["control_grid"],
        "weights": primary["weights"],
        "uv_support": primary.get("uv_support"),
        "base_curves": [],
        "occlusion_curves": [],
        "patches": patches,
        "metadata": {"source": "boundary_first", "scene": scene_name, "patch_count": len(patches)},
    }
