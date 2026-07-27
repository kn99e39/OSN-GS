from __future__ import annotations

"""Isolated end-to-end Boundary-first visible construction experiment.

It intentionally does not replace the legacy benchmark dispatcher.  Its only
path is observed support -> conservative recovered region -> boundary pair ->
support curves -> constrained multi-patch surface.
"""
from dataclasses import dataclass
from typing import Any
from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy
from osn_gs.surface.torch_surface_components import build_surface_components
from osn_gs.surface.torch_component_boundary import extract_component_boundary
from osn_gs.surface.torch_boundary_component_recovery import propose_boundary_first_component_recovery, materialize_boundary_first_recovery_regions
from osn_gs.surface.torch_boundary_first_visible_builder import build_boundary_first_visible_surface

@dataclass(frozen=True)
class BoundaryFirstSupportConstruction:
    raw_component_count: int
    recovery_edges: tuple[Any,...]
    recovered_regions: tuple[Any,...]
    boundary_results: tuple[Any,...]
    visible_results: tuple[Any,...]

def construct_boundary_first_support(scene: Any, *, curve_count: int=8, samples_per_curve: int=8, max_normalized_distance: float=2.0, min_normal_dot: float=0.9, boundary_resolution: int=96) -> BoundaryFirstSupportConstruction:
    hierarchy=build_voxel_gaussian_hierarchy(scene.points)
    component_set=build_surface_components(hierarchy,scene.points)
    edges=propose_boundary_first_component_recovery(component_set.components,scene.points,max_normalized_distance=max_normalized_distance,min_normal_dot=min_normal_dot)
    regions=materialize_boundary_first_recovery_regions(component_set.components,scene.points,edges)
    boundaries=tuple(extract_component_boundary(region.component,hierarchy,scene.points,resolution=boundary_resolution) for region in regions)
    results=tuple(build_boundary_first_visible_surface(boundary,component_points=scene.points[region.component.gaussian_indices],source_indices=region.component.gaussian_indices,curve_count=curve_count,samples_per_curve=samples_per_curve) for region,boundary in zip(regions,boundaries))
    return BoundaryFirstSupportConstruction(len(component_set.components),tuple(edges),tuple(regions),boundaries,results)