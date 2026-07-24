from osn_gs.surface.torch_nurbs import (
    SharedBoundaryConstraint,
    TorchCurveSet,
    TorchNURBSSurface,
    boundary_control_indices,
    build_torch_surface,
    fit_coupled_patch_graph_lsq,
    fit_torch_base_curves,
    fit_torch_visible_surface,
    fit_torch_visible_surface_lsq,
    pca_parameterize_points,
    predict_torch_occlusion_curves,
    project_torch_points_to_nurbs,
    sample_torch_occluded_surface,
)
from osn_gs.surface.torch_boundary_reconciliation import (
    PatchEdgePair,
    PatchReconciliationResult,
    fit_reconciled_patch_graph,
)
from osn_gs.surface.torch_aabb_broad_phase import BroadPhasePair, sweep_and_prune_pairs
from osn_gs.surface.torch_candidate_evidence import validate_candidate_observation_evidence
from osn_gs.surface.torch_continuation_domain import (
    ContinuationDomain,
    ContinuationDomainBuildError,
    build_continuation_domain,
)
from osn_gs.surface.torch_occluded_region_candidate import (
    ConflictEdge,
    CorrespondenceEdge,
    OccludedRegionCandidate,
    SupportChain,
    build_candidate_conflicts,
    build_geometric_region_candidates,
)
from osn_gs.surface.torch_observation_evidence import (
    CameraViewEvidence,
    EmptyVoxelSupportResult,
    ObservationEvidence,
    SampleEvidence,
    build_observation_evidence,
    classify_world_samples,
    query_empty_voxel_support,
)
from osn_gs.surface.torch_patch_boundary import (
    PatchBoundarySegment,
    build_rectangular_patch_edge,
    extract_trimmed_patch_boundaries,
)
from osn_gs.surface.torch_voxel_regions import TorchVoxelSurfaceRegions, build_torch_voxel_surface_regions

__all__ = [
    "BroadPhasePair",
    "CameraViewEvidence",
    "ConflictEdge",
    "ContinuationDomain",
    "ContinuationDomainBuildError",
    "CorrespondenceEdge",
    "EmptyVoxelSupportResult",
    "ObservationEvidence",
    "OccludedRegionCandidate",
    "PatchBoundarySegment",
    "PatchEdgePair",
    "PatchReconciliationResult",
    "SampleEvidence",
    "SharedBoundaryConstraint",
    "SupportChain",
    "TorchCurveSet",
    "TorchNURBSSurface",
    "TorchVoxelSurfaceRegions",
    "boundary_control_indices",
    "build_candidate_conflicts",
    "build_continuation_domain",
    "build_geometric_region_candidates",
    "build_observation_evidence",
    "build_rectangular_patch_edge",
    "build_torch_surface",
    "build_torch_voxel_surface_regions",
    "classify_world_samples",
    "extract_trimmed_patch_boundaries",
    "fit_coupled_patch_graph_lsq",
    "fit_reconciled_patch_graph",
    "fit_torch_base_curves",
    "fit_torch_visible_surface",
    "fit_torch_visible_surface_lsq",
    "pca_parameterize_points",
    "predict_torch_occlusion_curves",
    "project_torch_points_to_nurbs",
    "query_empty_voxel_support",
    "sample_torch_occluded_surface",
    "sweep_and_prune_pairs",
    "validate_candidate_observation_evidence",
]
