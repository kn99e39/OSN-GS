# WL151 compatibility matrix

| Canonical requirement | Renderer availability | Classification |
|---|---|---|
| local node positions | world XYZ | **COMPATIBLE** |
| canonical covariance/frame | covariance, log_scales, rotations, tangent scale, thickness | **INCOMPATIBLE_MISSING** |
| unique stable row identity | frozen event/row ID | **COMPATIBLE_BY_EXISTING_DETERMINISTIC_MAPPING** |
| structural reliability | intrinsic/contextual reliability classes | **INCOMPATIBLE_MISSING** |
| manifold affinity graph | same_surface/crease/parallel_separate/rejected edge relations | **INCOMPATIBLE_MISSING** |
| normal candidate | event normal | **SEMANTICALLY_DIFFERENT** |
| visibility/evidence state | renderer median depth and camera/pixel provenance | **SEMANTICALLY_DIFFERENT** |
| physical-sheet identity | manual control label / human review | **SEMANTICALLY_DIFFERENT** |
| region ownership | node_region_id / SurfaceRegionCandidate | **INCOMPATIBLE_MISSING** |
| member provenance | member/core/attached/rejected IDs | **INCOMPATIBLE_MISSING** |
| boundary candidate source | region-owned world halfedges | **INCOMPATIBLE_MISSING** |
| boundary adjacency/order | directed compatible halfedges → ordered component | **INCOMPATIBLE_MISSING** |
| closed/open/branch/ambiguous state | OrderedBoundaryComponent state | **INCOMPATIBLE_MISSING** |
| region-core interior | reliable core IDs/points for same region | **INCOMPATIBLE_MISSING** |
| pre-fit adapter ownership | source_region_id + component + ordered boundary/interior IDs | **INCOMPATIBLE_MISSING** |
| renderer depth → world position | camera/pixel/depth reconstruction | **COMPATIBLE_BY_EXISTING_DETERMINISTIC_MAPPING** |

## Gate

**CONTRACT_GAP** — Stop Condition A. Candidate C, synthetic contracts, and real-scene replay were not run.

`HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE` applies only to event 1527.
