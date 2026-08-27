"""Worklog 120 -- Observed / Occluded volumetric operationalization audit.

Four INDEPENDENT competing architecture hypotheses for partitioning the
camera-supported 3D reconstruction domain into OBSERVED / OCCLUDED (with
UNRESOLVED allowed only as a fail-closed implementation state):

    candidate_a_surface_hit          -- A. direct surface observation
    candidate_b_median_depth         -- B. median-depth partition
    candidate_c_geometric_visibility -- C. geometric line of sight
    candidate_d_renderer_reachability-- D. renderer reachability

They are NOT four tunable variants of one implementation. Each module owns
exactly one `classify_view(...)` function and nothing else decides its
semantics. Everything a candidate is NOT allowed to own -- query
representation, camera projection, the relevant-view contract, the frozen
global aggregation rule, metrics and serialization -- lives in `shared.py`,
which contains no visibility boundary of any kind. See the worklog's
"Shared-Code Semantic Audit" section.
"""
