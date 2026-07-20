from __future__ import annotations

"""Phase 4 §4.1 topology classification.

Routes a Phase 2 ``ComponentBoundaryResult`` to the chart layout Phase 4
knows how to build. Only ``annulus`` gets the new boundary-conforming O-grid
(§4.3); every other class falls back to Phase 3's trimmed rectangular
baseline (the plan's own "safe fallback" chart layout option), since this
project has no chart generator for those shapes yet.
"""

from typing import Any

TOPOLOGY_DISK_LIKE = "disk_like"
TOPOLOGY_ANNULUS = "annulus"
TOPOLOGY_MULTI_HOLE = "multi_hole"
TOPOLOGY_COMPLEX = "complex"
TOPOLOGY_NON_CHARTABLE = "non_chartable"

# A "hole" whose raster area is a smaller fraction of the outer loop's own
# area than this is treated as disk-like rather than annulus. Found empirically
# (2026-07-20): density_gradient's sparse-sampling region dips under the
# absolute density threshold and registers as a 17-cell "hole" against a
# 2468-cell outer loop (0.7%) -- not a real hole, but forcing an 8-slice
# O-grid onto it collapsed the fit (mean seam gap 0.267 vs planar_hole's
# real-hole 0.005). planar_hole's true hole is 262 cells against a
# 3202-cell outer loop (8.2%). 2% sits with margin on both sides of this
# one data point; there is no third labeled example yet to sharpen it further.
MIN_HOLE_AREA_FRACTION = 0.02


def classify_component_topology(
    topology: dict[str, Any], min_hole_area_fraction: float = MIN_HOLE_AREA_FRACTION
) -> str:
    """Classify a component from Phase 2's ``ComponentBoundaryResult.topology`` dict.

    - ``disk_like``: one outer loop, no hole (or only holes below
      ``min_hole_area_fraction`` of the outer loop's own area).
    - ``annulus``: one outer loop, exactly one hole clearing that fraction.
    - ``multi_hole``: one outer loop, two or more holes clearing it.
    - ``complex``: more than one outer loop (the refined support itself
      fragmented into disconnected pieces).
    - ``non_chartable``: no outer loop at all (empty support).

    Callers that only have the coarse ``outer_loop_count``/``hole_count``
    integers (no per-loop area) get the same classification as if every
    listed hole clears the fraction -- the area filter only activates when
    ``topology`` carries per-loop areas (``hole_loop_areas_cells``,
    ``outer_loop_area_cells``), which
    ``extract_component_boundary``'s caller is expected to add before
    calling this with a strict fraction; see ``classify_boundary_result``
    for the normal call path that fills these in automatically.
    """

    outer = int(topology.get("outer_loop_count", 0))
    hole_areas = topology.get("hole_loop_areas_cells")
    outer_area = topology.get("outer_loop_area_cells")
    if hole_areas is not None and outer_area:
        significant_holes = sum(1 for area in hole_areas if area / outer_area >= min_hole_area_fraction)
    else:
        significant_holes = int(topology.get("hole_count", 0))

    if outer == 0:
        return TOPOLOGY_NON_CHARTABLE
    if outer > 1:
        return TOPOLOGY_COMPLEX
    if significant_holes == 0:
        return TOPOLOGY_DISK_LIKE
    if significant_holes == 1:
        return TOPOLOGY_ANNULUS
    return TOPOLOGY_MULTI_HOLE


def classify_boundary_result(boundary_result: Any, min_hole_area_fraction: float = MIN_HOLE_AREA_FRACTION) -> str:
    """Classify directly from a Phase 2 ``ComponentBoundaryResult`` object.

    Preferred entry point: fills in per-loop areas from the result's own
    ``hole_loops``/``outer_loops`` descriptors so the significance filter
    above is always active, rather than relying on the caller to have
    pre-populated ``topology`` with area fields.
    """

    topology = dict(boundary_result.topology)
    topology["hole_loop_areas_cells"] = [loop.area_cells for loop in boundary_result.hole_loops]
    topology["outer_loop_area_cells"] = sum(loop.area_cells for loop in boundary_result.outer_loops)
    return classify_component_topology(topology, min_hole_area_fraction=min_hole_area_fraction)
