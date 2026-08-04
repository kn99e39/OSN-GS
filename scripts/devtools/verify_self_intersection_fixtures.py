"""Worklog 36: analytic fixture validation of validate_simple_closed_loop."""

from __future__ import annotations

import math

from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop


def rectangle():
    return [(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)]


def concave_simple():
    return [(0, 0, 0), (2, 0, 0), (2, 2, 0), (1, 1, 0), (0, 2, 0)]


def bow_tie():
    # a1-a2-a3-a4 where segment a1a2 crosses a3a4 (classic bow-tie).
    return [(0, 0, 0), (2, 2, 0), (2, 0, 0), (0, 2, 0)]


def figure_eight():
    return [(0, 0, 0), (2, 2, 0), (4, 0, 0), (2, -2, 0), (0, 0, 0), (2, 2, 0)][:5]  # degenerate repeat handled below


def repeated_vertex():
    return [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0), (0.5, 1, 0)]


def near_touching():
    return [(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1e-6, 0), (0, 1, 0)]


def two_disjoint_loops_combined_should_not_apply():
    # This function only validates ONE loop at a time -- not applicable here directly.
    pass


def main():
    cases = {
        "rectangle": (rectangle(), True),
        "concave_simple": (concave_simple(), True),
        "bow_tie": (bow_tie(), False),
        "repeated_vertex": (repeated_vertex(), False),
        "near_touching": (near_touching(), True),
    }
    for name, (points, expect_simple) in cases.items():
        report = validate_simple_closed_loop(points)
        status = "OK" if report.is_simple_polygon == expect_simple else "FAIL"
        print(f"{name}: is_simple={report.is_simple_polygon} expected={expect_simple} [{status}] reasons={report.reasons} winding={report.winding_number:.3f} turning={report.total_turning_angle:.3f}")


if __name__ == "__main__":
    main()
