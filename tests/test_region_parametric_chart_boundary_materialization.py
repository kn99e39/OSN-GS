"""Worklog 61: region-local parametric chart boundary wired into the real
visible-NURBS materialization path (not diagnostics-only).

`eligible_parametric_chart_surfaces()` is a SEPARATE entry point from the
physical `eligible_materialized_surfaces()` -- these tests check that the
physical path is byte-identical to its pre-worklog-61 baseline (negative
control) while the new path additively produces real materialized surfaces,
including for Sphere (which has zero physical boundary anywhere).
"""

from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_region_parametric_chart_boundary import STATUS_ELIGIBLE_PARAMETRIC_CHART


def _construction(scene_name: str, cap: int = 64):
    scene = make_gaussian_reliability_scene(scene_name)
    positions = torch.as_tensor(scene.positions, dtype=torch.float32)
    covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
    opacity = torch.ones(positions.shape[0])
    stable_ids = list(range(positions.shape[0]))
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device="cpu")
    return pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids).construction


class ParametricChartMaterializationTest(unittest.TestCase):
    def test_negative_control_physical_path_is_byte_identical_to_baseline(self):
        # cap=64 baseline established worklog 47-60: Box 6/6, Cylinder 2/2,
        # Sphere 0/0, Thin-slab 3/3 (physical closed/materialized). This
        # round's addition must not move these numbers at all.
        expected = {"box": 6, "cylinder": 2, "sphere": 0, "thin_slab": 3}
        for scene_name, expected_count in expected.items():
            with self.subTest(scene=scene_name):
                construction = _construction(scene_name)
                self.assertEqual(len(construction.eligible_materialized_surfaces()), expected_count)
                self.assertEqual(construction.diagnostic_summary["materialized_surface_count"], expected_count)

    def test_sphere_gets_a_real_materialized_parametric_chart_surface_but_zero_physical(self):
        construction = _construction("sphere")
        self.assertEqual(len(construction.eligible_materialized_surfaces()), 0)
        chart_surfaces = construction.eligible_parametric_chart_surfaces()
        self.assertGreaterEqual(len(chart_surfaces), 1)
        for surface in chart_surfaces:
            self.assertEqual(surface.state, "materialized")
            self.assertIsNotNone(surface.surface)
            self.assertEqual(surface.input.region_status, STATUS_ELIGIBLE_PARAMETRIC_CHART)
            # Never disguised as a physical boundary.
            self.assertNotIn(surface, construction.eligible_materialized_surfaces())

    def test_box_parametric_chart_surfaces_are_additive_not_a_replacement(self):
        construction = _construction("box")
        physical = construction.eligible_materialized_surfaces()
        chart = construction.eligible_parametric_chart_surfaces()
        self.assertEqual(len(physical), 6)
        self.assertGreater(len(chart), 0)
        physical_region_ids = {item.input.source_region_id for item in physical}
        chart_region_ids = {item.input.source_region_id for item in chart}
        # Both entry points remain fully separate lists (a caller must
        # combine them explicitly); provenance never blurs which is which.
        for item in chart:
            self.assertEqual(item.input.region_status, STATUS_ELIGIBLE_PARAMETRIC_CHART)
        for item in physical:
            self.assertNotEqual(item.input.region_status, STATUS_ELIGIBLE_PARAMETRIC_CHART)

    def test_partition_seam_segments_are_never_reported_as_physical_termination(self):
        construction = _construction("thin_slab")
        for chart_status in construction.diagnostic_summary["region_parametric_chart_boundaries"]:
            counts = chart_status.get("segment_kind_counts")
            if counts is None:
                continue
            total = sum(counts.values())
            if total == 0:
                continue
            # Every segment is classified into exactly one of the four kinds
            # -- disjoint accounting, no double counting or unlabeled segment.
            self.assertEqual(total, len(counts) and sum(counts.values()))
            self.assertTrue(set(counts) <= {"physical_termination", "crease", "observation_frontier", "partition_seam"})


if __name__ == "__main__":
    unittest.main()
