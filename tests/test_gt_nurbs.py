from __future__ import annotations

"""Boundary-conformal ground-truth NURBS chart validation.

The GT charts are the ideal target representation: they must lie on the true
surface, stay inside the analytic support predicate, cover it, and carry the
support topology in the parameterization itself (no trim masks).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


SCENES = (
    "plane", "sine", "crease", "density_gradient", "triangle", "u_shape",
    "crescent", "planar_hole", "elongated_plane", "mild_curved_sheet",
    "close_parallel_sheets",
)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class ConformalGTChartTest(unittest.TestCase):
    def _chart_samples(self, scene, samples: int = 48):
        """Evaluate every GT chart with the production NURBS evaluator."""

        from nurbs_constructor_benchmark.ground_truth import gt_nurbs_charts
        from osn_gs.surface.torch_nurbs import TorchNURBSSurface

        lin = torch.linspace(0.0, 1.0, samples)
        u, v = torch.meshgrid(lin, lin, indexing="ij")
        uv = torch.stack([u.reshape(-1), v.reshape(-1)], dim=1)
        per_chart = []
        for grid, degree_u, degree_v, kind in gt_nurbs_charts(scene):
            surface = TorchNURBSSurface(
                control_grid=grid,
                weights=torch.ones(grid.shape[0], grid.shape[1]),
                degree_u=degree_u,
                degree_v=degree_v,
            )
            per_chart.append((surface.evaluate(uv), kind))
        return per_chart

    def test_charts_lie_on_surface_and_inside_predicate(self):
        from nurbs_constructor_benchmark.scenes import make_scene

        for name in SCENES:
            scene = make_scene(name, 100)
            for points, kind in self._chart_samples(scene):
                residual, _ = scene.oracle(points)
                self.assertLess(
                    float(residual.abs().max()), 0.02,
                    f"{name}/{kind}: chart leaves the true surface",
                )
                inside = scene.support_predicate(points[:, :2]).float().mean()
                self.assertGreater(
                    float(inside), 0.95,
                    f"{name}/{kind}: chart spills outside the support predicate",
                )

    def test_charts_cover_the_support(self):
        from nurbs_constructor_benchmark.scenes import make_scene
        from nurbs_constructor_benchmark.support_domains import mask_on_grid

        resolution = 96
        for name in SCENES:
            scene = make_scene(name, 100)
            gt = mask_on_grid(scene.support_predicate, resolution)
            covered = torch.zeros_like(gt)
            for points, _ in self._chart_samples(scene, samples=192):
                cells = ((points[:, :2] + 1.0) * 0.5 * resolution).long().clamp(0, resolution - 1)
                covered[cells[:, 0], cells[:, 1]] = True
            coverage = float((covered & gt).sum()) / max(1, int(gt.sum()))
            self.assertGreater(coverage, 0.90, f"{name}: charts cover only {coverage:.2f} of GT support")

    def test_payload_is_conformal_without_trim_masks(self):
        from nurbs_constructor_benchmark.ground_truth import gt_nurbs_payload
        from nurbs_constructor_benchmark.scenes import make_scene

        for name in ("planar_hole", "crescent", "u_shape", "triangle", "plane"):
            payload = gt_nurbs_payload(make_scene(name, 100))
            self.assertEqual(payload["metadata"]["parameterization"], "boundary_conformal")
            self.assertIsNone(payload["uv_support"])
            for patch in payload["patches"]:
                self.assertIsNone(patch["uv_support"], f"{name}: GT patch carries a trim mask")

    def test_annulus_hole_is_a_chart_boundary(self):
        """The hole must come from the parameterization: no sample lands in it."""

        from nurbs_constructor_benchmark.scenes import make_scene

        scene = make_scene("planar_hole", 100)
        for points, _ in self._chart_samples(scene, samples=96):
            radii = points[:, :2].norm(dim=1)
            self.assertGreater(float(radii.min()), 0.30)
            self.assertLess(float(radii.max()), 0.91)


if __name__ == "__main__":
    unittest.main()
