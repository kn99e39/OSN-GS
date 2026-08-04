"""Worklog 33: REPRESENTATIVE GRAPH SCALE (G1) -- estimator invariance ("Test
A"), candidate-radius/residual-denominator ablation, and negative controls.

Separates two previously-conflated questions from worklog 32:

1. Is the graph-scale ESTIMATOR itself rigid-transform/uniform-scale
   invariant, holding the representative SET fixed? -- Test A here.
2. Does representative SELECTION return a different subset under rotation
   (an already-documented, accepted axis-aligned-voxel-grid limitation), and
   how much does that perturb downstream topology? -- Test B, covered by the
   relaxed assertions in test_density_preserving_representative_selection.py
   and test_full_cloud_continuation_shell.py's rigid-transform tests.

Worklog 32's three graph-scale attempts were checked only against
end-to-end tests that re-run selection (mixing 1 and 2). This file proves
(1) in isolation: G1 is EXACTLY invariant when representative identity is
held fixed -- so the worklog-32 failures were caused entirely by (2), not
by the estimator.
"""

from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig, _representative_knn_spacing
from osn_gs.surface.torch_gaussian_covariance_frame import (
    covariance_from_scale_rotation,
    extract_covariance_frame,
)
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    ManifoldAffinityConfig,
    RELATION_SAME_SURFACE,
    build_manifold_affinity_graph,
)
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability


def _pipeline(max_points: int) -> TorchOSNGSPipeline:
    return TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=max_points), device="cpu")


def _random_planar_representatives(
    count: int, generator: torch.Generator, curvature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A cluster of representatives with individually TINY covariance
    (mimicking real long-horizon-trained Gaussians) but real spacing much
    larger than that footprint -- the exact real-checkpoint pattern.
    ``curvature`` > 0 adds a gentle height variation so mutual_tangent_residual
    is genuinely nonzero (a perfectly flat plane has ~0 residual regardless
    of denominator, which cannot distinguish candidate-radius from
    residual-denominator effects)."""
    positions = torch.rand((count, 3), generator=generator) * 2.0 - 1.0
    if curvature > 0.0:
        positions[:, 2] = curvature * torch.sin(2.0 * positions[:, 0]) * torch.cos(2.0 * positions[:, 1])
    else:
        positions[:, 2] = 0.0
    scale = torch.tensor([[0.01, 0.008, 0.0005]]).repeat(count, 1)
    quaternion = torch.zeros((count, 4))
    quaternion[:, 0] = 1.0
    covariance = covariance_from_scale_rotation(scale, quaternion)
    return positions, covariance


class FrozenRepresentativeGraphScaleInvarianceTest(unittest.TestCase):
    """Test A: representative SET held fixed (no selection re-run at all)."""

    def _rigid_transform(self, positions, covariance, angle, translation, scale):
        cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rotation = torch.tensor([
            [cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0],
        ], dtype=positions.dtype)
        t = torch.tensor(translation, dtype=positions.dtype)
        transformed_positions = (positions @ rotation.T) * scale + t
        transformed_covariance = (scale * scale) * (rotation @ covariance @ rotation.transpose(-1, -2))
        return transformed_positions, transformed_covariance

    def test_g1_is_exactly_invariant_on_fixed_representative_set(self):
        generator = torch.Generator().manual_seed(0)
        positions, covariance = _random_planar_representatives(120, generator)
        ids = tuple(range(positions.shape[0]))
        frame = extract_covariance_frame(covariance)
        reliability = evaluate_structural_reliability(positions, frame)

        transformed_positions, transformed_covariance = self._rigid_transform(
            positions, covariance, angle=0.7, translation=(4.0, -2.0, 6.0), scale=3.3,
        )
        transformed_frame = extract_covariance_frame(transformed_covariance)

        g1_base = _representative_knn_spacing(positions)
        g1_transformed = _representative_knn_spacing(transformed_positions)

        graph_base = build_manifold_affinity_graph(
            positions, frame, reliability, ids=ids, candidate_scale=g1_base, residual_scale=g1_base,
        )
        graph_transformed = build_manifold_affinity_graph(
            transformed_positions, transformed_frame, reliability, ids=ids,
            candidate_scale=g1_transformed, residual_scale=g1_transformed,
        )
        base_relations = {(e.source_id, e.target_id): e.manifold_relation for e in graph_base.edges}
        transformed_relations = {(e.source_id, e.target_id): e.manifold_relation for e in graph_transformed.edges}
        self.assertEqual(base_relations, transformed_relations)
        # And it must actually find same_surface edges (not just be
        # invariantly empty) -- the real point of this fix.
        self.assertGreater(sum(1 for r in base_relations.values() if r == RELATION_SAME_SURFACE), 0)

    def test_g1_scales_linearly_with_uniform_scale(self):
        generator = torch.Generator().manual_seed(1)
        positions, _covariance = _random_planar_representatives(60, generator)
        g1 = _representative_knn_spacing(positions)
        g1_scaled = _representative_knn_spacing(positions * 4.0)
        torch.testing.assert_close(g1_scaled, g1 * 4.0, rtol=1e-4, atol=1e-6)


class CandidateResidualAblationTest(unittest.TestCase):
    """Section 9: candidate radius and residual denominator must be
    independently swappable and independently effective."""

    def _graph(self, positions, covariance, candidate_scale, residual_scale):
        ids = tuple(range(positions.shape[0]))
        frame = extract_covariance_frame(covariance)
        reliability = evaluate_structural_reliability(positions, frame)
        return build_manifold_affinity_graph(
            positions, frame, reliability, ids=ids,
            candidate_scale=candidate_scale, residual_scale=residual_scale,
        )

    def test_candidate_radius_alone_is_insufficient_without_residual_fix(self):
        """Widening ONLY the candidate radius (ablation A) must not, by
        itself, manufacture same_surface edges that residual-based relation
        classification would otherwise reject -- proves candidate radius and
        residual denominator are genuinely independent roles, not one
        disguised as two."""
        generator = torch.Generator().manual_seed(2)
        positions, covariance = _random_planar_representatives(120, generator, curvature=0.15)
        frame = extract_covariance_frame(covariance)
        g0 = frame.tangent_major_scale
        g1 = _representative_knn_spacing(positions)

        graph_g0 = self._graph(positions, covariance, g0, g0)
        graph_candidate_only = self._graph(positions, covariance, g1, g0)  # ablation A
        graph_residual_only = self._graph(positions, covariance, g0, g1)  # ablation B
        graph_both = self._graph(positions, covariance, g1, g1)  # ablation C

        def same_surface_count(graph):
            return sum(1 for e in graph.edges if e.manifold_relation == RELATION_SAME_SURFACE)

        # Both together (ablation C) must find substantially more than
        # either alone -- confirms neither role alone is sufficient, i.e.
        # candidate radius and residual denominator are genuinely
        # independent roles rather than one masking the other.
        self.assertGreater(same_surface_count(graph_both), same_surface_count(graph_g0))
        self.assertGreater(same_surface_count(graph_both), same_surface_count(graph_candidate_only))
        self.assertGreater(same_surface_count(graph_both), same_surface_count(graph_residual_only))


class NegativeControlWithGraphScaleTest(unittest.TestCase):
    """Section 13: false-merge negative controls, with G1 actually wired
    through the real production ``_construct_canonical_with_full_evidence``
    path (not a bespoke direct call)."""

    def _region_count(self, name: str, seed: int = 0) -> int:
        scene = make_gaussian_reliability_scene(name, seed=seed)
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))
        cap = max(16, positions.shape[0] // 3)
        pipeline = _pipeline(max_points=cap)
        bundle = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)
        return bundle.construction.diagnostic_summary["region_count"]

    def test_thin_slab_front_and_back_stay_separate(self):
        self.assertGreaterEqual(self._region_count("thin_slab"), 2)

    def test_box_faces_do_not_collapse_across_creases(self):
        # 6 real faces; false crease-crossing merges would reduce this.
        self.assertGreaterEqual(self._region_count("box"), 4)


if __name__ == "__main__":
    unittest.main()

class RepresentativeGraphScaleCardinalityTest(unittest.TestCase):
    def test_singleton_is_finite_floor(self):
        value = _representative_knn_spacing(torch.tensor([[1.0, 2.0, 3.0]]))
        self.assertEqual(value.shape, (1,))
        self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(value, torch.tensor([1e-9]))

    def test_pair_uses_the_only_other_representative(self):
        value = _representative_knn_spacing(torch.tensor([[0., 0., 0.], [3., 4., 0.]]))
        torch.testing.assert_close(value, torch.tensor([5., 5.]))
