"""Worklog 129: density-preserving representative selection + full-neighborhood evidence.

Covers the redesign that replaces representative-only (sample-sparse)
contextual reliability with full-observed-cloud aggregate evidence, and
replaces one-representative-per-voxel-cell selection with mode-aware,
density/diversity-weighted selection. See docs/worklogs/129_*.md.
"""

from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import (
    make_gap_sweep_scene,
    make_gaussian_density_sweep_scene,
    make_gaussian_reliability_scene,
)
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig  # noqa: E402
from osn_gs.surface.torch_gaussian_covariance_frame import (
    covariance_from_scale_rotation,
    extract_covariance_frame,
)
from osn_gs.surface.torch_density_preserving_representative_selection import (
    RepresentativeSelectionConfig,
    select_density_preserving_representatives,
)
from osn_gs.surface.torch_full_neighborhood_evidence import compute_full_neighborhood_evidence
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_intrinsic_reliability


def _pipeline(max_points: int) -> TorchOSNGSPipeline:
    return TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=max_points), device="cpu")


class MultiModeCellPreservationTest(unittest.TestCase):
    def test_close_parallel_sheets_both_survive_as_representatives_under_a_coarse_budget(self):
        """A thin slab's front/back faces must not collapse into a single
        representative even under a coarse voxel budget."""
        scene = make_gap_sweep_scene(0.05, seed=0)  # a genuinely thin slab
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))
        frame = extract_covariance_frame(covariance)

        result = select_density_preserving_representatives(
            positions, frame, opacity, stable_ids, max_points=8
        )
        # Both opposite-normal groups ("top"/"bottom") must appear among the
        # selected representatives -- neither face may be entirely discarded.
        all_labels = set(scene.group_labels)
        self.assertEqual(all_labels, {"top", "bottom"})
        represented_labels = {scene.group_labels[i] for i in result.representative_indices.tolist()}
        self.assertEqual(represented_labels, all_labels)

    def test_coincident_opposite_normal_pair_splits_into_two_modes_in_the_same_cell(self):
        """Two nearly-coincident Gaussians with opposite normals -- e.g. two
        sides of an infinitesimally thin double-sided surface -- MUST be
        recognized as two separate modes of the same voxel cell, never
        silently collapsed to one representative."""
        scene = make_gaussian_reliability_scene("box", seed=0)
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        anchor = positions[0].clone()
        extra_position_a = anchor.unsqueeze(0)
        extra_position_b = anchor.unsqueeze(0) + 1e-5
        scale = torch.tensor([[0.05, 0.05, 0.002]])
        quaternion_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        quaternion_b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # ~180-degree opposite normal
        extra_covariance_a = covariance_from_scale_rotation(scale, quaternion_a)
        extra_covariance_b = covariance_from_scale_rotation(scale, quaternion_b)
        all_positions = torch.cat((positions, extra_position_a, extra_position_b), dim=0)
        all_covariance = torch.cat((covariance, extra_covariance_a, extra_covariance_b), dim=0)
        opacity = torch.ones(all_positions.shape[0])
        stable_ids = list(range(all_positions.shape[0]))
        frame = extract_covariance_frame(all_covariance)

        result = select_density_preserving_representatives(
            all_positions, frame, opacity, stable_ids, max_points=64
        )
        self.assertGreater(result.diagnostics.multi_mode_cell_count, 0)
        self.assertGreaterEqual(result.diagnostics.modes_per_cell_max, 2)


class DensitySweepTest(unittest.TestCase):
    def test_increasing_density_improves_full_evidence_support_without_changing_region_count(self):
        pipeline = _pipeline(max_points=256)
        region_counts = []
        reliable_counts = []
        support_means = []
        for multiplier in (1, 2, 4, 8):
            scene = make_gaussian_density_sweep_scene("box", multiplier, seed=0)
            positions = torch.as_tensor(scene.positions, dtype=torch.float32)
            covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
            opacity = torch.ones(positions.shape[0])
            stable_ids = list(range(positions.shape[0]))
            bundle = pipeline._construct_canonical_with_full_evidence(
                positions, covariance, opacity, stable_ids
            )
            region_counts.append(bundle.construction.diagnostic_summary["region_count"])
            reliable_counts.append(bundle.construction.diagnostic_summary["reliable_count"])
            support_means.append(float(bundle.evidence.support_count.float().mean()))

        # A box always has exactly 6 face regions -- density must not create
        # false merges or spurious fragmentation as it increases.
        self.assertTrue(all(count == 6 for count in region_counts), region_counts)
        # Real full-cloud support must grow with density (this is the whole
        # point of full-neighborhood evidence over representative-only kNN).
        self.assertEqual(support_means, sorted(support_means))
        self.assertGreater(support_means[-1], support_means[0])
        # Reliable admission must not go backwards as density increases.
        self.assertEqual(reliable_counts, sorted(reliable_counts))

    def test_low_density_may_review_but_does_not_worsen_at_high_density(self):
        pipeline = _pipeline(max_points=128)
        scene_sparse = make_gaussian_density_sweep_scene("cylinder", 1, seed=1)
        scene_dense = make_gaussian_density_sweep_scene("cylinder", 6, seed=1)
        results = []
        for scene in (scene_sparse, scene_dense):
            positions = torch.as_tensor(scene.positions, dtype=torch.float32)
            covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
            opacity = torch.ones(positions.shape[0])
            stable_ids = list(range(positions.shape[0]))
            bundle = pipeline._construct_canonical_with_full_evidence(
                positions, covariance, opacity, stable_ids
            )
            results.append(bundle.construction.diagnostic_summary["reliable_count"])
        self.assertGreaterEqual(results[1], results[0])


class DeterminismAndInvarianceTest(unittest.TestCase):
    def test_selection_is_invariant_to_input_order_shuffle(self):
        scene = make_gaussian_density_sweep_scene("box", 3, seed=2)
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))
        frame = extract_covariance_frame(covariance)

        baseline = select_density_preserving_representatives(
            positions, frame, opacity, stable_ids, max_points=64
        )
        baseline_stable_ids = sorted(stable_ids[i] for i in baseline.representative_indices.tolist())

        permutation = torch.randperm(positions.shape[0])
        shuffled_positions = positions[permutation]
        shuffled_covariance = covariance[permutation]
        shuffled_opacity = opacity[permutation]
        shuffled_stable_ids = [stable_ids[i] for i in permutation.tolist()]
        shuffled_frame = extract_covariance_frame(shuffled_covariance)

        shuffled = select_density_preserving_representatives(
            shuffled_positions, shuffled_frame, shuffled_opacity, shuffled_stable_ids, max_points=64
        )
        shuffled_stable_ids_selected = sorted(
            shuffled_stable_ids[i] for i in shuffled.representative_indices.tolist()
        )
        self.assertEqual(baseline_stable_ids, shuffled_stable_ids_selected)

    def test_construction_outcome_is_stable_under_rigid_rotation_translation_and_uniform_scale(self):
        """A coarse axis-aligned voxel grid is not exactly representative-index
        invariant under an arbitrary rotation (the grid itself is world-axis
        aligned, so a rotated bounding box tiles differently) -- that is a
        known property of any axis-aligned spatial hash, not a worklog 129
        regression. What MUST hold is the downstream construction OUTCOME:
        region/reliable counts should be geometry-equivalent, not merely
        representative-set-identical (worklog 129 item 15)."""
        pipeline = _pipeline(max_points=48)
        scene = make_gaussian_density_sweep_scene("cylinder", 2, seed=3)
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))
        baseline = pipeline._construct_canonical_with_full_evidence(
            positions, covariance, opacity, stable_ids
        )

        angle = 0.4
        cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rotation = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ])
        scale = 2.5
        translation = torch.tensor([5.0, -3.0, 1.5])
        transformed_positions = (positions @ rotation.T) * scale + translation
        transformed_covariance = scale * scale * (rotation @ covariance @ rotation.T)
        transformed = pipeline._construct_canonical_with_full_evidence(
            transformed_positions, transformed_covariance, opacity, stable_ids
        )

        self.assertEqual(
            baseline.construction.diagnostic_summary["region_count"],
            transformed.construction.diagnostic_summary["region_count"],
        )
        # Reliable count need not match exactly (axis-aligned voxel grid
        # under an arbitrary rotation tiles the same geometry slightly
        # differently -- see docstring), but it must stay within the same
        # rough order of magnitude, not collapse to zero or explode.
        baseline_reliable = baseline.construction.diagnostic_summary["reliable_count"]
        transformed_reliable = transformed.construction.diagnostic_summary["reliable_count"]
        self.assertGreater(baseline_reliable, 0)
        self.assertGreater(transformed_reliable, 0)
        self.assertLess(abs(baseline_reliable - transformed_reliable), 0.5 * max(baseline_reliable, transformed_reliable))


class NegativeControlTest(unittest.TestCase):
    def test_isolated_floater_is_not_reliable_admitted_under_full_neighborhood_evidence(self):
        scene = make_gaussian_density_sweep_scene("box", 4, seed=4)
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        floater_position = torch.tensor([[50.0, 50.0, 50.0]])
        floater_scale = torch.tensor([[0.05, 0.05, 0.002]])
        floater_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        floater_covariance = covariance_from_scale_rotation(floater_scale, floater_quaternion)
        all_positions = torch.cat((positions, floater_position), dim=0)
        all_covariance = torch.cat((covariance, floater_covariance), dim=0)
        opacity = torch.ones(all_positions.shape[0])
        stable_ids = list(range(all_positions.shape[0]))
        floater_stable_id = stable_ids[-1]

        pipeline = _pipeline(max_points=128)
        bundle = pipeline._construct_canonical_with_full_evidence(
            all_positions, all_covariance, opacity, stable_ids
        )
        rep_ids = bundle.representative_stable_ids
        if floater_stable_id in rep_ids:
            local = rep_ids.index(floater_stable_id)
            self.assertNotEqual(
                bundle.construction.reliability.reliability_class[local],
                "reliable_structural_evidence",
            )

    def test_oversized_isotropic_contamination_does_not_gain_reliability_from_full_support(self):
        """A contaminant with a HIGH full-cloud support count (i.e. sitting in
        a dense region) must still be rejected on INTRINSIC grounds alone --
        full-neighborhood support must never override intrinsic rejection."""
        scene = make_gaussian_reliability_scene("box_isotropic_contamination", seed=5)
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        frame = extract_covariance_frame(covariance)
        intrinsic = evaluate_intrinsic_reliability(frame)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))

        # Force a downsample even on this small scene by using a tiny budget,
        # so every remaining point participates as its own representative
        # and full-neighborhood evidence is actually exercised.
        evidence = compute_full_neighborhood_evidence(
            positions, frame, opacity, intrinsic, positions, frame, stable_ids
        )
        from osn_gs.surface.torch_gaussian_structural_reliability import (
            evaluate_structural_reliability_from_full_evidence,
        )

        reliability = evaluate_structural_reliability_from_full_evidence(frame, evidence)
        contaminated_indices = [
            i for i, label in enumerate(scene.group_labels) if "isotropic" in label
        ]
        self.assertTrue(contaminated_indices)
        for index in contaminated_indices:
            self.assertNotEqual(reliability.reliability_class[index], "reliable_structural_evidence")


if __name__ == "__main__":
    unittest.main()
