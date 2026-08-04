"""Occluded Chart Ownership Foundation tests.

Covers the canonical ownership contract (osn_gs/gaussian/torch_surface_ownership.py):
identity, behavioral isolation of visible-patch read sites from occluded-
chart-owned uncertain Gaussians, ADC ownership transport, and the one-way
dependency invariant. Append-transaction-specific ownership tests (rollback,
duplicate rejection, receipt/sidecar match) live in
tests/test_uncertain_gaussian_append_adapter.py alongside the rest of that
transaction contract.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from unittest import mock

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.gaussian.torch_density_control import (
    TorchDensityControlConfig,
    apply_adaptive_density_control,
)
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.gaussian.torch_surface_ownership import (
    OCCLUDED_CHART_NAMESPACE_BASE,
    SURFACE_OWNER_OCCLUDED_CHART,
    SURFACE_OWNER_UNASSIGNED,
    SURFACE_OWNER_VISIBLE_PATCH,
    UNASSIGNED_OWNER_ID,
    OccludedChartOwnerCollisionError,
    commit_occluded_owner_binding,
    derive_default_ownership,
    is_visible_patch_owned,
    project_occluded_chart_owner_id,
    reject_visible_patch_id_in_occluded_namespace,
    rollback_occluded_owner_binding,
    validate_occluded_owner_binding_read_only,
    validate_surface_ownership_consistency,
)
from osn_gs.losses.torch_losses import nurbs_surface_loss


def _state(count: int = 81, seed: int = 7):
    torch.manual_seed(seed)
    axis = torch.linspace(-0.48, 0.48, 9)
    points = torch.stack(
        [torch.tensor([x, y, 0.0]) for x in axis for y in axis]
    )
    colors = torch.rand(len(points), 3)
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
    return pipeline, pipeline.initialize(points, colors)

def _append_occluded_chart_gaussian(model, *, source_chart_id: str = "chart-x", count: int = 2, patch_id_for_uv: int = 0):
    """Directly append `count` occluded-chart-owned uncertain Gaussians via the
    model-only API, bypassing the full proposal/candidate pipeline -- these
    tests are about whether PIPELINE/ADC/loss code respects ownership, not
    about re-testing adapter correctness (covered elsewhere)."""

    owner_id = project_occluded_chart_owner_id(source_chart_id)
    xyz = torch.rand(count, 3)
    features_dc = torch.zeros((count, 1, 3))
    features_rest = torch.zeros((count, model._features_rest.shape[1], 3))
    opacity = torch.zeros((count, 1))
    scaling = torch.zeros((count, 3))
    rotation = torch.zeros((count, 4))
    rotation[:, 0] = 1.0
    uncertain_confidence = torch.zeros((count, 1))
    uncertain_mask = torch.ones((count,), dtype=torch.bool)
    surface_uv = torch.rand((count, 2))
    cluster_ids = torch.full((count,), int(patch_id_for_uv), dtype=torch.long)  # compat-only, deliberately in-range
    owner_kind = torch.full((count,), SURFACE_OWNER_OCCLUDED_CHART, dtype=torch.long)
    owner_id_tensor = torch.full((count,), owner_id, dtype=torch.long)
    model.append_gaussians_model_only(
        xyz, features_dc, features_rest, opacity, scaling, rotation, uncertain_confidence,
        uncertain_mask, surface_uv, cluster_ids, owner_kind, owner_id_tensor,
    )
    return owner_id


class OwnershipIdentityTest(unittest.TestCase):
    def test_kind_constants_are_distinct(self):
        self.assertEqual(len({SURFACE_OWNER_UNASSIGNED, SURFACE_OWNER_VISIBLE_PATCH, SURFACE_OWNER_OCCLUDED_CHART}), 3)

    def test_same_chart_same_identity(self):
        a = project_occluded_chart_owner_id("chart-a")
        b = project_occluded_chart_owner_id("chart-a")
        self.assertEqual(a, b)

    def test_different_chart_different_identity(self):
        a = project_occluded_chart_owner_id("chart-a")
        b = project_occluded_chart_owner_id("chart-b")
        self.assertNotEqual(a, b)

    def test_different_schema_version_different_identity(self):
        a = project_occluded_chart_owner_id("chart-a", schema_version=1)
        b = project_occluded_chart_owner_id("chart-a", schema_version=2)
        self.assertNotEqual(a, b)

    def test_no_collision_with_visible_patch_namespace(self):
        owner_id = project_occluded_chart_owner_id("chart-a")
        # Any realistic visible patch_id (small, zero-based) is far below this floor.
        self.assertGreaterEqual(owner_id, OCCLUDED_CHART_NAMESPACE_BASE)
        for plausible_patch_id in range(0, 10_000):
            self.assertNotEqual(owner_id, plausible_patch_id)

    def test_empty_chart_id_rejected(self):
        with self.assertRaises(ValueError):
            project_occluded_chart_owner_id("")

    def test_identity_independent_of_python_hash_seed(self):
        # hashlib.sha256 (unlike the builtin salted hash()) must be identical
        # across process restarts with different PYTHONHASHSEED values.
        script = (
            "from osn_gs.gaussian.torch_surface_ownership import project_occluded_chart_owner_id;"
            "print(project_occluded_chart_owner_id('chart-a'))"
        )
        results = []
        for seed in ("0", "1", "12345"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            proc = subprocess.run(
                [sys.executable, "-c", script],
                env=env, cwd=os.getcwd(),
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            results.append(proc.stdout.strip())
        self.assertEqual(len(set(results)), 1, f"owner id varied across hash seeds: {results}")

    def test_is_visible_patch_owned_helper(self):
        kinds = torch.tensor([SURFACE_OWNER_VISIBLE_PATCH, SURFACE_OWNER_OCCLUDED_CHART, SURFACE_OWNER_UNASSIGNED])
        mask = is_visible_patch_owned(kinds)
        self.assertEqual(mask.tolist(), [True, False, False])


class BehavioralIsolationTest(unittest.TestCase):
    """Section 5: visible-patch behavioral read sites must ignore
    occluded-chart-owned uncertain Gaussians."""

    def test_support_mask_excludes_occluded_chart_owner(self):
        pipeline, state = _state()
        model = state.model
        before_masks = [
            None if p.uv_support_mask is None else p.uv_support_mask.clone()
            for p in state.surface_patches
        ]
        pipeline._assign_uv_support_masks(model, state.surface_patches)
        baseline_masks = [p.uv_support_mask.clone() for p in state.surface_patches]

        _append_occluded_chart_gaussian(model, patch_id_for_uv=0)
        pipeline._assign_uv_support_masks(model, state.surface_patches)
        after_masks = [p.uv_support_mask for p in state.surface_patches]

        for baseline, after in zip(baseline_masks, after_masks):
            torch.testing.assert_close(baseline, after)

    def test_support_mask_leak_would_have_been_visible_without_the_gate(self):
        # Sanity check that the fixture is actually capable of exposing a leak
        # (i.e. this isn't a false-negative test): reading cluster_ids without
        # the ownership gate DOES pick up the occluded-chart-owned row.
        pipeline, state = _state()
        model = state.model
        _append_occluded_chart_gaussian(model, patch_id_for_uv=0)
        cluster_ids = model.cluster_ids.detach()
        naive_assigned = cluster_ids == 0
        gated_assigned = is_visible_patch_owned(model.surface_owner_kind.detach()) & (cluster_ids == 0)
        self.assertGreater(int(naive_assigned.sum()), int(gated_assigned.sum()))

    def test_maintain_surface_from_certain_excludes_occluded_chart_owner(self):
        pipeline, state = _state()
        model = state.model
        uv_before = model.surface_uv[~model.is_uncertain].clone()
        _append_occluded_chart_gaussian(model, patch_id_for_uv=0)
        pipeline.maintain_surface_from_certain(state)

        certain_mask = ~model.is_uncertain
        # Certain rows are the original ones (occluded rows are appended after
        # and are uncertain), so this slice is exactly the pre-existing set.
        uv_after = model.surface_uv[certain_mask][: uv_before.shape[0]]
        torch.testing.assert_close(uv_before, uv_after)
        # The occluded-chart-owned rows must still carry their real ownership.
        owned_kind = model.surface_owner_kind[model.is_uncertain]
        self.assertTrue(bool((owned_kind == SURFACE_OWNER_OCCLUDED_CHART).all()))

    def test_nurbs_surface_loss_excludes_occluded_chart_owner(self):
        pipeline, state = _state()
        model = state.model
        state.iteration = 1
        baseline_loss = nurbs_surface_loss(state, weight=0.01, max_patches=0)
        _append_occluded_chart_gaussian(model, patch_id_for_uv=0)
        after_loss = nurbs_surface_loss(state, weight=0.01, max_patches=0)
        torch.testing.assert_close(baseline_loss, after_loss)

    def test_visible_only_results_unchanged_before_after_append(self):
        """Composite regression: adding an occluded-chart-owned Gaussian must
        not change ANY visible-only observable used above, all at once."""

        pipeline, state = _state()
        model = state.model
        pipeline._assign_uv_support_masks(model, state.surface_patches)
        masks_before = [p.uv_support_mask.clone() for p in state.surface_patches]
        state.iteration = 1
        loss_before = nurbs_surface_loss(state, weight=0.01, max_patches=0)

        _append_occluded_chart_gaussian(model, patch_id_for_uv=0)
        pipeline._assign_uv_support_masks(model, state.surface_patches)
        masks_after = [p.uv_support_mask for p in state.surface_patches]
        loss_after = nurbs_surface_loss(state, weight=0.01, max_patches=0)

        for before, after in zip(masks_before, masks_after):
            torch.testing.assert_close(before, after)
        torch.testing.assert_close(loss_before, loss_after)


class ADCOwnershipTransportTest(unittest.TestCase):
    """Section 5 (ADC): ownership must transport through clone/split/prune
    without any policy change -- pure passthrough."""

    def _grown_state(self):
        pipeline, state = _state(count=64)
        model = state.model
        model.xyz_gradient_accum[:] = 1.0
        model.denom[:] = 1.0
        return pipeline, state

    def test_clone_preserves_owner_kind_and_id(self):
        pipeline, state = self._grown_state()
        model = state.model
        config = TorchDensityControlConfig(
            densify_from_iter=0, densify_until_iter=1000, densification_interval=1,
            densify_grad_threshold=0.0, percent_dense=1.0,  # force clone, not split
        )
        before_count = len(model)
        report = apply_adaptive_density_control(model, config, scene_extent=10.0, iteration=1)
        self.assertGreater(report.cloned, 0)
        self.assertGreater(len(model), before_count)
        # All rows (original + cloned) must still be VISIBLE_PATCH-owned --
        # ADC only ever touches certain Gaussians.
        self.assertTrue(bool((model.surface_owner_kind == SURFACE_OWNER_VISIBLE_PATCH).all()))

    def test_split_preserves_owner_kind_and_id(self):
        pipeline, state = self._grown_state()
        model = state.model
        config = TorchDensityControlConfig(
            densify_from_iter=0, densify_until_iter=1000, densification_interval=1,
            densify_grad_threshold=0.0, percent_dense=0.0,  # force split, not clone
        )
        report = apply_adaptive_density_control(model, config, scene_extent=10.0, iteration=1)
        self.assertGreater(report.split, 0)
        self.assertTrue(bool((model.surface_owner_kind == SURFACE_OWNER_VISIBLE_PATCH).all()))

    def test_prune_preserves_row_alignment_for_occluded_chart_owner(self):
        pipeline, state = self._grown_state()
        model = state.model
        owner_id = _append_occluded_chart_gaussian(model, patch_id_for_uv=0, count=1)
        config = TorchDensityControlConfig(
            densify_from_iter=0, densify_until_iter=1000, densification_interval=1,
            densify_grad_threshold=1.0,  # no clone/split, opacity-only prune path exercised
            prune_opacity_threshold=1.1,  # prune every certain Gaussian (opacity in (0,1))
        )
        apply_adaptive_density_control(model, config, scene_extent=10.0, iteration=1)
        # Only the occluded-chart-owned row (opacity fixed at sigmoid(0)=0.5 <
        # threshold, but ownership kind must be untouched regardless) should
        # remain distinguishable by ownership.
        self.assertTrue(bool((model.surface_owner_kind == SURFACE_OWNER_OCCLUDED_CHART).any()) or len(model) == 0)

    def test_uncertain_owner_never_converted_to_visible_owner(self):
        pipeline, state = self._grown_state()
        model = state.model
        _append_occluded_chart_gaussian(model, patch_id_for_uv=0, count=2)
        config = TorchDensityControlConfig(
            densify_from_iter=0, densify_until_iter=1000, densification_interval=1,
            densify_grad_threshold=0.0, percent_dense=1.0,
        )
        apply_adaptive_density_control(model, config, scene_extent=10.0, iteration=1)
        occluded_rows = model.surface_owner_kind[model.is_uncertain]
        if int(occluded_rows.numel()) > 0:
            self.assertTrue(bool((occluded_rows == SURFACE_OWNER_OCCLUDED_CHART).all()))


class OneWayDependencyInvariantTest(unittest.TestCase):
    def test_uncertain_gaussian_not_in_certain_maintenance_indices(self):
        pipeline, state = _state()
        model = state.model
        _append_occluded_chart_gaussian(model, patch_id_for_uv=0)
        certain = ~model.is_uncertain
        # Direct structural check mirroring maintain_surface_from_certain's
        # own gate: an occluded-chart-owned row must never appear in `certain`.
        occluded_indices = torch.nonzero(model.surface_owner_kind == SURFACE_OWNER_OCCLUDED_CHART, as_tuple=False).reshape(-1)
        self.assertTrue(bool((~certain[occluded_indices]).all()))

    def test_source_chart_and_provenance_survive_ownership_read_sites(self):
        pipeline, state = _state()
        model = state.model
        owner_id = _append_occluded_chart_gaussian(model, source_chart_id="chart-z", patch_id_for_uv=0)
        pipeline._assign_uv_support_masks(model, state.surface_patches)
        pipeline.maintain_surface_from_certain(state)
        occluded_mask = model.surface_owner_kind == SURFACE_OWNER_OCCLUDED_CHART
        self.assertTrue(bool((model.surface_owner_id[occluded_mask] == owner_id).all()))


class WriteSiteRegressionTest(unittest.TestCase):
    """Section 5 (write-site regression): every cluster_ids write site keeps
    the VISIBLE_PATCH synchronization invariant, and occluded rows are
    correctly exempt from it."""

    def test_initialize_fallback_owner_id_matches_cluster_id(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.initialize(torch.rand(6, 3), torch.rand(6, 3), cluster_ids=torch.tensor([0, 0, 1, 1, 2, 2]))
        self.assertEqual(validate_surface_ownership_consistency(model), ())
        torch.testing.assert_close(model.surface_owner_id, model.cluster_ids)

    def test_adc_clone_split_owner_id_matches_cluster_id(self):
        pipeline, state = _state(count=64)
        model = state.model
        model.xyz_gradient_accum[:] = 1.0
        model.denom[:] = 1.0
        config = TorchDensityControlConfig(
            densify_from_iter=0, densify_until_iter=1000, densification_interval=1,
            densify_grad_threshold=0.0, percent_dense=1.0,
        )
        apply_adaptive_density_control(model, config, scene_extent=10.0, iteration=1)
        self.assertEqual(validate_surface_ownership_consistency(model), ())

    def test_prune_preserves_row_alignment_and_validator(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.initialize(torch.rand(6, 3), torch.rand(6, 3), cluster_ids=torch.tensor([0, 0, 1, 1, 2, 2]))
        model.prune(torch.tensor([True, False, True, False, True, False]))
        self.assertEqual(len(model), 3)
        self.assertEqual(validate_surface_ownership_consistency(model), ())
        torch.testing.assert_close(model.surface_owner_id, model.cluster_ids)

    def test_appended_occluded_row_owner_id_may_differ_from_cluster_id(self):
        pipeline, state = _state()
        model = state.model
        _append_occluded_chart_gaussian(model, source_chart_id="chart-diff", patch_id_for_uv=0)
        # cluster_ids compatibility value equals patch 0, but the real owner
        # id is the large synthetic value -- the validator must NOT flag this
        # for an OCCLUDED_CHART-owned row.
        occluded_mask = model.surface_owner_kind == SURFACE_OWNER_OCCLUDED_CHART
        self.assertTrue(bool((model.surface_owner_id[occluded_mask] != model.cluster_ids[occluded_mask]).all()))
        self.assertEqual(validate_surface_ownership_consistency(model), ())

    def test_snapshot_restore_preserves_validator_result(self):
        pipeline, state = _state()
        model = state.model
        _append_occluded_chart_gaussian(model, source_chart_id="chart-snap", patch_id_for_uv=0)
        self.assertEqual(validate_surface_ownership_consistency(model), ())
        snap = model.snapshot_state()
        model.surface_owner_id = torch.zeros_like(model.surface_owner_id)
        model.restore_state(snap)
        self.assertEqual(validate_surface_ownership_consistency(model), ())


class UnassignedOwnershipTest(unittest.TestCase):
    """Ownership Foundation Gate final-contract round: canonical UNASSIGNED
    semantics for negative ``cluster_ids`` (the canonical unassigned sentinel
    already leaves on Gaussians in inactive/skipped voxel leaves)."""

    def test_negative_cluster_id_initializes_to_unassigned(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.initialize(torch.rand(4, 3), torch.rand(4, 3), cluster_ids=torch.tensor([-1, 0, -1, 1]))
        expected_kind = torch.tensor(
            [SURFACE_OWNER_UNASSIGNED, SURFACE_OWNER_VISIBLE_PATCH, SURFACE_OWNER_UNASSIGNED, SURFACE_OWNER_VISIBLE_PATCH]
        )
        torch.testing.assert_close(model.surface_owner_kind, expected_kind)
        expected_owner_id = torch.tensor([UNASSIGNED_OWNER_ID, 0, UNASSIGNED_OWNER_ID, 1])
        torch.testing.assert_close(model.surface_owner_id, expected_owner_id)
        self.assertEqual(validate_surface_ownership_consistency(model), ())

    def test_explicit_patch_zero_assignment_is_visible_patch_owner_zero(self):
        # Patch id 0 is a legitimate real assignment, not a "no patch" marker
        # -- must NOT be confused with the -1 UNASSIGNED sentinel.
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.initialize(torch.rand(3, 3), torch.rand(3, 3), cluster_ids=torch.tensor([0, 0, 0]))
        self.assertTrue(bool((model.surface_owner_kind == SURFACE_OWNER_VISIBLE_PATCH).all()))
        self.assertTrue(bool((model.surface_owner_id == 0).all()))
        self.assertEqual(validate_surface_ownership_consistency(model), ())

    def test_replace_tensors_fallback_derives_same_default_as_initialize(self):
        # The checkpoint-load path (no ownership tensors provided) must reach
        # the identical migration default `initialize()` uses.
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.initialize(torch.rand(4, 3), torch.rand(4, 3), cluster_ids=torch.tensor([-1, 0, -1, 1]))
        model.replace_tensors(
            xyz=model._xyz.detach(), features_dc=model._features_dc.detach(),
            features_rest=model._features_rest.detach(), opacity=model._opacity.detach(),
            scaling=model._scaling.detach(), rotation=model._rotation.detach(),
            uncertain_confidence=model._uncertain_confidence.detach(), uncertain_mask=model.is_uncertain,
            surface_uv=model.surface_uv, cluster_ids=model.cluster_ids,
        )
        expected_kind = torch.tensor(
            [SURFACE_OWNER_UNASSIGNED, SURFACE_OWNER_VISIBLE_PATCH, SURFACE_OWNER_UNASSIGNED, SURFACE_OWNER_VISIBLE_PATCH]
        )
        torch.testing.assert_close(model.surface_owner_kind, expected_kind)
        self.assertEqual(validate_surface_ownership_consistency(model), ())

    def test_derive_default_ownership_matches_model_default(self):
        cluster_ids = torch.tensor([-3, -1, 0, 5])
        kind, owner_id = derive_default_ownership(cluster_ids)
        torch.testing.assert_close(
            kind,
            torch.tensor([SURFACE_OWNER_UNASSIGNED, SURFACE_OWNER_UNASSIGNED, SURFACE_OWNER_VISIBLE_PATCH, SURFACE_OWNER_VISIBLE_PATCH]),
        )
        torch.testing.assert_close(owner_id, torch.tensor([UNASSIGNED_OWNER_ID, UNASSIGNED_OWNER_ID, 0, 5]))

    def test_validator_rejects_visible_patch_with_negative_owner_id(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.initialize(torch.rand(2, 3), torch.rand(2, 3), cluster_ids=torch.tensor([0, 1]))
        model.surface_owner_id[0] = UNASSIGNED_OWNER_ID  # corrupt: VISIBLE_PATCH kind, UNASSIGNED sentinel id
        violations = validate_surface_ownership_consistency(model)
        self.assertTrue(any("negative surface_owner_id" in v for v in violations))

    def test_validator_rejects_unassigned_row_with_non_sentinel_owner_id(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.initialize(torch.rand(2, 3), torch.rand(2, 3), cluster_ids=torch.tensor([-1, 0]))
        model.surface_owner_id[0] = 5  # corrupt: UNASSIGNED kind but non-sentinel id
        violations = validate_surface_ownership_consistency(model)
        self.assertTrue(any("UNASSIGNED row" in v for v in violations))

    def test_negative_cluster_row_excluded_from_patch_zero_support_mask_catch_all(self):
        # The `_assign_uv_support_masks` patch-0 catch-all intentionally
        # absorbs negative-cluster rows, but ONLY if they are visible-patch
        # owned -- a genuinely UNASSIGNED row (the real pairing
        # `derive_default_ownership` now produces for cluster_id < 0) must
        # never contribute, even though its raw cluster_ids value still
        # satisfies the catch-all's naive `cluster_ids < 0` condition.
        pipeline, state = _state()
        model = state.model
        model.cluster_ids[0] = -1
        model.surface_owner_kind[0] = SURFACE_OWNER_UNASSIGNED
        model.surface_owner_id[0] = UNASSIGNED_OWNER_ID
        pipeline._assign_uv_support_masks(model, state.surface_patches)
        naive_patch_zero = (model.cluster_ids < 0) | (model.cluster_ids == 0)
        gated_patch_zero = is_visible_patch_owned(model.surface_owner_kind) & naive_patch_zero
        self.assertGreater(int(naive_patch_zero.sum()), int(gated_patch_zero.sum()))
        self.assertFalse(bool(gated_patch_zero[0]))

    def test_negative_cluster_row_never_matches_any_real_patch_id(self):
        # `maintain_surface_from_certain` and
        # `nurbs_surface_loss` all gate membership on `cluster_ids ==
        # patch_id`; -1 structurally never equals any real (>=0) patch id, so
        # a genuinely UNASSIGNED row is excluded from all three without
        # needing an explicit ownership-kind check at those sites.
        pipeline, state = _state()
        model = state.model
        model.cluster_ids[0] = -1
        model.surface_owner_kind[0] = SURFACE_OWNER_UNASSIGNED
        model.surface_owner_id[0] = UNASSIGNED_OWNER_ID
        for patch_id in range(len(state.surface_patches)):
            self.assertFalse(bool((model.cluster_ids == patch_id)[0]))
        state.iteration = 1
        loss = nurbs_surface_loss(state, weight=0.01, max_patches=0)
        self.assertTrue(bool(torch.isfinite(loss)))
        pipeline.maintain_surface_from_certain(state)

        self.assertEqual(int(model.cluster_ids[0]), -1)
        self.assertEqual(validate_surface_ownership_consistency(model), ())


class NamespaceCollisionTest(unittest.TestCase):
    """Section 4: owner-id namespace and collision defense."""

    def test_visible_patch_id_below_namespace_base_is_accepted(self):
        reject_visible_patch_id_in_occluded_namespace(0)
        reject_visible_patch_id_in_occluded_namespace(OCCLUDED_CHART_NAMESPACE_BASE - 1)  # must not raise

    def test_visible_patch_id_at_or_above_namespace_base_is_rejected(self):
        with self.assertRaises(ValueError):
            reject_visible_patch_id_in_occluded_namespace(OCCLUDED_CHART_NAMESPACE_BASE)
        with self.assertRaises(ValueError):
            reject_visible_patch_id_in_occluded_namespace(OCCLUDED_CHART_NAMESPACE_BASE + 1)

    def test_registered_projection_same_chart_repeated_is_fine(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        first, first_preexisted = validate_occluded_owner_binding_read_only(model, "chart-a")
        self.assertFalse(first_preexisted)
        commit_occluded_owner_binding(model, first, "chart-a")
        second, second_preexisted = validate_occluded_owner_binding_read_only(model, "chart-a")
        self.assertTrue(second_preexisted)
        self.assertEqual(first, second)
        self.assertEqual(model.occluded_chart_owner_registry[first], "chart-a")

    def test_registered_projection_different_charts_no_collision(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        a, _ = validate_occluded_owner_binding_read_only(model, "chart-a")
        commit_occluded_owner_binding(model, a, "chart-a")
        b, _ = validate_occluded_owner_binding_read_only(model, "chart-b")
        commit_occluded_owner_binding(model, b, "chart-b")
        self.assertNotEqual(a, b)
        self.assertEqual(model.occluded_chart_owner_registry, {a: "chart-a", b: "chart-b"})

    def test_forced_collision_raises_explicit_error(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        owner_id, _ = validate_occluded_owner_binding_read_only(model, "chart-a")
        commit_occluded_owner_binding(model, owner_id, "chart-a")
        # A real SHA-256 collision cannot be constructed in a test; force the
        # underlying projection to return an ALREADY-BOUND id for a different
        # chart id, and confirm the read-only validator catches it instead of
        # silently aliasing the two charts.
        with mock.patch(
            "osn_gs.gaussian.torch_surface_ownership.project_occluded_chart_owner_id",
            return_value=owner_id,
        ):
            with self.assertRaises(OccludedChartOwnerCollisionError):
                validate_occluded_owner_binding_read_only(model, "chart-b")
        # The registry must not have been corrupted by the failed rebind attempt.
        self.assertEqual(model.occluded_chart_owner_registry[owner_id], "chart-a")

    def test_different_models_have_independent_registries(self):
        model_a = TorchGaussianModel(sh_degree=1, device="cpu")
        model_b = TorchGaussianModel(sh_degree=1, device="cpu")
        owner_id, _ = validate_occluded_owner_binding_read_only(model_a, "chart-a")
        commit_occluded_owner_binding(model_a, owner_id, "chart-a")
        # Same chart id registered under a totally different (colliding, by
        # construction) owner_id on model_b must not see model_a's history.
        with mock.patch(
            "osn_gs.gaussian.torch_surface_ownership.project_occluded_chart_owner_id",
            return_value=owner_id,
        ):
            # Even the SAME chart id would normally be fine, but to prove
            # independence use a different chart id that model_a would reject.
            result, _ = validate_occluded_owner_binding_read_only(model_b, "chart-c")
            commit_occluded_owner_binding(model_b, result, "chart-c")
        self.assertEqual(result, owner_id)
        self.assertEqual(model_b.occluded_chart_owner_registry[owner_id], "chart-c")
        self.assertEqual(model_a.occluded_chart_owner_registry[owner_id], "chart-a")

    def test_rollback_removes_only_newly_created_binding(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        owner_id, preexisted = validate_occluded_owner_binding_read_only(model, "chart-a")
        commit_occluded_owner_binding(model, owner_id, "chart-a")
        self.assertFalse(preexisted)
        rollback_occluded_owner_binding(model, owner_id, was_preexisting=False)
        self.assertNotIn(owner_id, model.occluded_chart_owner_registry)

    def test_rollback_preserves_preexisting_binding(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        owner_id, _ = validate_occluded_owner_binding_read_only(model, "chart-a")
        commit_occluded_owner_binding(model, owner_id, "chart-a")
        # Simulate a second transaction seeing this binding as already present.
        _, preexisted = validate_occluded_owner_binding_read_only(model, "chart-a")
        self.assertTrue(preexisted)
        rollback_occluded_owner_binding(model, owner_id, was_preexisting=preexisted)
        self.assertEqual(model.occluded_chart_owner_registry[owner_id], "chart-a")

    def test_append_adapter_uses_registered_projection(self):
        # End-to-end: two batches from genuinely different charts append to
        # the same model without any collision (the common, expected path).
        from osn_gs.gaussian.torch_uncertain_append_adapter import (
            UncertainAppendInitialization,
            UncertainGaussianAppendAdapter,
        )
        from tests.test_uncertain_gaussian_append_adapter import make_batch

        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        batch_a = make_batch(supporting_patch_ids=(7, 3), batch_id_suffix="-a")
        batch_b = make_batch(supporting_patch_ids=(2, 9), batch_id_suffix="-b")
        batch_b.metadata["source_chart_id"] = "chart-b"  # genuinely different chart, not just a different batch id

        def _init(batch):
            n = len(batch.sample_ids)
            return UncertainAppendInitialization(
                torch.zeros((n, 1, 3)), torch.zeros((n, 3, 3)), torch.zeros((n, 1)), torch.full((n, 1), -1.0)
            )

        adapter.append(batch_a, model, _init(batch_a))
        adapter.append(batch_b, model, _init(batch_b))
        self.assertEqual(len(model.occluded_chart_owner_registry), 2)
        self.assertEqual(validate_surface_ownership_consistency(model), ())


if __name__ == "__main__":
    unittest.main()
