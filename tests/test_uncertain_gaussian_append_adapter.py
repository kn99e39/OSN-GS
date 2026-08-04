"""Uncertain Gaussian append adapter -- Gate follow-up hardening tests.

docs/worklogs/88_uncertain_gaussian_append_adapter_foundation.md. Each test
covers exactly one contract from the Gate verification request so a failure
points at a single cause; see the worklog's coverage table for the mapping.
"""

from types import SimpleNamespace
from unittest import mock
import unittest

import torch

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.gaussian.torch_surface_ownership import project_occluded_chart_owner_id
from osn_gs.gaussian.torch_uncertain_append_adapter import (
    CLUSTER_ID_PROJECTION_RULE,
    UncertainAppendInitialization,
    UncertainGaussianAppendAdapter,
)
from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.surface.torch_occluded_chart_hardening import OccludedChartSafetyResult
from osn_gs.surface.torch_uncertain_gaussian_proposal import (
    UncertainGaussianProposalConfig,
    generate_uncertain_gaussian_proposals,
)


def make_chart_and_safety(supporting_patch_ids=(7, 3)):
    grid = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    surface = TorchNURBSSurface(grid, torch.ones((2, 2)), degree_u=1, degree_v=1)
    chart = SimpleNamespace(
        chart_id="chart-a", source_candidate_id="candidate-a",
        supporting_patch_ids=list(supporting_patch_ids), supporting_domain_ids=["domain-a"],
        supporting_boundary_ids=["boundary-a"], state="validated", surface=surface,
    )
    safety = OccludedChartSafetyResult(
        "chart-a", "candidate-a", {}, {}, {"coverage_scope": "central_bridge_only"}, [], "eligible", [], {}, {}
    )
    return chart, safety


def make_batch(supporting_patch_ids=(7, 3), target_spacing=0.6, batch_id_suffix=None):
    chart, safety = make_chart_and_safety(supporting_patch_ids)
    batch = generate_uncertain_gaussian_proposals(
        chart, safety, config=UncertainGaussianProposalConfig(target_spacing=target_spacing)
    )
    if batch_id_suffix is not None:
        object.__setattr__(batch, "proposal_batch_id", batch.proposal_batch_id + batch_id_suffix)
    return batch


def initialization(batch, sh_degree=1):
    n = len(batch.sample_ids)
    rest = (sh_degree + 1) ** 2 - 1
    return UncertainAppendInitialization(
        torch.zeros((n, 1, 3)), torch.zeros((n, rest, 3)), torch.zeros((n, 1)), torch.full((n, 1), -1.0)
    )


def model_snapshot(model):
    return tuple(
        x.detach().clone()
        for x in (
            model._xyz, model._features_dc, model._features_rest, model._opacity, model._scaling,
            model._rotation, model._uncertain_confidence, model.is_uncertain, model.surface_uv, model.cluster_ids,
            model.surface_owner_kind, model.surface_owner_id,
        )
    )


def assert_model_unchanged(test, before, model):
    for old, new in zip(before, model_snapshot(model)):
        torch.testing.assert_close(old, new)


def adapter_snapshot(adapter, model):
    return frozenset(model.appended_uncertain_batch_ids), adapter.provenance_sidecar


def assert_adapter_unchanged(test, before, adapter, model):
    before_ids, before_sidecar = before
    test.assertEqual(before_ids, frozenset(model.appended_uncertain_batch_ids))
    test.assertEqual(before_sidecar, adapter.provenance_sidecar)


class ModelStateSnapshotTest(unittest.TestCase):
    """`TorchGaussianModel.snapshot_state`/`restore_state` themselves."""

    def test_snapshot_restore_round_trip_is_exact(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        batch = make_batch()
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, initialization(batch))
        before = model_snapshot(model)
        snap = model.snapshot_state()
        model._xyz = torch.nn.Parameter(torch.zeros((999, 3)))
        model.cluster_ids = torch.full((999,), -5, dtype=torch.long)
        model.restore_state(snap)
        assert_model_unchanged(self, before, model)


class AppendAdapterEligibilityTest(unittest.TestCase):
    """Section 4: each preflight gate rejected independently."""

    def test_eligible_proposal_is_appended(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertEqual(receipt.append_state, "appended")
        self.assertEqual(receipt.appended_sample_count, int(batch.valid_mask.sum()))

    def test_review_required_proposal_is_blocked(self):
        batch = make_batch()
        batch.metadata["eligibility"] = "review_required"
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before, adapter_before = model_snapshot(model), adapter_snapshot(adapter, model)
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("proposal_not_eligible", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")
        assert_model_unchanged(self, before, model)
        assert_adapter_unchanged(self, adapter_before, adapter, model)

    def test_ineligible_proposal_is_blocked(self):
        batch = make_batch()
        batch.metadata["eligibility"] = "ineligible"
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("proposal_not_eligible", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")

    def test_unsupported_proposal_is_blocked(self):
        batch = make_batch()
        batch.metadata["eligibility"] = "unsupported"
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("proposal_not_eligible", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")

    def test_known_free_contradiction_is_blocked(self):
        batch = make_batch()
        batch.metadata["safety_reasons"] = ["full_known_free_contradiction"]
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("known_free_contradiction", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")

    def test_missing_provenance_is_blocked(self):
        batch = make_batch()
        batch.metadata["source_candidate_id"] = ""
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("proposal_provenance_missing", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")

    def test_unsupported_schema_is_blocked(self):
        batch = make_batch()
        object.__setattr__(batch, "schema_version", 99)
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("unsupported_proposal_schema", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")

    def test_already_appended_state_is_blocked(self):
        batch = make_batch()
        object.__setattr__(batch, "append_state", "appended")
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("proposal_already_appended", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")

    def test_zero_valid_samples_is_explicit_rejection(self):
        batch = make_batch()
        batch.valid_mask[:] = False
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before = model_snapshot(model)
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("no_valid_samples", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")
        self.assertEqual(receipt.appended_sample_count, 0)
        assert_model_unchanged(self, before, model)

    def test_active_optimizer_model_is_blocked(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        model.optimizer = object()  # simulate an active optimizer without a real training_setup call
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("model_only_append_requires_no_optimizer", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")

    def test_missing_initialization_is_blocked(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before = model_snapshot(model)
        receipt = adapter.append(batch, model, None)
        self.assertIn("appearance_initialization_required", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")
        assert_model_unchanged(self, before, model)


class AppendAdapterConversionTest(unittest.TestCase):
    """Section 5: scale and rotation conversion contracts."""

    def test_scale_converts_to_log_scale_numerically(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, initialization(batch))
        expected = batch.linear_scale[batch.valid_mask]
        torch.testing.assert_close(torch.exp(model._scaling), expected, atol=1e-5, rtol=1e-5)

    def test_zero_scale_is_rejected(self):
        batch = make_batch()
        idx = torch.nonzero(batch.valid_mask, as_tuple=False)[0]
        batch.linear_scale[idx] = 0.0
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("nonpositive_proposal_scale", receipt.reasons)

    def test_negative_scale_is_rejected(self):
        batch = make_batch()
        idx = torch.nonzero(batch.valid_mask, as_tuple=False)[0]
        batch.linear_scale[idx] = -1.0
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("nonpositive_proposal_scale", receipt.reasons)

    def test_extremely_small_scale_is_rejected(self):
        batch = make_batch()
        tiny = batch.linear_scale.to(torch.float64).clone()
        tiny[:] = 1e-300  # positive in float64, underflows to exactly 0 once cast to float32
        object.__setattr__(batch, "linear_scale", tiny)
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertNotIn("nonpositive_proposal_scale", receipt.reasons)
        self.assertIn("proposal_scale_below_representable_minimum", receipt.reasons)
        self.assertEqual(receipt.append_state, "not_appended")

    def test_scale_dtype_device_and_ordering_preserved(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, initialization(batch))
        self.assertEqual(model._scaling.dtype, torch.float32)
        self.assertEqual(str(model._scaling.device), "cpu")

    def test_rotation_component_order_and_values_pass_through(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, initialization(batch))
        expected = batch.rotation_quaternion[batch.valid_mask]
        torch.testing.assert_close(model._rotation, expected)

    def test_nonnormalized_quaternion_is_rejected(self):
        batch = make_batch()
        idx = torch.nonzero(batch.valid_mask, as_tuple=False)[0]
        batch.rotation_quaternion[idx] = batch.rotation_quaternion[idx] * 5.0
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertIn("unnormalized_proposal_quaternion", receipt.reasons)

    def test_valid_mask_filtering_preserves_sample_id_alignment(self):
        batch = make_batch()
        valid_indices = torch.nonzero(batch.valid_mask, as_tuple=False).reshape(-1)
        self.assertGreater(int(valid_indices.numel()), 1)
        dropped = int(valid_indices[0])
        batch.valid_mask[dropped] = False
        remaining = torch.nonzero(batch.valid_mask, as_tuple=False).reshape(-1)
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        expected_ids = tuple(batch.sample_ids[i] for i in remaining.tolist())
        self.assertEqual(receipt.appended_sample_ids, expected_ids)
        torch.testing.assert_close(model._xyz, batch.position[remaining])


class AppendAdapterAppearanceOpacityTest(unittest.TestCase):
    """Section 5: appearance/opacity pass-through, no hidden defaults."""

    def test_initialization_values_pass_through_unmodified(self):
        batch = make_batch()
        n = len(batch.sample_ids)
        rest = 3
        init = UncertainAppendInitialization(
            torch.full((n, 1, 3), 0.25), torch.full((n, rest, 3), 0.5),
            torch.full((n, 1), 2.0), torch.full((n, 1), -3.0),
        )
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, init)
        valid = batch.valid_mask
        torch.testing.assert_close(model._features_dc, init.features_dc[valid])
        torch.testing.assert_close(model._features_rest, init.features_rest[valid])
        torch.testing.assert_close(model._opacity, init.opacity_logits[valid])
        torch.testing.assert_close(model._uncertain_confidence, init.uncertain_confidence_logits[valid])

    def test_no_hidden_default_appearance_or_opacity(self):
        # Distinct non-zero/non-default values must survive unchanged --
        # proves the adapter isn't quietly substituting a canned default.
        batch = make_batch()
        n = len(batch.sample_ids)
        init = UncertainAppendInitialization(
            torch.full((n, 1, 3), 7.0), torch.zeros((n, 3, 3)), torch.full((n, 1), 9.0), torch.full((n, 1), 4.0)
        )
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, init)
        self.assertTrue(bool((model._features_dc == 7.0).all()))
        self.assertTrue(bool((model._opacity == 9.0).all()))

    def test_initialization_digest_recorded_and_stable(self):
        batch = make_batch()
        init = initialization(batch)
        model_a = TorchGaussianModel(sh_degree=1, device="cpu")
        model_b = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter_a = UncertainGaussianAppendAdapter()
        adapter_b = UncertainGaussianAppendAdapter()
        receipt_a = adapter_a.append(batch, model_a, init)
        batch2 = make_batch(batch_id_suffix="-other")
        receipt_b = adapter_b.append(batch2, model_b, init)
        self.assertIsNotNone(receipt_a.initialization_digest)
        self.assertEqual(receipt_a.initialization_digest, receipt_b.initialization_digest)
        different_init = initialization(batch, sh_degree=1)
        different_init.features_dc[:] = 99.0
        receipt_c = adapter_a.append(make_batch(batch_id_suffix="-third"), model_a, different_init)
        self.assertNotEqual(receipt_a.initialization_digest, receipt_c.initialization_digest)


class AppendAdapterTransactionTest(unittest.TestCase):
    """Section 2-3: model/sidecar/ledger transactional atomicity."""

    def test_successful_append_is_atomic_across_model_sidecar_ledger(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertEqual(receipt.append_state, "appended")
        self.assertEqual(len(model), receipt.appended_sample_count)
        self.assertIn(batch.proposal_batch_id, adapter.provenance_sidecar)
        self.assertIn(batch.proposal_batch_id, model.appended_uncertain_batch_ids)

    def test_conversion_failure_before_model_commit_leaves_everything_untouched(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before, adapter_before = model_snapshot(model), adapter_snapshot(adapter, model)
        bad = initialization(batch)
        object.__setattr__(bad, "features_dc", torch.zeros((1, 1, 3)))  # wrong row count
        with self.assertRaises((RuntimeError, IndexError, ValueError)):
            adapter.append(batch, model, bad)
        assert_model_unchanged(self, before, model)
        assert_adapter_unchanged(self, adapter_before, adapter, model)

    def test_model_commit_failure_rolls_back_fully(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before, adapter_before = model_snapshot(model), adapter_snapshot(adapter, model)

        def _poisoned(*args, **kwargs):
            # Simulate exactly the non-atomic partial-mutation failure mode
            # the replace_tensors audit identified: some fields already
            # reassigned to the new (wrong) count before the exception.
            model._xyz = torch.nn.Parameter(torch.zeros((999, 3)))
            model.cluster_ids = torch.full((999,), -7, dtype=torch.long)
            raise RuntimeError("injected_model_commit_failure")

        model.append_gaussians_model_only = _poisoned
        with self.assertRaisesRegex(RuntimeError, "injected_model_commit_failure"):
            adapter.append(batch, model, initialization(batch))
        assert_model_unchanged(self, before, model)
        assert_adapter_unchanged(self, adapter_before, adapter, model)

    def test_sidecar_commit_failure_rolls_back_model(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before, adapter_before = model_snapshot(model), adapter_snapshot(adapter, model)
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_commit_sidecar", side_effect=RuntimeError("injected_sidecar_failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected_sidecar_failure"):
                adapter.append(batch, model, initialization(batch))
        assert_model_unchanged(self, before, model)
        assert_adapter_unchanged(self, adapter_before, adapter, model)

    def test_ledger_commit_failure_rolls_back_sidecar_and_model(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before, adapter_before = model_snapshot(model), adapter_snapshot(adapter, model)
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_commit_ledger", side_effect=RuntimeError("injected_ledger_failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected_ledger_failure"):
                adapter.append(batch, model, initialization(batch))
        assert_model_unchanged(self, before, model)
        assert_adapter_unchanged(self, adapter_before, adapter, model)
        self.assertNotIn(batch.proposal_batch_id, adapter.provenance_sidecar)

    def test_receipt_candidate_failure_leaves_everything_untouched(self):
        # Receipt is now built in stage 1, strictly BEFORE model/sidecar/
        # ledger commit begins -- a failure here must behave exactly like a
        # conversion failure: nothing committed at all, not a partial commit.
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before, adapter_before = model_snapshot(model), adapter_snapshot(adapter, model)
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_build_receipt", side_effect=RuntimeError("injected_receipt_failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected_receipt_failure"):
                adapter.append(batch, model, initialization(batch))
        self.assertEqual(len(model), 0)
        assert_model_unchanged(self, before, model)
        assert_adapter_unchanged(self, adapter_before, adapter, model)

    def test_sidecar_entry_build_failure_leaves_everything_untouched(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before, adapter_before = model_snapshot(model), adapter_snapshot(adapter, model)
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_build_sidecar_entry", side_effect=RuntimeError("injected_entry_failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected_entry_failure"):
                adapter.append(batch, model, initialization(batch))
        self.assertEqual(len(model), 0)
        assert_model_unchanged(self, before, model)
        assert_adapter_unchanged(self, adapter_before, adapter, model)

    def test_successful_commit_always_returns_success_receipt(self):
        # The strong guarantee: once all three commits succeed, append()
        # cannot do anything except return the pre-built success receipt --
        # there is no remaining code path between the ledger commit and the
        # return that could turn a successful commit into an exception.
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertEqual(receipt.append_state, "appended")
        self.assertEqual(len(model), receipt.appended_sample_count)
        self.assertIn(batch.proposal_batch_id, adapter.provenance_sidecar)
        self.assertIn(batch.proposal_batch_id, model.appended_uncertain_batch_ids)

    def test_failed_transaction_batch_id_can_be_retried(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_commit_ledger", side_effect=RuntimeError("injected")
        ):
            with self.assertRaises(RuntimeError):
                adapter.append(batch, model, initialization(batch))
        self.assertNotIn(batch.proposal_batch_id, model.appended_uncertain_batch_ids)
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertEqual(receipt.append_state, "appended")


class AppendAdapterProvenanceAndClusterTest(unittest.TestCase):
    """Section 6: sidecar provenance completeness and cluster-id projection."""

    def test_sidecar_preserves_full_provenance(self):
        batch = make_batch(supporting_patch_ids=(7, 3))
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        entry = adapter.provenance_sidecar[batch.proposal_batch_id]
        self.assertEqual(entry["proposal_batch_id"], batch.proposal_batch_id)
        self.assertEqual(entry["proposal_sample_ids"], receipt.appended_sample_ids)
        self.assertEqual(entry["source_chart_id"], batch.metadata["source_chart_id"])
        self.assertEqual(entry["source_candidate_id"], batch.metadata["source_candidate_id"])
        self.assertEqual(entry["source_patch_ids"], tuple(batch.metadata["source_patch_ids"]))
        self.assertEqual(entry["supporting_domain_ids"], tuple(batch.metadata["supporting_domain_ids"]))
        self.assertEqual(entry["supporting_boundary_ids"], tuple(batch.metadata["supporting_boundary_ids"]))
        self.assertEqual(entry["append_origin"], "uncertain_gaussian_append_adapter")
        self.assertIsNotNone(entry["initialization_digest"])
        self.assertEqual(entry["appended_index_range"], (0, len(model)))

    def test_cluster_id_projects_deterministic_min_not_first_element(self):
        batch = make_batch(supporting_patch_ids=(7, 3))
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertEqual(receipt.cluster_id, 3)  # min(7, 3), NOT source_patch_ids[0] == 7
        self.assertEqual(receipt.cluster_id_projection_rule, CLUSTER_ID_PROJECTION_RULE)
        self.assertTrue(bool((model.cluster_ids == 3).all()))

    def test_cluster_id_projection_independent_of_list_order(self):
        batch_forward = make_batch(supporting_patch_ids=(2, 9), batch_id_suffix="-fwd")
        batch_reversed = make_batch(supporting_patch_ids=(9, 2), batch_id_suffix="-rev")
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        r1 = adapter.append(batch_forward, model, initialization(batch_forward))
        r2 = adapter.append(batch_reversed, model, initialization(batch_reversed))
        self.assertEqual(r1.cluster_id, 2)
        self.assertEqual(r2.cluster_id, 2)

    def test_appended_rows_are_occluded_chart_owned_not_visible_patch_owned(self):
        from osn_gs.gaussian.torch_surface_ownership import (
            SURFACE_OWNER_OCCLUDED_CHART,
            project_occluded_chart_owner_id,
        )

        batch = make_batch(supporting_patch_ids=(7, 3))
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        expected_owner_id = project_occluded_chart_owner_id(batch.metadata["source_chart_id"])
        self.assertEqual(receipt.surface_owner_kind, SURFACE_OWNER_OCCLUDED_CHART)
        self.assertEqual(receipt.surface_owner_id, expected_owner_id)
        self.assertTrue(bool((model.surface_owner_kind == SURFACE_OWNER_OCCLUDED_CHART).all()))
        self.assertTrue(bool((model.surface_owner_id == expected_owner_id).all()))
        # min(source_patch_ids) is recorded as compatibility only -- it must
        # NOT equal the real owner id, and the real owner id must not equal
        # either raw source patch id (proving it isn't secretly one of them).
        self.assertNotEqual(receipt.cluster_id, receipt.surface_owner_id)
        self.assertNotIn(receipt.surface_owner_id, (7, 3))

    def test_receipt_and_sidecar_owner_identity_match(self):
        batch = make_batch(supporting_patch_ids=(7, 3))
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        entry = adapter.provenance_sidecar[batch.proposal_batch_id]
        self.assertEqual(entry["surface_owner_kind"], receipt.surface_owner_kind)
        self.assertEqual(entry["surface_owner_id"], receipt.surface_owner_id)

    def test_ownership_row_alignment_survives_valid_mask_filtering(self):
        from osn_gs.gaussian.torch_surface_ownership import (
            SURFACE_OWNER_OCCLUDED_CHART,
            project_occluded_chart_owner_id,
        )

        batch = make_batch(supporting_patch_ids=(7, 3))
        valid_indices = torch.nonzero(batch.valid_mask, as_tuple=False).reshape(-1)
        self.assertGreater(int(valid_indices.numel()), 1)
        batch.valid_mask[int(valid_indices[0])] = False
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        expected_owner_id = project_occluded_chart_owner_id(batch.metadata["source_chart_id"])
        self.assertEqual(len(model), receipt.appended_sample_count)
        self.assertTrue(bool((model.surface_owner_kind == SURFACE_OWNER_OCCLUDED_CHART).all()))
        self.assertTrue(bool((model.surface_owner_id == expected_owner_id).all()))


class AppendAdapterDuplicateLedgerTest(unittest.TestCase):
    """Section 7: duplicate append policy and ledger ownership/lifecycle."""

    def test_duplicate_batch_id_second_append_is_blocked(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, initialization(batch))
        before = model_snapshot(model)
        again = adapter.append(batch, model, initialization(batch))
        self.assertIn("duplicate_proposal_batch", again.reasons)
        self.assertEqual(again.append_state, "not_appended")
        assert_model_unchanged(self, before, model)

    def test_duplicate_block_ignores_payload_changes(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, initialization(batch))
        batch.metadata["source_chart_id"] = "different-chart-id"  # payload changed, ID unchanged
        again = adapter.append(batch, model, initialization(batch))
        self.assertIn("duplicate_proposal_batch", again.reasons)

    def test_different_batch_ids_same_geometry_both_append(self):
        # Intentional policy: dedup is batch-ID keyed only, not content keyed.
        batch_a = make_batch(batch_id_suffix="-a")
        batch_b = make_batch(batch_id_suffix="-b")
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        r1 = adapter.append(batch_a, model, initialization(batch_a))
        r2 = adapter.append(batch_b, model, initialization(batch_b))
        self.assertEqual(r1.append_state, "appended")
        self.assertEqual(r2.append_state, "appended")
        self.assertEqual(len(model), r1.appended_sample_count + r2.appended_sample_count)

    def test_failed_transaction_does_not_consume_batch_id(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_commit_sidecar", side_effect=RuntimeError("injected")
        ):
            with self.assertRaises(RuntimeError):
                adapter.append(batch, model, initialization(batch))
        self.assertNotIn(batch.proposal_batch_id, model.appended_uncertain_batch_ids)

    def test_successful_transaction_registers_batch_id(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, initialization(batch))
        self.assertIn(batch.proposal_batch_id, model.appended_uncertain_batch_ids)

    def test_duplicate_blocked_across_adapter_instances_same_model(self):
        """Fixed contract: the ledger is MODEL-owned, so a second, entirely
        fresh adapter instance still recognizes a batch ID already appended
        to the same model by a different adapter instance."""

        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter_1 = UncertainGaussianAppendAdapter()
        adapter_2 = UncertainGaussianAppendAdapter()
        r1 = adapter_1.append(batch, model, initialization(batch))
        before = model_snapshot(model)
        r2 = adapter_2.append(batch, model, initialization(batch))
        self.assertEqual(r1.append_state, "appended")
        self.assertEqual(r2.append_state, "not_appended")
        self.assertIn("duplicate_proposal_batch", r2.reasons)
        self.assertEqual(len(model), r1.appended_sample_count)  # NOT doubled
        assert_model_unchanged(self, before, model)

    def test_failed_transaction_retry_with_different_adapter_succeeds(self):
        """A transaction that fails via adapter A must not consume the batch
        ID in the model ledger, so a completely different adapter B can
        retry it successfully afterward."""

        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter_a = UncertainGaussianAppendAdapter()
        adapter_b = UncertainGaussianAppendAdapter()
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_commit_ledger", side_effect=RuntimeError("injected")
        ):
            with self.assertRaises(RuntimeError):
                adapter_a.append(batch, model, initialization(batch))
        self.assertNotIn(batch.proposal_batch_id, model.appended_uncertain_batch_ids)
        receipt = adapter_b.append(batch, model, initialization(batch))
        self.assertEqual(receipt.append_state, "appended")
        self.assertIn(batch.proposal_batch_id, model.appended_uncertain_batch_ids)

    def test_same_batch_id_different_models_are_independent(self):
        batch_a = make_batch()
        batch_b = make_batch()  # identical inputs -> identical proposal_batch_id
        self.assertEqual(batch_a.proposal_batch_id, batch_b.proposal_batch_id)
        model_a = TorchGaussianModel(sh_degree=1, device="cpu")
        model_b = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt_a = adapter.append(batch_a, model_a, initialization(batch_a))
        receipt_b = adapter.append(batch_b, model_b, initialization(batch_b))
        self.assertEqual(receipt_a.append_state, "appended")
        self.assertEqual(receipt_b.append_state, "appended")  # independent ledger per model
        self.assertIn(batch_a.proposal_batch_id, model_a.appended_uncertain_batch_ids)
        self.assertIn(batch_b.proposal_batch_id, model_b.appended_uncertain_batch_ids)
        self.assertNotIn(batch_a.proposal_batch_id, TorchGaussianModel(sh_degree=1, device="cpu").appended_uncertain_batch_ids)


class AppendAdapterOwnerRegistryTransactionTest(unittest.TestCase):
    """Ownership Foundation Gate final-contract round: owner registry is its
    own append() transaction stage (commit after sidecar, before ledger),
    with rollback that distinguishes a newly-created binding from one that
    predates the current transaction."""

    def test_convert_stage_does_not_mutate_registry(self):
        # `_convert()` (transaction stage 1) must only ever VALIDATE the
        # owner binding, never write it -- the actual write is `append()`'s
        # own later commit stage.
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter._convert(batch, model, initialization(batch))
        self.assertEqual(model.occluded_chart_owner_registry, {})

    def test_successful_append_commits_registry_entry(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch, model, initialization(batch))
        expected_owner_id = project_occluded_chart_owner_id(batch.metadata["source_chart_id"])
        self.assertEqual(model.occluded_chart_owner_registry, {expected_owner_id: batch.metadata["source_chart_id"]})

    def test_owner_registry_commit_failure_rolls_back_sidecar_and_model(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before, adapter_before = model_snapshot(model), adapter_snapshot(adapter, model)
        with mock.patch.object(
            UncertainGaussianAppendAdapter,
            "_commit_owner_registry",
            side_effect=RuntimeError("injected_owner_registry_failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected_owner_registry_failure"):
                adapter.append(batch, model, initialization(batch))
        assert_model_unchanged(self, before, model)
        assert_adapter_unchanged(self, adapter_before, adapter, model)
        self.assertEqual(model.occluded_chart_owner_registry, {})
        self.assertNotIn(batch.proposal_batch_id, model.appended_uncertain_batch_ids)

    def test_ledger_commit_failure_rolls_back_newly_created_registry_entry(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        before, adapter_before = model_snapshot(model), adapter_snapshot(adapter, model)
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_commit_ledger", side_effect=RuntimeError("injected_ledger_failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected_ledger_failure"):
                adapter.append(batch, model, initialization(batch))
        assert_model_unchanged(self, before, model)
        assert_adapter_unchanged(self, adapter_before, adapter, model)
        # This transaction created the binding, so a later-stage failure must
        # remove it again -- nothing should survive.
        self.assertEqual(model.occluded_chart_owner_registry, {})

    def test_ledger_commit_failure_does_not_remove_preexisting_registry_entry(self):
        # Two batches from the SAME occluded chart: the second transaction's
        # owner binding already existed (created by the first, already-
        # committed transaction) before the second transaction even started.
        batch_a = make_batch(batch_id_suffix="-a")
        batch_b = make_batch(batch_id_suffix="-b")
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        adapter.append(batch_a, model, initialization(batch_a))
        expected_owner_id = project_occluded_chart_owner_id(batch_a.metadata["source_chart_id"])
        self.assertEqual(model.occluded_chart_owner_registry, {expected_owner_id: batch_a.metadata["source_chart_id"]})
        before = model_snapshot(model)
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_commit_ledger", side_effect=RuntimeError("injected_ledger_failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "injected_ledger_failure"):
                adapter.append(batch_b, model, initialization(batch_b))
        assert_model_unchanged(self, before, model)
        # The binding created by batch_a's earlier, already-committed
        # transaction must survive batch_b's rollback untouched.
        self.assertEqual(model.occluded_chart_owner_registry, {expected_owner_id: batch_a.metadata["source_chart_id"]})

    def test_failed_transaction_retry_preserves_registry_consistency(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter_a = UncertainGaussianAppendAdapter()
        adapter_b = UncertainGaussianAppendAdapter()
        with mock.patch.object(
            UncertainGaussianAppendAdapter, "_commit_ledger", side_effect=RuntimeError("injected")
        ):
            with self.assertRaises(RuntimeError):
                adapter_a.append(batch, model, initialization(batch))
        self.assertEqual(model.occluded_chart_owner_registry, {})
        receipt = adapter_b.append(batch, model, initialization(batch))
        self.assertEqual(receipt.append_state, "appended")
        expected_owner_id = project_occluded_chart_owner_id(batch.metadata["source_chart_id"])
        self.assertEqual(model.occluded_chart_owner_registry, {expected_owner_id: batch.metadata["source_chart_id"]})


class AppendAdapterReceiptTest(unittest.TestCase):
    """Section 8: receipt contract."""

    def test_receipt_fields_on_success(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertEqual(receipt.proposal_batch_id, batch.proposal_batch_id)
        self.assertEqual(receipt.source_chart_id, batch.metadata["source_chart_id"])
        self.assertEqual(receipt.requested_sample_count, len(batch.sample_ids))
        self.assertEqual(receipt.valid_sample_count, int(batch.valid_mask.sum()))
        self.assertEqual(receipt.appended_sample_count, int(batch.valid_mask.sum()))
        self.assertEqual(receipt.rejected_sample_count, receipt.requested_sample_count - receipt.appended_sample_count)
        self.assertEqual(receipt.model_count_before, 0)
        self.assertEqual(receipt.model_count_after, receipt.appended_sample_count)
        self.assertEqual(receipt.appended_index_range, (0, receipt.appended_sample_count))
        self.assertEqual(len(receipt.appended_sample_ids), receipt.appended_sample_count)
        self.assertTrue(receipt.conversion_summary)
        self.assertEqual(receipt.append_state, "appended")
        self.assertEqual(receipt.reasons, ())
        self.assertIsNotNone(receipt.cluster_id)
        self.assertIsNotNone(receipt.initialization_digest)

    def test_receipt_fields_on_rejection(self):
        batch = make_batch()
        batch.metadata["eligibility"] = "ineligible"
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertEqual(receipt.append_state, "not_appended")
        self.assertEqual(receipt.appended_sample_count, 0)
        self.assertEqual(receipt.appended_index_range, None)
        self.assertEqual(receipt.appended_sample_ids, ())
        self.assertEqual(receipt.conversion_summary, ())
        self.assertIsNone(receipt.cluster_id)
        self.assertIsNone(receipt.initialization_digest)
        self.assertIn("proposal_not_eligible", receipt.reasons)

    def test_receipt_json_is_deterministic(self):
        batch = make_batch()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        adapter = UncertainGaussianAppendAdapter()
        receipt = adapter.append(batch, model, initialization(batch))
        self.assertEqual(receipt.stable_json(), receipt.stable_json())


if __name__ == "__main__":
    unittest.main()
