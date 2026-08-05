from __future__ import annotations

from unittest import mock
import unittest

import torch

from osn_gs.gaussian.torch_model import GaussianParameterGroups, TorchGaussianModel
from osn_gs.gaussian.torch_safe_uncertain_append_production import APPENDED, DUPLICATE
from osn_gs.gaussian.torch_uncertain_append_adapter import UncertainAppendInitialization, UncertainGaussianAppendAdapter
from osn_gs.gaussian.torch_uncertain_trainer_activation import (
    ACTIVATED,
    NOT_ACTIVATED,
    ROLLED_BACK,
    append_and_activate,
    masked_optimizer_step,
    run_safe_uncertain_proposals_append_and_activate,
)
from tests.test_safe_uncertain_append_production import _initialization, _safe_result


def _model_with_visible_rows_and_trained_optimizer(n: int = 3) -> TorchGaussianModel:
    """A model with `n` pre-existing (certain) rows and an optimizer that has
    already taken a real step -- so `exp_avg`/`exp_avg_sq` are non-zero,
    letting tests prove they survive activation/isolation bit-for-bit."""

    model = TorchGaussianModel(sh_degree=1, device="cpu")
    model.initialize(
        positions=torch.randn(n, 3), colors=torch.rand(n, 3),
        opacities=torch.full((n, 1), 0.5), scales=torch.full((n, 3), 0.02),
    )
    model.training_setup(GaussianParameterGroups())
    model.optimizer.zero_grad(set_to_none=True)
    loss = model.get_xyz.square().sum() + model._features_dc.square().sum()
    loss.backward()
    model.optimizer.step()
    return model


def _full_optimizer_state_snapshot(model) -> dict[str, dict[str, torch.Tensor]]:
    return {
        group["name"]: {
            key: value.clone()
            for key, value in model.optimizer.state[group["params"][0]].items()
            if torch.is_tensor(value)
        }
        for group in model.optimizer.param_groups
    }


class CompositeAppendActivateTest(unittest.TestCase):
    def test_success_extends_optimizer_in_place_with_identity_match(self):
        model = _model_with_visible_rows_and_trained_optimizer(3)
        old_optimizer_identity = model.optimizer
        old_xyz_state = _full_optimizer_state_snapshot(model)["xyz"]

        safe = _safe_result()
        result = append_and_activate(safe, model=model, initialization_provider=_initialization)

        self.assertEqual(result.attempts[0].status, ACTIVATED)
        self.assertIs(model.optimizer, old_optimizer_identity)
        self.assertGreater(len(model), 3)
        for group in model.optimizer.param_groups:
            (param,) = group["params"]
            self.assertIs(param, getattr(model, "_xyz" if group["name"] == "xyz" else f"_{group['name']}", param))
        new_xyz_state = model.optimizer.state[model._xyz]
        for key, value in old_xyz_state.items():
            if value.dim() > 0:
                torch.testing.assert_close(new_xyz_state[key][:3], value)

    def test_same_batch_rerun_causes_no_additional_activation(self):
        model = _model_with_visible_rows_and_trained_optimizer(2)
        safe = _safe_result()
        first = append_and_activate(safe, model=model, initialization_provider=_initialization)
        self.assertEqual(first.attempts[0].status, ACTIVATED)
        count_after_first = len(model)
        second = append_and_activate(safe, model=model, initialization_provider=_initialization)
        self.assertEqual(second.attempts[0].status, NOT_ACTIVATED)
        self.assertEqual(second.attempts[0].activated_row_count, 0)
        self.assertEqual(len(model), count_after_first)
        # `duplicate` is reachable through the underlying per-candidate append call.
        self.assertIn(DUPLICATE, {a.append_attempt.status for a in second.attempts} | {second.attempts[0].append_attempt.status})

    def test_rejected_and_no_candidate_paths_activate_nothing(self):
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        safe = _safe_result()
        result = append_and_activate(safe, model=model, initialization_provider=None)
        self.assertEqual(result.attempts[0].status, NOT_ACTIVATED)
        self.assertEqual(result.diagnostic_summary()["activated_count"], 0)
        self.assertIsNone(model.optimizer)
        self.assertEqual(len(model), 0)

    def test_sphere_raw_gaussian_entry_point_activates_nothing(self):
        from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
        scene = make_gaussian_reliability_scene("sphere")
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        result = run_safe_uncertain_proposals_append_and_activate(
            scene.positions, covariance=scene.covariances, stable_ids=tuple(range(len(scene.positions))),
            model=model, initialization_provider=_initialization,
        )
        self.assertEqual(result.attempts, ())
        self.assertEqual(result.diagnostic_summary()["activated_count"], 0)
        self.assertEqual(len(model), 0)

    def test_activation_failure_rolls_back_the_entire_composite_transaction(self):
        # Activation failing must undo the append itself too (model rows,
        # sidecar, owner registry, ledger, optimizer) -- the production path
        # never leaves a half-registered row (unlike the lower-level
        # `activate_appended_receipts`, which is allowed to).
        model = _model_with_visible_rows_and_trained_optimizer(3)
        old_optimizer = model.optimizer
        pre_xyz = model._xyz.detach().clone()
        pre_count = len(model)
        pre_ledger = frozenset(model.appended_uncertain_batch_ids)
        pre_registry = dict(model.occluded_chart_owner_registry)
        pre_state = _full_optimizer_state_snapshot(model)

        # The injected failure must fire ONLY for this module's own
        # post-append activation call, never for the (unaudited) append
        # transaction's own harmless internal `_preserve_optimizer_state`
        # no-op call (which also runs, with `self.optimizer is None`, deep
        # inside `replace_tensors()` during the append itself) -- patching it
        # unconditionally would incorrectly fail worklog 58's own transaction
        # instead of exercising this round's rollback path.
        real_preserve = TorchGaussianModel._preserve_optimizer_state

        def _fail_only_when_optimizer_active(self, *args, **kwargs):
            if self.optimizer is not None:
                raise RuntimeError("injected sync failure")
            return real_preserve(self, *args, **kwargs)

        safe = _safe_result()
        with mock.patch.object(
            TorchGaussianModel, "_preserve_optimizer_state", _fail_only_when_optimizer_active,
        ):
            result = append_and_activate(safe, model=model, initialization_provider=_initialization)

        attempt = result.attempts[0]
        self.assertEqual(attempt.status, ROLLED_BACK)
        self.assertEqual(attempt.append_attempt.status, APPENDED)  # the (unaudited) append itself DID succeed
        # ... but the composite transaction undid it: model back to pre-append.
        self.assertEqual(len(model), pre_count)
        torch.testing.assert_close(model._xyz.detach(), pre_xyz)
        self.assertEqual(frozenset(model.appended_uncertain_batch_ids), pre_ledger)
        self.assertEqual(dict(model.occluded_chart_owner_registry), pre_registry)
        # Optimizer: identity-correct against the model's CURRENT (freshly
        # restored, necessarily NEW `nn.Parameter`) objects, and state values
        # match pre-transaction exactly.
        self.assertIsNotNone(model.optimizer)
        for group in model.optimizer.param_groups:
            (param,) = group["params"]
            self.assertIs(param, {"xyz": model._xyz, "f_dc": model._features_dc, "f_rest": model._features_rest,
                                   "opacity": model._opacity, "scaling": model._scaling, "rotation": model._rotation,
                                   "uncertain_confidence": model._uncertain_confidence}[group["name"]])
        post_state = _full_optimizer_state_snapshot(model)
        for name, values in pre_state.items():
            for key, value in values.items():
                if value.dim() > 0:
                    torch.testing.assert_close(post_state[name][key], value)


class MaskedOptimizerStepTest(unittest.TestCase):
    def test_uncertain_only_step_leaves_visible_value_and_momentum_bit_for_bit(self):
        model = _model_with_visible_rows_and_trained_optimizer(3)
        pre_visible_xyz = model._xyz.detach()[:3].clone()
        pre_visible_state = {
            key: value[:3].clone()
            for key, value in model.optimizer.state[model._xyz].items()
            if torch.is_tensor(value) and value.dim() > 0
        }

        safe = _safe_result()
        activation = append_and_activate(safe, model=model, initialization_provider=_initialization)
        self.assertEqual(activation.attempts[0].status, ACTIVATED)

        model.optimizer.zero_grad(set_to_none=True)
        uncertain = model.is_uncertain
        loss = model.get_xyz[uncertain].square().sum()
        loss.backward()
        masked_optimizer_step(model, uncertain)

        post_visible_xyz = model._xyz.detach()[:3]
        post_visible_state = {
            key: value[:3]
            for key, value in model.optimizer.state[model._xyz].items()
            if torch.is_tensor(value) and value.dim() > 0
        }
        # Both the VALUE and the Adam MOMENTUM of excluded (visible) rows
        # must be exactly untouched -- not merely zero-gradient, which (with
        # this fixture's deliberately pre-seeded non-zero momentum) would
        # still let a plain `optimizer.step()` drift them.
        torch.testing.assert_close(post_visible_xyz, pre_visible_xyz)
        for key, value in pre_visible_state.items():
            torch.testing.assert_close(post_visible_state[key], value)
        # Uncertain rows genuinely moved.
        self.assertFalse(bool(torch.equal(model._xyz.detach()[3:], torch.zeros_like(model._xyz.detach()[3:]))))

    def test_visible_only_step_leaves_uncertain_value_and_momentum_bit_for_bit(self):
        model = _model_with_visible_rows_and_trained_optimizer(3)
        safe = _safe_result()
        activation = append_and_activate(safe, model=model, initialization_provider=_initialization)
        self.assertEqual(activation.attempts[0].status, ACTIVATED)
        pre_uncertain_xyz = model._xyz.detach()[3:].clone()
        pre_uncertain_state = {
            key: value[3:].clone()
            for key, value in model.optimizer.state[model._xyz].items()
            if torch.is_tensor(value) and value.dim() > 0
        }

        model.optimizer.zero_grad(set_to_none=True)
        certain = ~model.is_uncertain
        loss = model.get_xyz[certain].square().sum()
        loss.backward()
        masked_optimizer_step(model, certain)

        post_uncertain_xyz = model._xyz.detach()[3:]
        post_uncertain_state = {
            key: value[3:]
            for key, value in model.optimizer.state[model._xyz].items()
            if torch.is_tensor(value) and value.dim() > 0
        }
        torch.testing.assert_close(post_uncertain_xyz, pre_uncertain_xyz)
        for key, value in pre_uncertain_state.items():
            torch.testing.assert_close(post_uncertain_state[key], value)


class TrainerConnectionTest(unittest.TestCase):
    def _trainer_and_state(self):
        from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
        from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
        from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig
        from osn_gs.render.torch_fallback import TorchCamera

        torch.manual_seed(0)
        axis = torch.linspace(-0.48, 0.48, 5)
        points = torch.stack([torch.tensor([x, y, 0.0]) for x in axis for y in axis])
        colors = torch.rand(len(points), 3)
        # sh_degree=1 to match `_initialization`'s (worklog 58's own fixture
        # helper) rest_dim=3 appearance tensors.
        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(sh_degree=1), device="cpu")
        state = pipeline.initialize(points, colors)
        state.model.training_setup(GaussianParameterGroups())
        # Seed real (non-zero) Adam momentum, same rationale as the other
        # fixtures in this file -- proves isolation survives a genuinely
        # pre-trained optimizer, not a freshly-zero one.
        state.model.optimizer.zero_grad(set_to_none=True)
        (state.model.get_xyz.square().sum()).backward()
        state.model.optimizer.step()

        camera = TorchCamera(
            image_height=16, image_width=16, world_view_transform=torch.eye(4),
            full_proj_transform=torch.eye(4), camera_center=torch.zeros(3), FoVx=0.8, FoVy=0.8,
        )
        target = torch.rand(3, 16, 16)
        trainer = TorchOSNGSTrainer(
            pipeline_config=TorchPipelineConfig(),
            training_config=TorchTrainingConfig(
                iterations=1, progress_log_interval=0, timing_log_interval=0,
                prefer_cuda=False, write_output_files=False,
            ),
            rasterizer_config=GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True),
            device="cpu",
        )
        return trainer, state, camera, target

    def test_real_trainer_forward_backward_step_activates_and_trains_uncertain_rows(self):
        trainer, state, camera, target = self._trainer_and_state()
        pre_visible_xyz = state.model._xyz.detach().clone()
        pre_visible_state = _full_optimizer_state_snapshot(state.model)

        class _FakeSafeResult:
            def __init__(self, production):
                self.production = production

        with mock.patch(
            "osn_gs.gaussian.torch_uncertain_trainer_activation.run_safe_uncertain_proposals_from_gaussians",
            return_value=_FakeSafeResult(_safe_result()),
        ):
            activation, step_result = trainer.activate_and_train_uncertain_step(
                state, None, initialization_provider=_initialization, camera=camera, target=target,
            )

        self.assertEqual(activation.attempts[0].status, ACTIVATED)
        self.assertIsNotNone(step_result)
        self.assertIn("loss", step_result)
        self.assertTrue(torch.isfinite(torch.tensor(step_result["loss"])))
        self.assertGreater(len(state.model), pre_visible_xyz.shape[0])

        # Real render/backward touched the whole scene's gradient, but the
        # masked step must still have left every pre-existing visible row's
        # value AND Adam momentum untouched.
        torch.testing.assert_close(state.model._xyz.detach()[: pre_visible_xyz.shape[0]], pre_visible_xyz)
        post_state = _full_optimizer_state_snapshot(state.model)
        n = pre_visible_xyz.shape[0]
        for name, values in pre_visible_state.items():
            for key, value in values.items():
                if value.dim() > 0:
                    torch.testing.assert_close(post_state[name][key][:n], value)

    def test_sphere_through_trainer_activates_and_trains_nothing(self):
        trainer, state, camera, target = self._trainer_and_state()
        from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
        scene = make_gaussian_reliability_scene("sphere")
        activation, step_result = trainer.activate_and_train_uncertain_step(
            state, scene.positions, initialization_provider=_initialization, camera=camera, target=target,
            covariance=scene.covariances, stable_ids=tuple(range(len(scene.positions))),
        )
        self.assertEqual(activation.attempts, ())
        self.assertIsNone(step_result)


if __name__ == "__main__":
    unittest.main()
