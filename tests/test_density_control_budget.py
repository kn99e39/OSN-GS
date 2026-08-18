"""Behaviour of the `max_gaussians` VRAM guard when it binds.

The guard is not part of 3DGS or 2DGS -- it exists only so a 12 GB card can
finish the official 30,000-iteration schedule. That makes HOW it binds a
correctness question: an earlier revision spent the remaining budget on clones
first, which left `split` at zero on every capped step and measurably wrecked
the arm it bound for. These tests pin the two properties that keep a bound cap
from silently becoming an experimental variable.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.gaussian.torch_density_control import (
    TorchDensityControlConfig,
    _limited_indices,
    apply_adaptive_density_control,
)
from osn_gs.gaussian.torch_model import TorchGaussianModel


def _model(count: int, scale: float, seed: int = 5) -> TorchGaussianModel:
    torch.manual_seed(seed)
    model = TorchGaussianModel(sh_degree=1, device="cpu")
    model.initialize(
        positions=torch.rand((count, 3)),
        colors=torch.rand((count, 3)),
        opacities=torch.full((count, 1), 0.5),
        scales=torch.full((count, 3), scale),
        rotations=torch.rand((count, 4)),
    )
    model.xyz_gradient_accum = torch.full((count, 1), 1.0)
    model.denom = torch.ones((count, 1))
    return model


def _mixed_scale_model(count: int = 64, seed: int = 5) -> TorchGaussianModel:
    """Half the population above the dense-extent threshold, half below.

    Gives the ADC both clone candidates (small) and split candidates (large)
    at the same time, which is what a budget split has to arbitrate.
    """

    torch.manual_seed(seed)
    model = TorchGaussianModel(sh_degree=1, device="cpu")
    scales = torch.full((count, 3), 0.0005)
    scales[count // 2:] = 0.5
    model.initialize(
        positions=torch.rand((count, 3)),
        colors=torch.rand((count, 3)),
        opacities=torch.full((count, 1), 0.5),
        scales=scales,
        rotations=torch.rand((count, 4)),
    )
    model.xyz_gradient_accum = torch.full((count, 1), 1.0)
    model.denom = torch.ones((count, 1))
    return model


class BoundCapKeepsBothOperationsAliveTest(unittest.TestCase):
    def _report(self, max_gaussians: int):
        model = _mixed_scale_model()
        config = TorchDensityControlConfig(
            densify_until_iter=1000, densification_interval=1, max_gaussians=max_gaussians,
        )
        return apply_adaptive_density_control(model, config, scene_extent=1.0, iteration=100), model

    def test_uncapped_run_produces_both_clones_and_splits(self):
        report, _ = self._report(0)
        self.assertGreater(report.cloned, 0)
        self.assertGreater(report.split_parents, 0)

    def test_a_bound_cap_still_produces_splits(self):
        """The regression this file exists for: splits must not go to zero."""

        uncapped, _ = self._report(0)
        capped, model = self._report(80)
        self.assertLess(capped.cloned + capped.split, uncapped.cloned + uncapped.split)
        self.assertGreater(capped.cloned, 0)
        self.assertGreater(capped.split_parents, 0, "a bound cap must not suppress splitting")
        self.assertLessEqual(len(model), 80)

    def test_budget_is_split_in_proportion_to_demand(self):
        capped, _ = self._report(80)
        uncapped, _ = self._report(0)
        clone_share = capped.cloned / max(capped.cloned + capped.split, 1)
        demand_share = uncapped.cloned / max(uncapped.cloned + uncapped.split, 1)
        self.assertAlmostEqual(clone_share, demand_share, delta=0.15)

    def test_an_exhausted_cap_grows_nothing(self):
        model = _mixed_scale_model()
        config = TorchDensityControlConfig(
            densify_until_iter=1000, densification_interval=1, max_gaussians=len(model),
        )
        report = apply_adaptive_density_control(model, config, scene_extent=1.0, iteration=100)
        self.assertEqual(report.cloned, 0)
        self.assertEqual(report.split, 0)


class LimitedIndicesPriorityTest(unittest.TestCase):
    def test_unlimited_selection_is_unchanged(self):
        mask = torch.tensor([True, False, True, True])
        torch.testing.assert_close(_limited_indices(mask), torch.tensor([0, 2, 3]))

    def test_a_bound_limit_keeps_the_highest_priority_rows(self):
        mask = torch.tensor([True, True, True, True])
        priority = torch.tensor([0.1, 0.9, 0.2, 0.8])
        selected = _limited_indices(mask, 2, priority)
        self.assertEqual(sorted(selected.tolist()), [1, 3])

    def test_selection_stays_ascending(self):
        mask = torch.ones(10, dtype=torch.bool)
        priority = torch.tensor([9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0, 0.0])
        selected = _limited_indices(mask, 4, priority)
        self.assertEqual(selected.tolist(), sorted(selected.tolist()))

    def test_without_priority_it_falls_back_to_row_order(self):
        mask = torch.ones(5, dtype=torch.bool)
        self.assertEqual(_limited_indices(mask, 2).tolist(), [0, 1])

    def test_zero_limit_selects_nothing(self):
        mask = torch.ones(5, dtype=torch.bool)
        self.assertEqual(_limited_indices(mask, 0, torch.rand(5)).numel(), 0)


if __name__ == "__main__":
    unittest.main()
