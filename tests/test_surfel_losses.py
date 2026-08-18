"""2DGS geometric regularization: depth distortion and normal consistency.

Covers the activation staging of the official `train.py` and the exact
formulations recorded in `osn_gs/losses/torch_surfel_losses.py`, including the
measured gap between the paper's eq. 14 and the official code's aggregation.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.losses.torch_surfel_losses import (
    OFFICIAL_DIST_FROM_ITER,
    OFFICIAL_LAMBDA_NORMAL,
    OFFICIAL_NORMAL_FROM_ITER,
    PAPER_LAMBDA_DIST_BOUNDED,
    PAPER_LAMBDA_DIST_UNBOUNDED,
    SurfelRegularizationSchedule,
    depth_distortion_loss,
    normal_consistency_loss,
    normal_consistency_loss_paper_form,
    surfel_regularization_terms,
)


def _package(height: int = 5, width: int = 5, alpha_value: float = 1.0, seed: int = 1):
    torch.manual_seed(seed)
    alpha = torch.full((1, height, width), alpha_value)
    normal = torch.nn.functional.normalize(torch.rand((3, height, width)) - 0.5, dim=0)
    surface_normal_unit = torch.nn.functional.normalize(torch.rand((3, height, width)) - 0.5, dim=0)
    return {
        # Rasterized normal is alpha-weighted and therefore un-normalized.
        "rend_normal": normal * alpha,
        # The renderer pre-multiplies the depth-derived normal by detached alpha.
        "surf_normal": surface_normal_unit * alpha,
        "rend_alpha": alpha,
        "rend_dist": torch.rand((1, height, width)),
    }, normal, surface_normal_unit


class ActivationScheduleTest(unittest.TestCase):
    def test_default_schedule_is_the_official_one(self):
        schedule = SurfelRegularizationSchedule()
        self.assertEqual(schedule.dist_from_iter, OFFICIAL_DIST_FROM_ITER)
        self.assertEqual(schedule.normal_from_iter, OFFICIAL_NORMAL_FROM_ITER)
        self.assertEqual(schedule.dist_from_iter, 3000)
        self.assertEqual(schedule.normal_from_iter, 7000)
        self.assertAlmostEqual(schedule.lambda_normal, OFFICIAL_LAMBDA_NORMAL)
        self.assertAlmostEqual(schedule.lambda_normal, 0.05)
        # Official OptimizationParams default; the eval scripts pass the
        # paper's alpha explicitly.
        self.assertAlmostEqual(schedule.lambda_dist, 0.0)
        self.assertTrue(schedule.matches_official_staging())

    def test_depth_distortion_activates_strictly_after_iteration_3000(self):
        schedule = SurfelRegularizationSchedule(lambda_dist=PAPER_LAMBDA_DIST_UNBOUNDED)
        self.assertEqual(schedule.active_lambda_dist(1), 0.0)
        self.assertEqual(schedule.active_lambda_dist(3000), 0.0)
        self.assertEqual(schedule.active_lambda_dist(3001), PAPER_LAMBDA_DIST_UNBOUNDED)

    def test_normal_consistency_activates_strictly_after_iteration_7000(self):
        schedule = SurfelRegularizationSchedule()
        self.assertEqual(schedule.active_lambda_normal(7000), 0.0)
        self.assertEqual(schedule.active_lambda_normal(7001), OFFICIAL_LAMBDA_NORMAL)

    def test_rescaled_milestones_are_reported_as_non_official(self):
        schedule = SurfelRegularizationSchedule(dist_from_iter=300, normal_from_iter=700)
        self.assertFalse(schedule.matches_official_staging())

    def test_paper_lambda_dist_constants(self):
        self.assertAlmostEqual(PAPER_LAMBDA_DIST_BOUNDED, 1000.0)
        self.assertAlmostEqual(PAPER_LAMBDA_DIST_UNBOUNDED, 100.0)

    def test_weighted_terms_are_zero_before_activation_and_nonzero_after(self):
        package, _, _ = _package()
        schedule = SurfelRegularizationSchedule(lambda_dist=100.0, lambda_normal=0.05)

        dist_loss, normal_loss, ld, ln = surfel_regularization_terms(package, schedule, 100)
        self.assertEqual(ld, 0.0)
        self.assertEqual(ln, 0.0)
        self.assertEqual(float(dist_loss), 0.0)
        self.assertEqual(float(normal_loss), 0.0)

        dist_loss, normal_loss, ld, ln = surfel_regularization_terms(package, schedule, 8000)
        self.assertEqual(ld, 100.0)
        self.assertEqual(ln, 0.05)
        self.assertGreater(float(dist_loss), 0.0)
        self.assertNotEqual(float(normal_loss), 0.0)


class DepthDistortionTest(unittest.TestCase):
    def test_official_reduction_is_the_pixel_mean_of_rend_dist(self):
        package, _, _ = _package()
        torch.testing.assert_close(
            depth_distortion_loss(package), package["rend_dist"].mean()
        )

    def test_a_concentrated_ray_scores_lower_than_a_spread_one(self):
        """The CUDA accumulator's intent, checked on its own recursion.

        Reproduces `forward.cu`'s per-intersection update in Python for two
        synthetic rays with identical blending weights but different
        intersection depths, so the sign of the regularizer's preference is
        verified rather than assumed.
        """

        def distortion(depths, alphas):
            near_n, far_n = 0.2, 100.0
            transmittance, m1, m2, total = 1.0, 0.0, 0.0, 0.0
            for depth, alpha in zip(depths, alphas):
                weight = alpha * transmittance
                accumulated = 1 - transmittance
                m = far_n / (far_n - near_n) * (1 - near_n / depth)
                total += (m * m * accumulated + m2 - 2 * m * m1) * weight
                m1 += m * weight
                m2 += m * m * weight
                transmittance *= 1 - alpha
            return total

        alphas = [0.5, 0.5, 0.5]
        concentrated = distortion([3.00, 3.01, 3.02], alphas)
        spread = distortion([1.0, 3.0, 9.0], alphas)
        self.assertLess(concentrated, spread)
        self.assertLess(concentrated, 1e-4)


class NormalConsistencyTest(unittest.TestCase):
    def test_official_formulation_matches_the_official_expression(self):
        package, _, _ = _package()
        expected = (
            1 - (package["rend_normal"] * package["surf_normal"]).sum(dim=0)
        )[None].mean()
        torch.testing.assert_close(normal_consistency_loss(package), expected)

    def test_perfect_alignment_at_full_alpha_gives_zero_loss(self):
        normal = torch.nn.functional.normalize(torch.rand((3, 4, 4)) - 0.5, dim=0)
        alpha = torch.ones((1, 4, 4))
        package = {
            "rend_normal": normal * alpha,
            "surf_normal": normal * alpha,
            "rend_alpha": alpha,
            "rend_dist": torch.zeros((1, 4, 4)),
        }
        self.assertAlmostEqual(float(normal_consistency_loss(package)), 0.0, places=6)
        self.assertAlmostEqual(float(normal_consistency_loss_paper_form(package)), 0.0, places=6)

    def test_the_two_formulations_agree_at_saturated_alpha(self):
        package, _, _ = _package(alpha_value=1.0)
        torch.testing.assert_close(
            normal_consistency_loss(package),
            normal_consistency_loss_paper_form(package),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_the_two_formulations_disagree_at_partial_alpha(self):
        """The documented aggregation difference: 1 - A(R.N) vs A - R.N."""

        package, normal, surface_normal = _package(alpha_value=0.4)
        official = float(normal_consistency_loss(package))
        paper = float(normal_consistency_loss_paper_form(package))
        alignment = float((normal * surface_normal).sum(dim=0).mean())
        self.assertAlmostEqual(official, 1 - 0.4 * 0.4 * alignment, places=5)
        self.assertAlmostEqual(paper, 0.4 - 0.4 * alignment, places=5)
        self.assertGreater(abs(official - paper), 1e-3)

    def test_misalignment_increases_the_official_loss(self):
        normal = torch.nn.functional.normalize(torch.rand((3, 4, 4)) - 0.5, dim=0)
        alpha = torch.ones((1, 4, 4))
        aligned = {
            "rend_normal": normal, "surf_normal": normal,
            "rend_alpha": alpha, "rend_dist": torch.zeros((1, 4, 4)),
        }
        flipped = dict(aligned, surf_normal=-normal)
        self.assertLess(
            float(normal_consistency_loss(aligned)), float(normal_consistency_loss(flipped))
        )


class GradientFlowTest(unittest.TestCase):
    def test_both_terms_are_differentiable_in_the_rendered_quantities(self):
        alpha = torch.ones((1, 4, 4))
        normal = torch.nn.functional.normalize(torch.rand((3, 4, 4)) - 0.5, dim=0)
        rend_normal = (normal * alpha).clone().requires_grad_(True)
        rend_dist = torch.rand((1, 4, 4)).requires_grad_(True)
        package = {
            "rend_normal": rend_normal,
            "surf_normal": torch.nn.functional.normalize(torch.rand((3, 4, 4)) - 0.5, dim=0),
            "rend_alpha": alpha,
            "rend_dist": rend_dist,
        }
        (normal_consistency_loss(package) + depth_distortion_loss(package)).backward()
        self.assertTrue(bool(torch.isfinite(rend_normal.grad).all()))
        self.assertGreater(rend_normal.grad.abs().max().item(), 0.0)
        self.assertGreater(rend_dist.grad.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
