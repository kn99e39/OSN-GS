from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_gaussian_covariance_frame import (
    SHAPE_ISOTROPIC,
    SHAPE_NEEDLE,
    SHAPE_PLANAR,
    covariance_from_scale_rotation,
    extract_covariance_frame,
    orientation_insensitive_alignment,
)


class GaussianCovarianceFrameTest(unittest.TestCase):
    def test_planar_gaussian_minimum_axis_matches_expected_normal(self):
        # Flat disk in the xy-plane (tiny z scale) -> normal candidate should be +/- z.
        scale = torch.tensor([[1.0, 1.0, 0.01]])
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        covariance = covariance_from_scale_rotation(scale, quaternion)
        frame = extract_covariance_frame(covariance)
        self.assertEqual(frame.shape_class, (SHAPE_PLANAR,))
        alignment = orientation_insensitive_alignment(frame.normal_candidate, torch.tensor([[0.0, 0.0, 1.0]]))
        self.assertGreater(float(alignment[0]), 0.999)

    def test_eigenvector_sign_flip_does_not_change_alignment(self):
        scale = torch.tensor([[1.0, 1.0, 0.01]])
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        covariance = covariance_from_scale_rotation(scale, quaternion)
        frame = extract_covariance_frame(covariance)
        flipped = -frame.normal_candidate
        alignment = orientation_insensitive_alignment(frame.normal_candidate, flipped)
        self.assertAlmostEqual(float(alignment[0]), 1.0, places=5)

    def test_isotropic_gaussian_is_not_classified_planar(self):
        scale = torch.tensor([[0.5, 0.5, 0.5]])
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        covariance = covariance_from_scale_rotation(scale, quaternion)
        frame = extract_covariance_frame(covariance)
        self.assertEqual(frame.shape_class, (SHAPE_ISOTROPIC,))
        self.assertNotEqual(frame.shape_class[0], SHAPE_PLANAR)

    def test_needle_like_gaussian_is_distinguished_from_planar_surfel(self):
        needle_scale = torch.tensor([[2.0, 0.02, 0.02]])
        planar_scale = torch.tensor([[1.0, 1.0, 0.02]])
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        scale = torch.cat((needle_scale, planar_scale), dim=0)
        covariance = covariance_from_scale_rotation(scale, quaternion)
        frame = extract_covariance_frame(covariance)
        self.assertEqual(frame.shape_class[0], SHAPE_NEEDLE)
        self.assertEqual(frame.shape_class[1], SHAPE_PLANAR)
        self.assertNotEqual(frame.shape_class[0], frame.shape_class[1])

    def test_eigenvalue_ordering_and_rotation_are_correct(self):
        # A flat disk rotated 90 degrees about x so its normal points along y.
        scale = torch.tensor([[1.0, 1.0, 0.01]])
        half = math.radians(90.0) / 2.0
        quaternion = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])
        covariance = covariance_from_scale_rotation(scale, quaternion)
        frame = extract_covariance_frame(covariance)
        lambda1, lambda2, lambda3 = frame.eigenvalues[0].tolist()
        self.assertGreaterEqual(lambda1, lambda2)
        self.assertGreaterEqual(lambda2, lambda3)
        alignment = orientation_insensitive_alignment(frame.normal_candidate, torch.tensor([[0.0, 1.0, 0.0]]))
        self.assertGreater(float(alignment[0]), 0.999)


if __name__ == "__main__":
    unittest.main()
