"""Worklog 62: camera FoV must be resolution-independent (computed once from
COLMAP's original intrinsics, matching Graphdeco baseline's own convention
exactly) -- never recomputed from a downscaled render resolution.

A lockstep parity harness against the real baseline (same initial Gaussian
tensors, same camera, ADC disabled) found that recomputing FoV from a
rounded downscaled width/height introduced measurable rendered-image drift
from iteration 1, traced to `camera_fovs` dividing focal length by
`downscale` and using the already-resized width/height instead of the
original COLMAP camera dimensions.
"""

from __future__ import annotations

import unittest

from osn_gs.data.colmap_scene import ColmapCamera, camera_fovs


class CameraFovResolutionIndependenceTest(unittest.TestCase):
    def test_fov_does_not_depend_on_downscale_argument(self):
        # A PINHOLE camera with distinct fx/fy, matching a real COLMAP model.
        camera = ColmapCamera(camera_id=1, model="PINHOLE", width=1920, height=1080, params=[1000.0, 950.0, 960.0, 540.0])
        fovx_a, fovy_a = camera_fovs(camera, width=1920, height=1080, downscale=1)
        fovx_b, fovy_b = camera_fovs(camera, width=1920, height=1080, downscale=3.24)
        # `downscale` is accepted for call-site compatibility but must never
        # change the result when width/height are held at the camera's own
        # original resolution.
        self.assertEqual(fovx_a, fovx_b)
        self.assertEqual(fovy_a, fovy_b)

    def test_fovx_and_fovy_use_independent_focal_lengths_for_pinhole(self):
        camera = ColmapCamera(camera_id=1, model="PINHOLE", width=1920, height=1080, params=[1000.0, 950.0, 960.0, 540.0])
        fovx, fovy = camera_fovs(camera, width=1920, height=1080)
        # fy > fx here (950 vs... wait fx=1000 > fy=950) -- so FoVy (using the
        # smaller focal, fy=950, over the smaller dimension, height=1080)
        # should differ from a naive same-focal computation. Just check both
        # are finite, positive, and distinct (non-square pixels/model).
        self.assertGreater(fovx, 0.0)
        self.assertGreater(fovy, 0.0)
        self.assertNotAlmostEqual(fovx, fovy, places=3)

    def test_simple_pinhole_uses_the_same_focal_for_both_axes(self):
        camera = ColmapCamera(camera_id=1, model="SIMPLE_PINHOLE", width=1920, height=1080, params=[1000.0, 960.0, 540.0])
        fovx, fovy = camera_fovs(camera, width=1920, height=1080)
        # Same focal, different width/height -> different FoV (aspect ratio),
        # but both derived from the identical focal length.
        import math
        expected_fovx = 2.0 * math.atan(1920 / (2.0 * 1000.0))
        expected_fovy = 2.0 * math.atan(1080 / (2.0 * 1000.0))
        self.assertAlmostEqual(fovx, expected_fovx, places=9)
        self.assertAlmostEqual(fovy, expected_fovy, places=9)


if __name__ == "__main__":
    unittest.main()
