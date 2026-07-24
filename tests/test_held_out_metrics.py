from __future__ import annotations

"""Unit tests for osn_gs/eval/held_out_metrics.py."""

import unittest
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from osn_gs.eval.held_out_metrics import (
    evaluate_held_out_cameras,
    final_iteration_opacity_reset_applies,
)


@dataclass
class _FakeCamera:
    image_name: str


class _FakeRasterizer:
    """Renders back a fixed image regardless of camera/model, so tests can
    control exactly what the "rendered" output is."""

    def __init__(self, images: dict[str, Any]):
        self._images = images

    def render(self, camera, model, background):
        return {"render": self._images[camera.image_name]}


@unittest.skipUnless(torch is not None, "PyTorch is required")
class HeldOutMetricsTest(unittest.TestCase):
    def test_final_iteration_opacity_reset_predicate_matches_trainer_schedule(self):
        self.assertTrue(final_iteration_opacity_reset_applies(3000, 3000, 15_000))
        self.assertTrue(final_iteration_opacity_reset_applies(6000, 3000, 15_000))
        self.assertFalse(final_iteration_opacity_reset_applies(15_000, 3000, 15_000))
        self.assertFalse(final_iteration_opacity_reset_applies(2999, 3000, 15_000))
        self.assertFalse(final_iteration_opacity_reset_applies(3000, 0, 15_000))

    def test_perfect_match_gives_infinite_psnr_and_high_ssim(self):
        target = torch.rand(3, 16, 16)
        cameras = [_FakeCamera("a.jpg"), _FakeCamera("b.jpg")]
        images = [target.clone(), target.clone()]
        rasterizer = _FakeRasterizer({"a.jpg": target.clone(), "b.jpg": target.clone()})

        result = evaluate_held_out_cameras(rasterizer, model=None, test_cameras=cameras, test_images=images, device="cpu")

        self.assertEqual(result["camera_count"], 2)
        self.assertEqual(result["psnr_mean"], float("inf"))
        self.assertGreater(result["ssim_mean"], 0.99)
        for entry in result["per_camera"]:
            self.assertIn("image_name", entry)
            self.assertIn("psnr", entry)
            self.assertIn("ssim", entry)
            self.assertIn("mse", entry)
            self.assertEqual(entry["mse"], 0.0)

    def test_noisy_render_gives_finite_psnr_and_lower_ssim(self):
        torch.manual_seed(0)
        target = torch.rand(3, 16, 16)
        rendered = target + 0.5 * torch.randn_like(target)
        cameras = [_FakeCamera("a.jpg")]
        images = [target]
        rasterizer = _FakeRasterizer({"a.jpg": rendered})

        result = evaluate_held_out_cameras(rasterizer, model=None, test_cameras=cameras, test_images=images, device="cpu")
        self.assertTrue(0.0 < result["psnr_mean"] < 100.0)
        self.assertLess(result["ssim_mean"], 0.99)

    def test_mismatched_lengths_raises(self):
        rasterizer = _FakeRasterizer({})
        with self.assertRaises(ValueError):
            evaluate_held_out_cameras(
                rasterizer, model=None,
                test_cameras=[_FakeCamera("a.jpg")], test_images=[], device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
