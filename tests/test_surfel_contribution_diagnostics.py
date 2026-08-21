from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import torch

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
    from osn_gs.render.diff_surfel_loader import get_diff_surfel_backend
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig
    from osn_gs.render.torch_surfel_contribution_diagnostics import (
        RendererContributionEvidence,
        accumulate_renderer_contribution_evidence,
        compute_renderer_contribution_for_view,
    )
    from tests.test_surfel_rasterization_cuda import FOVX, FOVY, HEIGHT, WIDTH, _camera

    BACKEND_AVAILABLE = get_diff_surfel_backend() is not None
else:  # pragma: no cover - environment dependent
    BACKEND_AVAILABLE = False

requires_cuda = unittest.skipUnless(
    CUDA_AVAILABLE and BACKEND_AVAILABLE,
    "CUDA and the vendored diff_surfel_rasterization extension are required",
)


def _identity_camera() -> "TorchCamera":
    return _camera(torch.eye(3, dtype=torch.float32), torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))


def _flat_surfel(z: float, opacity: float, scale: float = 2.0, color=(0.5, 0.5, 0.5)):
    model = TorchGaussianSurfelModel(sh_degree=0, device="cuda")
    model.initialize(
        positions=torch.tensor([[0.0, 0.0, z]], dtype=torch.float32),
        colors=torch.tensor([list(color)], dtype=torch.float32),
        opacities=torch.tensor([[opacity]], dtype=torch.float32),
        scales=torch.tensor([[scale, scale]], dtype=torch.float32),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    return model


def _multi_flat_surfel(specs: list[tuple[float, float, float]]):
    """`specs` = [(z, opacity, scale), ...], one primitive per entry, all
    facing the camera identically (identity rotation)."""

    model = TorchGaussianSurfelModel(sh_degree=0, device="cuda")
    positions = torch.tensor([[0.0, 0.0, z] for z, _, _ in specs], dtype=torch.float32)
    colors = torch.tensor([[0.5, 0.5, 0.5] for _ in specs], dtype=torch.float32)
    opacities = torch.tensor([[o] for _, o, _ in specs], dtype=torch.float32)
    scales = torch.tensor([[s, s] for _, _, s in specs], dtype=torch.float32)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0] for _ in specs], dtype=torch.float32)
    model.initialize(positions=positions, colors=colors, opacities=opacities, scales=scales, rotations=rotations)
    return model


@requires_cuda
class RendererContributionDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.camera = _identity_camera()
        self.rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())

    def test_canonical_forward_outputs_unchanged_by_diagnostic_call(self):
        """Directive section 5: instrumentation must not alter rendering."""

        model = _flat_surfel(z=2.0, opacity=0.8)
        canonical = self.rasterizer.render(self.camera, model)
        canonical_render = canonical["render"].detach().clone()
        canonical_depth = canonical["depth"].detach().clone()

        contributed, weight, diagnostic_package = compute_renderer_contribution_for_view(self.camera, model, self.rasterizer)

        torch.testing.assert_close(diagnostic_package["render"].detach(), canonical_render)
        torch.testing.assert_close(diagnostic_package["depth"].detach(), canonical_depth)

        # A second, completely independent canonical call after the
        # diagnostic gradient computation must also still match exactly --
        # proves no persistent state (e.g. a stray .grad) was mutated.
        replay = self.rasterizer.render(self.camera, model)
        torch.testing.assert_close(replay["render"].detach(), canonical_render)
        torch.testing.assert_close(replay["depth"].detach(), canonical_depth)

    def test_diagnostic_call_does_not_mutate_parameter_grad(self):
        model = _flat_surfel(z=2.0, opacity=0.8)
        self.assertIsNone(model._features_dc.grad)
        compute_renderer_contribution_for_view(self.camera, model, self.rasterizer)
        self.assertIsNone(model._features_dc.grad)
        self.assertIsNone(model._opacity.grad)
        self.assertIsNone(model._xyz.grad)

    def test_visible_contributing_surfel_is_detected(self):
        model = _flat_surfel(z=2.0, opacity=0.9, scale=3.0)
        contributed, weight, _ = compute_renderer_contribution_for_view(self.camera, model, self.rasterizer)
        self.assertTrue(bool(contributed[0]))
        self.assertGreater(float(weight[0]), 0.0)

    def test_fully_occluded_surfel_is_not_falsely_counted_as_contributing(self):
        """Three stacked, fully covering, near-opaque surfels in front of a
        far surfel: the vendored kernel's own transmittance-termination floor
        (test_T < 0.0001) must exhaust before the far surfel is ever reached
        -- directive section 9's explicit contract, verified against real
        official rasterizer semantics, not a synthetic approximation."""

        model = _multi_flat_surfel([
            (2.0, 0.99, 8.0), (2.0, 0.99, 8.0), (2.0, 0.99, 8.0),  # near, wide/fully covering, stacked
            (5.0, 0.99, 0.2),  # far, small footprint concentrated at frame centre -- would be visible if unoccluded
        ])
        contributed, weight, _ = compute_renderer_contribution_for_view(self.camera, model, self.rasterizer)
        self.assertTrue(bool(contributed[0]))  # the nearest of the stack does contribute
        self.assertFalse(bool(contributed[3]))  # the far surfel must not
        self.assertEqual(float(weight[3]), 0.0)

    def test_unreachable_surfel_far_outside_frustum_does_not_contribute(self):
        model = _multi_flat_surfel([(2.0, 0.9, 3.0), (2.0, 0.9, 3.0)])
        model._xyz.data[1] = torch.tensor([500.0, 500.0, 2.0], dtype=torch.float32, device="cuda")
        contributed, weight, _ = compute_renderer_contribution_for_view(self.camera, model, self.rasterizer)
        self.assertTrue(bool(contributed[0]))
        self.assertFalse(bool(contributed[1]))

    def test_aggregate_accounting_is_deterministic_across_repeated_runs(self):
        """contributing_view_count / ever_contributed (booleans) must be
        bit-identical; accumulated_weight_sum only needs to agree within the
        tolerance the CUDA kernel's own order-dependent atomicAdd summation
        allows (a known, harmless floating-point non-associativity -- not a
        semantic non-determinism in which surfels count as contributing)."""

        model = _multi_flat_surfel([(2.0, 0.9, 3.0), (5.0, 0.9, 3.0)])
        cameras = [self.camera, self.camera]
        first = accumulate_renderer_contribution_evidence(cameras, model, self.rasterizer)
        second = accumulate_renderer_contribution_evidence(cameras, model, self.rasterizer)
        self.assertTrue(torch.equal(first.contributing_view_count, second.contributing_view_count))
        self.assertTrue(torch.equal(first.ever_contributed, second.ever_contributed))
        torch.testing.assert_close(first.accumulated_weight_sum, second.accumulated_weight_sum, rtol=1e-3, atol=1e-2)

    def test_multi_view_aggregation_counts_views_correctly(self):
        model = _flat_surfel(z=2.0, opacity=0.9, scale=3.0)
        cameras = [self.camera, self.camera, self.camera]
        evidence = accumulate_renderer_contribution_evidence(cameras, model, self.rasterizer)
        self.assertEqual(int(evidence.contributing_view_count[0]), 3)
        self.assertTrue(bool(evidence.ever_contributed[0]))
        self.assertGreater(float(evidence.accumulated_weight_sum[0]), 0.0)
        self.assertGreaterEqual(float(evidence.max_single_view_accumulated_weight[0]), 0.0)

    def test_primitive_tensors_remain_unchanged_by_diagnostics(self):
        model = _flat_surfel(z=2.0, opacity=0.9, scale=3.0)
        xyz_before = model.get_xyz.detach().clone()
        opacity_before = model.get_opacity.detach().clone()
        accumulate_renderer_contribution_evidence([self.camera, self.camera], model, self.rasterizer)
        torch.testing.assert_close(model.get_xyz.detach(), xyz_before)
        torch.testing.assert_close(model.get_opacity.detach(), opacity_before)


class DiagnosticModuleIsolationTest(unittest.TestCase):
    """No CUDA required -- pure source-text check that the diagnostic module
    is never imported by the canonical training/pipeline path."""

    def test_canonical_training_modules_do_not_import_the_diagnostic_module(self):
        repo_root = Path(__file__).resolve().parent.parent
        for relative in ("osn_gs/core/torch_pipeline.py", "osn_gs/core/torch_trainer.py"):
            path = repo_root / relative
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module)
            self.assertFalse(
                any("torch_surfel_contribution_diagnostics" in name for name in names),
                f"{relative} must not import the diagnostic-only contribution module",
            )
