import torch

from nurbs_constructor_benchmark.scenes import SCENE_NAMES, make_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig


def test_default_synthetic_dataset_uses_depth_bearing_3d_scenes() -> None:
    assert SCENE_NAMES == ("saddle_shell", "spherical_cap", "folded_roof", "wave_annulus")
    for name in SCENE_NAMES:
        scene = make_scene(name, 240, seed=13)
        extent = scene.points.max(dim=0).values - scene.points.min(dim=0).values
        assert float(extent[2]) > 0.35
        assert float(extent[2] / extent[:2].max()) > 0.18


def test_synthetic_covariance_is_flattened_and_tangent_aligned() -> None:
    scene = make_scene("saddle_shell", 240, seed=3)
    scales = scene.covariance_scales
    ratio = scales.max(dim=1).values / scales.min(dim=1).values
    assert float(ratio.median()) > 2.5
    assert float((ratio >= 2.0).float().mean()) > 0.75
    assert torch.allclose(
        scene.covariance_rotations.norm(dim=1),
        torch.ones(len(scene.points)),
        atol=1e-5,
    )


def test_pipeline_preserves_observed_synthetic_covariance() -> None:
    scene = make_scene("wave_annulus", 96, seed=4)
    pipeline = TorchOSNGSPipeline(
        TorchPipelineConfig(
            nurbs_constructor_mode="legacy",
            base_curve_count=2,
            visible_surface_resolution_u=2,
            visible_surface_resolution_v=2,
        ),
        device="cpu",
    )
    state = pipeline.initialize(
        scene.points,
        scene.colors,
        covariance_scales=scene.covariance_scales,
        covariance_rotations=scene.covariance_rotations,
    )
    assert torch.allclose(state.model.get_scaling.detach().cpu(), scene.covariance_scales, atol=1e-6)
    assert torch.allclose(state.model._rotation.detach().cpu(), scene.covariance_rotations, atol=1e-6)

def test_synthetic_covariance_thin_axis_tracks_surface_normal() -> None:
    scene = make_scene("saddle_shell", 256, seed=4)
    w, x, y, z = scene.covariance_rotations.unbind(dim=1)
    local_z_in_world = torch.stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=1,
    )
    alignment = (local_z_in_world * scene.covariance_normals).sum(dim=1)
    assert float(alignment.min()) > 0.999