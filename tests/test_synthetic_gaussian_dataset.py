import torch

from nurbs_constructor_benchmark.scenes import SCENE_NAMES, make_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig


def test_default_synthetic_dataset_uses_volumetric_3d_solids() -> None:
    assert SCENE_NAMES == ("box", "cylinder", "sphere")
    for name in SCENE_NAMES:
        scene = make_scene(name, 240, seed=13)
        extent = scene.points.max(dim=0).values - scene.points.min(dim=0).values
        assert float(extent[2]) > 0.35
        assert float(extent[2] / extent[:2].max()) > 0.18
        # Every point must lie exactly on the analytic solid's true boundary.
        residual, _ = scene.oracle(scene.points)
        assert float(residual.abs().max()) < 1e-4
        assert scene.faces is not None and len(scene.faces) == scene.gt_patch_count


def test_synthetic_covariance_is_flattened_and_tangent_aligned() -> None:
    scene = make_scene("box", 240, seed=3)
    scales = scene.covariance_scales
    ratio = scales.max(dim=1).values / scales.min(dim=1).values
    assert float(ratio.median()) > 2.5
    assert float((ratio >= 2.0).float().mean()) > 0.75
    assert torch.allclose(
        scene.covariance_rotations.norm(dim=1),
        torch.ones(len(scene.points)),
        atol=1e-5,
    )


def test_pipeline_preserves_observed_covariance_on_canonical_surface() -> None:
    from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_curvature_sweep_scene
    from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame

    scene = make_curvature_sweep_scene(10.0)
    frame = extract_covariance_frame(scene.covariances)
    scales = torch.stack((frame.tangent_major_scale, frame.tangent_minor_scale, frame.normal_thickness), dim=1)
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(), device="cpu")
    rotation_matrix = torch.stack((frame.tangent_u, frame.tangent_v, frame.normal_candidate), dim=-1)
    tangent_v = frame.tangent_v * torch.where(torch.linalg.det(rotation_matrix) < 0.0, -1.0, 1.0).unsqueeze(1)
    rotation_matrix = torch.stack((frame.tangent_u, tangent_v, frame.normal_candidate), dim=-1)
    rotations = pipeline._quaternion_from_rotation_matrix(rotation_matrix)
    state = pipeline.initialize(scene.positions, torch.full_like(scene.positions, 0.5), covariance_scales=scales, covariance_rotations=rotations)
    assert state.visible_surface_construction.construction_state == "constructed"
    assert torch.allclose(state.model.get_scaling.detach().cpu(), scales, atol=1e-6)
def test_synthetic_covariance_thin_axis_tracks_surface_normal() -> None:
    scene = make_scene("box", 256, seed=4)
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
