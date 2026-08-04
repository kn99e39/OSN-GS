"""Worklog 35: verify the recovered closed loop for box_face follows the
analytic square boundary (not an interior shortcut, not a self-intersecting
path), and that floater/contamination scenes never include the
excluded/contaminated points in a materialized loop."""

from __future__ import annotations

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig


def check_box_face_boundary_accuracy():
    scene = make_gaussian_reliability_scene("box_face", seed=0)
    stable_ids = tuple(range(scene.positions.shape[0]))
    opacity = torch.ones(scene.positions.shape[0])
    config = TorchPipelineConfig(canonical_construction_max_points=27)
    pipeline = TorchOSNGSPipeline(config, device="cpu")
    bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
    construction = bundle.construction

    closed = [c for c in construction.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]
    print(f"closed components: {len(closed)}")
    for component in closed:
        ids = component.ordered_source_ids
        idx = torch.tensor(list(ids))
        pts = scene.positions[idx]
        # box_face is a single pz face spanning roughly [-hx,hx] x [-hy,hy] at z=hz.
        z = pts[:, 2]
        xy_radius = (pts[:, 0].abs().amax().item(), pts[:, 1].abs().amax().item())
        print(f"  loop len={len(ids)} z_std={float(z.std()):.5f} xy_extent={xy_radius}")
        # winding / self-intersection sanity: consecutive-edge polyline should
        # not have any two non-adjacent segments crossing in the xy-plane for a
        # convex square boundary -- cheap check: total turning angle ~ +-2*pi.
        centroid = pts.mean(dim=0)
        angles = torch.atan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        diffs = (angles.roll(-1) - angles)
        diffs = (diffs + torch.pi) % (2 * torch.pi) - torch.pi
        total_turning = float(diffs.sum())
        print(f"  total_turning_radians={total_turning:.3f} (expect near +-2*pi={2*3.14159:.3f} for simple loop)")


def check_negative_controls_exclude_outliers():
    for scene_name, excluded_hint in [("box_isolated_floater", "floater"), ("box_isotropic_contamination", "contaminated")]:
        scene = make_gaussian_reliability_scene(scene_name, seed=0)
        stable_ids = tuple(range(scene.positions.shape[0]))
        opacity = torch.ones(scene.positions.shape[0])
        config = TorchPipelineConfig(canonical_construction_max_points=2048)
        pipeline = TorchOSNGSPipeline(config, device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
        construction = bundle.construction
        closed = [c for c in construction.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]
        all_loop_ids = set()
        for c in closed:
            all_loop_ids.update(c.ordered_source_ids)
        labels = scene.group_labels
        outlier_ids_in_loop = [
            sid for sid in all_loop_ids
            if sid < len(labels) and excluded_hint in labels[sid]
        ]
        print(f"{scene_name}: closed_loops={len(closed)} outlier_ids_in_any_closed_loop={outlier_ids_in_loop}")


if __name__ == "__main__":
    check_box_face_boundary_accuracy()
    check_negative_controls_exclude_outliers()
