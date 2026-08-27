"""Export the fixed Worklog 125 visualization pair for the Worklog 123 contract.

This is deliberately a diagnostic-only export.  It evaluates every trained
2DGS Gaussian *centre* as an ordinary world-space ``VolumetricQuery`` through
the frozen Worklog 123 Candidate B / global aggregation path.  A centre is not
silently promoted to a renderer median event, so it carries no event
provenance.  The output contains exactly the checkpoint's Gaussians twice:

* ``ORIGINAL_SCENE``: learned appearance;
* ``OBSERVED_OCCLUDED``: identical Gaussian rows with only their colour
  replaced by the global Candidate-B state.

No marker Gaussian, light, overlay, opacity change, geometry change, epsilon,
ULP acceptance band, or altered aggregation is present here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for entry in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from coverage_first_surfel_partition_export import (  # noqa: E402
    PRIMITIVE_SURFEL_2D, _rgb_to_f_dc, checkpoint_primitive, load_primitive_model,
    write_ppm, write_surfel_ply,
)
from maximal_visible_connectivity_export import load_all_train_cameras  # noqa: E402
from observed_occluded import candidate_b_median_depth as candidate_b  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    STATE_NON_RELEVANT, STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED,
    project_queries,
)

_ITERATION_DIR = "iteration_0000001"
_REPORT_NAME = "wl123_fixed_observed_occluded_visualization_report.json"
_WORKLOG = "127_novel_view_observed_occluded_inspection_correction.md"

# Fixed Worklog 125 palette.  It encodes a query classification only; it does
# not claim physical first hit, surface ownership, trust, or continuity.
_OBSERVED_RGB = (0.10, 0.85, 0.35)
_OCCLUDED_RGB = (0.92, 0.18, 0.18)
_UNRESOLVED_RGB = (0.60, 0.60, 0.62)


def _progress(message: str) -> None:
    print(f"[wl123-fixed-visualization] {message}", flush=True)


def global_state_from_accumulators(
    observed_any: torch.Tensor,
    has_relevant: torch.Tensor,
    has_unresolved: torch.Tensor,
) -> torch.Tensor:
    """Frozen ANY-OBSERVED aggregation in streamed form.

    This is algebraically identical to ``shared.aggregate_global`` but avoids
    allocating an N-by-161 state matrix for the 1.19M Gaussian-centre queries.
    """

    result = torch.full_like(observed_any, STATE_UNRESOLVED, dtype=torch.int8)
    result[has_relevant & ~has_unresolved & ~observed_any] = STATE_OCCLUDED
    result[observed_any] = STATE_OBSERVED
    return result


def state_colours(global_state: torch.Tensor) -> torch.Tensor:
    """Return one colour per existing Gaussian; never appends a row."""

    count = int(global_state.numel())
    colours = torch.tensor(_UNRESOLVED_RGB, dtype=torch.float32, device=global_state.device).reshape(1, 3).expand(count, 3).clone()
    colours[global_state == STATE_OBSERVED] = torch.tensor(_OBSERVED_RGB, dtype=torch.float32, device=global_state.device)
    colours[global_state == STATE_OCCLUDED] = torch.tensor(_OCCLUDED_RGB, dtype=torch.float32, device=global_state.device)
    return colours



def novel_inspection_candidates(cameras: list[Any], positions: torch.Tensor) -> list[Any]:
    """Make a deterministic outer orbit, separate from the 161 query views.

    The state contract is evaluated only against ``cameras``. These cameras are
    render-only review poses: they are outside the capture radius and no member
    is one of the dataset cameras. A review pose can consequently expose
    Gaussian centres occluded relative to every frozen query view.
    """

    from osn_gs.data.colmap_scene import projection_matrix
    from osn_gs.render.torch_fallback import TorchCamera

    device = positions.device
    target = torch.quantile(positions.to(torch.float32), 0.5, dim=0)
    centres = torch.stack([camera.camera_center.to(device=device, dtype=torch.float32) for camera in cameras])
    capture_radius = torch.quantile(torch.linalg.vector_norm(centres - target, dim=1), 0.90)
    scene_radius = torch.quantile(torch.linalg.vector_norm(positions - target, dim=1), 0.90)
    radius = torch.maximum(capture_radius * 1.20, scene_radius * 1.25)

    # COLMAP camera y is down. Average opposite y axes to obtain a stable up
    # vector, then make a deterministic horizontal frame.
    down_axes = torch.stack([camera.world_view_transform.T[:3, :3][1].to(device) for camera in cameras])
    up = -down_axes.mean(dim=0)
    up = up / torch.linalg.vector_norm(up).clamp_min(1e-8)
    seed = centres[0] - target
    seed = seed - up * torch.dot(seed, up)
    if float(torch.linalg.vector_norm(seed)) < 1e-6:
        seed = torch.tensor((1.0, 0.0, 0.0), device=device)
        seed = seed - up * torch.dot(seed, up)
    axis_x = seed / torch.linalg.vector_norm(seed).clamp_min(1e-8)
    axis_y = torch.linalg.cross(up, axis_x, dim=0)
    axis_y = axis_y / torch.linalg.vector_norm(axis_y).clamp_min(1e-8)

    reference = min(cameras, key=lambda camera: str(camera.image_name))
    projection = projection_matrix(0.01, 100.0, reference.FoVx, reference.FoVy, device=str(device)).transpose(0, 1).contiguous()
    candidates = []
    # Fixed before rendering: eight azimuths half a sector away from the first
    # capture ray, each at two elevations. These are not query views.
    for elevation_degrees in (12.0, 28.0):
        elevation = float(np.deg2rad(elevation_degrees))
        for index in range(8):
            azimuth = 2.0 * np.pi * (index + 0.5) / 8.0
            horizontal = np.cos(azimuth) * axis_x + np.sin(azimuth) * axis_y
            outward = np.cos(elevation) * horizontal + np.sin(elevation) * up
            centre = target + radius * outward
            forward = target - centre
            forward = forward / torch.linalg.vector_norm(forward).clamp_min(1e-8)
            right = torch.linalg.cross(up, forward, dim=0)
            right = right / torch.linalg.vector_norm(right).clamp_min(1e-8)
            down = torch.linalg.cross(forward, right, dim=0)
            rotation = torch.stack((right, down, forward), dim=0)
            translation = -rotation @ centre
            conventional = torch.eye(4, dtype=torch.float32, device=device)
            conventional[:3, :3] = rotation
            conventional[:3, 3] = translation
            world_view = conventional.T.contiguous()
            candidates.append(TorchCamera(
                image_height=reference.image_height, image_width=reference.image_width,
                world_view_transform=world_view,
                full_proj_transform=world_view @ projection,
                camera_center=centre, FoVx=reference.FoVx, FoVy=reference.FoVy,
                image_name=f"NOVEL_OUTER_ORBIT_e{int(elevation_degrees):02d}_a{index:02d}",
            ))
    return candidates


def red_dominant_pixel_count(image: torch.Tensor) -> int:
    """Presentation-only review score; never feeds classification."""

    rgb = image.detach()
    red = (rgb[0] > rgb[1] * 1.15) & (rgb[0] > rgb[2] * 1.15) & (rgb[0] > 0.10)
    return int(red.sum().item())


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _write_view_readme(folder: Path, body: str, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    footer = (
        "\n---\n"
        "분류 계약: Worklog 123의 frozen Candidate B `depth <= stored_median_depth`와 "
        "frozen ANY-OBSERVED global aggregation. Gaussian 중심은 임의 world-space 질의이며 "
        "renderer-event provenance를 붙이지 않았다.\n"
        "변경 금지: Gaussian 행 수·위치·scale·rotation·opacity·조명은 바꾸지 않았고, "
        "epsilon/ULP band도 없다.\n"
        f"Gaussian 수: {count:,} (두 view 동일) · 전체 리포트: `../{_REPORT_NAME}` · "
        f"Worklog: `docs/worklogs/{_WORKLOG}`\n"
    )
    (folder / "README.md").write_text(body + footer, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=65536,
                        help="resource-only streaming chunk; does not alter classification arithmetic")
    arguments = parser.parse_args()
    if arguments.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")

    started = time.time()
    arguments.out.mkdir(parents=True, exist_ok=True)
    _progress(f"loading checkpoint {arguments.checkpoint}")
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    if checkpoint_primitive(payload) != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(f"{arguments.checkpoint} is not a 2DGS surfel checkpoint")
    cameras, camera_meta = load_all_train_cameras(
        arguments.source_path, arguments.images, arguments.sparse_dir,
        arguments.resolution, arguments.llffhold, arguments.device,
    )
    if len(cameras) != 161:
        raise ValueError(f"expected the frozen 161 training cameras, got {len(cameras)}")

    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe
    from osn_gs.render.surfel_rasterizer import OSNSurfelRasterizer, SurfelRasterizerConfig

    device = model.device
    positions = model.get_xyz.detach()
    count = int(positions.shape[0])
    geometry_hashes_before = {
        "xyz": _tensor_sha256(positions),
        "scaling": _tensor_sha256(model._scaling),
        "rotation": _tensor_sha256(model._rotation),
        "opacity": _tensor_sha256(model._opacity),
    }
    observed_any = torch.zeros((count,), dtype=torch.bool, device=device)
    has_relevant = torch.zeros((count,), dtype=torch.bool, device=device)
    has_unresolved = torch.zeros((count,), dtype=torch.bool, device=device)
    representative_union = torch.zeros((count,), dtype=torch.bool, device=device)

    _progress(f"classifying {count:,} existing Gaussian centres across {len(cameras)} frozen views")
    with torch.no_grad():
        for view_index, camera in enumerate(cameras):
            package = render_with_query_depth_probe(camera, model, query_depths=None)
            median_flat = candidate_b.median_depth_map(package["out_others"]).reshape(-1)
            representative = package["representative_id"].reshape(-1).to(torch.int64)
            representative = representative[representative >= 0]
            if representative.numel():
                representative_union[torch.unique(representative)] = True
            for first in range(0, count, arguments.query_batch_size):
                last = min(first + arguments.query_batch_size, count)
                geometry = project_queries(camera, positions[first:last])
                states = candidate_b.classify_view(geometry, median_flat)["states"]
                observed_any[first:last] |= states == STATE_OBSERVED
                has_relevant[first:last] |= states != STATE_NON_RELEVANT
                has_unresolved[first:last] |= states == STATE_UNRESOLVED
            del package, median_flat
            if (view_index + 1) % 10 == 0 or view_index + 1 == len(cameras):
                _progress(f"classified view {view_index + 1}/{len(cameras)}")

        global_state = global_state_from_accumulators(observed_any, has_relevant, has_unresolved)
        colours = state_colours(global_state)
        if int(colours.shape[0]) != count:
            raise AssertionError("state colouring changed the Gaussian row count")

        rasterizer = OSNSurfelRasterizer(SurfelRasterizerConfig())
        review_candidates = novel_inspection_candidates(cameras, positions)

        # The temporary override changes ONLY appearance tensors in memory. It
        # is restored byte-for-byte before this process exits; checkpoint and
        # all geometry/opacity tensors remain untouched.
        original_dc = model._features_dc.detach().clone()
        original_rest = model._features_rest.detach().clone()
        original_degree = model.active_sh_degree
        try:
            model._features_dc.copy_(_rgb_to_f_dc(colours).unsqueeze(1))
            model._features_rest.zero_()
            model.active_sh_degree = 0
            candidate_renders = [(camera, rasterizer.render(camera, model)["render"]) for camera in review_candidates]
            preview_camera, classified_render = max(candidate_renders, key=lambda entry: red_dominant_pixel_count(entry[1]))
            red_candidate_scores = {
                str(camera.image_name): red_dominant_pixel_count(render)
                for camera, render in candidate_renders
            }
        finally:
            model._features_dc.copy_(original_dc)
            model._features_rest.copy_(original_rest)
            model.active_sh_degree = original_degree
        _progress(f"rendering fixed pair from novel inspection camera {preview_camera.image_name}")
        original_render = rasterizer.render(preview_camera, model)["render"]

    geometry_hashes_after = {
        "xyz": _tensor_sha256(model.get_xyz),
        "scaling": _tensor_sha256(model._scaling),
        "rotation": _tensor_sha256(model._rotation),
        "opacity": _tensor_sha256(model._opacity),
    }
    if geometry_hashes_before != geometry_hashes_after:
        raise AssertionError("a protected Gaussian tensor changed during visualization export")

    original_folder = arguments.out / "ORIGINAL_SCENE"
    classified_folder = arguments.out / "OBSERVED_OCCLUDED"
    original_ply = original_folder / _ITERATION_DIR / "point_cloud.ply"
    classified_ply = classified_folder / _ITERATION_DIR / "point_cloud.ply"
    write_ppm(original_folder / "render.ppm", original_render)
    write_ppm(classified_folder / "render.ppm", classified_render)
    written_original = write_surfel_ply(
        original_ply, model.get_xyz, model._features_dc[:, 0, :], model._opacity.reshape(-1), model._scaling, model._rotation,
    )
    written_classified = write_surfel_ply(
        classified_ply, model.get_xyz, _rgb_to_f_dc(colours), model._opacity.reshape(-1), model._scaling, model._rotation,
    )
    if written_original != count or written_classified != count:
        raise AssertionError("PLY export did not retain every checkpoint Gaussian")

    _write_view_readme(original_folder, """# ORIGINAL_SCENE

학습된 2DGS Gaussian **전부**를 renderer query에 쓰지 않은 novel outer-orbit inspection camera에서 원래 학습된 SH appearance로 렌더링했다. 색상 부호화·추가 Gaussian·광원·overlay가 없다.
""", count)
    _write_view_readme(classified_folder, """# OBSERVED_OCCLUDED

`ORIGINAL_SCENE`와 정확히 같은 Gaussian 행을 renderer query에 쓰지 않은 novel outer-orbit inspection camera에서 렌더링했다. 위치·shape·opacity는 원본 그대로이며, 오직 Gaussian 색상만 Worklog 123 frozen global query state로 교체했다.

- 초록: global `OBSERVED`
- 빨강: global `OCCLUDED`
- 회색: global `UNRESOLVED`

이는 Gaussian 중심의 arbitrary world-space 분류이다. 색상은 물리적 first hit, surface ownership, trust, 연속성, 또는 독립 hidden-surface evidence를 뜻하지 않는다.
""", count)

    state_np = global_state.detach().cpu().numpy().astype(np.int8)
    np.savez_compressed(
        arguments.out / "gaussian_center_global_states.npz",
        global_state=state_np,
        observed_any=observed_any.detach().cpu().numpy(),
        has_relevant=has_relevant.detach().cpu().numpy(),
        has_unresolved=has_unresolved.detach().cpu().numpy(),
    )
    state_counts = {
        "OBSERVED": int((state_np == STATE_OBSERVED).sum()),
        "OCCLUDED": int((state_np == STATE_OCCLUDED).sum()),
        "UNRESOLVED": int((state_np == STATE_UNRESOLVED).sum()),
    }
    report: dict[str, Any] = {
        "batch": "Worklog 127 — novel-view fixed Gaussian visualization of Worklog 123 query contract",
        "checkpoint": str(arguments.checkpoint),
        "camera_meta": camera_meta,
        "preview_camera": str(preview_camera.image_name),
        "novel_inspection_camera": {
            "not_in_frozen_query_camera_set": True,
            "construction": "fixed 8 azimuth x 2 elevation outer-orbit candidates; selected only for visible red review area",
            "red_dominant_pixel_scores": red_candidate_scores,
        },
        "classification": {
            "query": "each existing checkpoint Gaussian centre as arbitrary world-space x",
            "renderer_event_provenance": "absent for every centre; no identity is invented",
            "per_view": "frozen Worklog 123 Candidate B classify_view, stored canonical median depth",
            "global": "frozen ANY-OBSERVED aggregation, streamed exactly",
            "epsilon_or_boundary_policy": "none",
            "state_counts": state_counts,
        },
        "fixed_visualization_contract": {
            "views": ["ORIGINAL_SCENE", "OBSERVED_OCCLUDED"],
            "same_gaussian_count": written_original == written_classified == count,
            "gaussian_count": count,
            "colour_only_override": True,
            "marker_gaussians_added": 0,
            "lighting_added": False,
            "overlay_added": False,
            "protected_geometry_hashes": geometry_hashes_after,
            "protected_geometry_unchanged": geometry_hashes_before == geometry_hashes_after,
        },
        "cross_check": {
            "renderer_median_representative_union": int(representative_union.sum().item()),
            "worklog_119_123_expected_union": 785937,
        },
        "outputs": {
            "ORIGINAL_SCENE": {"render_ppm": str(original_folder / "render.ppm"), "point_cloud_ply": str(original_ply)},
            "OBSERVED_OCCLUDED": {"render_ppm": str(classified_folder / "render.ppm"), "point_cloud_ply": str(classified_ply)},
        },
        "elapsed_seconds": time.time() - started,
    }
    (arguments.out / _REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(f"complete in {report['elapsed_seconds']:.1f}s; states={state_counts}")


if __name__ == "__main__":
    main()
