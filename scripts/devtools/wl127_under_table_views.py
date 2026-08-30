"""Worklog 127 supplemental -- under-table camera views for the evidence-bounded
TSDF Visible Surface (Arm B only).

The dataset's 161 training cameras never look up from underneath the table (a
common capture-rig blind spot), so the six preview cameras the main driver
picks are all downward/eye-level looking views. This script builds two
synthetic look-at cameras positioned near floor height inside the table's leg
footprint, looking upward at the tabletop underside, and renders the SAME
cached Arm B mesh (`mesh.npz`, unchanged, no recomputation) through those new
cameras using the same z-buffered mesh rasterizer as the main driver.

No geometry is recomputed: `field.npz`/`mesh.npz` are loaded read-only from
the existing `--cache` directory, and the report's already-measured numbers
are reused verbatim in the README. This is a targeted diagnostic render, not
a rerun of the batch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from maximal_visible_connectivity_export import load_all_train_cameras  # noqa: E402
from osn_gs.render.torch_fallback import TorchCamera  # noqa: E402
from evidence_bounded_tsdf import mesh_ops  # noqa: E402
import evidence_bounded_tsdf_stages as stages  # noqa: E402

LOW_SUPPORT_COUNT = 1


def _progress(message: str) -> None:
    print(f"[wl127-under-table] {message}", flush=True)


def build_lookat_camera(
    eye: torch.Tensor, target: torch.Tensor, *, up_hint: torch.Tensor,
    image_height: int, image_width: int, fovx: float, fovy: float, device: str, name: str,
) -> TorchCamera:
    """A synthetic camera in the SAME world-to-camera convention as
    `osn_gs.data.colmap_scene.camera_matrices`: camera looks down +Z in
    camera space, world_view_transform stores the transposed [R|t] so that
    `x_cam = R @ x_world + t`, and full_proj_transform = world_view @ projection
    (both already transposed for row-vector convention, matching the loader)."""

    from osn_gs.data.colmap_scene import projection_matrix

    forward = (target - eye)
    forward = forward / forward.norm()
    right = torch.linalg.cross(forward, up_hint)
    right = right / right.norm()
    true_up = torch.linalg.cross(right, forward)
    true_up = true_up / true_up.norm()
    # Camera space: x = right, y = -true_up (image y grows downward), z = forward.
    rotation = torch.stack([right, -true_up, forward], dim=0)  # (3,3), rows = camera axes in world
    translation = -rotation @ eye

    world_view_np = torch.eye(4, dtype=torch.float32)
    world_view_np[:3, :3] = rotation
    world_view_np[:3, 3] = translation
    world_view = world_view_np.transpose(0, 1).contiguous().to(device)
    projection = projection_matrix(0.01, 100.0, fovx, fovy, device=device).transpose(0, 1).contiguous()
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    center = eye.to(device)

    return TorchCamera(
        image_height=image_height, image_width=image_width,
        world_view_transform=world_view, full_proj_transform=full_projection,
        camera_center=center, FoVx=fovx, FoVy=fovy, image_name=name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, type=Path, help="the WL127 run's --cache directory (reads mesh.npz, field.npz)")
    parser.add_argument("--report", required=True, type=Path, help="the WL127 run's report JSON (numbers only, read-only)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--images", default="images_8")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--llffhold", type=int, default=8)
    arguments = parser.parse_args()

    device = arguments.device
    output_root: Path = arguments.out
    preview_root = output_root / "preview_png"
    preview_root.mkdir(parents=True, exist_ok=True)

    report = json.loads(arguments.report.read_text(encoding="utf-8"))

    _progress("loading one training camera for intrinsics (image size / FoV) only")
    cameras, _meta = load_all_train_cameras(
        arguments.source_path, arguments.images, arguments.sparse_dir,
        arguments.resolution, arguments.llffhold, device,
    )
    reference = cameras[0]

    _progress(f"loading cached mesh from {arguments.cache / 'mesh.npz'}")
    mesh_cached = np.load(arguments.cache / "mesh.npz", allow_pickle=True)
    vertices = mesh_cached["vertices"]
    faces = mesh_cached["faces"]
    vertex_support_count = mesh_cached["vertex_support_count"]
    vertices_gpu = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_gpu = torch.tensor(faces, dtype=torch.int64, device=device)
    _progress(f"  mesh: {vertices.shape[0]:,} vertices, {faces.shape[0]:,} faces")

    # ---- table geometry, established by direct inspection of mesh.npz:
    # camera centers converge near world (0.089, 0.105, 0.056); within a small
    # radius of that point the densest horizontal vertex band sits at
    # y in [1.30, 1.55] (tabletop) with table legs continuing to y ~ 0.55.
    # This scene's COLMAP convention has +Y pointing DOWN (camera "up" vectors
    # point toward -Y and camera centers range y in [-2.67, 2.94], consistent
    # with downward-looking capture cameras sitting at larger +Y). So "looking
    # up at the underside" means placing the eye at LARGER y (near floor,
    # below the tabletop) and looking toward SMALLER y (up toward the
    # tabletop's underside).
    table_center_xz = (0.089, 0.9)
    tabletop_y = 1.4
    eye_y = 1.9  # below tabletop, near floor between the legs
    target = torch.tensor([table_center_xz[0], tabletop_y - 0.05, table_center_xz[1]], device=device)
    up_hint = torch.tensor([0.0, -1.0, 0.0], device=device)  # world "up" (toward the tabletop) in this convention

    synthetic_cameras = [
        build_lookat_camera(
            torch.tensor([table_center_xz[0], eye_y, table_center_xz[1]], device=device), target,
            up_hint=up_hint, image_height=reference.image_height, image_width=reference.image_width,
            fovx=reference.FoVx, fovy=reference.FoVy, device=device, name="UNDER_TABLE_CENTER",
        ),
        build_lookat_camera(
            torch.tensor([table_center_xz[0] + 0.6, eye_y - 0.1, table_center_xz[1] - 0.3], device=device), target,
            up_hint=up_hint, image_height=reference.image_height, image_width=reference.image_width,
            fovx=reference.FoVx, fovy=reference.FoVy, device=device, name="UNDER_TABLE_OBLIQUE",
        ),
    ]

    support_cap = float(np.percentile(vertex_support_count, 95)) if vertex_support_count.size else 1.0
    low = vertex_support_count <= LOW_SUPPORT_COUNT

    def render_mesh_view(name: str, colours: torch.Tensor, body: str, *, shaded: bool = True) -> None:
        folder = output_root / name
        for camera in synthetic_cameras:
            image = mesh_ops.rasterize_mesh_shaded(camera, vertices_gpu, faces_gpu, colours, shaded=shaded)
            array = image.detach().cpu().numpy()
            if camera is synthetic_cameras[0]:
                stages.write_png(folder / "render.png", array)
            stages.write_png(preview_root / f"{name}__{camera.image_name}.png", array)
        footer = (
            "\n---\n"
            f"체크포인트: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt`\n"
            f"본 산출물의 근거: `../evidence_bounded_projective_tsdf_report.json` (수치는 이번 배치에서 재계산하지 "
            "않고 원본 리포트를 그대로 재사용했다)\n"
            "이 뷰들의 카메라는 학습 데이터셋에 없는 **합성(synthetic) look-at 카메라**다 — 161개 학습 뷰 중 "
            "테이블 아래쪽을 보는 카메라가 실제로 존재하지 않아서, 진단 목적으로 테이블 다리 사이 바닥 높이에서 "
            "천장 방향(테이블 상판 밑면)을 올려다보는 각도로 직접 구성했다.\n"
        )
        (folder / "README.md").write_text(body + footer, encoding="utf-8")
        _progress(f"  wrote {name}")

    tsdf_vertex_colours = torch.tensor([0.20, 0.85, 0.40], device=device).reshape(1, 3).expand(
        vertices_gpu.shape[0], 3
    ).contiguous()
    render_mesh_view(
        "UNDER_TABLE_NEW_TSDF_VISIBLE_SURFACE", tsdf_vertex_colours,
        f"""# UNDER_TABLE_NEW_TSDF_VISIBLE_SURFACE

## 색상 의미
- **초록** (`0.20, 0.85, 0.40`) 음영: `NEW_TSDF_VISIBLE_SURFACE`와 동일한 evidence-bounded TSDF 메시(정점 {vertices.shape[0]:,}, 삼각형 {faces.shape[0]:,})를, 학습 데이터셋에 없는 테이블 아래쪽 시점에서 z-buffer로 렌더링한 것
- **어두운 남색 배경**: 메시가 없는 픽셀 -- 이 시점에서는 UNKNOWN(관측 부재)이 background와 시각적으로 구분되지 않으므로, 화면이 비어 있다고 해서 "빈 공간"을 뜻하지 않는다. field 자체가 절대 diffuse/fill되지 않는 sparse authority 계약이기 때문이다

## 이 이미지가 보여주는 것
사용자가 지적한 대로, 161개 학습 카메라 중 테이블 아래(다리 사이, 상판 밑면)를 올려다보는 카메라가 하나도 없다 -- 흔한 캡처 리그의 사각지대다. evidence-bounded TSDF는 렌더러가 실제로 관측한 median depth로만 fuse되므로, 관측이 없는 영역은 authoritative voxel 자체가 존재하지 않고 mesh도 추출되지 않는다. 이 렌더가 정직하게 비어 보인다면 그것은 버그가 아니라 이 아키텍처가 의도한 대로 "모른다"를 hallucination 없이 보여주는 것이다.

## 분석 및 평가
원본 배치 리포트 기준(`../evidence_bounded_projective_tsdf_report.json`): 전체 메시는 연결 성분 {report['reconstruction']['connected_components']:,}개, 총 표면적 {report['reconstruction']['total_surface_area']:.1f}, renderer median event의 {report['renderer_evidence_reproduction']['all_events']['fraction_within_h']:.2%}가 표면 h 이내, raycast hit coverage {report['raycast_self_consistency']['ray_hit_coverage']:.2%}. 이 수치들은 161개 학습 뷰 전체에 대한 평균이며, 테이블 아래처럼 애초에 관측이 없는 국소 영역의 부재를 상쇄하지 않는다 -- 오히려 그 평균에 이 빈 영역의 기여가 이미 반영되어 있다(관측이 없으니 그 영역은애초에 authoritative voxel로 카운트되지 않는다). 즉 이 렌더가 비어 보이는 것과 위 커버리지 수치가 높은 것은 모순이 아니다.
""",
    )

    support_colours_full = torch.tensor(
        stages.support_to_rgb(vertex_support_count, support_cap), dtype=torch.float32, device=device
    )
    render_mesh_view(
        "UNDER_TABLE_TSDF_SUPPORT_COUNT", support_colours_full,
        f"""# UNDER_TABLE_TSDF_SUPPORT_COUNT

## 색상 의미
- **빨강 → 청록** 램프(메시 전체를 z-buffer로 렌더링): 각 정점 위치 voxel의 support_count. 빨강 = 1개 뷰, 청록 = {support_cap:.0f}개 이상
- **어두운 남색 배경**: 메시가 없는 픽셀

## 이 이미지가 보여주는 것
`TSDF_SUPPORT_COUNT`와 동일한 진단을, 테이블 아래쪽 시점에서 본 것이다. 이 각도에서 보이는 표면이 있다면(예: 테이블 다리, 상판 가장자리처럼 다른 학습 뷰에서도 스치듯 관측된 부분) 그 support_count 색으로 "몇 개 뷰가 그 지점을 지지했는지"를 바로 확인할 수 있다.

## 분석 및 평가
원본 배치 기준 authoritative voxel의 {report['field']['fraction_support_count_1']:.2%}가 support_count = 1이고 평균 support는 {report['field']['support_count_distribution']['mean']:.2f}다(`field`). 테이블 아래에서 보이는 파편(있다면)은 대부분 이 저-support 꼬리에 속할 가능성이 높다 -- 테이블 아래를 정면으로 본 카메라가 없으므로, 보이는 조각은 옆/위 시점에서 스치듯 관측된 것뿐이기 때문이다.
""",
        shaded=False,
    )

    low_vertex_mask = torch.tensor(low, device=device)
    low_colours = torch.where(
        low_vertex_mask.unsqueeze(1),
        torch.tensor([0.95, 0.25, 0.75], device=device).reshape(1, 3),
        torch.tensor([0.18, 0.19, 0.22], device=device).reshape(1, 3),
    )
    render_mesh_view(
        "UNDER_TABLE_TSDF_LOW_SUPPORT_SURFACE", low_colours,
        f"""# UNDER_TABLE_TSDF_LOW_SUPPORT_SURFACE

## 색상 의미
- **자홍** (`0.95, 0.25, 0.75`): support_count <= {LOW_SUPPORT_COUNT}인 삼각형
- **어두운 회색** (`0.18, 0.19, 0.22`): 같은 메시의 나머지(support 충분) 부분

## 이 이미지가 보여주는 것
`TSDF_LOW_SUPPORT_SURFACE`를 테이블 아래쪽 시점에서 본 것이다. **hallucination 후보 검토용**이며 삭제하지 않고 그대로 내보낸다.

## 분석 및 평가
원본 배치 기준 support<=1 표면은 삼각형 수로 {report['hallucination_audit']['fraction_low_support']:.2%}, sampled point 기준 {report['hallucination_audit']['low_support_sampled_points']:,}/{report['hallucination_audit']['sampled_mesh_points']:,}점을 차지한다(`hallucination_audit`). 테이블 아래 시점에서 자홍 비중이 유독 높게 보인다면, 그 영역이 전역 평균보다 훨씬 얇은 증거 위에 서 있다는 뜻이므로 사람이 직접 판단해야 한다.
""",
        shaded=False,
    )

    _progress(f"done -- 3 views x 2 cameras written under {output_root}")


if __name__ == "__main__":
    main()
