from __future__ import annotations

"""Project frozen W172 support/boundaries over existing canonical scene renders."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts/devtools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from devtools.demo import worklog_172_fragmentation_gate_vs_structural_fit_audit as w172
from observed_occluded.shared import project_queries

w171 = w172.w171
OUT = ROOT / "output/172_render_view_projection_review"
RENDERS = ROOT / "output/confirmed/164_canonical_renderer_contributor_primitive_observability_historical_candidate_b_controlled_ab"


def project(points, camera):
    geometry = project_queries(camera, torch.as_tensor(points, dtype=torch.float32))
    return np.column_stack((geometry.pixel_x.numpy(), geometry.pixel_y.numpy())), geometry.relevant.numpy()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(out=OUT):
    from matplotlib.collections import LineCollection

    plt = w171._prepare_matplotlib()
    source_paths = sorted(p for p in w172.OUT.rglob("*") if p.is_file())
    source_manifest = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    w171_manifest = w172.frozen_manifest()
    baseline = json.loads((w172.OUT / "worklog_172_report.json").read_text(encoding="utf-8"))
    render_report = json.loads((RENDERS / "worklog_164_report.json").read_text(encoding="utf-8"))
    cameras, camera_metadata = w171.w155._build_named_cameras(w171.DEFAULT_DATASET, "images_8", "sparse/0", -1, 8, "cpu")
    complete = w171._load_real_runtimes(w171.DEFAULT_W154, w171.DEFAULT_W145)
    payloads = {}
    for name in w172.CASES:
        case_dir = w172.OUT / "case_artifacts" / name
        with np.load(case_dir / "selected_support.npz") as data:
            selected, indices = data["world_xyz"].copy(), data["complete_row_indices"].copy()
        full = complete[name].samples.world_xyz.numpy()
        assert np.array_equal(full[indices], selected)
        assert w171._sha256_arrays(complete[name].samples.source_cell_keys.numpy(), full) == baseline["cases"][name]["baseline"]["support"]["hash_after_analysis"]
        excluded = np.ones(len(full), dtype=bool)
        excluded[indices] = False
        with np.load(case_dir / "boundary_chart.npz") as data:
            offsets, vertices = data["loop_offsets"].copy(), data["loop_world_xyz"].copy()
        payloads[name] = (full, excluded, offsets, vertices)
    out.mkdir(parents=True, exist_ok=True)
    accounting = {}
    render_manifest = {}
    for camera_name, camera in cameras.items():
        stem = Path(camera_name).stem
        original_path = RENDERS / "review_views/original_scene" / f"{stem}.png"
        state_path = RENDERS / "review_views/contributor_aware_global_state" / f"{stem}.png"
        for path in (original_path, state_path):
            render_manifest[str(path.relative_to(ROOT))] = sha(path)
        with Image.open(original_path) as image:
            original = np.asarray(image.convert("RGB"))
        with Image.open(state_path) as image:
            state = np.asarray(image.convert("RGB"))
        assert original.shape == state.shape == (camera.image_height, camera.image_width, 3)
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))
        fig.subplots_adjust(left=.015, right=.985, bottom=.11, top=.9, wspace=.035, hspace=.15)
        for ax in axes.flat:
            ax.imshow(original)
            ax.set(xlim=(-.5, camera.image_width-.5), ylim=(camera.image_height-.5, -.5))
            ax.set_axis_off()
        axes[0, 0].set_title("Original Scene — frozen W164 render")
        axes[1, 0].images[0].set_data(state)
        axes[1, 0].set_title("Observed/Occluded — W164 historical state")
        case_accounting = {}
        for row, name in enumerate(w172.CASES):
            full, excluded, offsets, vertices = payloads[name]
            pixels, valid = project(full, camera)
            selected_valid, excluded_valid = valid & ~excluded, valid & excluded
            overview, boundary_view = axes[row, 1], axes[row, 2]
            overview.scatter(*pixels[excluded_valid].T, s=1.1, c=w171.CONTEXT_RGB, alpha=.85, linewidths=0)
            overview.scatter(*pixels[selected_valid].T, s=.5, c=w171.SUPPORT_RGB, alpha=.35, linewidths=0)
            boundary_view.scatter(*pixels[selected_valid].T, s=.5, c=w171.SUPPORT_RGB, alpha=.20, linewidths=0)
            loop_pixels, loop_valid = project(vertices, camera)
            segments = []
            total_segments = 0
            for start, end in zip(offsets[:-1], offsets[1:]):
                ids = np.arange(start, end)
                next_ids = np.roll(ids, -1)
                segment_valid = loop_valid[ids] & loop_valid[next_ids]
                total_segments += len(ids)
                segments.extend(np.stack((loop_pixels[ids[segment_valid]], loop_pixels[next_ids[segment_valid]]), axis=1))
            boundary_view.add_collection(LineCollection(segments, colors=w171.BOUNDARY_RGB, linewidths=.4, alpha=.8))
            title = "Tabletop" if row == 0 else "Curved/vase target"
            overview.set_title(f"{title}: complete support + diagnostic component\n{selected_valid.sum():,} selected / {excluded_valid.sum():,} other rows in view")
            loop_count = len(offsets)-1
            boundary_view.set_title(f"{title}: selected support + chart boundary\n{loop_count:,} loops; NURBS not materialized")
            case_accounting[name] = {"complete_count": len(full), "selected_projected": int(selected_valid.sum()),
                "excluded_projected": int(excluded_valid.sum()), "outside_frustum_or_near": int((~valid).sum()),
                "loop_count": loop_count, "boundary_segments": total_segments, "boundary_segments_projected": len(segments),
                "display_subsampling": False, "depth_visibility_culling": False}
        fig.suptitle(f"W172 — {camera_name} — canonical renderer pixel projection", fontsize=13)
        fig.text(.5, .06, "Cyan: diagnostic largest component   |   Gray: other W171 components   |   Orange: chart-plane boundary", ha="center", fontsize=10)
        fig.text(.5, .025, "Reference state: green=OBSERVED, red=OCCLUDED, gray=UNRESOLVED (W164). Overlays are not depth-culled or visibility labels.", ha="center", fontsize=9)
        w171._save_figure(fig, out / f"{stem}.png")
        plt.close(fig)
        accounting[camera_name] = {"camera": camera_metadata[camera_name], "cases": case_accounting}
    assert source_manifest == {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    assert w171_manifest == w172.frozen_manifest()
    assert all(sha(ROOT / path) == digest for path, digest in render_manifest.items())
    readme = """# W172 렌더링 뷰 투영 검토

각 `<camera_name>.png`는 동일한 fixed camera의 2x3 panel이다. 왼쪽 위는 Original Scene, 왼쪽 아래는 기존 W164 Observed/Occluded다. 가운데 위/아래는 각각 W172 tabletop/curved-vase complete support와 diagnostic largest component의 투영, 오른쪽은 해당 선택 component와 모든 ordered chart boundary의 투영이다.

입력 geometry는 W171/W172에서 고정한 zero-set XYZ와 저장된 boundary_chart.npz다. Cyan은 진단 component, gray point는 W171 complete support의 나머지 fragment, orange line은 canonical chart plane 위의 boundary다. 이 세 색은 visibility나 신뢰도가 아니다. NURBS가 materialize되지 않았으므로 NURBS/control net/residual overlay는 없다.

왼쪽의 mandatory Gaussian pair는 W164의 기존 PNG를 변경 없이 입력으로 사용했다. 원래 checkpoint iteration 30000, 1,190,469 Gaussian rows, learned SH Original Scene, 동일 camera/648x420 calibration, OSNSurfelRasterizer, black background다. State panel의 green=(0.10,0.85,0.35) OBSERVED, red=(0.92,0.18,0.18) OCCLUDED, gray=(0.60,0.60,0.62) UNRESOLVED는 W164 contributor-aware historical state다. W172의 새 판정이나 physical blocking certificate가 아니다. Gaussian geometry/appearance를 재렌더하거나 바꾸지 않았다. 각 panel은 동일 전체 image extent를 사용하고 white figure margin만 추가했다.

투영은 canonical renderer와 동일한 `project_queries`를 사용한다. `pixel=((ndc+1)*size-1)/2`이며 y를 뒤집지 않는다. 기존 W155/W171 RGB-overlay helper의 y-flip 및 `(size-1)` mapping과 다르므로 이번 렌더 투영에는 그 helper를 사용하지 않는다. W171 원본 그림/코드/결과는 수정하지 않았다. Camera calibration은 기존 fixed DSC07960/DSC08003/DSC08043을 사용했다.

모든 support row를 투영하고 canonical frustum/near-plane 밖 row만 표시하지 않는다. Point subsampling이나 component 삭제가 없다. Boundary segment는 양 끝 vertex가 유효한 경우만 그리므로 image edge/near plane을 가로지르는 일부 segment는 표시되지 않을 수 있다. Scene depth와 비교하는 occlusion culling은 없으며, 앞 물체 위로 뒤쪽 support가 겹쳐 보일 수 있다. Orange boundary는 planar chart의 투영이지 실제 physical surface edge가 아니다. Curved/vase는 기존 case 이름이며 vase 자체의 분리를 보장하지 않는다.

PNG는 secondary image-space review다. W172의 common-world view와 함께 판단한다. Support/boundary/fit을 다시 만들지 않았고 W171/W172 baseline 및 기존 render의 실행 전후 hash를 보존했다. Source 경로/hash, checkpoint 정보, 카메라별 투영 개수는 `projection_report.json`에 기록한다. GPU 작업 및 replay cache 복사는 없다.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    report = {"status": "COMPLETE", "source_w172_files": source_manifest, "source_renders": render_manifest,
              "source_files_unchanged": True, "checkpoint": render_report["inputs"]["checkpoint"],
              "checkpoint_sha256": render_report["inputs"]["checkpoint_sha256"], "iteration": render_report["inputs"]["iteration"],
              "projection": "canonical observed_occluded.shared.project_queries; no y flip; renderer half-pixel convention",
              "camera_accounting": accounting, "png_count": 3, "gpu_rendering": False}
    w171._write_json(out / "projection_report.json", report)
    for stem in (Path(name).stem for name in cameras):
        with Image.open(out / f"{stem}.png") as image:
            image.verify()
    print(json.dumps({"status": report["status"], "png_count": 3, "out": str(out)}))
    return report


if __name__ == "__main__":
    run()
