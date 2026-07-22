# Worklog 59: Baseline 10k A/B를 위한 Graphdeco held-out split/해상도 로직 vendoring

날짜: 2026-07-22

상태: **vendoring 완료 및 실측 검증 완료. 실제 10k A/B 학습 실행은 아직 하지 않음.**

## 배경

`TODO.md` 최상단 항목("baseline 3DGS 대비 Scene 품질 격차 — 남은 동해상도 A/B 검증")을 수행하기 전에 두 가지 공정성 문제를 확인했다:

1. OSN-GS의 데이터 로더(`osn_gs/data/colmap_scene.py`, `torch_scene.py`)에는 baseline의 `--eval`/`llffhold` 같은 held-out test camera 분리 개념이 아예 없다 — 전체 카메라를 다 학습에 쓴다.
2. baseline은 원본 폭이 1600px를 넘으면 `orig_w/1600` 배율로 연속적으로(포터블 정수 아님) 자동 축소하는데, OSN-GS는 `--low_vram`(고정 2배) 또는 정수 `image_downscale`만 지원해서 두 프레임워크의 실제 학습 해상도를 정확히 맞추기 어려웠다.

사용자 지시: 이 로직을 직접 재구현하지 말고, 이미 rasterizer를 vendoring한 것과 같은 방식으로 baseline 코드를 그대로 가져오되(bit-identical 보장), 그 사실을 명시적으로 문서화해둘 것.

## 구현

### `osn_gs/data/vendor/graphdeco_scene_split.py` (신규, license header 보존)

`gaussian-splatting/scene/dataset_readers.py`의 `readColmapSceneInfo` 중 `llffhold` 분기와, `gaussian-splatting/utils/camera_utils.py`의 `loadCam` 중 해상도 결정 분기를 **그대로** 포팅했다(제어 흐름 변경 없음, upstream의 `CameraInfo`/`args` 객체 대신 평범한 값(이름 리스트/너비/높이)을 받는 독립 함수로만 재포장).

- `select_llff_holdout_test_names(image_names, scene_path=None, eval=True, llffhold=8)`: 정렬된 이름 리스트에서 `idx % llffhold == 0`인 이름을 held-out으로 선택. `"360" in path`일 때 llffhold=8로 강제하는 upstream 고유의 mip-360 휴리스틱까지 그대로 포함.
- `resolve_graphdeco_resolution(orig_w, orig_h, resolution=-1, resolution_scale=1.0)`: `resolution in (1,2,4,8)` 명시적 분기와 `resolution=-1`(자동, `orig_w>1600`이면 `orig_w/1600`배 축소) 분기를 그대로 포팅, `(width, height, scale)` 반환.

### `osn_gs/data/colmap_scene.py`

- `load_image_tensor`에 `target_size` 파라미터 추가 — upstream이 계산한 `(width, height)`를 그대로 resize target으로 써서, `downscale` 배율을 다시 나눠 계산할 때 생길 수 있는 반올림 오차(1px)를 없앴다.
- 신규 `load_colmap_scene_with_eval_split(...)`: 위 두 벤더 함수를 사용해 train-only `TorchScene`과 held-out test camera/image 리스트를 분리해서 반환하는 `EvalSplitScene` dataclass. 기존 `load_colmap_scene`(분리 없음)은 변경하지 않고 그대로 유지 — 일반 학습 경로는 영향 없음.

## 검증

- **분리 리스트가 upstream과 정확히 일치하는지** 실제 `DATASET/`(185장)에서 직접 확인: vendored 함수 출력과 `gaussian-splatting/scene/dataset_readers.py`의 실제 `readColmapSceneInfo`를 그 자리에서 import해 나란히 실행 → held-out 24장의 파일명 리스트가 **완전히 동일**(`EXACT MATCH: True`).
- **해상도도 upstream의 실제 계산과 동일**: 원본 5187×3361 → 양쪽 다 `(1600, 1036)`, `scale=3.241875`.
- `load_colmap_scene_with_eval_split('DATASET', device='cpu')` 전체 파이프라인 스모크 실행: train=161, test=24(합 185, 정확히 분리), 반환된 이미지 텐서 shape `(3, 1036, 1600)` 확인.
- 신규 단위테스트 `tests/test_colmap_scene_vendor.py`(8개, 정렬 후 n번째 선택/eval=False/기본 llffhold=8/"360" 휴리스틱/자동축소/축소없음/명시적 배율/resolution_scale 복합)와 전체 스위트 `python -m unittest discover -s tests -p "test_*.py"` → **158 passed, 1 skipped**(기존 150 + 8 신규, 회귀 없음).

## 문서화

`AGENTS.md`의 "Current Rendering Structure" 뒤에 "Vendored Baseline Scene-Split Logic" 절을 추가해, 이 코드가 upstream에서 그대로 가져온 것이며 `gaussian-splatting/`은 런타임에 전혀 import/수정되지 않는다는 점을 명시했다.

## 남은 사항 (이번 세션에서 하지 않음)

- 학습 루프(`train.py`/`scripts/train_osn_gs_torch.py`)를 이 `load_colmap_scene_with_eval_split`을 쓰도록 연결하는 작업.
- held-out camera에 대한 PSNR/SSIM 사후 평가 헬퍼(체크포인트 렌더링 → 정답 이미지와 비교) — 기존 `osn_gs/losses/torch_losses.py::ssim`, `osn_gs/utils/torch_ops.py::psnr_from_mse`를 재사용할 수 있음, 아직 미구현.
- 실제 10k iteration OSN-GS/Graphdeco A/B 실행 자체 — 인프라(CUDA GPU, 데이터셋, 양쪽 코드베이스)는 모두 준비/검증됐지만 아직 실행하지 않음.
