# Worklog 68: Baseline 1000-iter 저장 + render.ppm 추가, 두 프레임워크 동일 카메라 고정

날짜: 2026-07-23

상태: **완료. 카메라 일치는 데이터 로딩만으로 확인, 전체 학습 재실행은 아직 안 함.**

## 배경

사용자가 OSN-GS의 `render.ppm`(iteration마다 저장되는 미리보기 이미지)을 직접 눈으로 보고 품질을 판단해왔는데, baseline(Graphdeco)은 `scene.save(iteration)`이 point cloud만 저장하고 렌더링 미리보기를 만들지 않았다. 사용자가 baseline도 1000 iteration마다 저장하고, baseline 자체 렌더러로 `render.ppm`을 만들어서 직접 비교할 수 있게 해달라고 요청했다. 추가로, 같은 iteration의 두 결과를 비교하려면 반드시 **같은 카메라 시점**이어야 한다는 지적이 있었다.

## 문제: 카메라가 프레임워크마다 다르게 뽑히고 있었음

- OSN-GS: 기존 코드는 주기적 저장 시 `batch.cameras[0]` — 즉 **그 iteration에 샘플링된 학습 미니배치의 첫 카메라**를 썼다. 미니배치 샘플링이 매 iteration 셔플되므로, iteration 1000과 2000의 미리보기가 서로 다른 카메라일 수 있었다(OSN-GS 자기 자신과도 비교 불가능한 상태).
- Baseline: `scene.getTrainCameras()`는 `Scene.__init__`에서 `random.shuffle()`이 적용된 순서라, 인덱스로 카메라를 고르면 실행마다/프레임워크마다 다른 이미지가 나온다.

## 해결

두 프레임워크 모두 **"학습 카메라를 이름순으로 정렬해서 첫 번째"**라는 동일한 결정론적 규칙으로 고정 미리보기 카메라를 고른다. OSN-GS 로더(`osn_gs/data/colmap_scene.py`)는 애초에 이미지를 이름순으로 로드하고, baseline은 셔플되므로 별도로 이름순 정렬이 필요하다. 두 프레임워크가 정확히 같은 COLMAP 데이터 + 같은 llffhold 분할을 쓰므로, 이 규칙을 각자 독립적으로 적용해도 항상 같은 이미지로 수렴한다.

- `osn_gs/core/torch_trainer.py`: `_preview_camera(scene)` 헬퍼 추가 (`min(scene.cameras, key=lambda c: c.image_name)`). 주기적 저장(`_train_loop` 내 `save_outputs` 호출)과 최종 저장 모두 이 고정 카메라를 쓰도록 변경 — 기존 `batch.cameras[0]`/`scene.cameras[0]` 대체.
- `gaussian-splatting/train.py`: `Scene` 생성 직후 동일한 규칙으로 `preview_camera`를 한 번 계산해 고정. `save_iterations`에 도달할 때마다 `scene.save(iteration)` 다음에 이 카메라로 `render()`를 호출하고 `point_cloud/iteration_N/render.ppm`에 저장하는 `save_render_ppm()` 함수를 추가했다(OSN-GS `_save_ppm`과 동일한 P6 포맷).

## 검증

- 학습 없이 두 로더만 돌려서 이름순 첫 카메라가 실제로 일치하는지 확인: 둘 다 `DSC07957.JPG`.
- `gaussian-splatting/train.py`는 `py_compile`로 문법 확인.
- `osn_gs` 전체 pytest: `205 passed, 1 skipped`, 회귀 없음.
- **전체 10k 학습으로 실제 `render.ppm` 파일들을 끝까지 만들어 본 것은 아직 안 함** — 다음 A/B 실행 때 `--save_iterations 1000 2000 ... 9000`(baseline은 기본값이 `[7000, 30000]`이라 명시적으로 넘겨야 함, `--iterations`는 자동으로 추가됨)을 넘겨서 실행하면 된다.

## 부수 정리

사용자 지시로 이전 baseline 전용 산출물(`output/graphdeco_ab_10k`, `output/graphdeco_ab_10k_log.txt`)을 삭제했다. OSN-GS 쪽 산출물은 그대로 유지.
