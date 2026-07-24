# Worklog 72: scene_extent 이원화 — point cloud 기반 유지 + camera 기반 calibration 분리

날짜: 2026-07-23

상태: **구현 완료, 실제 학습 검증은 다음 단계.**

## 배경

Worklog 71에서 확인한 근본 원인: OSN-GS의 `_scene_extent()`(point cloud 기반)와 baseline의 `cameras_extent`(camera 위치 기반)가 이 데이터셋에서 2.5배 차이 나고, baseline에서 그대로 가져온 ADC 상수(`percent_dense`, `max_scale_ratio`)가 point cloud 기반 값에 적용되면서 Gaussian scale이 baseline보다 20~50% 크게 나왔다(iteration 3000 실측).

사용자 결정: **point cloud 기반 로직은 버리지 않고 유지**하되, 파이프라인을 수정해서 문제를 해결한다. 단순히 camera 기반으로 되돌리면 baseline을 복제하는 것뿐이라 논문에 쓸 내용이 없지만, point cloud 기반을 유지한 채 제대로 보정하면 "3DGS의 camera-trajectory 기반 scene-scale 추정이 walkthrough형 캡처에서 scene scale을 과소평가한다"는 방법론적 주장이 가능해진다.

## 구현: 두 개의 서로 다른 "scene scale" 질문을 분리

- **`scene_extent`** (point cloud 기반, 기존 `_scene_extent()` 유지): "실제로 관측한 것들이 얼마나 멀리 떨어져 있는가" — position 학습 스텝 크기(`spatial_lr_scale`)에 계속 사용.
- **`calibration_extent`** (신규 `_calibration_extent()`, camera 위치 기반, baseline `getNerfppNorm` verbatim port): "어느 정도 크기의 splat이 '너무 크다'고 볼 기준인가" — baseline이 `percent_dense`/`max_scale_ratio` 상수를 튜닝할 때 전제한 바로 그 값. ADC의 clone-vs-split 임계값과 world-size pruning 임계값에 이제 이걸 쓴다.

### 변경 파일

- `osn_gs/data/vendor/graphdeco_scene_split.py`: `estimate_camera_extent(camera_centers)` 추가 — `getNerfppNorm`의 radius 계산(평균 카메라 중심으로부터 최대 거리 × 1.1)을 verbatim port. 기존 held-out split/resolution 로직과 같은 파일에 vendoring 관례대로 추가.
- `osn_gs/core/torch_trainer.py`:
  - `_calibration_extent(scene)` 추가 — `scene.cameras`의 `camera_center`들로 `estimate_camera_extent` 호출.
  - `train()`에서 `scene_extent`와 `calibration_extent`를 각각 계산해 `_train_loop`에 둘 다 전달.
  - `apply_adaptive_density_control(...)` 호출의 세 번째 인자를 `scene_extent` → `calibration_extent`로 교체 (spatial_lr_scale은 그대로 `scene_extent` 사용).
  - `_scene_extent` docstring을 갱신해 이제 ADC 임계값에는 안 쓰인다는 점을 명시.

### 손대지 않은 것

`osn_gs/core/torch_pipeline.py::_scene_scale`(covariance 초기화 max-scale clamp에 쓰이는, bbox 대각선 기반의 별도 함수)은 이번 범위에 포함하지 않았다. 같은 outlier-민감성 문제가 있어 보이지만, pipeline이 카메라 정보 없이 point/color만으로 동작하도록 설계돼 있어 camera 기반 값을 넘기려면 별도 배선이 필요하다. 후속 과제로 남긴다.

## 검증

- 신규 유닛 테스트(`tests/test_colmap_scene_vendor.py::CameraExtentTest`): 수동 계산 대조, 합성 카메라 링으로 실측값(4.94) 재현, 빈 입력 fallback.
- 실제 DATASET으로 `_calibration_extent` 직접 호출 → **4.9229**, baseline이 보고한 `cameras_extent`(4.94)와 사실상 일치 확인.
- 전체 pytest: `208 passed, 1 skipped`. 회귀 없음.
- **실제 재학습으로 Gaussian scale 분포가 baseline과 맞춰지는지, blur가 개선되는지는 아직 확인 안 함** — 다음 단계.

## 다음 작업

1. iteration 3000 A/B를 다시 돌려서 Gaussian scale 분포(mean/median/p90)를 baseline과 재비교.
2. render.ppm 육안 비교로 blur 개선 확인.
3. 개선이 확인되면 held-out PSNR/SSIM으로 정량 확인(단, `--iterations`가 `opacity_reset_interval`의 배수와 겹치지 않게 iteration 수 선택 — TODO.md 최하위 우선순위 항목 참고).

## 채택 결정 및 보류된 대안 (2026-07-23)

이 fix(`scene_extent`는 LR에, camera 기반 `calibration_extent`는 ADC 임계값에 사용)를 **채택**한다.

논의했던 대안: camera 데이터를 아예 안 쓰고, point cloud의 전역(global) percentile 대신 **각 Gaussian 주변의 local nearest-neighbor 간격**으로 ADC 임계값을 대체하는 방향. 하지만 이건 raw point cloud KNN 기반으로 하면 3DGS 계열에서 이미 나온 일반적인 ADC 개선이라 OSN-GS 고유의 기여로 보기 어렵다고 판단해 보류했다.

**단, 이걸 OSN-GS가 이미 만들고 있는 NURBS surface 표현 자체에서 local scale 기준을 끌어오는 방향으로 바꾸면 얘기가 달라진다** — 예를 들어 각 Gaussian이 속한 patch의 local UV footprint/곡률 기반으로 "이 위치에서 Gaussian이 너무 큰가"를 판단하는 식. 이러면 "이미 구축한 surface 표현을 ADC 쪽에도 활용했다"는 논리가 서서 논문에 넣을 만해진다. 다만 NURBS 파이프라인이 아직 Phase 5(occluded surface)까지 안 끝난 상태라 지금 착수하기엔 이르고, 훨씬 큰 별도 연구 과제다. **나중에 착수할 가능성이 있어 여기 기록만 해두고, 이번엔 진행하지 않는다.**
