# Worklog 77-B: anisotropy 격차 parity ablation — 초기 covariance·position LR·ADC gradient lifecycle 판정

날짜: 2026-07-23

상태: **3개 3k ablation 완료. Production 기본값은 유지. 초기 covariance와 ADC gradient lifecycle은 주원인으로 기각했고, position LR scale은 anisotropy를 크게 좌우하지만 camera extent로의 단순 교체는 held-out 품질을 악화시켰다.**

## 목적과 공통 조건

Worklog 76의 세 후보를 하나씩만 바꾼 실제 DATASET 3,000 iteration A/B로 검증했다.

- 공통: `--eval --llffhold 8 --resolution -1 --no-low_vram`, train 161 / held-out 24 cameras, effective resolution `(1600, 1036)`.
- 기준 OSN-GS: point-cloud `scene_extent` position LR, 1-NN covariance, ADC survivor gradient 보존.
- baseline PLY: `output/graphdeco_ab_3k/point_cloud/iteration_3000/point_cloud.ply`.
- 모든 OSN-GS ablation에서 ADC 100-iteration window마다 `grad_source=screen:100 fallback:0 unavailable:0`이었다. 이번 비교에서 xyz-gradient fallback은 실제로 발동하지 않았으므로 anisotropy 격차의 원인이 아니다.

## 구현한 ablation interface

- `--covariance_init graphdeco_knn`: Graphdeco `simple-knn::distCUDA2`와 같은 최근접 3개 squared distance 평균을 chunked torch path로 계산한다. 기존 `knn`은 1-NN 기본값으로 유지한다.
- `--position_lr_extent_mode calibration`: ADC calibration은 그대로 두고 xyz position LR scale만 camera-based `calibration_extent`로 바꾼다. 기본 `scene`은 기존 point-cloud robust extent다.
- `--adc_drop_survivor_gradients`: ADC shape replacement 뒤 survivor parameter gradient를 버려 Graphdeco lifecycle을 재현한다. 기본은 기존 OSN-GS처럼 보존한다.
- ADC 로그는 `clone_parents`, `split_parents`, `split_children`, 각 parent group anisotropy, screen/fallback gradient source를 분리해 기록한다.

## 결과

저장 시점은 세 프레임워크 모두 iteration 3000의 ADC/reset 이전 상태다.

| 실행 | Gaussian 수 | anisotropy mean | 중앙값 | p90 | 최소축 중앙값 | held-out PSNR | held-out SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1,822,948 | 15.0987 | 5.3921 | 19.2291 | 0.003339 | - | - |
| OSN 기준 (1-NN, scene LR) | 2,100,604 | 4.7986 | 3.3194 | 8.4219 | 0.005367 | 7.9819 | 0.1202 |
| `graphdeco_knn` | 1,920,305 | 4.8599 | 3.4543 | 8.6064 | 0.005784 | 8.0921 | 0.1279 |
| `calibration` position LR | 1,861,007 | 10.7615 | 4.6834 | 16.4903 | 0.003617 | 7.8460 | 0.1114 |
| `adc_drop_survivor_gradients` | 2,094,169 | 4.7848 | 3.3361 | 8.4462 | 0.005370 | 7.9920 | 0.1214 |

## 판정

### 1. 초기 covariance 1-NN/3-NN

`graphdeco_knn`은 초기 covariance parity와 Gaussian 수를 baseline 쪽으로 이동시키고 held-out PSNR/SSIM을 각각 `+0.1102` / `+0.0077` 개선했다. 그러나 anisotropy 중앙값은 `3.3194 -> 3.4543`, p90은 `8.4219 -> 8.6064`로 미세한 변화에 그쳤다.

따라서 1-NN/3-NN 불일치는 실제 parity 차이지만 **anisotropy 격차의 주원인은 아니다**. 기본값을 즉시 변경하지 않고 명시적 ablation mode로 유지한다. 향후 quality recipe 선택에서는 별도 반복 seed/장기 run으로 작은 held-out 개선의 재현성을 확인해야 한다.

### 2. point-cloud scene LR vs camera calibration LR

position LR만 `calibration_extent`로 바꾸면 최소축 중앙값이 `0.005367 -> 0.003617`로 baseline `0.003339`에 가까워지고 anisotropy p90도 `8.4219 -> 16.4903`까지 크게 상승했다. 즉 **position LR scale이 최소축 contraction과 anisotropy의 강한 제어 변수**라는 점은 확인됐다.

하지만 held-out PSNR/SSIM은 각각 `-0.1359` / `-0.0088` 악화됐다. 단순 camera extent 복귀는 anisotropy를 맞추는 대신 generalization을 악화시키므로 production 기본값으로 채택하지 않는다. Worklog 72의 dual-scale 설계를 바로 되돌릴 근거도 없다.

### 3. ADC survivor gradient lifecycle

Graphdeco처럼 ADC iteration의 survivor gradient를 버려도 anisotropy 중앙값/p90은 `3.3361 / 8.4462`로 기준과 사실상 같고 held-out도 `+0.0101 / +0.0012` 수준이다. 이 차이는 주원인으로 기각한다. OSN-GS 기본의 gradient 보존 정책은 유지한다.

## 해석

이번 결과는 “baseline과 같은 anisotropy가 곧 더 좋은 held-out 품질”이라는 가정을 반박한다. camera-based position LR은 baseline과 비슷한 flat Gaussian을 만들지만 이 scene의 held-out 성능은 더 낮다. 반대로 OSN-GS의 point-cloud scene LR은 anisotropy가 작아도 held-out에서는 더 낫다.

따라서 남은 문제는 baseline 수치를 기계적으로 복제하는 것이 아니라, **position LR scale이 world-space 위치 수렴·view-dependent scale-axis gradient·ADC parent selection을 어떻게 함께 바꾸는지**를 분리하는 것이다. 다음 계측은 per-axis scaling gradient의 sign/quantile, position update norm, parent age/clone lineage를 동일 camera sequence에서 기록해야 한다. 새 regularizer나 기본값 변경은 이 계측 전에는 도입하지 않는다.

## 검증

- `tests.test_training_regressions`: `24 passed`.
- 전체: `python -B -m unittest discover -s tests -p "test_*.py"` -> `233 passed, 1 skipped`.
- 세 3k output과 held-out JSON:
  - `output/osn_gs_ab_3k_graphdeco_knn/`
  - `output/osn_gs_ab_3k_calibration_lr/`
  - `output/osn_gs_ab_3k_drop_adc_gradients/`

## 남은 위험

- 각 실행은 단일 deterministic recipe/run이다. 작은 held-out 차이는 seed 반복 없이 채택 근거가 될 수 없다.
- 3k는 screen-size pruning이 아직 실제로 발동하지 않는 구간이므로 장기 안정성은 판정하지 못한다.
- baseline과 OSN-GS의 view sampling order가 동일하지 않으므로 절대 수치의 완전한 bitwise parity가 아니라, 코드-path ablation의 방향성을 해석한 결과다.
