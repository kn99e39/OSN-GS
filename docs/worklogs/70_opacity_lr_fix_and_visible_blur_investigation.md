# Worklog 70: opacity_lr baseline parity 수정 + visible Gaussian 블러 원인 조사

날짜: 2026-07-23

상태: **opacity_lr 수정 완료. 블러 원인은 미확정 — 유력 후보 배제만 진행됨.**

## 배경

`osn_gs_ab_10k_fix`(iteration 10000)와 baseline(iteration 6000)의 render.ppm을 직접 비교했을 때, baseline이 더 적은 iteration으로도 훨씬 선명한 반면 OSN-GS는 테이블/화병/나뭇잎 등 **non-occluded(카메라에 직접 보이는) 영역**까지 눈에 띄게 블러가 있었다. 사용자가 지적한 대로, `nurbs_surface_loss`가 이미 Gaussian 위치를 detach하여 visible Gaussian은 baseline처럼 image loss만으로 학습되어야 하므로("occluded surface 배치를 위한 NURBS이지 visible surface 학습에 영향을 주면 안 됨"), NURBS 미완성은 이 블러의 변명이 될 수 없다. 원인을 코드 레벨에서 조사했다.

## 수정: opacity_lr baseline 불일치

`osn_gs/gaussian/torch_model.py::GaussianParameterGroups`와 baseline `arguments/__init__.py`를 항목별로 대조:

| 항목 | baseline | OSN-GS(수정 전) | 결과 |
|---|---:|---:|---|
| position_lr_init/final | 0.00016 / 0.0000016 | 동일 | 일치 |
| feature_lr | 0.0025 | 동일 | 일치 |
| **opacity_lr** | **0.025** | **0.05** | **2배 불일치** |
| scaling_lr | 0.005 | 동일 | 일치 |
| rotation_lr | 0.001 | 동일 | 일치 |

다른 5개 학습률은 소수점까지 정확히 일치하는데 opacity만 정확히 2배였다 — 의도된 설계가 아니라 실수로 판단, `2.5e-2`로 수정했다. 다만 opacity LR은 보통 blur보다는 opacity flicker/불안정을 유발하는 파라미터라, 이것만으로 블러 전체를 설명한다고 보기는 어렵다.

## 배제한 후보들

- `lambda_dssim`(0.2), `percent_dense`(0.01), `densify_grad_threshold`(0.0002), `prune_opacity_threshold`(0.005): baseline과 완전히 동일.
- SSIM 구현(`osn_gs/losses/torch_losses.py` vs `gaussian-splatting/utils/loss_utils.py`): window_size=11, sigma=1.5, 수식까지 동일.
- `antialiasing`: 둘 다 False.
- 초기 opacity(0.12 vs baseline 0.1)와 covariance 초기화(`covariance_scale_multiplier=1.0`, KNN 기반 nearest-neighbor 방식): 구조적으로 동등, 큰 차이 아님.
- `nurbs_surface_loss`의 `xyz.detach()` 확인 — 문서/docstring 의도대로 visible Gaussian 위치에 실제로 gradient가 안 흘러들어감. 설계가 깨진 상태는 아니었다.
- batch_size=1, xyz LR decay `max_steps=30000`: 둘 다 동일.

## ADC clone/split 비율이 비정상적으로 높다는 가설 — 반증됨

이전 worklog(67)에서 OSN-GS의 clone 카운트가 매 ADC step마다 tracked 대비 3~8% 수준으로 계속 높게 유지되는 걸 이상 신호로 의심했었다. 이번에 baseline `densify_and_prune`에 동일한 진단(`grads.norm() >= max_grad`인 비율)을 임시로 추가해 실제 DATASET, 2000 iteration 스모크로 직접 측정했다:

```
tracked=147728 ... over_threshold=27059 (18.317%)
tracked=210414 ... over_threshold=42081 (19.999%)
tracked=302635 ... over_threshold=57297 (18.933%)
...
```

**baseline도 tracked 대비 10~20%가 threshold를 넘는다** — OSN-GS의 3~8%보다 오히려 더 높다. 즉 OSN-GS의 높은 clone/split 카운트는 이상 현상이 아니라 이 데이터셋(고해상도 실제 정원 씬) 자체의 정상적인 ADC 거동이었다. 이 가설은 기각한다. 진단 코드는 측정 후 baseline에서 원복했다.

## 검증

- 전체 pytest: `205 passed, 1 skipped`. opacity_lr 수정에 대한 회귀 없음.
- 실제 재학습으로 블러가 개선되는지는 아직 확인 안 함.

## 남은 후보 / 다음 조사 방향

블러의 결정적 원인은 아직 못 찾았다. 남은 방향:
- opacity_lr 수정 반영 후 실제 A/B 재실행으로 시각적 개선 여부 확인.
- Gaussian이 실제로 baseline만큼 작고 뾰족하게 수렴하는지 — `get_scaling` 분포를 iteration별로 baseline과 직접 비교.
- SH 최고 차수 활성화 시점/렌더링에 실제로 반영되는지(`active_sh_degree` 증가와 `model.get_features` 사용 경로) 확인.
- surface_loss가 detach돼 있어도, 같은 `total.backward()` 호출 안에서 surface/uncertain loss들이 optimizer state(Adam의 exp_avg/exp_avg_sq)에 간접적으로 영향을 주는 경로가 있는지(예: 공유 텐서, in-place 연산) 재확인.
