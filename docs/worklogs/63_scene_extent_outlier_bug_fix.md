# Worklog 63: `_scene_extent`(spatial_lr_scale) outlier bug — 실 데이터에서만 드러난 품질 저하 원인

날짜: 2026-07-22

상태: **root-cause 확인 및 수정 완료. 수정 반영 상태로 실제 10k A/B 재실행 예정.**

## 배경

Worklog 59/60에서 완성한 same-condition A/B 인프라로 실제 `DATASET`(185장, 정원 씬)에서 OSN-GS 10k 학습을 시작했는데, 사용자가 iteration 5000 시점 렌더링(`render.ppm`)을 직접 확인하고 "품질이 심하게 나쁘다"고 지적했다. 로그를 보니 iteration 5000~7000 구간에서 PSNR이 19~22 사이에서 정체돼 있었고, opacity reset 직후 76만 개 중 21만 5천 개가 한 번에 pruning되는 등 뚜렷한 수렴 실패 징후가 있었다. 사용자는 "이 작업(vendoring/eval-split) 시작 전부터 이랬다"고 확인해줘서, 오늘 작성한 신규 로더 코드가 원인이 아니라 기존에 있던 문제임이 확정됐다.

## Root Cause

`osn_gs/core/torch_trainer.py::_scene_extent()`가 `spatial_lr_scale`(xyz position learning rate 배율, 3DGS 표준 설계와 동일하게 `xyz_lr * spatial_lr_scale` 형태로 쓰임)을 계산하는 데, **raw COLMAP sparse point cloud의 bounding-box 대각선 길이**를 그대로 사용하고 있었다:

```python
span = pts.max(dim=0).values - pts.min(dim=0).values
extent = norm(span)
```

실측 (`DATASET`, 138766개 sparse point):

| 지표 | 값 |
|---|---:|
| **버그 버전 `_scene_extent`(bbox 대각선)** | **124.5** |
| centroid 기준 median 거리 | 4.99 |
| centroid 기준 90th percentile 거리 | 11.19 |
| centroid 기준 max 거리 | 61.85 |
| baseline 3DGS의 `cameras_extent`(카메라 위치 기반 radius) | 4.94 |
| 코드베이스에 이미 있던 robust 버전 `estimate_scene_extent`(90th percentile*1.1) | 12.3 |

median 거리는 5.0인데 bbox 대각선은 124.5 — COLMAP SfM reconstruction에 흔히 섞이는 소수의 noisy 원거리 outlier point가 bbox를 왕창 부풀린 것이다. 결과적으로 xyz 위치 learning rate가 **baseline 대비 약 25배**(`124.5/4.94`) 커져서, Gaussian 위치가 매 gradient step마다 과도하게 튀어 안정된 위치로 수렴하지 못하고 그게 그대로 "Gaussian이 70만 개나 있는데도 계속 흐린" 증상으로 나타났다.

**왜 지금까지 안 걸렸나**: 이 함수는 `TorchOSNGSTrainer.train()`이 실제 다중-iteration 학습 루프를 시작하기 직전 딱 한 번만 호출된다. 이번 세션을 포함해 지금까지의 모든 검증은 `nurbs_constructor_benchmark`의 `initialize()`(0회 학습, constructor 정확성만 측정)만 사용했고, 게다가 synthetic oracle-Gaussian 벤치마크 씬에는 COLMAP 특유의 노이즈성 outlier point가 애초에 존재하지 않는다. 즉 실제 COLMAP 데이터로 진짜 학습 루프를 여러 iteration 돌려야만 드러나는, 지금까지의 검증 범위 바깥에 있던 버그였다.

## 수정

`_scene_extent`가 이미 코드베이스에 존재하던 `osn_gs/data/colmap_scene.py::estimate_scene_extent`(mean-center 후 90th percentile 거리 * 1.1)를 재사용하도록 변경. 새 공식을 별도로 만들지 않고 이미 검증된 기존 유틸리티를 그대로 가져다 썼다.

## 검증

- 수정 후 동일 데이터에서 `_scene_extent` 재계산 → **12.31**(기존 124.5에서 정상 범위로 복귀, baseline의 4.94와 같은 자릿수).
- 전체 테스트 스위트 `python -m unittest discover -s tests -p "test_*.py"` → **182 passed, 1 skipped**, 회귀 없음.
- **실제 DATASET, 1000 iteration 스모크 재실행** (수정 전과 동일 조건: `--eval --llffhold 8 --resolution -1 --no-low_vram`):
  - train PSNR **24.454**, held-out PSNR **21.663** / SSIM **0.5653** — 버그 버전이 5000~7000 iteration에서도 PSNR 19~22에 정체돼 있던 것을, 수정 버전은 **1/5~1/7의 iteration만으로 이미 앞질렀다.**
  - 렌더링을 직접 육안으로 비교해도 차이가 뚜렷하다: 버그 버전은 형체만 겨우 보이는 수준이었는데, 수정 버전은 1000 iteration만으로 테이블/화병/마른 꽃/돌바닥/생울타리가 또렷하게 구분된다.

## 남은 사항

- 이 수정을 반영한 상태로 실제 10k iteration OSN-GS/Graphdeco baseline A/B를 재실행할 예정(다음 작업).
- ADC clone 카운트(iteration당 수만 개)가 여전히 baseline 대비 많아 보이는 경향은 있으나, 이번 수정으로 핵심 증상(정체된 PSNR, 시각적 blur)이 해소되는 것을 확인했으므로 별도 후속 조사 없이 A/B를 진행한다. 필요하면 A/B 결과를 보고 추가로 판단한다.
