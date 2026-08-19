# Worklog 106 — Coverage-first Gaussian Subset partition, checkpoint 정정 재실측

## 상태

**완료 — 이 배치도 architecture 성공/실패 판단을 내리지 않는다.** Worklog 105의 구현(모듈·테스트·export 스크립트)은 전혀 변경하지 않았다. 이 배치는 **checkpoint를 바로잡은 재실측**일 뿐이다.

## 1. 정정 사유

사용자가 Worklog 105의 `render.ppm`을 보고 "제대로 진행된 3DGS scene이 전혀 아니다"라고 지적했다. Worklog 105는 `output/extent_ab/val64/baseline_compatible/final`(iteration 3100, PSNR 20.1, opacity reset 직후 — Worklog 94~104가 전부 참조했던 것과 같은 checkpoint)을 썼는데, 이 checkpoint는 `docs/Urgent_Work/HANDOFF_2026-08-19.md` §0이 이미 "의도한 scene이 맞는지 미확인"으로 표시해 둔 것이었다. 사용자가 **`output/osn_gs_scene/3000`을 쓰라고 명시적으로 지시**했다.

두 checkpoint 비교:

| | `extent_ab/val64/baseline_compatible/final` (구) | `osn_gs_scene/3000` (신, 정정) |
|---|---:|---:|
| iteration | 3100 | 3000 |
| PSNR | 20.11 | **23.92** |
| Gaussian 수 | 1,685,549 | 1,033,693 |
| opacity reset 직후 여부 | 예(`cumulative_adc_opacity_reset_count=1`) | 미기록(정상 궤적) |

`output/osn_gs_scene/3000/render.ppm`을 직접 확인한 결과 정원 테이블·화분·주변 식생이 선명하게 재구성된 정상 3DGS scene이다. `output/osn_gs_scene/final`은 실제로는 iteration 10000이고 PSNR이 19.8로 3000보다 낮으므로, "final"이 항상 최선이라고 가정하지 않는다 — 사용자가 지목한 `3000`을 그대로 쓴다.

## 2. 재실측 결과

Worklog 105와 **동일한** 스크립트(`scripts/devtools/coverage_first_subset_partition_export.py`)를 `--checkpoint output/osn_gs_scene/3000`으로만 바꿔 재실행했다. 코드 변경 없음.

### 회계 (coverage 계약)

| 지표 | 값 |
|---|---:|
| Visible Gaussian(= 분할 입력) | **1,033,693** |
| assigned / unassigned / multiply-owned | 1,033,693 / **0** / **0** |
| spatially disconnected subset | **0** |
| `coverage_identity_holds` | **true** |

### Subset 분포

| 지표 | 값 |
|---|---:|
| Subset 수 | **29,944** |
| min / median / mean / p95 / max | 1 / 1 / 34.52 / 7 / **857,342** |
| 최대 subset 비율 | **82.94%**(전체 Gaussian의) |
| Singleton(크기 1) | 21,612개(subset의 72.17%, Gaussian의 2.09%) |
| 크기 ≤8 | 28,644개(subset의 95.66%, Gaussian의 4.19%) |

### Edge / fallback

| 지표 | 값 |
|---|---:|
| Candidate edge | 5,346,738 |
| Spatial edge | 4,464,080 |
| Normal-compatibility cut edge | 1,121,675(spatial의 **25.13%**) |
| Accepted edge | 3,342,405 |
| Fallback ownership | 21,612개(**2.09%**) = normal 비호환 20,998 + 공간 이웃 없음 614 |
| 축 분리 가능성(진단) | well_defined 443,787 / tangent_axes_degenerate 563,819 / normal_axis_degenerate 17,990 / isotropic 8,097 / non_finite 0 |
| Local spacing | min 0.00246 / median 0.03740 / mean 0.04774 / p95 0.11456 / max 2.45926 |
| Cut ratio(Gaussian별) | mean 0.2505 / median 0.1667 / p95 0.8182, 완전 절단 20,998 / 전혀 안 잘림 357,200 |

### Worklog 105(잘못된 checkpoint)와의 비교

정상 학습된 scene이 훨씬 낮은 fallback 비율(2.09% vs 6.40%)과 낮은 normal-cut 비율(25.1% vs 46.1%)을 보인다 — opacity reset 직후의 noisy/degraded checkpoint가 orientation 신호 자체를 훼손해 분할 품질에 영향을 준다는 정황과 일치하지만, **이 비교 자체는 정성적 관찰일 뿐 이번 배치의 architecture 판단이 아니다.**

## 3. Review export (정정본)

경로: `output/osn_gs_coverage_first_subset_partition_v2/`(Worklog 105의 원래 잘못된 checkpoint export는 `output/osn_gs_coverage_first_subset_partition/`에 그대로 보존 — 정정 전/후 비교 가능). 구조는 Worklog 105와 동일(4개 full-scene view + `render.ppm` + `partition_report.json`).

시각 확인: ORIGINAL_SCENE render.ppm이 checkpoint 자신의 `output/osn_gs_scene/3000/render.ppm`과 픽셀 일치. GAUSSIAN_SUBSET_PARTITION은 테이블·화분이 뚜렷한 파란 단일 subset으로, 바닥·산울타리 대부분이 하나의 거대한 subset(82.9%)으로 나타난다. NORMAL_ORIENTATION_VIEW는 바닥이 균일한 초록(위쪽 normal), 테이블 다리가 붉은 계열(수평 normal)로 구분되어 해석 가능하다. SUBSET_BOUNDARY_VIEW는 산울타리·테이블 가장자리에서 cut ratio가 높고 평평한 바닥에서 낮다.

## 4. 재현 명령

```
python scripts/devtools/coverage_first_subset_partition_export.py \
    --checkpoint output/osn_gs_scene/3000 \
    --out output/osn_gs_coverage_first_subset_partition_v2 \
    --device cuda --source-path DATASET
```

런타임 54.8초(RTX 5080, Worklog 105의 1,685,549개 대비 Gaussian 수가 적어 더 빠름).

## 5. 검증

코드 변경이 없으므로 Worklog 105의 focused 테스트 25개와 전체 회귀(`1152 passed, 1 skipped`)를 재실행하지 않았다.

## 6. Memory 갱신

`output/extent_ab/val64/baseline_compatible/final`을 향후 실측에 다시 쓰지 않도록 신규 memory(`feedback_correct_replay_checkpoint`)를 세션 memory와 `docs/agent_memory/`에 추가했다 — 다음 실측은 사용자가 다른 checkpoint를 지목하지 않는 한 `output/osn_gs_scene/3000`을 쓴다.

## 결론 없음

이 worklog도 coverage-first architecture의 성공/실패 판단을 내리지 않는다. 사용자가 `output/osn_gs_coverage_first_subset_partition_v2/`의 GAUSSIAN_SUBSET_PARTITION을 직접 시각적으로 검토한 뒤에야 다음 단계를 결정한다.
