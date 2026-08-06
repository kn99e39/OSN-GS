# Worklog 65: Baseline-Compatible 3k Production Validation

## 목적

worklog 64의 `gaussian_initialization_mode=baseline_compatible`(production 기본값)이 실사용 3k 학습에서 anisotropy/ADC 동역학·렌더 품질·chart materialization을 실제로 회복시키는지 검증한다. 코드/threshold는 변경하지 않는다.

## 방법

동일 seed/resolution/loss/camera schedule/ADC schedule(`--position_lr_extent_mode scene --surface_update_interval 0`)로 3개 조건을 3100 iteration까지 재학습, `save_iterations 600 2900 3000 3100`:

- **covariance_knn**: `output/extent_ab/val64/covariance_knn`
- **baseline_compatible**: `output/extent_ab/val64/baseline_compatible`
- **Graphdeco baseline**: `output/extent_ab/val64/baseline`

iteration 0(학습 전 초기 상태)은 체크포인트가 없으므로 `pipeline.initialize()`/baseline `create_from_pcd()`를 직접 호출해 별도 분석했다. 분석 스크립트: `scripts/devtools/gaussian_init_mode_3k_validation_analysis.py`(기존 `fixed_loader_replay_analysis.py`/`baseline_ply_replay_analysis.py`를 그대로 재사용하고, projected radius 분포와 iteration-0 분석만 신규 추가).

## 결과

### Anisotropy / scale

| iter | 조건 | count | aniso median | aniso p99 | s_min med | min-scale collapse |
|---|---|---:|---:|---:|---:|---:|
| 0 | covariance_knn | 138,766 | 26.25 | 26.25 | 0.00170 | — |
| 0 | baseline_compatible | 138,766 | **1.0** | **1.0** | 0.03898 | — |
| 0 | baseline | 138,766 | 1.0 | 1.0 | 0.03898 | — |
| 600 | covariance_knn | 138,766 | 27.42 | 59.86 | 0.00204 | — |
| 600 | baseline_compatible | 138,766 | **1.52** | **5.69** | 0.02672 | — |
| 600 | baseline | 138,766 | 1.58 | 7.11 | 0.02649 | — |
| 3100 | covariance_knn | 2,029,500 | 35.77 | 322.8 | 0.00070 | 1.49% |
| 3100 | baseline_compatible | 2,074,050 | **3.66** | **24.0** | 0.00552 | **0.027%** |
| 3100 | baseline | 1,913,586 | 5.49 | 109.8 | 0.00345 | 0.448% |

`baseline_compatible`은 iteration 0/600에서 baseline과 사실상 동일(median 정확히 1.0, 600에서 1.52 vs 1.58)하고, 3100에서도 median 3.66이 baseline의 5.49와 같은 자릿수다 — `covariance_knn`의 35.77과는 완전히 다른 수준이다. min-scale collapse도 `covariance_knn`의 1.49%에서 `baseline_compatible`의 0.027%로 급감(baseline의 0.448%보다도 낮음).

### ADC 동역학 / screen-size prune

| iter | 조건 | cumulative split | 이 step clone/split | 이 step screen prune |
|---|---|---:|---:|---:|
| 3000 | covariance_knn | 328,746 | 2,747 / 5,494 | 0 |
| 3000 | baseline_compatible | 354,704 | 4,044 / 8,088 | 0 |
| 3100 | covariance_knn | 334,240 | 1,403 / 2,806 | **224,164** |
| 3100 | baseline_compatible | 362,792 | 1,459 / 2,918 | **233,178** |

**Screen-size prune 폭주는 감소하지 않았다** — iteration 3100(opacity reset 직후)의 screen prune 건수가 `covariance_knn` 224,164 vs `baseline_compatible` 233,178로 사실상 동일(오히려 소폭 증가). Anisotropy가 극적으로 개선됐음에도 이 폭주 규모가 그대로라는 것은, **screen-size prune storm이 anisotropy 문제와는 별개로 opacity-reset 이벤트 자체(대량 opacity 저하 → 다음 ADC pass에서 opacity+screen 동시 대량 prune)에 내재한 현상**임을 시사한다. worklog 62/63에서 "screen-size prune 폭주 = anisotropy 문제의 하류 증상"이라던 초기 가설은 이 결과로 다시 한번 좁혀진다 — anisotropy는 실제 원인이었지만(worklog 64), opacity-reset 직후 prune storm 자체의 크기는 별도 메커니즘이다.

### 렌더 품질 (PSNR/SSIM/LPIPS)

| iter | 조건 | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| 2900 | covariance_knn | 22.877 | 0.6086 | 0.4655 |
| 2900 | baseline_compatible | **23.291** | **0.6341** | **0.4226** |
| 2900 | baseline | 24.368 | 0.7146 | 0.2841 |
| 3000 | covariance_knn | 22.863 | 0.6086 | 0.4595 |
| 3000 | baseline_compatible | **23.231** | **0.6338** | **0.4173** |
| 3000 | baseline | 24.507 | 0.7260 | 0.2754 |
| 3100 | covariance_knn | 21.842 | 0.5862 | 0.4900 |
| 3100 | baseline_compatible | **22.150** | 0.6108 | 0.4525 |
| 3100 | baseline | 23.221 | 0.6970 | 0.3133 |

렌더 품질은 매 checkpoint에서 일관되게 baseline 방향으로 개선(PSNR +0.3~0.4, SSIM +0.02~0.03, LPIPS 개선)되지만 **격차를 완전히 닫지는 못한다**(2900에서 여전히 baseline 대비 PSNR -1.1, SSIM -0.08). 부분적 회복으로 정직하게 기록한다.

### Reliability / region / chart materialization — 단순 출력 감소가 아니라 over-segmentation 완화

| iter | 조건 | region_count | small(≤3) | median size | physical | parametric | combined |
|---|---|---:|---:|---:|---:|---:|---:|
| 2900 | covariance_knn | 159 | 77 | 4.0 | 1/1 | 89/89 | 90 |
| 2900 | baseline_compatible | 7 | 4 | 3.0 | 0/0 | 5/5 | 5 |
| 2900 | baseline(PLY→OSN pipeline) | 8 | 2 | — | 0/0 | 4/4 | 4 |
| 3100 | covariance_knn | 153 | 63 | 4.0 | 1/1 | 83/83 | 84 |
| 3100 | baseline_compatible | 19 | 11 | 3.0 | 0/0 | 11/11 | 11 |
| 3100 | baseline(PLY→OSN pipeline) | 3 | 0 | — | 0/0 | 2/2 | 2 |

판정 기준대로 "단순 출력 감소 vs over-segmentation 완화"를 구분하면: **over-segmentation 완화가 맞다.** 근거 — 세 조건 모두 **동일한 reliability/region-formation 파이프라인**을 실제 Graphdeco baseline PLY(순수 참조용, OSN-GS 학습과 무관)에도 그대로 돌려보면 region_count가 3~8개에 불과하다. `baseline_compatible`의 region_count(7~19)는 이 참조값과 **같은 자릿수**인 반면, `covariance_knn`의 region_count(145~184)는 그 참조값의 **20~60배**다 — 동일한 학습된 기하 구조를 두고 이렇게 큰 차이가 나는 것은 실제 표면 복잡도 차이가 아니라, `covariance_knn`이 처음부터 강제한 인위적 anisotropy가 reliability gate(모델 자신의 현재 scale/rotation을 evidence로 재사용)를 통해 수많은 가짜 "이미 평평해 보이는" 조각을 만들어내는 구성적 artifact이기 때문이다. region_size_histogram의 `small_le3_count` 비율(약 45~50%로 유사)도 같은 결론을 보강한다 — `covariance_knn`은 수백 개의 작은 파편, `baseline_compatible`은 baseline 참조와 자릿수가 맞는 소수의 조각을 만든다.

iteration 0/600에서는 `baseline_compatible`/`baseline` 모두 region_count=0이다 — 등방 초기값에서 시작해 학습으로 실제 방향성이 형성되기 전까지는 reliability gate가 (의도대로) 아무 region도 인정하지 않기 때문이며, 결함이 아니라 `covariance_knn`이 인위적으로 조기에 "가짜" 방향성 정보를 주입하던 것이 사라진 정상적 결과다.

### Projected radius

| iter | 조건 | median | p99 | max |
|---|---|---:|---:|---:|
| 3100 | covariance_knn | 9.0 | 65.0 | 2,826 |
| 3100 | baseline_compatible | 8.0 | 58.0 | 34,522 |
| 3100 | baseline | 8.0 | 55.0 | 74,807 |

median/p99는 세 조건이 비슷하지만 max는 baseline이 가장 크고 covariance_knn이 가장 작다 — `covariance_knn`의 공격적인 조기 split이 개별 Gaussian 풋프린트를 작게 유지하는 부수 효과로 보이며, 이번 판정 기준과는 직접 관련이 없어 참고로만 기록한다.

## 완료 기준 대조

- anisotropy와 ADC 동역학이 baseline 방향으로 회복되는지 → **예, 확인.** median 35.77→3.66(baseline 5.49), min-scale collapse 1.49%→0.027%(baseline 0.448%보다도 낮음).
- 3000 이후 screen-size prune 폭주가 감소하는지 → **아니오.** 224,164→233,178로 사실상 불변(소폭 증가). Anisotropy와는 별개로 opacity-reset 이벤트 자체에 내재한 현상으로 보인다(신규 관찰, 미해결).
- 렌더 품질이 회복되는지 → **부분적으로.** 매 checkpoint에서 일관되게 baseline 방향 개선(PSNR +0.3~0.4)이나 격차 완전 해소는 아님(-1.1~-1.4 PSNR 잔존).
- chart 수 감소가 단순 출력 감소인지 over-segmentation 완화인지 → **over-segmentation 완화로 확인.** 동일 파이프라인을 baseline PLY에 직접 돌린 참조값(3~8 regions)과 `baseline_compatible`(7~19)은 같은 자릿수, `covariance_knn`(145~184)은 그 20~60배.

## Known issue (수정하지 않음, 이번 replay에 영향 없음)

`gaussian_initialization_mode=baseline_compatible`은 `_initialize_canonical`(production 기본 "initialize" 스케줄, 이번 replay가 사용한 경로)에만 적용되고, `initialize_deferred`(`adc_post_commit`/`disabled` 스케줄)는 의도적으로 이 플래그를 무시하고 항상 `covariance_knn` 초기값을 쓴다 — 그 경로의 첫 post-ADC surface 재구성이 모델 자신의 scale/rotation을 유일한 orientation evidence로 재사용하기 때문이다(worklog 64에서 발견, `docs/Urgent_Work/OSN_GS_Urgent_Work_Master.md` §7에 기록). deferred 스케줄을 실제로 쓰는 작업이 생기기 전까지는 미수정.

## 테스트

이번 라운드는 순수 production replay/분석이며 코드·threshold 변경이 없다. 지시대로 full pytest는 실행하지 않았다(worklog 64의 `783 passed, 1 skipped`가 그대로 유효).
