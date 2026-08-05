# Worklog 63: Fixed-loader 3k Production Replay

## 목적

worklog 62에서 수정한 `camera_fovs()`/resize-filter fix가 실제 3000-iteration production 학습에서도 anisotropy/screen-size prune 폭주를 회복시키는지 검증한다. 별도 알고리즘·threshold는 변경하지 않는다.

## 방법

`--surface_update_interval 0`(기존과 무관한 사전 결함 회피, 이전 라운드부터 유지), `--position_lr_extent_mode scene`으로 다음 3개를 동일 seed/resolution/loss/ADC schedule로 비교했다.

- **Before**: fix 이전 OSN-GS 3k(`output/extent_ab/A_scene_extent`, `A_scene_extent_ext`)
- **After**: fix 이후 OSN-GS 3k(`output/extent_ab/A_scene_extent_fixed_loader`, iteration 2900/3000/3100 재학습)
- **Baseline**: Graphdeco 3k(`gaussian-splatting/output/scene_2900_3100`)

분석은 `scripts/devtools/fixed_loader_replay_analysis.py`(OSN-GS checkpoint)와 `scripts/devtools/baseline_ply_replay_analysis.py`(baseline PLY)로 수행했다.

## 결과 (iteration 3100)

| 지표 | Before | After | Baseline |
|---|---:|---:|---:|
| Gaussian count | 1,867,920 | 2,032,216 | 1,911,848 |
| anisotropy median | 35.06 | **35.68** | **5.46** |
| anisotropy p99 | 300.6 | 306.3 | 107.3 |
| min-scale collapse 비율 | 1.36% | 1.36% | 0.44% |
| 이 step screen prune | 215,089 | **224,114**(↑) | — |
| PSNR / SSIM / LPIPS | 23.04 / 0.618 / 0.455 | 21.84 / 0.587 / 0.490 | 23.23 / 0.696 / 0.315 |

2900/3000 구간도 동일 패턴(anisotropy median fix 전후 35 부근에서 고정, baseline은 5.3~5.5 유지). Visible NURBS materialization(fix 후 checkpoint, cap=2048): 2900 region 153/reliable 3/physical 1/parametric 76/combined 77, 3000 region 142/reliable 2/physical 0/parametric 80/combined 80, 3100 region 142/reliable 2/physical 2/parametric 69/combined 71 — worklog 61의 이전 cap-1024 대체값을 이 수치로 교체한다.

## 완료 기준 대조

- 수정 후 anisotropy/ADC 동역학이 baseline 방향으로 회복 → **아니오**. anisotropy median이 fix 전후로 사실상 동일(35.06→35.68)하며 baseline(5.46)과의 격차(약 6.5배)는 그대로다.
- 3000 이후 screen-size pruning 폭주가 사라지거나 크게 감소 → **아니오**. 오히려 소폭 증가(215,089→224,114).

## 결론

과제의 fallback 지시("결과가 회복되지 않으면 loader 결함은 최초 발산 원인 중 하나로만 인정하고, 남은 training-core 차이를 별도로 추적하라")에 따라: worklog 62의 FoV/resize fix는 lockstep(ADC 비활성화, byte-identical transplanted tensor) 조건에서 real, 확인된 최초 발산 원인이지만, **실사용 3k+ 규모 anisotropy/screen-prune 폭주의 지배적 원인은 아니다.** lockstep 하네스는 baseline의 tensor를 그대로 이식해 시작했기 때문에 OSN-GS 자신의 Gaussian 초기화 파이프라인은 검증 범위 밖이었다 — 이 지점이 다음 라운드(worklog 64)에서 확인·수정한 실제 지배적 원인이다.

## 테스트

이번 라운드는 순수 replay/분석이며 `osn_gs/` 코드 변경이 없다. worklog 62의 `775 passed, 1 skipped, 1 warning, 18 subtests passed`가 그대로 유효하다.
