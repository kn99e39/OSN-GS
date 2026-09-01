# Worklog 141 — Oracle Single-Surface Support / Renderer-Native Appearance Evidence

## 의도와 격리

이번 배치는 자동 `Surface Membership`를 구현하지 않고, WL127 raw Visible Surface Evidence의 혼합 membership와 WL139 representative family 문제를 분리하는 oracle upper-bound 진단이다. 새 코드는 `devtools/demo/oracle_single_surface_support_appearance_evidence.py`, 테스트는 `tests/test_oracle_single_surface_support_appearance_evidence.py`, 산출물은 `output/141_oracle_single_surface_support_appearance_evidence/`에 격리했다. WL127 geometry, frozen checkpoint, canonical renderer, WL139 graphness/fitter는 수정하지 않았다.

## Phase 0 — WL140 정성 보정 기록

- `historical_wl139_curved_rim_alignment_control`은 table rim semantic evidence로 승격하지 않았다.
- `adjacent_table_side`와 `patio_ground_planar`도 multi-view Gaussian overlay만으로 실제 semantic surface라고 확정하지 않았다.
- WL140 `curved_table_rim` raw population은 둘 이상의 physical trend를 포함할 가능성이 확인됐다.
- 따라서 `graphness PASS != same physical surface`를 WL141의 해석 계약으로 고정했다.

## Stage A — oracle support

세 후보 broad surface에 대해 candidate spatial crop과 별도로 3개 고정 camera의 수동 polygon을 동결하고, 3개 중 2개 이상 mask에 투영되는 WL127 row만 oracle support로 선택했다. row ID, row hash, camera ID, polygon, 투표 수는 `frozen_oracle_support_manifest.json`에 fit 전에 기록했다. 대표면, graphness, SH/color, continuation, Candidate B는 support 선택에 사용하지 않았다.

| case | candidate rows | oracle rows | oracle/candidate | baseline graphness | oracle graphness | 대표면 |
|---|---:|---:|---:|---|---|---|
| `tabletop_top_oracle` | 1,554 | 1,367 | 87.97% | FAIL 0.2667 | FAIL 0.2963 | 미생성 |
| `curved_table_rim_oracle` | 20,181 | 17,842 | 88.41% | PASS 0.0891 | PASS 0.0705 | 생성 |
| `paver_ground_oracle` | 7,552 | 6,220 | 82.36% | PASS 0.0579 | PASS 0.0556 | 생성 |

세 support 모두 기계적 `SUPPORT_ALIGNMENT_PASS`를 얻었지만, 이는 동일 physical sheet를 증명하지 않으며 `PENDING_HUMAN_QUALITATIVE_REVIEW`로 남겼다. 사람 검토용 pre-fit alignment 출력은 각 camera의 `pre_fit_alignment/`에 있다.

대표면은 두 graphness-PASS oracle arm에만 WL139 고정 설정(8×4, degree 2/2, smoothness/tikhonov 1e-4)으로 생성했다. curved rim oracle의 진단용 raw→representative median/p95는 `1.644h / 10.655h`, representative→raw는 `3.034h / 32.553h`였다. ground oracle은 raw→representative `1.546h / 2.791h`였지만 representative→raw `34.578h / 80.538h`로 full rectangular domain의 unsupported 영역 문제가 컸다. 이 값들은 withheld geometry error가 아니며, 정성 판단을 대체하지 않는다.

지원 domain 진단은 fit geometry를 trim하지 않고 annotation만 했다. curved rim은 supported chart vertex fraction 55.57%, ground는 11.85%였다. full rectangular representative와 실제 oracle raw support domain을 구분해 PLY/PNG/NPZ로 내보냈다.

## Stage A 판정

자동 실행에서는 image alignment와 macro-shape의 사람 검토를 수행할 수 없으므로 최종 gate를 `F. MIXED / INCONCLUSIVE — human image alignment and macro-shape review required`로 기록했다. tabletop은 oracle support 이후에도 graphness가 fail했으며, curved rim/ground는 대표면이 생성됐지만 정성적 physical-sheet alignment와 unsupported domain의 영향이 아직 분리되지 않았다. 따라서 `SINGLE-SURFACE SUPPORT IS A VALIDATED MISSING LAYER`를 자동 승격하지 않았다.

## Stage B — provenance fail-closed

WL127 PLY header에는 `x/y/z`, `f_dc_0..2`, opacity/scale/rotation만 있고 renderer event, primitive ID, contributor ID/weight가 없다. provenance audit 결과는 `NO_VALID_PRIMITIVE_PROVENANCE`다. 따라서 nearest-Gaussian center proxy를 사용하지 않았고 SH-DC, view-conditioned SH, appearance gradient, frozen edge dataset, AUROC/AUPRC를 계산하지 않았다. appearance가 surface membership나 occlusion evidence라는 주장도 하지 않는다. Stage B는 `NOT_EXECUTED_UNTIL_STAGE_A_HUMAN_PASS`로 닫았다.

## 검증

- focused tests: `23 passed`
- 실제 실행: `.venv\\Scripts\\python.exe -B devtools\\demo\\oracle_single_surface_support_appearance_evidence.py --device cuda`
- 실제 실행 failures: `[]`
- pre-fit raw alignment PNG 9개, 최종 raw/representative/annotation PNG 및 PLY/NPZ 생성

## 결론과 다음 결정

이번 배치는 자동 membership나 최종 architecture를 만들지 않았다. 현재 증거는 “수동 oracle support를 별도 층으로 진단할 수 있고, 일부 graph-like arm에서는 WL139 representative를 재생성할 수 있다”까지다. appearance는 유효 provenance가 추가되고 Stage A 정성 gate가 통과할 때만 다음 배치에서 독립 separation evidence로 평가한다. continuation, Occluded Surface, region growing, geometry+appearance fusion은 수행하지 않았다.