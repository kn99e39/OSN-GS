# Worklog 76: boundary_support_spacing scale-domain 계약과 결정

## 목적

Worklog 72~74가 남긴 dense boundary-support 연결 실패의 지배 원인(half-line의 68%가 "local scale 안에 continuation 후보 없음"으로 종료)이 **scale domain의 단위 오류**인지 아닌지를 한 배치에서 종결한다. 연결되는 대상은 필터링된 **boundary-support candidate**인데 거리 게이트는 **full-evidence sampling spacing**을 쓰고 있었다(worklog 74 실측: candidate spacing이 full-evidence spacing의 2.08~3.72배).

Normal-source 작업은 유예 상태이며 covariance_normal 경로는 그대로 둔다. connectivity certificate(단계 순서, 2.5배 거리 배수, 0.1배 ambiguity 허용, reason/tangent/normal predicate, mutuality)는 재설계하지 않았다.

## 구현

신규 `osn_gs/surface/torch_boundary_support_spacing.py`가 세 spacing을 **의미적으로 분리**해 명시한다: `full_evidence_spacing`, `representative_spacing`(worklog 32의 값, **report-only** — 연결 scale로 절대 쓰지 않음), `boundary_support_spacing`. 비교 대상 estimator는 셋뿐이다.

- **A `full_evidence_spacing`** — 현행 production, 미변경 baseline
- **B `region_boundary_support_spacing`** — region 단위 robust(median) candidate NN spacing
- **C `local_boundary_support_spacing`** — candidate별 robust local spacing(자기 최근접 3개까지의 median)

**거리 배수 2.5와 ambiguity 허용 0.1은 세 모드에서 고정했고 모드별로 따로 조정하지 않았다.** 목적은 올바른 scale domain을 정하는 것이지 loop을 만드는 threshold를 찾는 것이 아니다.

production 경로에는 **가산적(additive)** 으로만 손댔다. `_connect(..., connectivity_scale=None)`와 `diagnose_dense_boundary_connectivity(..., connectivity_scale=None)`에 선택적 per-candidate scale 인자를 추가했고, `None`(기본값)은 이전 동작을 그대로 재현한다. estimator가 퇴화하면(candidate 2개 미만 등) 항상 full-evidence spacing으로 fail-safe fallback한다.

안전성 검증을 위해 `measure_edge_support_occupancy`를 추가했다: 채택된 각 edge를 자기 축 방향으로 full-evidence spacing 폭 bin으로 나누고, 축에서 full-evidence spacing 안에 evidence가 투영되는 bin을 occupied로 센다. **순수 disclosure 지표이며 어떤 edge도 수정·수용·연결하지 않는다.**

## 회귀 확인

production 기본 경로(`connectivity_scale=None`)의 real 7개 region 결과는 `open_or_ambiguous` **621**, closed loop **0** 으로, worklog 72가 발표한 수치와 **정확히 일치**한다. 가산 변경이 production 동작을 건드리지 않았음을 실측으로 확인했다.

## Scale-mode 비교 (real baseline_compatible@2900, 7개 region)

| region | cand | mode | scale/full-ev | no_candidate | both/one/neither | edges d/n/t/mut | closed | 미지지 edge | crossing |
|---|---:|---|---:|---:|---|---|---:|---|---:|
| r0 | 20 | A | 1.00 | 34/40 | 0/6/14 | 6/6/6/3 | 0 | 0/3 | 0 |
| | | B | 3.72 | 2/40 | 10/8/2 | 182/178/136/11 | 0 | **7/11** | 0 |
| | | C | 4.61 | 0/40 | 11/8/1 | 251/242/176/11 | 0 | **7/11** | 0 |
| r1 | 126 | A | 1.00 | 167/252 | 13/44/69 | 118/114/84/33 | 0 | 1/33 | 0 |
| | | B | 2.49 | 42/252 | 66/45/15 | 736/692/520/64 | 0 | **33/64** | 0 |
| | | C | 3.97 | 4/252 | 71/53/2 | 1713/1623/1219/67 | 0 | **34/67** | 0 |
| r2 | 134 | A | 1.00 | 180/268 | 11/47/76 | 102/102/76/33 | 0 | 3/33 | 0 |
| | | B | 2.28 | 51/268 | 70/42/22 | 592/590/456/77 | 0 | **36/77** | 0 |
| | | C | 3.51 | 2/268 | 91/38/5 | 1688/1572/1144/82 | 0 | **42/82** | 0 |
| r3 | 23 | A | 1.00 | 28/46 | 4/4/15 | 48/46/26/4 | 0 | 0/4 | 0 |
| | | B | 2.77 | 7/46 | 9/8/6 | 128/114/68/7 | 0 | 1/7 | 0 |
| | | C | 3.53 | 0/46 | 12/9/2 | 233/148/93/9 | 0 | 2/9 | 0 |
| r4 | 224 | A | 1.00 | 307/448 | 12/61/151 | 196/144/98/38 | 0 | 2/38 | 0 |
| | | B | 2.52 | 55/448 | 98/90/36 | 1350/804/586/102 | 0 | **57/102** | 0 |
| | | C | 3.65 | 7/448 | 126/78/20 | 3597/2240/1580/104 | 0 | **58/104** | 0 |
| r5 | 77 | A | 1.00 | 99/154 | 10/31/36 | 88/86/82/20 | 0 | 0/20 | 0 |
| | | B | 2.08 | 38/154 | 30/33/14 | 338/314/254/33 | 0 | **13/33** | 0 |
| | | C | 3.29 | 5/154 | 49/26/2 | 872/799/587/39 | 0 | **19/39** | 0 |
| r6 | 181 | A | 1.00 | 253/362 | 8/59/114 | 172/150/106/33 | 0 | 2/33 | 0 |
| | | B | 2.64 | 43/362 | 78/66/37 | 1174/934/630/81 | **1** | **42/81** | 0 |
| | | C | 4.10 | 2/362 | 104/63/14 | 3109/2081/1435/85 | **1** | **46/85** | 0 |

합성 fixture(`box_face`, `cylinder` 3개 region)에서도 같은 방향이었다: C가 `no_candidate`를 12→1, 4→0, 20→0, 4→0으로 줄였고 cylinder bottom_cap에서는 미지지 edge 0·crossing 0인 **안전한 closed loop 1개**를 실제로 복원했다(A는 0). 즉 합성 소규모 표면에서는 C가 순수한 개선으로 보인다.

### 집계 (합성 4 + real 7 = 11 region)

| 지표 | A full_evidence | B region_support | C local_support |
|---|---:|---:|---:|
| candidate | 826 | 826 | 826 |
| `no_candidate_within_local_scale` | 1108/1652 (67%) | 254 (15%) | **21 (1.3%)** |
| both / one / neither 방향 | 70/270/486 | 377/306/143 | **496/284/46** |
| 생존 edge d/n/t/mut | 780/698/526/185 | 4620/3704/2714/397 | 11751/8917/6397/427 |
| branch component | 0 | 0 | 0 |
| proper crossing | 0 | 0 | 0 |
| closed loop | 0 | 1 | 2 |
| **미지지(빈 구간 포함) edge** | **9/185 (4.9%)** | **191/397 (48.1%)** | **211/427 (49.4%)** |
| 최장 미지지 구간 비율(real 최대) | 0.50 | 0.75~0.83 | 0.67~0.92 |
| recovered loop `interior_outside_boundary` | — | 0.9978 | 0.4989(합성 1건 포함) |

## 독립 boundary_support_spacing에 대한 근거

**찬성 근거(진단으로서는 옳았다).** 단위 오류는 실재했다. candidate spacing은 full-evidence spacing의 2.08~4.61배이고, scale domain을 바꾸자 `no_candidate`가 67%→1.3%로 떨어지며 방향 연속성이 크게 회복된다(neither 486→46). branch explosion도 crossing도 전혀 발생하지 않았다(둘 다 0). 합성 소규모 표면에서는 C가 안전한 loop을 실제로 하나 만들어낸다.

**반대 근거(production 채택 불가).** 회복된 연속성의 출처가 문제다. 미지지 edge 비율이 4.9% → **48.1%/49.4%** 로 10배 뛴다. 이는 단발성 빈 bin 잡음이 아니다 — real 전 region에서 **단일 edge 길이의 75~92%가 관측된 빈 공간을 지나는 사례**가 나오고, 채택 edge의 중앙값조차 길이의 20~33%가 미지지다(A는 중앙값 0.0, p90 0.0). 즉 두 독립 scale은 **관측되지 않은 gap을 가로질러 연결함으로써** 연속성을 사는 것이며, 이는 명시적으로 금지된 gap bridging이다.

게다가 그 대가로 얻는 것이 거의 없다. real에서 늘어난 closed loop은 region6의 **1개**뿐이고, 그 loop의 `interior_outside_boundary`는 **99.78%** — worklog 70/71과 동일한 퇴화 결과라 downstream에서 쓸 수 없다.

수용 기준("branch explosion·unsupported gap bridging·crossing·containment 악화 없이 국소 연속성을 회복할 때만 채택")에서 gap bridging 항목이 명확히 위반된다.

## Production 결정: **full_evidence_spacing 유지 (독립 scale 미채택)**

production 연결 scale은 변경하지 않는다. `boundary_support_spacing_mode`는 기본값 `None`(=현행 동작)으로 남기고, 세 spacing을 의미적으로 분리한 **계약 자체는 유지**한다(이번 측정 결과와 재개 조건을 모듈 docstring에 명시). 남은 실패는 scale domain이 아니라 **boundary-support predicate/evidence 자체**에 귀속한다.

## 결정 이후 real-checkpoint boundary/UV/NURBS 결과

production 경로가 변경되지 않았으므로 real baseline_compatible@2900의 dense boundary 결과는 그대로다: **closed loop 0 / `open_or_ambiguous` 621**(worklog 72와 동일, 위 회귀 확인). 따라서 reduced UV gate와 6×6 NURBS fitting에 도달하는 recovered boundary가 하나도 없고, 결과는 **`valid_supported` 0 / `extrapolative` 0 / `partition_materialization_required` 0** 이다. 이 0들은 gate를 통과하지 못해서가 아니라 **gate에 입력될 boundary 자체가 없기 때문**이며, 두 상태를 혼동해 기록하지 않는다.

## 이번 라운드가 확정한 것

worklog 73이 지목한 "68% 거리 실패"는 **단위 오류가 가린 회복 가능한 연속성이 아니었다.** 단위를 바로잡으면 그 실패는 사라지지만(1.3%), 그때 연결되는 상대는 관측 근거가 없는 먼 candidate다 — 즉 candidate들이 애초에 **공간적으로 인접하지 않다.** **scale은 현재 병목에서 확정적으로 제거된다.** worklog 75가 normal source를 제거한 데 이어, 이번 라운드는 scale domain을 제거한다. 남은 후보는 boundary-support predicate 자체(어떤 점을 boundary support로 인정하는가)와 그 evidence 밀도다.

지시대로 추가 scale-tuning 라운드는 시작하지 않는다. hull·PCA rectangle·bounding box·forced closure·gap bridging·cross-region merge·geometric fallback은 도입하지 않았고, region formation·ownership gating·representative membership·visible Gaussian photometric training·covariance_normal 경로는 전부 미변경이다.

## 검증

신규 `tests/test_boundary_support_spacing.py` 13개 + 변경 모듈을 소비하는 기존 테스트 포함 **42 passed**. 공개 계약은 기본값이 이전 동작과 동일한 가산 인자 2개뿐이라 full pytest는 지시대로 실행하지 않았다.
