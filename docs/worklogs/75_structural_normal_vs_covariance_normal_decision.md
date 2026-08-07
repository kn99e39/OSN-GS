# Worklog 75: Normal-Source 아키텍처 결정 (covariance_normal vs structural_normal)

## 목적

Worklog 74가 남긴 결론("covariance-derived tangent이 cycle을 깨므로 다음 실험은 explicit structural-normal/tangent A/B여야 한다")을 **단일 bounded 비교로 종결**한다. 이 라운드는 추가 진단 라운드를 만들지 않고 adopt/reject 결정을 반환하는 것이 목적이다.

## 방법

신규 `osn_gs/surface/torch_structural_normal.py` (실험 경로 전용, 어떤 production 모듈도 이것을 import하지 않음):

- `compute_structural_normals(points)`: region-owned observed 점의 **위치만**으로 local PCA(kNN 위치 공분산의 최소 고유벡터). scale·rotation·covariance·SH·opacity·renderer·optimizer state를 일절 읽지 않는다.
- `rebuild_candidate_orientation(...)`: **이미 추출된 frozen candidate 집합**의 orientation frame만 재계산한다. 기존 `torch_region_owned_dense_boundary_support`의 missing-sector/outward-direction 구성을 그대로 재사용해 `tangent = cross(normal, outward)`를 다시 만들고, candidate id·position·boundary_reason·full_evidence_scale은 그대로 통과시킨다. **B는 candidate를 독립적으로 재추출하지 않는다.**

A/B 사이에서 고정한 것: candidate id와 xyz, region ownership(양 모드 실행 전에 1회 계산), boundary reason, candidate extraction, connectivity scale과 distance threshold, ambiguity/mutuality 로직, topology acceptance. 두 모드 모두 **동일한 미변경** `_connect`와 worklog 73/74 진단 함수를 통과한다.

Scene은 기존 fixture/checkpoint만 사용했다(신규 dataset 없음): `box_face`(단순 평면), `cylinder`(곡면 side + 평면 cap이 crease로 만나는 orientation-민감 케이스), 그리고 boundary-closure 실패를 실제로 보이는 real `baseline_compatible@2900`(worklog 72~74가 측정한 그 checkpoint) 4개 region.

## Scene별 A/B

각 셀은 `A(covariance) → B(structural)`. edges는 distance/normal/tangent/mutuality 단계 생존 directional proposal, cycles는 distance-stage cycle의 단계별 생존.

### Scene 1 — box_face (단순 평면, 81점)

| region | cand | normal rej | tangent rej | edges d/n/t/mut | cycles d/n/t/mut | closed loops |
|---|---:|---|---|---|---|---|
| region | 14 | 0 → 0 | 0 → 0 | 24/24/22/8 → 24/24/22/8 | 1/1/1/0 → 1/1/1/0 | 0 → 0 |

normal 각도 불일치 median 0.22°(max 0.40°). **모든 지표가 완전히 동일하다.**

### Scene 2 — cylinder (crease/곡면, orientation 민감)

| region | cand | normal rej | tangent rej | edges d/n/t/mut | cycles d/n/t/mut | closed loops |
|---|---:|---|---|---|---|---|
| bottom_cap | 8 | 0 → 0 | 0 → 0 | 12/12/12/6 → 동일 | 0/0/0/0 → 동일 | 0 → 0 |
| side | 12 | 0 → 0 | 0 → 0 | 4/4/4/2 → 동일 | 0/0/0/0 → 동일 | 0 → 0 |
| top_cap | 7 | 0 → 0 | 0 → 0 | 10/10/10/5 → 동일 | 0/0/0/0 → 동일 | 0 → 0 |

불일치 median: cap 0.16~0.24°, 곡면 side 7.31°. **단순/곡면 합성면에서 B는 A를 전혀 열화시키지 않는다**(동일). 다만 side에서는 B의 normal 기준으로 12/12 candidate가 missing-sector admission을 통과하지 못했을 것으로 기록됐다(기록만 하고 candidate 집합은 고정 유지).

### Scene 3 — real baseline_compatible@2900 (boundary-closure 실패 checkpoint)

| region | cand | normal rej | tangent rej | edges d/n/t/mut | cycles d/n/t/mut | closed loops |
|---|---:|---|---|---|---|---|
| region0 (93점) | 20 | 0 → **2** | 0 → 0 | 6/6/6/3 → 6/**4**/**4**/**2** | 0/0/0/0 → 0/0/0/0 | 0 → 0 |
| region1 (519점) | 126 | 3 → **14** | 11 → **5** | 118/114/84/33 → 118/**94**/**76**/**31** | 2/2/0/0 → 2/**4**/**2**/0 | 0 → 0 |
| region2 (510점) | 134 | 0 → **18** | 19 → **15** | 102/102/76/33 → 102/**78**/**58**/**28** | 2/2/0/0 → 2/2/0/0 | 0 → 0 |
| region3 (92점) | 23 | 2 → **6** | 3 → **2** | 48/46/26/4 → 48/**18**/**12**/4 | 0/0/0/0 → 0/0/**1**/0 | 0 → 0 |

normal 각도 불일치 median 30.8~63.6°, p90 76.9~82.8°, max ~90°. B 기준 admission 재검 시 A의 candidate 중 12/20, 72/126, 39/134, 5/23이 boundary candidate로 인정되지 않았을 것으로 기록됐다.

**Containment/interior-outside: 양 모드 모두 valid closed loop이 0개이므로 측정 대상이 없다(`None`).**

## 집계 (3 scene · 8 region)

| 지표 | A covariance_normal | B structural_normal | 변화 |
|---|---:|---:|---|
| candidate 수 | 344 | 344 | 고정(설계대로) |
| normal rejection | 5 | **40** | **8배 악화** |
| tangent rejection | 33 | **22** | 33% 개선 |
| distance-valid edge | 324 | 324 | 동일(distance 단계는 normal 무관) |
| tangent 생존 edge | 240 | **198** | **−17.5%** |
| mutuality 생존 edge | 94 | **86** | **−8.5%** |
| distance-stage cycle | 5 | 5 | 동일 |
| tangent 단계 생존 cycle | 1 | **4** | 개선 |
| **final closed loop** | **0** | **0** | **변화 없음** |

candidate 위치에서의 normal 각도 불일치 median(전체) 19.06°.

## Runtime / memory

| 항목 | A | B |
|---|---:|---:|
| normal 생성 단계 합계 | 1.119s | 0.246s |
| boundary connectivity 합계 | 0.386s | 0.386s |
| structural normal 계산 자체(region당) | — | 0.0002~0.0029s |
| 추가 메모리 | — | N×3×4 byte (region당 324~6,228 byte) |
| 영속 state | — | 없음(호출 시 재계산, model/checkpoint/optimizer에 저장 안 함) |

connectivity 시간은 두 모드가 동일한 `_connect`를 타므로 사실상 같다(0.386s vs 0.386s). B의 normal 생성이 더 빠른 것은 A 쪽 측정에 candidate 추출 전체가 포함되기 때문이며, structural normal 계산 자체의 절대 비용은 region당 3ms 미만으로 무시할 수준이다.

## Rendering 동일성

Real checkpoint에서 실제 CUDA rasterizer(`installed package` backend)로 고정 view를 A/B 실행 **전후** 렌더링해 비교했다: `torch.equal` **bitwise identical = True**(sum 4965.23376378417 동일). structural normal은 renderer/optimizer 경로에 도달하지 않는다.

## 판정: **KEEP covariance_normal**

### 주요 근거

1. **어느 모드도 closed loop을 만들지 못한다(0 vs 0, 8개 region 전부).** structural normal은 boundary-closure 실패를 해결하지 못한다. 결정 기준의 1차 목표가 충족되지 않는다.
2. **연결성은 일관되게 개선되지 않고 오히려 악화된다.** tangent rejection이 33→22로 줄어든 것은 사실이나, 그 손실이 사라진 게 아니라 **normal 단계로 이동**했을 뿐이다(5→40, 8배). 순 생존 edge는 tangent 240→198(−17.5%), mutuality 94→86(−8.5%)로 real region 4개 중 3개에서 명확히 감소했다. tangent 단계 cycle이 1→4로 늘어난 것도 전부 mutuality 이전에 소멸해 최종 결과로 이어지지 않았다.
3. **새로운 불안정성이 도입된다.** B의 normal 기준으로 재검하면 A가 이미 인정한 candidate의 30~60%(cylinder side는 12/12)가 boundary candidate admission을 통과하지 못한다. 즉 structural normal은 기존 candidate 계약에 대한 drop-in 교체가 아니며, 채택하면 "어디가 boundary인가"에 대한 판단 자체가 조용히 바뀐다.
4. **합성면에서는 두 모드가 사실상 구별되지 않는다**(불일치 0.16~0.42°, 곡면 7.31°, 모든 지표 동일). 반면 real 데이터에서만 불일치가 median 30~64°로 폭발한다 — 실학습 Gaussian cloud가 국소적으로 깨끗한 surface sheet가 아니라 volumetric/noisy하기 때문에 **위치만 쓰는 local PCA가 real 데이터에서 불안정**하다는 뜻이다. 즉 structural normal의 문제는 개념이 아니라 입력 데이터의 국소 구조다.

비용(runtime·memory·rendering)은 문제가 아니다 — B는 오히려 싸고, 렌더링은 bit-identical이며 영속 state도 없다. 그럼에도 결정 기준상 "consistently preserves/improves connectivity"가 충족되지 않고 "새 불안정성 도입"에 해당하므로 채택하지 않는다.

### Trade-off 정직한 기술

이 결과는 covariance normal이 **좋다는** 증거가 아니다. worklog 74의 가설(covariance tangent이 cycle을 깬다)은 부분적으로 확인됐다 — normal source를 바꾸면 tangent rejection은 실제로 줄어든다. 그러나 그 개선이 최종 topology로 이어지지 않고 손실이 normal 단계로 재배치될 뿐이라는 것이, 이번 라운드가 확정한 사실이다. **normal source는 현재 boundary-closure 실패의 구속 조건(binding constraint)이 아니다.** 두 orientation source가 real 데이터에서 median 30~64° 어긋나면서도 최종 결과가 0 vs 0으로 같다는 것은, 병목이 orientation 정확도가 아니라 그 상류(candidate/support 밀도·scale domain)에 있음을 가리킨다.

이번 라운드에서 `boundary_support_spacing`은 활성화·재설계하지 않았고, connectivity threshold도 변경하지 않았으며, hull·PCA rectangle·forced closure·gap bridging·cross-region merge·geometric fallback은 도입하지 않았다.

## 검증

신규 `tests/test_structural_normal.py` 11개 + 소비하는 기존 모듈 테스트(`test_region_owned_dense_boundary_support.py`, `test_dense_boundary_connectivity_diagnostics.py`, `test_dense_boundary_scale_diagnostics.py`) 포함 **15 passed**. 지시대로 focused tests만 실행했고 full pytest는 수행하지 않았다.
