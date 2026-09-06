# Worklog 176 — Historical Local Surface Decomposition 계약 복원과 W154 Architecture Drift 귀속

## 상태

완료. 이번 배치는 production code, historical implementation, checkpoint,
W154/W175 artifact를 수정하지 않은 read-only architecture audit이다.
최종 판정은 **`CONTRACT_SPLIT_ACROSS_MULTIPLE_HISTORICAL_BRANCHES`**이다.

W96–W100/W154의 intrinsic `t_w` 계열과 W10/W31–W38/W150의
`form_surface_regions` 계열은 서로 다른 역사적 branch다. 전자는 W154의
실제 active Gaussian identity 경로지만 최종 architecture contract로 승인됐다는
증거가 없다. 후자는 consensus·bridge/path·parallel-separation을 포함한
구조적 safeguard를 갖지만 covariance-derived frame을 사용한다. 두 branch의
장점을 이 배치에서 합성하지 않았다.

## 1. Intent alignment

다음 항목을 보존했다.

- W96/W97/W98/W99/W100, W150, W154/W155, W171–W175의 문서와 구현
- historical parameter와 current production behavior
- W175가 확정한 `W97 subset_id → W154 region_id → W171 ownership` exact lineage
- `Gaussian Local Surface Decomposition = WHO`, `TSDF zero-set = WHERE` 의미 분리

다음 항목은 실행하지 않았다.

- W97 tuning, W150 자동 채택, W98/W99/W100 자동 promotion
- 새 hybrid decomposition, same-checkpoint re-partition, TSDF ownership 변경
- NURBS fit, continuation, Eligibility, UNRESOLVED, component filtering

## 2. Historical contract lineage

### W10 및 W31–W38: covariance-based structural constructor

W10은 pairwise `same_surface` connected component의 한계를 보완하기 위해
shared-neighbor consensus, bridge veto, tangent/path consistency와 ambiguity
state를 도입한 isolated foundation이었다. 당시 자체적으로 production integration을
주장하지 않았지만, W31–W38에서 reliability/affinity, seed와 merge의 분리,
two-phase DSU, component-pair support, bridge veto 의미가 정제됐다.

특히 W38은 raw-component bridge exemption이 bridge veto를 100% 우회하는
tautology임을 확인하고 이를 diagnostic-only로 되돌렸다. `seed_strong_edge`와
weak bridge merge를 분리한 two-phase semantics가 이후 보존된 구조적 safeguard다.

### W96–W100: intrinsic-normal experimental branch

- **W96**: learned 2DGS intrinsic `t_w`를 orientation authority로 사용했으나
  pairwise single-linkage가 74.70% giant subset을 만들었다.
- **W97**: 같은 local graph에서 region orientation concentration을 union gate로
  추가해 최대 subset을 21.20%로 줄였지만 real curved surface 과분할이 남았고,
  architecture 결론은 유보했다.
- **W98**: shape-operator residual과 positional normal-vs-tangent cut으로
  curvature/discontinuity를 분리하려 했지만 real scene에서 94.51% giant subset이
  재발했다. synthetic PASS는 production promotion이 아니었다.
- **W99**: W97 안전 초기화와 W98 interface majority merge를 결합했지만 real
  background가 53.86%로 합쳐졌다. 결과는 mixed이며 final decision은 없었다.
- **W100**: region-conditioned bilateral AND certificate로 22.91%까지 줄였고
  W99의 5-step patio→hedge lineage를 모두 기각했지만, 역시 diagnostic result로
  종료했고 W154 contract로 승격하지 않았다.

따라서 W96–W100은 normal-only failure에 대한 중요한 safeguard 실험 계열이지만,
그중 어느 하나도 보존된 기록상 최종 승인 contract가 아니다.

### W115–W116 및 W124: renderer-native 계열과 canonical inventory

W115–W116은 renderer median representative, image-space topology, ambiguity와
NURBS materialization을 분리했다. 이는 Gaussian intrinsic-normal Local Surface
constructor를 확정한 문서가 아니다. W124도 `form_surface_regions`를 포함한
covariance/affinity/reliability 계열을 current canonical A로, W97
region-coherent partition을 provisional B로 구분하면서 visible topology/NURBS의
미해결 영역을 별도로 남겼다.

### W150 및 W151–W175

W150은 Boundary First가 fit 전에 `form_surface_regions`의 region ownership을
사용해야 한다고 source call graph로 명시했다. W151–W155는 renderer event가
physical-sheet membership를 자동으로 제공하지 않음을 확인했고, W155/W175는
W154의 W97 stable-ID mapping을 exact replay했다.

W171–W174는 TSDF support fragmentation, boundary/domain, raw zero-set triangle
구조를 귀속했으며, W175는 W97 identity가 W171 ownership까지 보존됨을 확정했다.
동시에 W175는 W150 richer contract가 W154에서 호출되지 않았고 두 branch의
same-checkpoint crosswalk가 없음을 명시했다.

## 3. Decomposition contract matrix

상태 값은 `IMPLEMENTED`, `OPTIONAL`, `ABSENT`, `DIAGNOSTIC_ONLY`만 사용했다.
정적 report의 전체 matrix는 [worklog_176_report.json](../../output/176_local_surface_contract_audit/worklog_176_report.json)에 저장된다.

| 후보 | Normal source | Structural safeguard | 상태 |
|---|---|---|---|
| W96 | intrinsic `t_w` | 없음; pairwise graph | DIAGNOSTIC_ONLY |
| W97 | intrinsic `t_w` | region concentration, non-bridging propagation | DIAGNOSTIC_ONLY |
| W98 | intrinsic `t_w` | shape residual, positional/discontinuity cuts | DIAGNOSTIC_ONLY |
| W99 | intrinsic `t_w` | full-interface majority merge | DIAGNOSTIC_ONLY |
| W100 | intrinsic `t_w` | region-conditioned bilateral interface certificate | DIAGNOSTIC_ONLY |
| W10/W31–W38/W150 | covariance-derived frame | consensus, parallel separation, bridge veto, tangent/path, two-phase seed/merge | IMPLEMENTED in its own lineage |

세부 항목별 matrix는 script가 요청된 KNN, locality gate, sign-independent
normal, positional/normal separation, tangent consistency, discontinuity cut,
multi-edge consensus, concentration, interface, bridge/path, singleton,
ambiguity, deterministic tie-breaking을 모두 기록한다.

## 4. Normal-alone failure와 safeguard history

핵심 역사적 failure는 “각 local normal pair가 그럴듯하다”는 사실이 서로 다른
surface를 하나의 connected region으로 만든다는 점이었다. W96에서 74.70% giant
subset으로 드러났고, W97은 region-level global state로 완화했다. 그러나 W97은
smooth curvature를 과분할했다. W98은 curvature-aware local differential signal을
도입했지만 edge-local cut만으로는 dense graph percolation을 막지 못했다. W99와
W100은 complete-interface와 bilateral/region-conditioned semantics로 각각
재검증했으며, synthetic PASS와 real-scene 개선을 모두 보고했지만 architecture
promotion은 하지 않았다.

별도의 covariance branch에서는 W10/W31–W38이 parallel-sheet/phase-alias/weak
bridge를 shared-neighbor consensus, bridge veto, tangent/path transport,
two-phase seed/merge로 다뤘다. 이 branch가 구조적 safeguard를 역사적으로
구현한 것은 맞지만, 그 normal authority는 intrinsic `t_w`가 아니다.

## 5. Intended final Local Surface contract

사용자 intent의 A–D를 대조하면 다음과 같다.

| 요구 | 역사적 근거 |
|---|---|
| A. intrinsic Gaussian `t_w` authority | W96–W100/W154 |
| B. TSDF 이전 Gaussian branch에서 WHO 결정 | W96–W100/W154 및 W150 constructor |
| C. normal-only aliasing을 막는 승인된 structural mechanism | W10/W31–W38/W150 계열 |
| D. TSDF connectivity가 Surface identity를 재발견하지 않음 | W154/W175의 보존된 의미 |

A+B+D는 W154 active W97 branch에서 보이지만 C의 승인된 mechanism과 결합되어
있지 않다. C를 가진 W150 branch는 covariance-derived frame이라 A를 만족하지
않는다. 그러므로 기존 구현 중 A–D를 동시에 만족하는 하나는 발견되지 않았다.

최종 결과는 **`CONTRACT_SPLIT_ACROSS_MULTIPLE_HISTORICAL_BRANCHES`**이며,
이 배치에서 W97과 W150/W100을 조합한 새 구현을 만들지 않았다.

## 6. 왜 W154가 W97을 사용하는가

실제 W154 runner의 [active call](../../devtools/demo/candidate_f_gaussian_region_owned_tsdf.py:449)은
`derive_surface_orientation_from_surfel` 이후
`partition_surfels_region_coherent(active_orientation, RegionCoherenceConfig())`를
호출한다. `form_surface_regions` 호출은 없다. 반면 W150이 추적한 기존
[constructor](../../osn_gs/surface/torch_visible_surface_construction.py:207)는
covariance frame/reliability/affinity 뒤에 `form_surface_regions`를 호출한다.

따라서 W154가 W97을 선택한 것은 **명시적인 path difference**이자 정확한 drift
지점이다. 보존된 Worklog에는 “W97이 최종 승인돼 W150을 대체했다”는 후속 결정이
없다. W175의 `CANONICAL_LOCAL_SURFACE_IDENTITY_PRESERVED`는 W97 계열 identity가
보존됐다는 뜻이지 W150 contract와 동치라는 뜻이 아니다.

## 7. Conditional same-checkpoint control

실행하지 않았다. 이미 승인된 단일 implementation이 A–D를 만족하지 않기
때문이다. W150만 실행하면 intrinsic `t_w` 요구를 바꾸고, W100만 실행하면
diagnostic branch를 승인 contract로 오인하며, 둘을 합치면 금지된 새 hybrid가
된다.

## 8. W174 witness와 tabletop baseline

W175가 exact stored-ID로 재확인한 W97 baseline은 다음과 같다.

- W171 complete support: 17,965 rows, selected support: 15,189 rows
- fixed support의 subset 수: 1 (`subset_id=1`)
- distinct owner Gaussian: complete 3,315, selected 2,887
- W174 row 4043/4051 owner: 모두 subset 1, `core`, `ambiguous=false`
- reference path의 W97 subset boundary crossing: 0

비교 contract는 존재하지 않으므로 witness classification은
`NOT_COMPARABLE`이다. W150 또는 W100을 임의로 comparison arm으로 실행하지
않았다. nearest-Gaussian ownership과 TSDF connectivity도 재계산하지 않았다.

## 9. Zero-set semantic implication

Gaussian decomposition은 **WHO**, TSDF zero-set sample은 **WHERE**다. 하나의
Gaussian subset이 spatially sparse한 zero-set sample을 갖는 것은 가능하며, missing
sample/native component gap/empty chart cell을 새 Local Surface identity로
해석하지 않는다. Structural NURBS Carrier와 Observed Evidence Membership도
동일시하지 않는다.

## 10. Visual review

이번 배치에서는 새 비교 시각화를 생성하지 않았다. 요청된 A/B visualization은
“approved historical comparison contract”가 있을 때만 의미가 있는데, 이번
감사 결과는 단일 contract가 아니라 split이기 때문이다. W175의 W97 lineage
visuals와 W150/W154–W175의 기존 artifact는 변경 없이 retained 상태다.

## 11. Architecture attribution

W154의 W97→W171 identity lineage는 보존됐다. 그러나 W154는 W150이 기록한
`form_surface_regions`의 covariance/reliability/affinity/consensus/bridge/path
계약을 소비하지 않는다. 따라서 **W154는 더 오래된/다른 W97 contract를 소비하며,
W150 관점에서는 architecture drift가 존재한다.** 이것은 W97 identity가 틀렸다는
판정도, W150을 자동으로 대체 contract로 채택하자는 판정도 아니다.

## 12. Promoted / Retained / Rejected / Open

- **Promoted:** W10/W31–W38 `form_surface_regions` 구조적 semantics는 covariance
  branch 내부에서 구현·보존됐다. W96/W97 intrinsic `t_w`는 W154 active baseline
  identity로만 유지된다.
- **Retained:** W97/W154/W171 exact lineage, Gaussian WHO/TSDF WHERE 분리,
  W98–W100 diagnostic evidence, W150 bypass evidence.
- **Rejected:** W175 identity 보존만으로 W97을 final이라고 선언, W150 자동 대체,
  W98/W99/W100 자동 promotion, TSDF component-based identity, 새 hybrid 구현.
- **Open:** intrinsic `t_w`와 승인된 structural safeguard를 함께 만족하는
  architecture contract, W97↔W150 same-checkpoint crosswalk, 최종 normal authority.

## 검증

정적 contract audit focused tests:

```text
tests/test_worklog_176_local_surface_contract_audit.py
```

이번 배치는 checkpoint replay나 full regression을 실행하지 않았다. Worklog
completion condition에 따라 historical intent와 W154 active path를 판정하고,
contract split/drift 지점에서 중단했다.
