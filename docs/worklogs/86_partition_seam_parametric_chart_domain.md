# Worklog 86: partition-seam 방식의 parametric chart-domain 계약 — 최종 go/no-go

## 목적

Worklog 85는 엄격한 물리 perimeter 재구성 경로를 닫았다: coherent evidence의 83%가 유효한 manifold topology를 형성하지 못했다 — 메커니즘 결함이 아니라(합성 데이터로 직접 검증), real evidence 자체가 하나의 지배적 perimeter가 아니라 다수의 작은 조각/열린 fragment로 흩어져 있기 때문이었다. 그러나 real 데이터는 coherent chart-scale surface unit(worklog 82+83+84) 자체는 회수 가능함을 보여준다 — 문제는 "coherent surface가 있는가"가 아니라 "그 전체 물리 경계를 항상 관측된 boundary-support Gaussian의 닫힌 loop로 요구할 수 있는가"였다.

이번 배치는 **physical surface termination과 parametric chart boundary가 동등하지 않다**는 전제로 parametric chart-domain 계약 자체를 재설계한다: chart 경계는 physical_termination/crease/observation_frontier뿐 아니라 **evidence로 정당화된 partition_seam**을 포함할 수 있다. 이 seam은 fitting 이전에, 오직 관측된 topology/evidence로만 결정되며, fit 오차·held-out p95·NURBS 품질·원하는 patch 개수는 seam 결정에 절대 관여하지 않는다.

상류 계약 유지: Region ownership, worklog 82 surface-consistent evidence unit, worklog 83 chart-scale assembly, worklog 84 coherence audit, physical_termination/crease/observation_frontier provenance, worklog 79 coverage, PCA-UV/6×6 NURBS, visible Gaussian 학습 — 전부 미변경.

## 구현

신규 `osn_gs/surface/torch_chart_unit_partition_seam.py`는 worklog 85의 물리 재구성(`materialize_chart_unit_boundary`)을 **완전히 미변경으로 감싸는(wrap) 구조**다:

1. **먼저 물리 경계만으로 시도**(worklog 85, 미변경) — 이미 닫힌 loop이 있으면 그것이 chart 경계다(`boundary_composition="physical_only"`, seam 시도 없음).
2. 물리 재구성이 `NO_VALID_LOOP_TOPOLOGY`(닫힌 loop 없음)를 반환하고, 그 안에서 **정확히 하나의 열린 physical fragment**(느슨한 끝 두 개를 가진 path)를 발견했을 때만 partition seam을 시도한다: fragment의 두 끝점을 **unit 자신의 interior local 2-manifold adjacency graph**(`build_same_surface_adjacency`, worklog 82의 원래 interior-mesh 기본값 k=8/cap=12, 미변경 — worklog 85가 curve 전용으로 쓴 unrestricted/cap=2 그래프와는 다른, 원래 의도된 interior 용도로 재사용)를 통한 **최단 경로**로 잇는다. seam의 모든 edge는 실제 관측된 두 점 사이의 진짜 same_surface adjacency다 — 빈 공간을 가로지르는 chord가 아니다. 다른 boundary candidate는 seam 중간 경유지에서 제외한다(seam은 진짜 interior를 통과해야지, 미확보된 다른 boundary 후보를 임의로 경유해서는 안 된다). 기존 typed crease veto가 이 그래프에도 동일하게 적용되므로 seam은 기존 crease를 절대 가로지르지 못한다 — physical/crease/frontier는 hard constraint로 유지된다.
3. **열린 fragment가 2개 이상이면 seam을 시도하지 않는다** — 여러 fragment를 어떻게 이을지는 조합적으로 모호하므로 추측하지 않고 `STATE_MULTI_FRAGMENT_UNRESOLVED`로 명시 disclosure한다.
4. seam을 찾으면, physical fragment + seam이 정확히 하나의 cycle을 이루므로(둘 다 같은 두 끝점을 공유하는 열린 path이기 때문) worklog 85와 **동일한 두 개의 독립 안전장치**로 검증한다 — self-intersection(`evaluate_closed_loop_geometry`, 미변경)과 observed-support occupancy(`measure_edge_support_occupancy`, worklog 76, 미변경) — 그 다음 worklog 79 coverage 계약(미변경).

fit 오차·held-out p95·NURBS 품질·원하는 patch 개수는 이 모듈의 어디에서도 읽지 않는다 — seam 결정과 검증은 파라미터화·fitting이 시작되기 전에 전부 끝난다. 모든 segment는 physical/partition_seam으로 명시 구분되어(`is_partition_seam`) materialized domain이 갖지 않은 physical evidence를 절대 암묵적으로 주장하지 않는다.

`tests/test_chart_unit_partition_seam.py` 15개: 완전히 닫힌 물리 disc는 seam을 절대 시도하지 않음(pass-through), ambiguous/no-dense-support는 그대로 통과, **`_find_open_paths` 직접 검증**(단일 open path 발견, closed cycle은 path로 오분류 안 함, 두 개의 분리된 path 모두 발견, branching component는 path로 취급 안 함), **`_find_partition_seam` 직접 검증**(dense coherent interior를 통한 seam 발견, 진짜로 끊긴 경우 None 반환, 제외된 index가 seam 중간에 절대 등장하지 않음), multi-fragment 상태의 도달 가능성 검증. (평탄한 synthetic disc에서는 모든 점이 같은 normal을 공유해 worklog 85의 "무제한 탐색 폭"이 먼 후보끼리도 same_surface를 통과시켜 spurious closure가 나거나, 곡률을 주면 coherence audit 자체가 기각하는 좁은 경계만 존재함을 직접 실측으로 확인했다 — 그래서 seam 메커니즘 자체는 hand-built adjacency로 직접 검증했다.)

## 실측 (real baseline_compatible@2900, 7개 region 전체, evidence 3526점)

| reg | evid | units | coherent% | domain% | valid% | phys_only | phys+seam | phys segs | seam segs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 93 | 3 | 92.5 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| 1 | 519 | 24 | 89.4 | 0.8 | 0.0 | 0 | 1 | 1 | 1 |
| 2 | 510 | 22 | 96.5 | 0.8 | 0.8 | 1 | 0 | 4 | 0 |
| 3 | 92 | 9 | 80.4 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| 4 | 1035 | 56 | 86.6 | 1.3 | 0.9 | 1 | 2 | 5 | 2 |
| 5 | 375 | 7 | 93.3 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| 6 | 902 | 57 | 82.7 | 1.7 | 0.7 | 2 | 1 | 9 | 1 |

**전체 가중 합계: coherent chart-unit coverage 88.1%, partitioned parametric-domain coverage 1.0%, valid_supported 0.5%, extrapolative 0.0%, unsafe_geometry 0.5%, unresolved 87.1%.**

boundary 구성: **physical_only unit 4개, physical_plus_partition_seam unit 4개** — worklog 85의 materialized 4개(전부 physical_only)에서 **materialized unit 총수가 4→8로 두 배** 됐고 valid_supported도 2→4로 두 배 됐다. seam이 실제로 유효한 domain을 새로 만들어냈다는 것은 실측으로 확인된다.

178개 unit 전체의 상태 분포: `no_dense_support` 71, `unsupported_closure` 33, `coverage_failed` 29, `multi_fragment_unresolved` 23, `materialized` 8, `seam_not_found` 8, `self_intersecting` 5, `no_valid_loop_topology`(branch만) 1.

## 판정

**NO-GO — partition seam 메커니즘은 건전하고 실제로 evidence-backed domain을 늘리지만(fabrication 없음), 산출량이 production 채택에 필요한 규모에 크게 못 미친다. 현재 학습된 Gaussian evidence는 의도한 parametric visible-surface 표현에 불충분하다.**

세부적으로:

- **메커니즘 자체는 정당하다.** materialized unit이 4→8(2배), valid_supported가 2→4(2배)로 실측 증가했고, 모든 seam edge가 진짜 관측된 same_surface adjacency임을 전용 테스트로 직접 검증했다(hull·bounding box·gap bridging·centroid 정렬 없음, fit 품질 미참조). "physical termination과 parametric chart boundary는 다르다"는 이번 배치의 핵심 전제는 real 데이터에서 실제로 작동한다.
- **그러나 산출량은 여전히 매우 작다.** coherent evidence의 88.1% 중 partitioned domain에 도달한 것은 1.0%뿐이다. 87.1%는 여전히 unresolved다. 원인 분포를 보면 seam이 해결할 수 없는 근본 문제가 지배적이다:
  - `no_dense_support`(71/178, 40%): boundary 후보가 아예 admitted되지 않은 unit — seam은 물리 fragment의 끝을 잇는 메커니즘이므로 애초에 물리 evidence가 전혀 없으면 적용할 수 없다.
  - `multi_fragment_unresolved`(23/178, 13%): fragment가 2개 이상이라 어느 쌍을 이을지 근거 없이 정할 수 없어 의도적으로 미시도.
  - `unsupported_closure`+`coverage_failed`+`self_intersecting`(33+29+5=67/178, 38%): loop(물리 단독이든 physical+seam이든)은 만들어졌지만 evidence 밀도·정합성이 최종 안전장치(occupancy/coverage/self-intersection)를 통과하기에 부족 — seam을 추가해도 이 실패들의 근본 원인(evidence 자체의 밀도·정합성)은 바뀌지 않는다.
  - `seam_not_found`(8/178, 4%): fragment는 정확히 하나지만 그 두 끝이 interior graph로도 전혀 연결되지 않는 진짜 단절.

즉 seam 메커니즘이 다루는 "단일 fragment + 연결 가능한 interior"라는 조건 자체가 real evidence에서 드물다 — 대부분의 unit은 물리 evidence가 아예 없거나(40%), fragment가 여럿이거나(13%), loop은 만들어져도 evidence 자체가 최종 검증을 통과할 만큼 밀집·정합적이지 않다(38%). 이는 seam 알고리즘을 더 다듬는다고 해결되는 문제가 아니라, **관측된 Gaussian evidence의 밀도·연속성·정합성 자체가 chart-scale parametric domain을 대량으로 지지하지 못한다**는 것을 가리킨다.

## 결론: boundary-first visible constructor 재설계 중단

지시대로, evidence-backed partition seam이 usable real parametric domain을 산출하지 못했으므로 **visible-constructor 재설계를 여기서 멈춘다.** worklog 79부터 86까지(chart-domain coverage 계약 → 표현 재설계 → parameterization 결정 → chart-unit 분해 → 조립 → coherence/evidence-scale boundary → partition seam) 매 라운드가 실제 결함을 찾아 고쳤고 매번 실측으로 검증했지만, 최종적으로 도달한 결론은 일관되다: **현재 학습된 Gaussian evidence는 의도한 boundary-first parametric visible-surface 표현에 불충분하다.** 이는 다음 boundary 실험으로 이어지지 않는다 — 이 결론 자체가 이번 라인의 최종 산출물이다.

## 검증

`tests/test_chart_unit_partition_seam.py` 신규 15개 전부 통과. 관련 기존 테스트(`test_chart_unit_evidence_scale_boundary.py`, `test_dense_surface_consistency_components.py`, `test_dense_chart_unit_assembly.py`, `test_boundary_support_spacing.py`) 재실행 51개 전부 통과. **NO-GO 판정이므로 canonical Region→Chart 계층을 교체하지 않았고 전체 regression은 실행하지 않았다**(지시: go 판정일 때만 통합+전체 regression). hull·PCA rectangle·bounding box·alpha shape·강제 폐쇄·gap bridging·region merge·centroid 정렬·fit-품질 기반 분할은 전혀 사용하지 않았다. worklog 77 predicate, worklog 82 same_surface 기준(0.85/0.35, k=8/cap=12), worklog 85의 curve 전용 그래프(unrestricted/cap=2)는 전부 미변경 그대로 재사용했다 — 새 threshold는 도입하지 않았다. visible Gaussian photometric 학습과 상류 region ownership은 손대지 않았다.
