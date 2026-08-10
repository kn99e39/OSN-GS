# Worklog 87: partition_seam을 1급 parametric-boundary 타입으로 — 최종 go/no-go

## 목적

Worklog 86은 partition seam을 검증했지만 "열린 physical fragment 정확히 1개 + interior seam 1개"라는 제한된 경우만 구현했다 — physical boundary-support 존재를 암묵적 전제조건으로 취급했다(candidate 0~2개, fragment 2개 이상인 unit은 아예 시도조차 안 함). 이번 배치는 **partition_seam을 physical_termination/crease/observation_frontier와 동등한 1급 parametric-boundary 타입**으로 일반화하고, 이것이 boundary-first visible constructor의 최종 go/no-go임을 확정한다.

상류 계약 유지: Region ownership, worklog 82 micro-component, worklog 83 assembly, worklog 84 coherence audit, observed/certain evidence만 사용, physical_termination/crease/observation_frontier는 hard constraint, worklog 79 coverage, PCA-UV/6×6 NURBS, visible Gaussian 학습 — 전부 미변경.

## 구현

`osn_gs/surface/torch_chart_unit_partition_seam.py`를 전면 재작성했다(worklog 85의 물리 재구성 메커니즘 자체는 재사용하되, 더 이상 감싸는 wrapper가 아니라 직접 구성):

1. **모든 독립적으로 닫힌 physical loop**(worklog 85의 degree-2-regular cycle detection, 미변경)은 각각 자기 자신의 domain이 된다 — **한 unit이 여러 개의 독립 domain을 낼 수 있다.**
2. 소비되지 않고 남은 candidate(physical loop에 들어가지 못한 fragment·고립점)는 **결정론적 fragment-chain stitching**으로 최대 1개의 추가 domain으로 닫는다: 각 조각(fragment 또는 고립 candidate — 고립점은 시작=끝이 같은 길이-1 조각으로 취급)을 **자신의 첫 candidate stable id**로 정렬(worklog 114의 deterministic-cap 관례, worklog 85의 `min(component)` 시작점과 같은 종류의 재현 가능한 tie-break, 기하·각도 아님)한 뒤, 조각 i의 끝을 조각 (i+1)의 시작에 seam으로 잇고 마지막은 조각 0의 시작으로 되돌아온다. 이는 다음을 전부 하나의 메커니즘으로 지원한다: fragment 1개+seam 1개(worklog 86의 N=1 특수 사례), fragment 2개 이상+seam 여러 개, physical loop이 전혀 없고 seam만으로 닫히는 **seam-dominated** domain(`partition_seam_segment_count > physical_segment_count`일 때 보고). 체인 중 **어느 한 seam이라도 존재하지 않으면 전체 시도가 fail-closed**된다 — 다른 순서나 부분 봉합을 시도하지 않는다(그 자체가 "또 다른 heuristic"이 되므로).
3. **candidate 최소 개수를 3→2로 낮췄다** — 닫힌 물리 loop만 고려하던 시절의 `<3` 제약은 fragment 기반 seam 구조에서는 근거가 없다(fragment는 2점만 있어도 성립). 이는 임의 완화가 아니라 새 아키텍처에 맞는 제약 정정이다. (실측: real data에는 candidate 1~2개인 unit이 존재하지 않아 이 자체는 결과에 영향 없음 — 아래 참고.)

### 구현 중 실측으로 발견·수정한 자체 결함

첫 구현은 seam이 "다른 모든 boundary candidate를 경유지에서 제외"하도록 했다(worklog 86의 설계를 그대로 일반화). real unit(140점, candidate 47개)에서 직접 측정한 결과, 이 제외 규칙 자체가 병목이었다: 제외 없이 interior graph를 BFS하면 두 fragment 끝점이 실제로 연결되는데(38/140 노드 도달, 목표점 포함), "다른 candidate 전부 제외" 정책 아래서는 그 유일한 경로가 막혀 seam-not-found로 실패했다. 과제 제약 어디에도 "seam이 다른 typed candidate를 경유하면 안 된다"는 요구는 없었다 — 그 경유 edge도 여전히 실제 관측된 same_surface adjacency이지 발명된 chord가 아니다. **제외 범위를 "현재까지 체인에 이미 배치된 노드"로만 좁혔고**, 그로 인한 유일한 실질적 위험(아직 처리되지 않은 다른 조각의 노드를 경유지로 써버려 나중에 중복 정점이 생기는 것)은 **최종 체인에서 중복 stable id를 명시적으로 검사해 발견 시 전체 시도를 fail-closed**하는 것으로 방어했다. 이 수정 후에도(아래 실측 참고) 최종 산출량은 크게 달라지지 않았다 — 실패 지점이 이동했을 뿐(같은 unit이 이제 다른 pair에서 실패), 여러 조각을 모두 체인으로 이어야 하는 구조 자체가 real data에서 거의 항상 어느 한 구간에서 끊긴다.

`tests/test_chart_unit_partition_seam.py` 12개: 고립 candidate 탐지 직접 검증, **`_stitch_pieces_into_domain` 직접 검증**(2개 고립점이 2개 seam으로 이어짐, 입력 순서와 무관하게 출력이 결정론적, 2점짜리 조각이 정확히 1개 내부 physical edge를 만듦, 도달 불가능한 끝점은 이유와 함께 fail-closed), 완전히 닫힌 physical disc는 정확히 1개 physical_only domain, **두 개의 독립적인 물리 loop은 별도로 탐지·검증됨**(worklog 79 coverage가 "이 unit 전체 evidence의 절반 이상을 덮는가"를 각 domain마다 검사하므로, 인위적으로 멀리 떨어진 두 evidence를 억지로 합친 unit에서는 각자 절반씩만 덮어 정당하게 coverage_failed — 이것이 coverage 계약이 실제로 하는 일이지 다중-loop 탐지의 결함이 아님을 별도 테스트로 명시), candidate 2개 미만은 새 기준으로 no_dense_support, ambiguous/over-merged는 domain 0개.

## 실측 (real baseline_compatible@2900, 7개 region 전체, evidence 3526점)

| reg | evid | units | coherent% | domain% | valid% | physical% | mixed% | seam-dom% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 93 | 3 | 92.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 519 | 24 | 89.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 510 | 22 | 96.5 | 0.8 | 0.8 | 0.8 | 0.0 | 0.0 |
| 3 | 92 | 9 | 80.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 4 | 1035 | 56 | 86.6 | 2.0 | 0.5 | 0.2 | 0.5 | 1.4 |
| 5 | 375 | 7 | 93.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 6 | 902 | 57 | 82.7 | 1.2 | 0.7 | 0.9 | 0.0 | 0.3 |

**전체 가중 합계: coherent chart-unit coverage 88.1%, partitioned parametric-domain coverage 1.0%, valid_supported 0.4%, extrapolative 0.3%, unsafe_geometry 0.3%, unresolved 87.1%.** boundary 구성(evidence 가중): physical_only 0.4%, mixed 0.1%, **seam_dominated 0.5%** — seam이 완전히 지배하는 domain이 실제로 real 데이터에서 나타난다(seam-dominated가 physical_only와 비슷한 비중), 일반화가 실질적으로 다른 종류의 domain을 만들어낸다는 증거다.

178개 unit 전체 domain 상태: `materialized` 8개(3 valid_supported + 1 extrapolative + 4 unsafe_geometry) — worklog 86의 8개(4 valid + 4 unsafe)와 **총량은 거의 동일**하다.

### worklog 86의 no_dense_support / multi_fragment_unresolved 회수 여부

| worklog 86 분류 | unit 수 | 이번에 materialize된 수 |
|---|---:|---:|
| `no_dense_support`(candidate<3, 대부분 candidate=0) | 71 | **0** |
| `multi_fragment_unresolved`(열린 fragment 2개 이상) | 23 | **0** |

`no_dense_support`는 구조적으로 회수 불가능함을 실측으로 확인했다: 71개 중 **64개가 candidate 정확히 0개**다(1~2개인 경우는 real data에 아예 없음) — seam은 물리 candidate 최소 1개 쌍이 있어야 시작할 수 있는데, 애초에 이을 물리 anchor 자체가 없다. "physical evidence를 전제조건으로 삼지 않는다"는 것은 seam 시도의 **트리거 조건**에 관한 것이지, seam이 물리 evidence 없이도 처음부터 경계를 만들어낼 수 있다는 뜻이 될 수는 없다 — 그렇게 하면 evidence 없이 geometry를 발명하는 것이 되어 명시적으로 금지된 항목("invented geometric chord")을 어기게 된다.

`multi_fragment_unresolved`는 **일반화된 daisy-chain으로 정말로 시도됐지만** 조각이 여러 개일수록 체인의 **모든** 구간(seam)이 다 성공해야 하나의 domain이 되는 구조상, real data에서는 어느 한 구간이라도 끊길 확률이 누적돼 전체가 거의 항상 실패한다 — 자체 결함(과도한 candidate 제외) 수정 후에도 실패 지점만 이동했을 뿐 최종 결과는 그대로였다(같은 실측으로 확인).

## 판정

**NO-GO. partition_seam을 1급 타입으로 일반화하는 아키텍처는 완성했지만, real 관측 evidence에 적용한 실질 산출량은 worklog 86의 제한된 버전과 거의 같다(8/178 unit, evidence 가중 1.0%). 관측된 Gaussian evidence의 절대다수(87.1%)는 이 일반화된 표현으로도 parametric domain에 도달하지 못한다.**

세부적으로:

- **일반화 자체는 실재하는 새 능력을 만든다.** seam-dominated domain(evidence-only 0.5%)이 실제로 나타나 physical evidence가 지배적이지 않은 domain도 만들 수 있음을 증명했고, 자체 발견한 exclusion 버그를 수정해 seam 탐색 자체의 정확성도 개선했다(같은 제약 위반 없이 더 많은 경로를 찾음).
- **그러나 두 개의 근본적 병목이 이 일반화로는 풀리지 않는다.** (1) `no_dense_support`(전체 unit의 40%)는 seam의 시작점이 될 physical evidence 자체가 없어 원천적으로 회수 불가능 — 이것은 "prerequisite 제거"로 해결할 수 있는 종류의 문제가 아니라 evidence 자체의 부재다. (2) `multi_fragment_unresolved`(13%)는 daisy-chain의 **모든 구간이 동시에 성공**해야 하는 구조적 요구 때문에, fragment가 많을수록 성공 확률이 급격히 낮아진다 — 이는 seam 알고리즘의 결함이 아니라 real evidence가 애초에 fragment 여러 개로 흩어져 있다는 것 자체가 문제의 근원임을 재확인한다.
- worklog 79~86이 이미 도달한 결론과 **완전히 일치한다**: 병목은 표현/알고리즘이 아니라 **관측된 Gaussian evidence 자체의 밀도·연속성**이다.

## 결론: boundary-first visible constructor 재설계 최종 종료

**worklog 79부터 87까지, chart-domain coverage 계약 → 표현 재설계 → parameterization 결정 → chart-unit 분해 → 조립 → coherence audit → evidence-scale boundary → local surface topology → 제한된 partition seam → 일반화된 partition seam까지, 매 라운드가 실제 결함을 찾아 고쳤고 매번 실측으로 검증했다. 최종 결론은 아홉 번째 라운드에 걸쳐 일관되게 하나였다: 현재 학습된 Gaussian evidence는 의도한 parametric visible-surface 표현에 불충분하다.** 지시대로 이 결과를 또 다른 고립된 boundary 실험으로 잇지 않는다.

## 검증

`tests/test_chart_unit_partition_seam.py` 신규 12개 전부 통과. 관련 기존 테스트(`test_chart_unit_evidence_scale_boundary.py`, `test_dense_surface_consistency_components.py`, `test_dense_chart_unit_assembly.py`, `test_boundary_support_spacing.py`, `test_dense_parametric_chart_support.py`, `test_region_owned_full_evidence.py`) 재실행 79개 전부 통과. **NO-GO 판정이므로 canonical Region→Chart 계층을 교체하지 않았고 전체 regression은 실행하지 않았다**(지시: go 판정일 때만 통합+전체 regression). hull·PCA rectangle·bounding box·alpha shape·강제 폐쇄·gap bridging·region merge·centroid 정렬·fit-품질 기반 분할·원하는 patch 개수를 향한 threshold 조정은 전혀 사용하지 않았다. worklog 77 predicate, worklog 82 same_surface 기준(0.85/0.35), worklog 85의 curve 전용 그래프(unrestricted search/cap=2), worklog 82의 interior-mesh 기본값(k=8/cap=12)은 전부 미변경 그대로 재사용했다 — 변경한 것은 seam 경유지 제외 범위(자체 결함 수정, 중복-정점 가드로 안전성 보강)와 candidate 최소 개수(3→2, 아키텍처 정정)뿐이며 둘 다 "원하는 patch 수를 향한 튜닝"이 아니라 새 아키텍처에 맞는 제약 정정임을 실측으로 뒷받침했다. visible Gaussian photometric 학습과 상류 region ownership은 손대지 않았다.
