# Worklog 159 — Partial Zero-Set Topology와 명시적 Ambiguity 계약 감사

## 상태

완료. 최종 verdict는 **`AMBIGUITY_IS_MACRO_TOPOLOGY_CRITICAL`**이다. W158의 `ZERO_SET_CONNECTIVITY_CONTRACT_GAP`을 그대로 반복하지 않고, deterministic multi-patch와 실제 topology ambiguity를 분리한 뒤 ambiguity leverage를 측정했다.

## 수행 내용

- W158 Region 0/2/5 Candidate G NPZ의 node/edge/component/raw ambiguous 수치를 재현했고 primary count가 모두 exact 일치했다.
- 동일 cell 내부 triangle의 shared zero-crossing lattice-edge로 local patch를 계산했다. 두 개 이상인 cell은 `DETERMINISTIC_MULTI_PATCH`로 분류하고, exact zero vertex/edge/face, exact bilinear face-decider tie, W158 invalid incidence 및 local triangle 부재만 ambiguity 후보로 남겼다.
- Candidate H는 deterministic cell의 shared entity만 사용한 partial lower-bound graph다. `GUARANTEED_CONNECT`, `GUARANTEED_DISCONNECT`, `TOPOLOGY_AMBIGUOUS`를 분리했으며 ambiguity를 graph edge로 승격하지 않았다.
- Region 0/2/5 결과는 각각 deterministic multi-patch `129,810/41,068/40,153`, genuine ambiguity `10,787/3,744/3,517`이다. ambiguity interface는 `378/447/129`개이고, 서로 다른 H component를 동시에 접하는 셀은 `3/2/1`개로 macro identity 변경 가능성이 확인됐다. hypothetical envelope의 component reduction은 `5/2/1`이지만 이는 production topology가 아니다.
- macro topology가 안정적이지 않아 Boundary First/WL139 replay는 조건부로 보류했다. W159는 ambiguity가 있어도 real-scene common-world A–H PNG view를 생성했다.
- W155 canonical Gaussian `Original Scene`/`Observed-Occluded` pair를 PNG-only로 보존했다. W159 output은 PNG 26개, PPM 0개이며 시각화 하위 directory 전체에 UTF-8 README를 추가했다.
- 이전 요청에 따라 W153 output→temp mirror는 `replay_cache/`를 제외하도록 수정했고 제외 목록을 결과에 기록한다.

## 평가

- 합성 A–H contract와 W159 focused 6개, 기존 W158 5개, W153 회귀를 통과했다.
- W158의 다수 ambiguous accounting이 단순한 ambiguity가 아니라 deterministic multi-patch로 설명될 수 있음을 확인했지만, 소수의 unresolved interface가 여러 H component identity에 실제 leverage를 가지므로 전체 topology 승격은 불가하다.
- native Candidate F/6-face membership, ownership, scalar field, zero-surface eligibility, one-cell gap은 변경하지 않았다. global mesh, 18/26-neighbor bridge, radius matching, gap fill, Boundary First tuning, NURBS tuning은 수행하지 않았다.

## 산출물과 잔여 위험

- Report: `output/159_partial_zero_set_topology_ambiguity_contract_audit/worklog_159_report.json`
- Candidate H arrays: `candidate_h_region_000000.npz`, `candidate_h_region_000002.npz`, `candidate_h_region_000005.npz`
- Review: `output/159_partial_zero_set_topology_ambiguity_contract_audit/review_views/`
- 잔여 위험은 local scalar field만으로 unresolved interface의 physical-sheet identity를 결정할 수 없다는 점이다. 추가 field sample 또는 별도 physical prior 없이는 ambiguous join을 선택할 수 없다.