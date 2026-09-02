# Worklog 151 — Renderer Evidence와 Canonical Local Surface / Boundary First Ownership 호환성

## 상태

완료. WL150의 `ARCHITECTURE_BYPASS` 결과를 전제로, renderer median event를
기존 canonical ownership 계약에 의미 변경 없이 넣을 수 있는지 먼저 감사했다.
호환성 gate가 닫혀 Candidate C나 새 inference를 실행하지 않았다.

## 1. 현재 질문

질문은 다음과 같다.

> renderer-event evidence가 새 의미를 발명하지 않고 기존 canonical Local Surface
> Decomposition / Boundary First 계약에 합법적으로 들어갈 수 있는가?

## 2. WL150 baseline reconciliation

WL149/WL150 frozen baseline을 먼저 replay했다.

- event count: `1586`
- event-union SHA-256: `79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78`
- event 1527: `DSC08003.JPG`, pixel `(259,169)`, `v_min` owner 유지
- representative shape: `3840 x 3`
- support: `314`, support mask hash `23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9`
- all-four supported cells: `211`

baseline reconciliation은 exact PASS이며 historical 배열을 변경하지 않았다.

## 3. Canonical Local-Surface contract

실행 소스상 canonical constructor는 positions와 함께 covariance 또는
`log_scales + rotations`, unique stable IDs를 요구한다. 그 뒤 covariance frame,
structural reliability, manifold affinity graph를 만들고 `form_surface_regions`가
same-surface/crease/parallel-separate/rejected 관계, shared-neighbor consensus,
bridge/path consistency를 이용해 `node_region_id`와
`SurfaceRegionCandidate.region_id/member_ids/core_member_ids`를 만든다.

기계적 입력과 물리적 의미를 분리하면 다음과 같다.

- 기계적 입력: XYZ, covariance/frame, reliability, graph, stable IDs.
- 물리적 의미: 같은 surface 관계, local topology, normal/tangent coherence,
  region별 member/core/ambiguous/rejected provenance.

`event_id`는 unique row ID로는 deterministic mapping이 가능하지만 region ownership은
아니다.

## 4. Canonical Boundary-First contract

canonical path는 accepted region과 oriented normal/canonical frame에서 world-space
boundary halfedge를 만들고, directed compatibility/order를 통해
`OrderedBoundaryComponent`를 만든다. `ordered_closed_loop`이며 outer role이고 branch가
없는 component만 eligible하다. 이후 simple-loop 검사를 통과한 boundary points와 같은
region의 core interior points가 `materialize_visible_boundary_component`에 **fit 전**
전달된다.

open/branch/ambiguous boundary에는 synthetic rectangular closure를 만들지 않는다.
adapter가 요구하는 ownership은 source region ID, component ID, ordered boundary IDs,
interior IDs, supporting source IDs이다.

## 5. Renderer-event evidence contract

각 frozen event에는 world XYZ, renderer median depth, source camera/pixel, event normal,
per-view provenance가 직접 있다. WL149 row order로 unique event ID를 replay할 수 있다.

다음은 없다.

- Gaussian covariance/log-scales/rotations와 canonical reliability 결과
- primitive/contributor identity
- visible topology identity
- local region/member/core identity
- adjacency, ordered boundary, boundary ownership
- executable physical-sheet identity

주의할 의미 차이는 다음과 같다.

- renderer contribution provenance ≠ physical-sheet membership
- manual polygon/control label ≠ physical-sheet identity
- camera co-visibility ≠ surface topology
- event normal ≠ region identity
- spatial proximity ≠ ownership
- event row ID ≠ canonical member/region ID

event 1527에 대해서만 외부 검토 결과를 유지한다.

`HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE`

## 6. Compatibility matrix

호환성 matrix는 산출물 `compatibility_matrix.json/.md`에 전체 항목을 저장했다.
주요 결과는 다음과 같다.

- `COMPATIBLE`: local positions 1개
- `COMPATIBLE_BY_EXISTING_DETERMINISTIC_MAPPING`: event row ID 및 이미 수행된
  camera/pixel/depth→XYZ 복원 2개
- `SEMANTICALLY_DIFFERENT`: event normal, renderer evidence state,
  physical-sheet control label 3개
- `INCOMPATIBLE_MISSING`: covariance/reliability/graph/region/member/boundary/core
  contract 10개

특히 manual polygon을 physical-sheet membership로, camera agreement를 topology로,
normal을 region identity로 바꾸는 것은 기존 deterministic mapping이 아니다.

## 7. Adapter eligibility verdict

**STOP CONDITION A — `CONTRACT_GAP`**

기존 canonical mechanism을 호출하려면 physical-sheet membership, local region
ownership, manifold adjacency/topology, boundary ownership/order, Gaussian
covariance/reliability 정합을 새로 정의해야 한다. 이는 representation plumbing이
아니라 새 semantic/inference contract다.

따라서 adapter eligibility는 `false`다.

## 8. Conditional candidate implementation

실행하지 않았다. `Candidate C — Canonical Pre-Fit Ownership Restoration`은
compatibility gate가 PASS일 때만 허용되므로 구현하지 않았다. canonical constructor에
renderer event를 억지로 넣거나, 새 KNN/distance/normal/correspondence/classifier를
만드는 작업도 하지 않았다.

## 9. Extrema owner accounting

historical WL149 owner `947`, `795`, `1104`, `1527` 모두 별도 기록했지만 Candidate C
region assignment는 `NOT_RUN_DUE_TO_CONTRACT_GAP`이다. 1527은 여전히 human review상
off-surface이며 frozen `v_min` owner다. 네 샘플에서 rejection rule을 만들지 않았다.

## 10. Synthetic contracts

`NOT_RUN`. Stop Condition A 이후 synthetic adapter fixture를 실행하면 missing
semantics를 임의로 채우게 되므로 실행하지 않았다.

## 11. Real-scene quantitative result

Candidate C가 없으므로 `NOT_RUN`. 기존 WL149 quantitative 결과와 historical Arm A/B는
보존했으며, 이번 batch에서 새 geometry나 metrics를 생성하지 않았다.

## 12. Real-scene qualitative review

Candidate C geometry가 없으므로 `NOT_RUN`. 기존 WL149 visualization은 historical
산출물로 남아 있고, 이번 compatibility audit를 위해 수정하지 않았다.

## 13. Architecture verdict

**A. CONTRACT_GAP**

Renderer event는 rendering provenance와 geometry는 제공하지만 canonical local
surface/boundary ownership을 의미 그대로 제공하지 않는다. 따라서 WL150에서 확인한
bypass를 이 batch에서 복원하지 않는 것이 맞다.

## 14. Retained / Rejected / Open

- Retained: WL139/WL145/WL148/WL149/WL150, 1586 union, event 1527, renderer median
  semantics, PCA/representative/support historical outputs, canonical source.
- Rejected: blind constructor reconnection, event 삭제, new membership/adjacency/
  correspondence heuristic, threshold tuning, Candidate C 구현, synthetic/real-scene
  success claim.
- Open: renderer event를 canonical region에 연결할 physical-sheet membership와
  covariance/reliability semantics를 별도 architecture batch에서 정의할 수 있는지,
  그리고 future boundary owner가 어떤 validated evidence를 가져야 하는지.

## Intent alignment

Stop Condition A를 적용했다. geometry, PCA, NURBS, support, renderer, canonical
production behavior는 변경하지 않았고 event 1527도 삭제하지 않았다.

## Implementation fidelity

소스 함수/라인 inventory와 historical commit, frozen WL149 baseline을 산출물에
기록했다. Candidate C, synthetic, real replay, qualitative export는 의도적으로
실행하지 않았다. 새 threshold, classifier, outlier rule, primitive inference,
appearance/SH, continuation, Occluded Surface는 추가하지 않았다.

## Architecture result

`A. CONTRACT_GAP`. 이번 batch의 완료 조건은 compatibility NO를 정확히 증명하는
것이며, missing semantics를 발명하지 않고 중단하는 것이다. 새 산출물은
`output/151_renderer_event_canonical_surface_compatibility_audit/`와
`temp/151_renderer_event_canonical_surface_compatibility_audit/`에 있다.

Focused tests: `4 passed`.
