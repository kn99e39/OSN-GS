# OSN-GS Direction Reset Plan
## From Global Surface Decomposition to Boundary-Conditioned Occluded Surface Construction

**문서 목적**  
이 문서는 OSN-GS의 기존 visible-surface 중심 개발 방향을 재검토하고, 프로젝트의 본래 목표인 **Occluded Surface의 NURBS 기반 구축**에 맞춰 methodology와 구현 우선순위를 재정립하기 위한 Agent 작업 지침이다.

Agent는 이 문서를 기존 Phase 1–5 문서 및 worklog보다 상위의 방향성 재정의 문서로 읽되, 기존 구현을 임의로 삭제하거나 production 경로를 변경해서는 안 된다. 먼저 현재 구조를 감사하고, 기존 방향과 새 방향의 차이를 명확히 문서화한 뒤, 새 방향으로의 전환 계획을 제안하고 멈춘다.

---

# 1. 프로젝트의 본래 목표

OSN-GS의 최종 목적은 visible surface 전체를 완벽하게 복원하거나 scene-wide surface segmentation을 수행하는 것이 아니다.

```text
Observed Gaussian surface evidence
→ local visible NURBS patches
→ extension-relevant visible boundaries
→ occluded-region hypotheses
→ constrained occluded NURBS charts
→ occluded Gaussian placement
```

Visible-surface decomposition, topology recovery, patch coupling은 모두 **Occluded Surface 구축을 위한 보조 수단**이다.

---

# 2. 기존 방향성

기존 방향은 다음 문제를 중심에 두었다.

> Sparse/curved Gaussian observations에서 실제로 동일한 visible surface component를 안정적으로 복원한다.

기존 흐름:

```text
Observed Gaussians
→ voxel leaves
→ surface components
→ topology classification
→ topology-aware visible NURBS charts
→ occluded extension
```

이를 위해 adaptive voxel hierarchy, face adjacency, local plane compatibility, boundary extraction, topology routing, annulus O-grid, coupled patch-boundary fitting, proxy-based decomposition, Gaussian-native support continuity를 조사했다.

## 기존 방향의 한계

- Sparse/density-gradient로 끊겨 보이는 동일 surface와 실제 disconnected surface를 local pairwise evidence로 안정적으로 구분하지 못했다.
- Global component correctness가 본래 목적보다 과도한 upstream requirement가 되었다.
- Facing direction과 normal similarity를 강한 gate로 사용할 경우 직교·사선 surface 사이의 valid occluded transition을 놓칠 수 있다.
- Voxel/contact topology가 actual surface continuation을 대신 판단하는 구조가 curved surface에서 실패했다.

---

# 3. 새 방향성

새 methodology의 중심 문제는 다음이다.

> Visible NURBS boundary에서 확장한 parametric continuation domains가 공통의 비관측 공간을 지지할 때 occluded-surface candidate를 생성하고, constrained NURBS chart의 geometric validity와 uncertainty를 통해 후보를 평가한다.

새 흐름:

```text
Observed Gaussians
→ local visible NURBS patches
→ adjacent patch-boundary reconciliation
→ remaining open boundary extraction
→ parametric continuation domain generation
→ high-recall occluded-region candidate formation
→ constrained occluded NURBS chart construction
→ validity and uncertainty evaluation
→ occluded Gaussian placement
```

---

# 4. 기존 방향과 새 방향 비교

| 항목 | 기존 방향 | 새 방향 |
|---|---|---|
| 핵심 목표 | 정확한 visible component 복원 | occluded NURBS surface 구축 |
| voxel 역할 | connectivity 결정 | local support partition |
| decomposition | global topology 전제 | local patch support 생성 |
| artificial boundary | upstream 오류 | downstream reconciliation 가능 |
| boundary 처리 | semantic 3분류 | mergeable / unsupported / extension candidate |
| candidate 생성 | strict pair compatibility | high-recall continuation-domain overlap |
| facing direction | 강한 gate | soft evidence |
| normal similarity | 강한 gate | curvature·uncertainty evidence |
| 직교 surface | 쉽게 제외 | valid candidate로 허용 |
| validation | connectivity correctness | chart validity·visibility·uncertainty |

---

# 5. 새 Methodology 요구사항

## 5.1 Local Visible NURBS Patch Representation

각 local patch는 다음 정보를 제공해야 한다.

- boundary curve
- control points와 knot
- first/second derivatives
- patch-local UV orientation
- boundary-adjacent internal isocurve
- patch adjacency/provenance

완벽한 global component segmentation은 요구하지 않는다.

## 5.2 Patch-Boundary Reconciliation

Local decomposition 또는 chart partition 때문에 생긴 인접 boundary는 다음 근거로 재통합한다.

- scale-normalized proximity
- boundary tangent compatibility
- local derivative/normal compatibility
- coupled shared-boundary fitting
- post-fit Jacobian validity

금지:

- voxel ID만으로 자동 병합
- 모든 adjacent voxel patch 일괄 통합
- absolute distance threshold 단독 사용
- scene-specific merge rule

Voxel occupancy를 occlusion domain과 동일시하지 않는다.

## 5.3 Extension Boundary Extraction

Reconciliation 후 남은 open boundary를 다음 상태로 다룬다.

```text
1. reconciled internal boundary
2. unsupported open boundary
3. occluded-surface extension candidate
```

True object boundary semantic classifier는 필수 요구사항이 아니다.

## 5.4 Parametric Continuation Domain

Visible NURBS surface \(S(u,v)\)에서 boundary derivative를 직접 사용한다.

예를 들어 boundary가 \(u=0\)이면:

- boundary tangent: \(S_v\)
- boundary-normal surface tangent: \(S_u\)
- surface normal: \(S_u 	imes S_v\)

필수 작업:

- outward parametric direction 결정
- derivative orientation 통일
- inner isocurve 확보
- first-order continuation strip 생성
- optional curvature-aware strip 생성
- local confidence 기록

Local geometry를 Gaussian에서 다시 추정하지 않는다.

## 5.5 High-Recall Occluded-Region Candidate Formation

두 개 이상의 continuation domains가 다음 중 하나를 만족하면 candidate로 유지한다.

- 교차
- 근접
- 동일한 bounded empty/unobserved region을 공동 지지

Hard gate는 최소화한다.

Hard gate 후보:

- 동일 local neighborhood
- finite extension
- visible geometry로 완전히 점유되지 않음
- chart 생성 가능성

Soft evidence:

- facing 정도
- normal similarity
- tangent alignment
- curvature agreement
- support symmetry
- boundary distance

직교·사선 surface도 허용한다.

## 5.6 Bounded Occlusion Scope

초기 canonical implementation은 two-sided 또는 multi-sided visible support가 있는 bounded occlusion region에 한정한다.

- one-sided surface를 scene 외부라고 단정하지 않는다.
- 다만 unsupported one-sided extrapolation은 초기 scope에서 제외한다.
- one-sided extrapolation은 후속 연구 범위다.

## 5.7 Constrained Occluded NURBS Construction

Candidate region마다 visible NURBS boundary를 고정 조건으로 사용하는 occluded chart를 만든다.

필수 조건:

- boundary C0 exact/near-exact
- shared control geometry
- independent post-hoc seam overwrite 금지
- jointly constrained fitting
- arbitrary boundary-angle support
- fold/self-intersection 검증
- visible geometry penetration 검증

G1/G2는 초기 hard constraint가 아니다.

## 5.8 Result-Based Validation

Candidate를 사전에 과도하게 제거하지 않고 생성 결과를 평가한다.

필수 diagnostics:

- C0 error
- tangent/normal mismatch
- Jacobian flips
- minimum singular value
- condition p95/p99/max
- self-intersection
- visible-surface penetration
- extension length/area
- curvature magnitude/change
- empty-space consistency
- camera visibility consistency
- candidate conflict

Candidate 상태:

```text
candidate
validated
rejected
```

## 5.9 Uncertainty

Uncertainty 후보:

- parametric boundary distance
- supporting boundary count
- continuation-domain overlap
- boundary prediction agreement
- required curvature
- normal-angle difference
- extension length
- chart condition
- visibility support
- candidate conflict

추후 Gaussian density, opacity, scale, optimization freedom, pruning priority에 사용한다.

---

# 6. 우선 해결해야 할 문제

## Priority 1 — Current Pipeline Requirement Audit

현재 구현이 새 방향에 필요한 데이터를 제공하는지 조사한다.

- open boundary
- boundary control points
- inner isocurve
- derivatives
- UV orientation
- patch adjacency
- coupled-boundary metadata
- camera visibility
- local empty-space query

## Priority 2 — Artificial Boundary Reconciliation Audit

확인할 질문:

- Step 5-A가 같은 component 내부 seam만 해결하는가?
- 다른 local patch/component 간 coupling이 가능한가?
- patch integration 없이 downstream boundary loop 조립이 가능한가?
- coupled fitting 후 crack/flip이 제거되는가?

## Priority 3 — Continuation Domain Prototype

필수 fixture:

- planar boundary
- curved boundary
- rotated patch
- orthogonal patch pair
- oblique patch pair
- annulus inner boundary
- artificial seam boundary

## Priority 4 — Occluded Candidate Region Builder

출력:

```text
candidate_id
supporting_boundary_ids
candidate_domain
overlap/near-overlap evidence
boundary angles
estimated extension scale
visible occupancy conflict
visibility support
confidence diagnostics
```

## Priority 5 — Minimal Constrained NURBS Bridge

필수 fixture:

- facing parallel boundaries
- orthogonal boundaries
- oblique boundaries
- curved-to-curved boundaries
- annular bounded gap
- asymmetric boundary lengths
- noisy boundary samples

## Priority 6 — Benchmark and Refinement

False-positive 유형을 먼저 관찰한 뒤 refinement한다.

- valid bridge
- visible penetration
- excessive curvature
- self-intersection
- unsupported long extension
- unrelated patch connection
- multi-candidate conflict
- open-scene extrapolation

---

# 7. 당장 필요하지 않은 문제

- complete global visible-surface connectivity
- all-pair sparse/disconnected classification
- watertight scene reconstruction
- object instance segmentation
- true object boundary semantic classification
- global manifold reconstruction
- single global UV atlas
- one-sided occlusion extrapolation
- G2 continuity
- learned covariance-based connectivity
- general point-cloud segmentation 최적화

---

# 8. 기존 Proxy-Based Decomposition Branch 처리

Stage 0–3 및 Stage 3-R 결과는 삭제하지 않는다.

- production 미적용
- active path에서 중단
- diagnostics/ablation/reference로 보존
- deprecated research branch로 문서화

보존 이유:

- face adjacency의 한계
- pairwise connectivity 식별 불가능성
- global component recovery가 과도한 requirement라는 근거

---

# 9. Agent 작업 지침

## Step 0 — Read and Compare

다음을 모두 읽는다.

- 현재 Architecture
- Phase 1–5 plans
- coupled-boundary fitting worklogs
- proxy decomposition Stage 0–3
- Stage 3-R
- TODO와 active urgent-work 문서

## Step 1 — Current Interface Audit

코드 변경 없이 조사한다.

- visible NURBS boundary 데이터 흐름
- patch/component/chart relation
- derivative access
- open-boundary representation
- visibility data availability
- empty-space query 가능성
- coupled fitting 재사용 가능성

## Step 2 — Migration Plan

다음을 제안한다.

- 유지할 모듈
- deprecated할 모듈
- 수정할 interface
- 신규 모듈
- 최소 end-to-end prototype
- benchmark fixtures
- production adoption gates

## Step 3 — Stop

사용자 승인 없이 구현하지 않는다.

---

# 10. Agent 금지 사항

- production default 변경
- 기존 Phase 1 제거
- proxy branch production 적용
- scene-specific logic
- true object boundary classifier 구현
- strict facing-angle gate
- strict normal-similarity gate
- 모든 adjacent patch 자동 통합
- voxel occupancy를 occluded area로 간주
- unsupported one-sided extrapolation
- GT topology/component count runtime 사용
- Phase 5 본편 무단 착수

---

# 11. Agent 최종 보고 형식

1. 기존 방향성 요약
2. 새 방향성 요약
3. 근본적 차이
4. 유지 가능한 모듈
5. 중단/deprecate할 모듈
6. 새 methodology에 필요한 데이터
7. 현재 코드에 존재하는 데이터
8. 누락 interface/data
9. 최소 end-to-end prototype
10. 단계별 구현 난이도
11. 예상 위험
12. 새 plan 파일 제안
13. 구현 착수 전 승인 질문

---

# 12. Agent에게 전달할 최초 명령

```text
OSN-GS의 active methodology를 global visible-surface decomposition 중심에서 boundary-conditioned occluded-surface construction 중심으로 재정립한다.

첨부된 `OSN_GS_Direction_Reset_Plan.md`를 최상위 방향성 문서로 읽고, 기존 Phase 1–5 plan, coupled-boundary worklog, proxy-decomposition Stage 0–3 및 Stage 3-R worklog, 현재 Architecture/TODO 문서를 모두 감사하라.

이번 작업에서는 구현하지 마라.

먼저 다음을 수행하라.

1. 기존 방향과 새 방향의 차이를 코드/문서 기준으로 정리
2. 현재 visible NURBS patch에서 boundary curve, control points, derivatives, inward isocurve, UV orientation을 어디까지 얻을 수 있는지 확인
3. artificial patch boundary reconciliation에 기존 coupled fitting을 어느 범위까지 재사용할 수 있는지 확인
4. camera visibility 및 empty-space evidence가 현재 pipeline에 존재하는지 확인
5. continuation domain과 occluded-region candidate를 만들기 위해 부족한 데이터와 interface를 식별
6. 기존 proxy-based decomposition branch를 production 미적용 deprecated diagnostics branch로 정리할 방법 제안
7. minimal end-to-end prototype과 단계별 migration plan 작성

새 방향에서는 facing direction과 normal similarity를 hard gate로 사용하지 않는다. 직교·사선 boundary 관계도 candidate로 허용하고, high-recall candidate generation 후 constrained NURBS 결과의 validity와 uncertainty로 평가한다.

모든 조사 결과를 한국어 worklog 및 새 implementation plan 초안으로 작성한 뒤 멈춰라. 사용자 승인 없이 코드 구현, production 변경, Phase 5 본편 착수를 수행하지 마라.
```

---

# 13. 성공 기준

- 필요한 upstream data 식별
- global component recovery 필요성 재판정
- continuation domain 생성 가능성 확인
- artificial boundary reconciliation 범위 확인
- visibility/empty-space evidence 확인
- minimal Phase 5 prototype 범위 확정
- 기존 branch와 새 active branch 관계 문서화
- 사용자 승인 전 production 변경 없음
