# Worklog 75: Boundary-Conditioned Phase A–B 구현 및 검증

날짜: 2026-07-23

상태: **Phase A–B 구현·검증 완료. Gate B 사용자 검토 대기. Phase C 이후, production integration, global component 변경은 미착수.**

## 1. 승인 범위

사용자가 승인한 `OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`의 Phase A–B만 수행했다.

- NURBS knot/second-derivative interface
- ordered/oriented patch boundary와 inner isocurve
- Boundary-First state/export lifecycle
- generic shared-control constraint assembly
- isolated artificial patch-boundary reconciliation

Camera/free-space, continuation domain, occluded candidate/bridge, uncertain Gaussian append는 수행하지 않았다.

## 2. Phase A 구현

### `osn_gs/surface/torch_nurbs.py`

- `knots_u`, `knots_v`, `knot_vectors()` read-only snapshot API를 추가했다.
- 기존 lazy knot cache는 `_knot_vectors_internal()`로 단일화했다.
- analytic rational NURBS `evaluate_with_second_derivatives()`를 추가했다.
  - 반환: `S`, `S_u`, `S_v`, `S_uu`, `S_uv`, `S_vv`
  - degree 0/1 축의 second partial은 0으로 처리한다.
- Public knot snapshot을 수정해도 internal cached knot/evaluation은 바뀌지 않는다.

Finite-difference 최대 절대오차:

| derivative | max abs error |
|---|---:|
| `S_uu` | `1.38e-08` |
| `S_uv` from `d(S_v)/du` | `3.31e-09` |
| `S_uv` from `d(S_u)/dv` | `6.59e-09` |
| `S_vv` | `1.63e-08` |

### `osn_gs/surface/torch_patch_boundary.py`

신규 stable boundary contract를 추가했다.

- 상태: `unclassified`, `reconciled_internal`, `unsupported`, `extension_candidate`
- deterministic `boundary_id`와 patch/provenance
- ordered UV/world curve
- supported interior가 항상 curve의 left가 되는 orientation
- adjacent inner isocurve
- NURBS-derived tangent, inward surface tangent, normal
- patch/control-edge adjacency
- JSON payload와 local confidence

Trim mask는 cell-union boundary edge를 interior-left 방향으로 trace한다. Closed loop는 canonical start vertex로 회전해 동일 입력의 ID/순서가 결정적이다. Inner isocurve는 boundary 안쪽 target에 가장 가까운 supported cell center를 사용하므로 항상 current support mask 내부에 있다.

Annulus chart는 patch마다 다음 네 edge를 명시적으로 보존한다.

- `u0`, `u1`: artificial chart seam
- `v0`: observed inner boundary
- `v1`: observed outer boundary

Step 5-A가 켜진 기본 annulus에서는 artificial seam을 `reconciled_internal`로 기록한다. 독립 fit ablation에서는 `unclassified`로 남긴다.

### State와 export

- `BoundaryFirstState`가 `component_boundaries`와 `patch_boundaries`를 보존한다.
- Boundary-First/main NURBS JSON에 root/per-patch knot를 추가했다.
- Boundary-First renderer JSON에 `patch_boundaries`와 per-patch `boundary_ids`를 추가했다.
- Main `TorchPipelineState`의 membership/default는 변경하지 않았다.

## 3. Phase B 구현

### Generic coupled solver

`SharedBoundaryConstraint`, `boundary_control_indices()`, `fit_coupled_patch_graph_lsq()`를 추가했다.

- Explicit patch graph
- Full/partial control edge
- Same/reversed parameter direction
- Deterministic union-find variable map
- Shared control은 첫 LSQ solve 전부터 하나의 global unknown
- Patch interior와 unconstrained boundary는 private
- Post-hoc control overwrite 없음

기존 production `fit_coupled_wedge_ring_lsq()`는 변경하지 않았다. 새 generic solver는 isolated prototype에서만 사용하므로 Step 5-A의 수치 경로와 기본 결과를 보존한다.

### Artificial boundary reconciliation

`osn_gs/surface/torch_boundary_reconciliation.py`를 추가했다.

- Caller가 제공한 local adjacency만 평가한다. All-pair global graph를 만들지 않는다.
- Hard evidence:
  - finite curve
  - curve-length overlap
  - scale-normalized gap
- Soft evidence:
  - tangent angle
  - normal angle
- Candidate만 generic joint fit으로 전달한다.
- Post-fit patch-wide Jacobian finite/flip/near-degenerate 검사를 통과해야 `reconciled_internal`이 된다.
- 한 patch라도 validity/C0를 실패하면 전체 joint transaction을 independent fit으로 rollback한다.
- Patch membership union이나 production component 변경은 없다.

## 4. 정량 결과

| fixture | pre normalized gap RMS | mean normal angle | joint fit | post C0 max | 결과 |
|---|---:|---:|---|---:|---|
| coplanar artificial seam | `0.01123` | `0.0°` | yes | `2.46e-07` | `reconciled_internal` |
| orthogonal shared edge | `0.00835` | `89.9996°` | yes | `2.38e-07` | `reconciled_internal` |
| disconnected gap `0.2` | `0.18384` | `0.0°` | no | n/a | `unclassified` / `scale_normalized_gap` |

직교 fixture가 약 90° normal mismatch에도 통과했으므로 normal similarity를 hard gate로 사용하지 않는 새 methodology 계약을 직접 검증했다. 모든 joint-fit positive patch는 Jacobian validity를 통과했다.

Curved 3-patch fixture는 adjacency 입력 순서를 뒤집어도 decision order와 control grid가 `torch.equal`로 일치했다.

## 5. Boundary-First smoke

다음 scene을 현재 production Boundary-First constructor로 실행했다.

```text
plane
planar_hole
curved_annulus
mild_curved_sheet
close_parallel_sheets
```

주요 결과:

- `planar_hole`: 8 patches, seam gap `0`, orientation flip `0`, Jacobian condition p95 약 `2.86`
- `curved_annulus`: 기존과 동일하게 2 components (`disk_like`, `complex`)
- `mild_curved_sheet`: 기존과 동일하게 spurious annulus 8 patches
- `close_parallel_sheets`: 2 components, topology ARI `1.0`

즉 Phase A–B가 기존 global Phase 1 membership/topology blocker를 몰래 변경하지 않았다.

Renderer artifact:

- `plane`: 1 patch, 2 boundary loops, knot lengths U/V `15/15`
- `planar_hole`: 8 patches, 32 boundary records, knot lengths U/V `11/6`
- 동일 입력 반복 SHA-256 byte match:
  - `plane`: `CE3999F66859A8C1C11AA6610C9E38890FA9306C5E7D3C4C6809739F31018F4D`
  - `planar_hole`: `D845515BFF700F21BB04872925BE25D08451070A0738A518C079B3AB430476C8`

## 6. 검증

- Phase A–B 핵심 집중: `28 tests` 통과
- 관련 모듈 집중 회귀: `93 tests` 통과
- 전체 unittest: `221 tests` 통과, `1 skipped`
- 전체 pytest: `220 passed, 1 skipped, 8 subtests passed`
- 기존 `torch_nurbs.py` requires-grad tensor scalar conversion warning 1건은 그대로이며 이번 변경과 무관하다.
- `git diff --check` 통과

## 7. 평가

Phase A–B gate의 목적은 달성했다.

- Visible NURBS가 knot와 first/second derivative를 직접 제공한다.
- Trim/chart boundary가 stable patch-local record로 유지된다.
- Inward orientation과 inner isocurve를 Gaussian local fit으로 재추정하지 않는다.
- Artificial boundary는 normal/facing hard gate 없이 joint shared-control fit으로 검증할 수 있다.
- Scale-normalized gap이 큰 disconnected pair는 production/global merge 없이 남는다.
- Existing Step 5-A와 production constructor 결과는 유지된다.

## 8. 남은 제한과 위험

- Trim boundary와 inner isocurve는 현재 support-grid quantization을 따른다. Smooth parametric boundary curve fitting은 아직 없다.
- Generic solver는 canonical Phase B 범위대로 동일 resolution/degree를 전제로 한다. Degree elevation, knot insertion, unequal control count correspondence는 미지원이다.
- Partial control-edge constraint는 solver에서 지원하지만 reconciliation evidence는 현재 full edge sampling 기준이다.
- `max_normalized_gap=0.05`는 isolated prototype 기본값이며 production threshold가 아니다.
- `unclassified` boundary를 `unsupported` 또는 `extension_candidate`로 분류하지 않았다.
- Camera visibility, known free space, empty-space consistency는 아직 없다.
- Self-intersection/visible penetration/candidate conflict는 Phase F 범위이며 아직 없다.
- Boundary records는 Boundary-First benchmark state에 연결됐다. Main training state에는 knot export만 추가했으며 occlusion lifecycle은 연결하지 않았다.

## 9. 중단 및 다음 승인

계획대로 Gate B에서 멈춘다. 다음 단계는 **Phase C — Observation Evidence와 Free-Space Query**이며 별도 사용자 승인 없이는 시작하지 않는다.