# OSN-GS Phase F.1 — Occluded Chart Hardening Design

상태: **승인됨. Phase F chart를 변경하지 않는 deterministic sampled safety gate.**

## 범위

`OccludedChartResult + visible surface registry + other chart results`에서 sampled triangle mesh를 만들고 self-intersection, visible-surface penetration, chart conflict, Phase G 입력 자격만 계산한다. analytic/CAD 수준의 연속 surface 보증은 제공하지 않는다.

## 계약

- 기존 chart state(`fitted`/`validated`/`unsupported`/`rejected`)와 control grid는 변경하지 않는다.
- `OccludedChartSafetyResult`는 eligibility(`eligible`/`review_required`/`ineligible`/`unsupported`)와 raw safety payload를 별도로 보존한다.
- `OccludedChartConflictEdge`는 unresolved conflict만 기록하며 ranking·제거·해결은 하지 않는다.
- sampled grid의 fixed diagonal triangle mesh와 기존 `sweep_and_prune_pairs`를 쓰며 narrow phase는 CPU float64 tolerance-aware triangle 검사다.
- coverage scope는 항상 `central_bridge_only`, `transition_surface_modeled=false`로 기록한다.

## Phase G 전제

초기 Phase G source는 `eligible`만 허용한다. `validated` chart라도 sampled self-intersection/penetration이 unchecked이거나 unresolved conflict, support coverage failure, hard evidence contradiction이 있으면 proposal source가 될 수 없다. `review_required`는 자동 승격하지 않는다.

## 비범위

analytic self-intersection proof, CAD robust predicate, conflict resolution, global selection/ranking, cyclic/multi-sided chart, transition strip fitting, Gaussian proposal/append, production integration은 구현하지 않는다.
