﻿﻿﻿﻿﻿﻿﻿﻿# OSN-GS Urgent Work Master

상태: **ACTIVE — 이 문서가 `docs/Urgent_Work/`의 유일한 현행 canonical entry point다.** 기존 Urgent Work 문서는 이 통합으로 제거한다. 구현·감사 증거는 `docs/worklogs/`와 Git 이력에 보존하며, 현재 작업 지시에는 이 문서만 적용한다.

갱신일: 2026-07-26

## 1. 현재 방향

OSN-GS의 현행 방법론은 global visible-surface decomposition이나 global component recovery가 아니다. 관측된 local visible NURBS patch의 boundary를 출발점으로, 두 개 이상의 support가 있는 유한 bounded occluded region만 가설화하고 검증하는 **boundary-conditioned occluded-surface construction**이다.

```text
certain Gaussian
→ visible NURBS patch / boundary
→ continuation domain
→ bounded pairwise occluded-region candidate
→ constrained occluded NURBS chart
→ deterministic sampled safety gate
→ immutable uncertain Gaussian proposal
→ model-only append adapter boundary
→ [중단: appearance/opacity initialization 및 production integration은 별도 승인]
```

고정 원칙:

- Certain Gaussian은 image loss로만 움직이며 NURBS 경로의 reverse gradient를 받지 않는다.
- Voxel은 local evidence/partition이며 occluded area의 정답이나 global merge 근거가 아니다.
- facing, normal, tangent, curvature는 soft evidence다. strict hard gate로 쓰지 않는다.
- one-sided extrapolation, global ranking/selection, conflict resolution, cyclic/multi-sided topology, uncertain-to-certain promotion은 현행 범위 밖이다.
- 모든 결과는 provenance와 raw diagnostic을 보존하며, calibrated 단일 confidence로 조기 압축하지 않는다.

## 2. 상위 문서와 문서 지위

| 문서 | 현재 지위 | 역할 |
|---|---|---|
| 이 문서 | **현행 canonical** | 방향, 승인 상태, 현재 범위, 다음 Gate를 한 곳에서 정의 |
| 제거된 기존 Urgent Work 문서 | Git/Worklog 이력 | Direction Reset, Boundary Conditioned Plan, Phase F/F.1 설계의 결정·계약은 아래 요약과 Worklog에 흡수됨 |
| `docs/worklogs/` | 증거 기록 | 구현·검증·Gate 결과의 append-only 근거 |

기존 Urgent Work 문서는 제거하며, 세부 구현 증거는 Worklog와 Git 이력으로 추적한다. 상태 판단은 이 문서와 가장 최근 Worklog를 우선한다.

## 3. 완료·승인 상태

| 범위 | 상태 | 증거 |
|---|---|---|
| Phase A: boundary/NURBS interface | 구현·검증·승인 완료 | `worklogs/75_boundary_conditioned_phase_ab.md` |
| Phase B: artificial boundary reconciliation | 구현·검증·Gate B 승인 완료 | `worklogs/75_boundary_conditioned_phase_ab.md` |
| Phase C: observation/free-space evidence | 구현·검증·Gate C 승인 완료 | `worklogs/77a`–`79` |
| Phase D: continuation domain | 구현·검증·Gate D 승인 완료 | `worklogs/81_phase_d_continuation_domain_implementation.md` |
| Phase E: pairwise bounded candidate | 구현·검증·Gate E 승인 완료 | `worklogs/82_phase_e_bounded_candidate_implementation.md` |
| Phase F: constrained open pairwise chart | 구현·검증·Gate F 승인 완료 | `worklogs/83_phase_f_constrained_occluded_chart_implementation.md` |
| Phase F.1: sampled chart hardening | 구현·검증·Gate F.1 승인 완료 | `worklogs/84_phase_f1_chart_hardening_implementation.md`, `85`, `86` |
| Phase G: immutable uncertain Gaussian proposal | 구현·검증·Gate G 사용자 승인 완료 | `worklogs/87_phase_g_uncertain_gaussian_proposal_foundation.md` |
| Phase G 후속: model-only append adapter foundation | 구현·검증 완료, **사용자 Gate 검토 대기** | `worklogs/88_uncertain_gaussian_append_adapter_foundation.md` |

## 3.1 Visible Surface Construction 품질 상태와 Gate 해석

Phase A–G Gate 승인에는 **visible-surface constructor의 benchmark 품질 승인이나 production adoption이 포함되지 않는다.** 이 Gate들은 이미 존재하는 local visible NURBS patch/boundary를 입력으로 하는 interface, continuation, bounded occluded chart, sampled safety, proposal 및 append mutation boundary를 검증한 것이다.

`curved_annulus`와 `mild_curved_sheet`의 known blocker는 여전히 남아 있다.

- `curved_annulus`는 곡률과 hole이 있는 GT annulus가 Phase 1 `build_surface_components`에서 component 둘로 분리되어 annulus topology를 안정적으로 만들지 못한다.
- `mild_curved_sheet`는 hole이 없는 단일 곡면을 반대로 annulus/O-grid로 과분할한다.
- 원인은 curved geometry에서 Phase 1/2 component·loop/topology 판정이 불안정한 구조적 한계다. Phase F/F.1 또는 occluded-chart 후단에서 국소 보정할 문제가 아니다.
- 현재 `construct_boundary_first`는 `classify_boundary_result(boundary) == annulus`일 때만 `build_annulus_chart` O-grid를 호출하고, 그 밖의 모든 topology에는 `fit_trimmed_component` rectangle fallback을 호출한다. 저장된 `curved_annulus` payload는 `component_count=2`, 각 component `topology=disk_like`, `chart=trimmed_rect_fallback`으로 기록되어 O-grid를 전혀 타지 않았다. 즉 true annulus라는 구조가 component/boundary 추정 단계에서 소실된 뒤 다른 builder로 routing된 것이다.

따라서 현재 상태는 “visible NURBS 입력을 소비하는 occluded-surface 연구 경로가 Gate를 통과했다”이지, “Visible Surface Construction이 curved benchmark에서 품질적으로 궤도에 올랐다”가 아니다. 이 blocker는 production integration의 선행 품질 과제로 유지한다. 근거: `TODO.md`의 NURBS 표면 생성 품질 항목, `docs/worklogs/60b_proxy_decomposition_stage0_baseline.md`, `docs/worklogs/77b_anisotropy_gap_parity_ablation_results.md`.
## 3.2 Boundary-first constructor의 통일 원칙 — 사용자 지시

모든 visible-surface topology는 동일한 Boundary-first 구성 원리를 따라야 한다. topology별 chart layout, seam 수, parameterization은 달라도 되지만, 구축의 source of truth는 항상 다음이다.

```text
명시적 observed boundary loop 또는 boundary pair
→ inner/outer 또는 대응 support boundary 확정
→ 두 boundary 사이의 ordered support curves / isocurves
→ boundary-constrained NURBS surface 또는 multi-patch surface
```

annulus만 `O-grid`를 쓰고 그 밖의 구조를 box/trimmed-rectangle fallback으로 fit하는 현재 dispatcher는 이 원칙에 부합하지 않는다. `curved_annulus`의 실패는 topology 오분류뿐 아니라, 오분류 뒤 boundary 구조를 보존하지 않는 rectangle fallback으로 전환되는 설계 문제를 드러낸다.

향후 visible-surface constructor 재설계의 필수 조건:

- 모든 topology에서 boundary provenance, loop correspondence, support curves를 보존한다.
- control grid는 boundary/support constraints에서 파생하며, voxel box 또는 PCA rectangle은 partition/initialization 보조 수단일 뿐 final chart source of truth가 아니다.
- annulus는 inner/outer loop 사이의 radial support-curve family와 seam layout을 사용할 수 있으나, 이것은 일반 원리의 특수 parameterization이지 별도 methodology가 아니다.
- disk, strip, non-convex, multi-loop 등도 대응 boundary/loop와 support-curve network를 먼저 만들며, 구조를 표현하지 못하는 경우에는 fallback surface를 조용히 만들지 않고 `unsupported` 또는 명시적 diagnostic을 반환한다.

이 요구는 기존 append adapter Gate와 별개인 visible-surface constructor 방향 재정의다. 2026-07-27 현재 isolated `curved_annulus` proof path은 100% 완료다. 새 support-network, constrained-surface, component-recovery, visible-builder, isolated benchmark entry point는 split component를 immutable recovered region으로 복구하고 pre-surface inner/outer loop에서 explicit seam multi-patch를 만든다. 증거는 `docs/worklogs/89_boundary_first_support_curve_foundation.md` → `90_boundary_first_pre_surface_visible_builder.md` → `91_boundary_first_curved_annulus_recovery_proof.md`다. 기존 dispatcher integration 준비는 약 88%다. Worklog 95의 isolated review runner는 15개 전체 benchmark scene을 기록하며, 6개 constructed와 9개 unsupported를 같은 artifact에 명시한다. 이 결과는 curved annulus만을 성공 기준으로 삼지 않으며, universal constructor 완성을 주장하지도 않는다. Worklog 94는 materialized support sample·cyclic seam·corner Jacobian의 deterministic 수치 측정 게이트를 추가했다. 이 수치는 원본 관측 경계 전체의 shape fidelity를 뜻하지 않으며, 해당 fidelity gate는 남은 범위다. curved annulus positive control과 plane/parallel false-hole negative control gate는 `docs/worklogs/92_boundary_first_negative_control_routing_gate.md`에 고정했다. 이어서 point count 400/600 및 seed 0/1/2 recovery sweep에서 curved annulus는 모두 허용되고 close parallel sheets는 모두 거부됨을 `docs/worklogs/93_boundary_first_recovery_sweep_gate.md`에 고정했다. disk/non-convex/multi-loop support-network, dispatcher feature gate와 production integration은 별도 범위다. Worklog 97은 curved annulus를 universal success proxy로 쓰지 않고, ordered observed boundary loop와 provenance 있는 interior support/central-cap 계약을 모든 topology의 다음 공통 구현 전제로 고정한다.
## 4. 현재 수행 중인 작업

현재 허용된 범위는 **Uncertain Gaussian append adapter foundation의 Gate 검토**다.

입력 경계:

```text
approved immutable proposal batch
→ read-only append preflight
→ model-compatible tensor conversion
→ atomic model-only append
→ immutable append receipt
```

이미 구현·검증한 계약:

- `eligible` proposal, 완전한 chart/candidate/domain/boundary provenance, valid sample, finite position/rotation/scale, strictly positive scale, normalized quaternion, 지원 schema 및 유효 deterministic ID만 통과한다.
- `review_required`, `ineligible`, `unsupported`, known-free contradiction, 누락 provenance, zero valid sample, 재-append batch는 append하지 않는다.
- model이 요구하는 appearance/opacity는 proposal의 unset 상태를 임의 RGB/SH/opacity 값으로 채우지 않는다. 명시적 initialization이 없으면 `appearance_initialization_required` blocker를 반환한다.
- linear scale은 model의 log scaling으로 변환하고 canonical quaternion은 raw rotation tensor로 보존한다. valid-mask filtering, sample ordering, dtype/device 정렬을 유지한다.
- optimizer가 있는 model은 model-only boundary에서 차단한다. 성공 commit은 모든 per-Gaussian tensor를 함께 증가시키며, preflight 또는 conversion 실패 시 model은 불변이다.
- 동일 process의 proposal batch ID duplicate는 ledger로 차단한다. batch/sample ID와 chart/candidate/patch/domain/boundary provenance는 batched sidecar에 보존한다. checkpoint-persistent ledger는 deferred schema gap이다.

현재 Gate의 확인 사항은 mutation boundary 독립성, rollback, duplicate 차단, provenance 보존, appearance/opacity blocker 비우회, production 미변경이다.

## 5. 명시적으로 deprecated 또는 비활성인 범위

### Deprecated research diagnostics

다음은 실패 근거·ablation·회귀 재현을 위해 남기되 active continuation/candidate/chart/append 경로에 import하거나 production에 연결하지 않는다.

- `torch_surface_proxy.py`
- `torch_surface_candidate_graph.py`
- `torch_surface_decomposition.py`
- `torch_gaussian_support_continuity.py`
- Proxy Stage 0–3 및 Stage 3-R script, test, artifact

### 완료되어 현행 지시가 아닌 문서/범위

- Direction Reset의 “현재 구조 감사 후 migration plan을 제안하고 멈춘다” 지시
- Phase F constrained-chart 설계·구현 착수 지시
- Phase F.1 hardening 설계·Gate F.1 승인 요청 지시
- Proposal-only Phase G Gate 검증 지시

이들은 deprecated 코드가 아니라 완료된 역사적 계약·증거다. 세부 알고리즘 근거가 필요할 때만 reference로 읽는다.

### 아직 승인되지 않았거나 시작하지 않은 범위

- appearance/opacity initialization policy
- optimizer parameter registration 및 optimizer state expansion
- trainer/training step/loss/renderer 연결
- densification, pruning, ADC, uncertain-to-certain promotion
- checkpoint save/load 및 persistent append ledger
- global chart ranking/selection, conflict resolution, review workflow
- cyclic/multi-sided chart, one-sided extrapolation, production default 변경

## 6. 다음 순서와 Gate 경계

1. append adapter foundation Gate를 사용자 검토·승인한다.
2. 승인 뒤에도 production integration을 자동 시작하지 않는다.
3. appearance/opacity initialization policy와 optimizer/trainer/renderer/checkpoint integration은 별도의 설계·승인 Gate로 분리한다.
4. Phase H production adoption은 positive/negative geometry·visibility regression, deterministic sweep, runtime/resource bound, no reverse gradient, explicit failure reporting을 모두 갖춘 뒤에만 계획한다.

다음 production 입력의 최소 조건은 계속 다음과 같다.

```text
chart.state == validated AND safety.eligibility == eligible
```

이 조건은 proposal source 자격이며, appearance/opacity initialization 또는 model append의 자동 승인 근거가 아니다.

## 7. 검증 기준과 최신 수치

- append adapter + Phase G/D/E/F/F.1 지정 회귀군: 173 passed
- 전체 pytest: 353 passed, 1 skipped, 2 warnings
- warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad scalar conversion 1건과 Windows `.pytest_cache` 권한 경고 1건이다.

## 8. 작업 규칙

- 새 substantial 작업은 완료 영역마다 한국어 Worklog를 새 번호로 추가하고 `docs/README.md`에 상태 링크를 남긴다.
- Korean Markdown은 UTF-8 또는 UTF-8 BOM으로 보존한다.
- 이 문서에 없는 production mutation, global selection/resolution, appearance/opacity heuristic은 사용자 승인 전 시작하지 않는다.

2026-07-27 현재 ordered loop, observed-anchor central cap, concave false-fill gate, multi-component review export, source RMS feature gate까지 격리 경로에 구현했다. docs/worklogs/100_boundary_first_feature_gated_review.md의 0.1 RMS review에서 15 scene은 constructed 8, review_required 1, unsupported 6이다. multi-hole planar-domain decomposition, normal/curvature/pole regularity gate, 전체 regression 해소 전에는 기본 dispatcher나 production integration을 변경하지 않는다.

## 2026-07-27 Boundary-first 역할 계약 보완

- visible builder는 topology별 surface methodology를 고르지 않는다. 관측 loop는 `outer_boundary`와 `interior_boundary` 역할을, hole-free support는 `outer_boundary`와 observed `interior_anchor` 역할을 제공하며 모두 공통 Boundary-first materialization 계약으로 처리한다.
- sine에서 발생한 작은 KDE enclosed hole은 실제 multi-hole 증거가 아니다. 유의미한 면적 기준 미만의 artifact는 provenance에 남기면서 support validation mask에서만 복원했다. material hole은 복원하지 않는다.
- 이 변경은 isolated Boundary-first 경로에 한정된다. default dispatcher, production integration, global selection/resolution은 아직 시작하지 않았다. 다음 범위는 source-boundary fidelity review gate다. 근거: `docs/worklogs/102_boundary_first_role_contract_and_false_hole_gate.md`.
## 2026-07-27 Source-boundary fidelity review 보완

- isolated builder provenance는 ordered observed boundary와 동일 component raw point 사이의 minimum/median/mean/maximum 거리 및 local-spacing 정규화를 기록한다. 이는 현재 자동 eligibility 차단이 아닌 deterministic review metric이다.
- `artifacts/boundary_first_support_review_20260727_v5_role_fidelity/`의 15-scene review는 constructed 12, review_required 1, unsupported 2다. sine은 constructed이나 normalized median distance 2.826로 경계 충실도 tuning 우선 검토 대상으로 남긴다.
- 다음 범위는 fidelity parameter sweep과 quality gate이며, dispatcher/production integration은 그대로 미변경이다. 근거: `docs/worklogs/103_boundary_first_source_boundary_fidelity_payload.md`.
## 2026-07-27 Resolution 96 isolated review 보완

- isolated observed-support raster 기본값은 96이며 runner에서 `--boundary-resolution`으로 재현한다. anchor ray 검사는 raster cell 1개만 tolerance로 허용하고 값을 provenance에 남긴다.
- 새 artifact에서 sine/curved_annulus의 normalized median boundary distance는 각각 2.379/2.113으로 개선됐고 U-shape는 여전히 unsupported다. 기본 dispatcher/production은 미변경이다. 근거: `docs/worklogs/104_boundary_first_resolution96_raster_tolerance.md`.
## 2026-07-27 Boundary-first 전체 회귀 checkpoint

- isolated 변경 후 전체 pytest는 475 passed, 기존 실패 3건, 1 skipped, 1 warning, 8 subtests passed다. 실패는 annulus independent-fit detection guard 1건과 trimmed fitter degenerate fraction 2건으로 이번 변경 파일과 분리돼 있다.
- source-boundary fidelity/tiny-hole/role-contract 회귀는 통과했다. 기본 dispatcher 및 production은 미변경이다. 근거: `docs/worklogs/105_boundary_first_full_regression_checkpoint.md`.
## 2026-07-27 Resolution/tolerance sweep

- isolated resolution 96/1-cell tolerance는 sine·curved_annulus 12개 positive 조합을 모두 구성하고 U-shape 6개 concave negative 조합을 모두 거부했다. 거부 사유 변화는 허용하되 silent central fill은 없다. dispatcher/production은 미변경이다. 근거: `docs/worklogs/106_boundary_first_resolution_tolerance_sweep.md`.
## 2026-07-27 Multi-loop role evidence

- multi-loop은 `outer_boundary + 복수 interior_boundary` 역할을 모두 보존하되, non-overlapping planar-domain partition 증거 없이는 review_required로 유지한다. outer loop를 hole별 annulus로 복제하는 overlapping chart는 금지한다. dispatcher/production은 미변경이다. 근거: `docs/worklogs/107_boundary_first_multi_loop_role_evidence.md`.
## 2026-07-27 Multi-loop 이후 전체 회귀 checkpoint

- 전체 pytest는 478 passed, 기존 실패 3건, 1 skipped, 1 warning, 8 subtests passed다. 새 multi-loop role evidence와 Boundary-first sweep 증가는 통과했다. 다음 범위는 non-overlapping planar-domain partition 불변식이며 dispatcher/production은 미변경이다. 근거: `docs/worklogs/108_boundary_first_multi_loop_full_regression_checkpoint.md`.
## Canonical topology constraint 반영 (2026-07-27)

- common Boundary-first role contract가 유일한 construction entry이며 topology는 observed loop 해석, ownership, ordering, support sufficiency, parameterization/quality 진단에만 사용한다.
- O-grid, observed-anchor fan, multi-loop partition은 별도 dispatcher가 아니라 같은 role contract의 materialization form이다.
- annulus seed=14 orientation fixture의 기대값 8은 frame-margin 없는 extractor에만 성립하는 stale expectation이었다. HEAD/현재/parameter isolation으로 원인을 확인했고 canonical frame-margin path의 geometry validity expectation(0 flips, no overlap, no near-degeneracy)으로 갱신했다.
- false-hole correction은 아직 multiresolution/KDE persistence, raw-support, local-spacing area와 genuine-small-hole negative control을 모두 갖추지 못했다. 따라서 해당 보정의 자동 canonical 승격 및 dispatcher 연결은 금지한다.