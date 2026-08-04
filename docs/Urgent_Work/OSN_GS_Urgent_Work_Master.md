# OSN-GS Urgent Work Master

최종 갱신: 2026-08-04

이 문서는 현재 진행 방향과 승인 경계를 정의하는 canonical master다. 과거 실험의 상세 경과는 Git 이력에 보존하며, 현재 판단에 필요 없는 작업로그는 `docs/worklogs/`에서 제거했다. 작업로그 번호는 `docs/worklogs/README.md`의 최신 인덱스(1부터 재번호, 현재 53까지)를 기준으로 한다.

## 1. 목표 모델과 불변 조건

OSN-GS에서 NURBS는 관측 가능한 표면을 설명하고 가려진 영역의 불확실 Gaussian 생성에만 기하 정보를 제공하는 중간 표현이다. visible/certain Gaussian의 위치는 NURBS가 아니라 영상 손실로 최적화한다.

- visible surface는 topology별 별도 방법론으로 분기하지 않는다.
- 모든 topology는 observed boundary loop, boundary role, source provenance, interior support를 공통 입력 계약으로 사용한다.
- multi-hole은 outer loop와 모든 interior loop를 보존한다. 비중첩 planar partition 증거가 없으면 `review_required`이며 임의 central fill 또는 hole별 overlapping annulus 복제는 금지한다.
- artifact의 chart가 생성되었다는 사실은 품질·안전·사용 가능성을 뜻하지 않는다.
- **evidence가 없으면 강제로 닫지 않는다.** gap 보간, 임의 candidate 생성, closure만을 위한 낮은-evidence edge 선택은 어떤 단계에서도 금지한다 — 이 원칙은 아래 2절 전체에서 반복적으로 검증·재확인됐다.

## 2. 현재 활성 작업 A — Canonical visible-NURBS production pipeline (full-cloud continuation → representative selection → boundary ordering → materialization)

**2026-07-30(worklog 18) 기준으로 이 경로가 `train.py`/benchmark의 유일한 production 경로가 됐다.** `legacy`, `voxel_patch_stage1`, IDW/local split fallback과 해당 CLI는 전부 제거됐다. 이전 절(과거 버전)에서 "isolated, dispatcher 미연결"로 기술했던 world-space boundary half-edge → ordered graph → materialization adapter 계열 작업(worklog 12-15)은 이 병합 이후 더 이상 isolated가 아니다 — 지금은 `construct_visible_nurbs_from_gaussians` / `TorchOSNGSPipeline._construct_canonical_with_full_evidence`가 곧 production 경로다.

### 2.1 파이프라인 단계

1. **Representative selection** (`torch_density_preserving_representative_selection.py`): 대량 Gaussian을 `canonical_construction_max_points`(기본 2048) 예산으로 mode-aware farthest-point sampling. worklog 49에서 boundary-evidence swap-in(예산 경쟁으로 탈락한 진짜 orientation-divergent evidence를 안전하게 복구)을 추가했다.
2. **Reliability + region formation** (`torch_gaussian_structural_reliability.py`, `torch_gaussian_surface_region_formation.py`): full-cloud contextual evidence 기반 reliability, seed/merge DSU 기반 region consolidation.
3. **Boundary candidate 추출** (`torch_full_cloud_continuation_shell.py`, `torch_boundary_support_termination.py`): representative 주변 same-mode angular gap을 관측해 `observed_support_termination`/`no_gap`/`parallel_sheet_conflict`/`crease_discontinuity`/`ambiguous_continuation`/`reliability_frontier`/`unresolved_sampling_gap`으로 분류. worklog 47(cross-surface leakage), 48(fold/gap-crossing locality), 50(single-radius over-reach)에서 순차적으로 오분류를 좁게 수정했다.
4. **Directed ordering / materialization** (`torch_directed_boundary_ordering.py`): region별 compatible edge에 대한 exact one-in/one-out Hungarian matching으로 closed cycle과 open path를 결정론적으로 복원. worklog 53에서 downstream-invalid 2-cycle이 capacity를 낭비하는 결함과, 그로 인해 노출된 direct/reverse tangent quality 비교의 self-intersection 인식 결함을 수정했다.

### 2.2 현재 결론 (worklog 40~53에 걸친 6단계 소진 감사)

Real 3k/5k/10k checkpoint(cap 2048) replay에서 **5k만 2개 region을 닫고(closed/materialized 2/2), 3k와 10k는 0개**다. worklog 40부터 53까지 다음 단계를 각각 독립적으로 감사했고, 전부 실제 결함을 찾아 좁게 수정했지만 **어느 것도 closed-loop 개수를 바꾸지 못했다**:

| worklog | 감사 대상 | 결과 |
|---|---|---|
| 40 | region-pair 단위 sphere seam 오탐 | 수정, sphere 22 false candidate 제거 |
| 45 | directed compatibility gate의 target tangent 부호 | 수정, Box 6번째 face + real 5k 0→2 closed 회복 |
| 47 | no_gap의 cross-surface(mode) leakage | 수정, 효과는 재분류만 |
| 48 | no_gap의 candidate-local fold/gap-crossing | 수정, 3k/5k 각 1 node만 재분류 |
| 49 | representative selection의 FPS 예산 손실 | 수정(swap-in), real swap 22/51/76건이나 closed 불변 |
| 50 | 단일 4x 반경의 원거리 support 과다 신뢰 | 수정(scale-persistence), 76/36/39 node 재분류, closed 불변 |
| 51 | raw full-cloud에 representative 축약으로 잃은 chain이 있는지 | 기각(3곳 raw에도 chain 없음) + 2곳은 Hungarian 경쟁 문제로 재분류 |
| 52 | 그 Hungarian "경쟁" 주장 자체의 정확성 | 정정 — edge는 실제로 matching에 포함돼 있었음, 진짜 원인은 topology 불가능(region 52) 또는 evidence 열세(region 56) |
| 53 | region 56의 진짜 원인(2-cycle 낭비) | 수정, fragment 2개→4-node open path 1개로 개선(닫히진 않음, 진짜 최댓값이 open이므로) |

**결론: 남은 병목은 파이프라인 결함이 아니라 candidate evidence 자체의 밀도/위상이다.** 3k/10k의 큰 region은 perimeter 전체를 덮을 만큼 충분히 독립적인 observed-termination evidence가 없다 — representative를 더 정확한 위치로 옮기거나(49), 오분류를 고치거나(47/48/50), matching 낭비를 없애도(53) 이 밀도 자체는 바뀌지 않는다. 5k가 성공하는 이유는 이런 결함이 없어서가 아니라 해당 region이 작아서(candidate 4개)다.

### 2.3 검증된 negative control (모든 worklog가 반복 확인)

cap=64 기준: Box `physical=51 closed=6 materialized=6`, Cylinder `16/2/2`, Sphere `14/0/0`, Thin slab `37/3/3`. 이 수치는 worklog 47부터 53까지 매 라운드 재확인했다 — 새 작업은 이 표를 반드시 재현해야 하며, 벗어나면 즉시 원인을 규명한다(worklog 53의 2차 self-intersection 결함이 이렇게 발견됐다).

### 2.4 다음 착수 후보 (미승인, 방향 제안일 뿐)

- Candidate GENERATION 밀도 자체(왜 큰 region이 candidate 3~6개에 그치는지) — 지금까지 전부 candidate 전달/분류/정렬 단계만 감사했고 생성 자체의 근본 원인은 아직 안 봤다.
- ordering/quality 비교(`recover_directed_boundary_components`)가 direct/reverse 전체 scene 단위로 이뤄지는 구조 자체의 재검토 — worklog 53에서 이 결합이 예상치 못한 부작용을 만든 사례가 있었다.

## 3. 현재 활성 작업 B — Isolated Boundary-first hardening (구 Section 2, 현재 사실상 비활성)

과거(2026-07 초) 이 절이 기술하던 exporter/cubic seam wedge/observed-anchor central cap 작업(worklog 4-6)은 그 자체로는 dispatcher/production에 연결된 적이 없다. 2026-07-30(worklog 18) 이후 production은 §2의 canonical 경로로 통합됐고, 이 isolated hardening 라인은 이후 세션에서 별도로 재개되지 않았다. 재개할 경우 다음을 여전히 유지한다: exporter의 outer/interior/support/seam/chart 명시적 분리, sampled crossing/fidelity gate, degree-1/fan 임시면의 최종 근거 사용 금지, false-hole evidence 부족 시 `review_required` 유지.

## 4. 현재 활성 작업 C — Uncertain Gaussian model foundations

Phase G proposal, model-only append adapter, occluded chart ownership foundation은 각각 구현·검증된 계약으로 유지한다. 이들은 visible-surface quality를 대신 증명하지 않으며, append 대상의 appearance/opacity와 downstream lifecycle은 여전히 명시적 차단 조건이다.

- 현재는 model-only 범위다.
- optimizer, trainer, renderer, checkpoint schema, global ranking/selection, conflict resolution 및 production integration은 시작하지 않는다.
- Gaussian append가 허용되려면 chart state와 safety eligibility를 포함한 상위 gate가 충족되어야 한다.

근거: `docs/worklogs/1_phase_g_uncertain_gaussian_proposal_foundation.md`, `docs/worklogs/2_uncertain_gaussian_append_adapter_foundation.md`, `docs/worklogs/3_occluded_chart_ownership_foundation.md`.

## 5. 명시적 비범위

다음은 현재 착수 금지다.

- §2 canonical 경로의 candidate/threshold를 scene-specific하게 튜닝하는 일 (box/cylinder/sphere/thin_slab처럼 shape별 예외를 만드는 것 포함)
- gap 보간, 임의 candidate 생성, 또는 closure만을 위한 낮은-evidence edge/cycle 강제 선택
- representative cap 증가로 evidence 부족을 우회하는 일
- optimizer/trainer/renderer/checkpoint 통합 범위 확대
- global chart ranking·selection 또는 conflict resolution
- 불완전한 false-hole evidence를 이용한 자동 topology 확정
- benchmark artifact만으로 visible surface 품질이 해결되었다고 선언하는 일

## 6. 현재 검증 상태와 알려진 위험

Repository-wide pytest 최신 기준(worklog 53): `720 passed, 1 skipped, 1 warning, 8 subtests passed`. §2.3의 negative-control 표와 함께 매 라운드 재확인한다.

알려진 위험:
- §2.2 표의 real 3k/10k closed-loop 부재는 미해결이며, §2.4 후보(candidate 생성 밀도) 전에는 착수 승인이 없다.
- §2 파이프라인의 direct/reverse tangent quality 비교는 scene 전체 단위 선택이라, 한 region의 수정이 다른 region의 결과를 바꿀 수 있다(worklog 53에서 실측). 이 구조를 변경하는 작업은 반드시 §2.3 negative-control 전체 재확인을 동반한다.
- §3(Isolated Boundary-first)은 재개 시 이 문서를 먼저 갱신하고 시작한다 — 현재는 활성 작업자가 없다.

다음 작업자는 먼저 이 문서와 `docs/worklogs/README.md`의 최신 인덱스, 그리고 가장 최근 작업로그(현재 `docs/worklogs/53_downstream_valid_directed_matching_repair.md`)를 읽고 이어서 작업한다. 과거 방향의 세부 기록은 필요할 때 Git history로만 조회한다.
