# OSN-GS Urgent Work Master

최종 갱신: 2026-07-29

이 문서는 현재 진행 방향과 승인 경계를 정의하는 canonical master다. 과거 실험의 상세 경과는 Git 이력에 보존하며, 현재 판단에 필요 없는 작업로그는 `docs/worklogs/`에서 제거했다.

## 1. 목표 모델과 불변 조건

OSN-GS에서 NURBS는 관측 가능한 표면을 설명하고 가려진 영역의 불확실 Gaussian 생성에만 기하 정보를 제공하는 중간 표현이다. visible/certain Gaussian의 위치는 NURBS가 아니라 영상 손실로 최적화한다.

- visible surface는 topology별 별도 방법론으로 분기하지 않는다.
- 모든 topology는 observed boundary loop, boundary role, source provenance, interior support를 공통 입력 계약으로 사용한다.
- multi-hole은 outer loop와 모든 interior loop를 보존한다. 비중첩 planar partition 증거가 없으면 `review_required`이며 임의 central fill 또는 hole별 overlapping annulus 복제는 금지한다.
- artifact의 chart가 생성되었다는 사실은 품질·안전·사용 가능성을 뜻하지 않는다.

## 2. 현재 활성 작업 A — Isolated Boundary-first visible-surface hardening

기본 dispatcher, trainer, renderer, production constructor에는 연결하지 않은 isolated 경로다. 현재 목표는 모든 topology에서 관측 경계와 interior support를 우선 보존하는 곡선 surface construction과 review export를 확립하는 것이다.

현재 구현은 closed-loop correspondence, cubic seam wedge, observed-anchor central cap, role/provenance payload 및 review geometry를 포함한다. 그러나 최신 v6 artifact는 선형 fan/wedge 인상, source-boundary fidelity 부족, inner/outer boundary 식별 불명확성을 드러냈다. 따라서 다음을 우선한다.

1. exporter가 observed outer boundary, interior boundary, support curve, seam, chart를 명시적으로 분리해 출력한다.
2. sampled crossing/near-contact와 source-boundary fidelity를 hard/review gate로 일관되게 반영한다.
3. degree-1 또는 fan 기반의 임시 면을 최종 품질 근거로 사용하지 않고, 경계 제약 고차 fitting과 bidirectional fidelity를 검증한다.
4. false-hole 판정은 충분한 evidence 전에는 자동 확정하지 않고 `review_required` 또는 `unsupported`를 유지한다.

현재 진척은 isolated construction hardening 기준 약 55%다. 이는 production integration 진척도가 아니며, dispatcher/production 연결은 승인 범위 밖이다. 최신 근거는 `docs/worklogs/110_boundary_first_review_geometry_semantics_and_crossing_gate.md`다.

NURBS Construction benchmark의 기본 입력은 평면형 point sample이 아니라 depth-bearing 3D shell과 baseline-like tangent covariance다. 이 교체는 constructor의 XY/planar 가정을 더 엄격하게 검증하기 위한 것이며, observed Gaussian covariance는 pipeline과 renderer export까지 보존한다. 근거: `docs/worklogs/111_nurbs_construction_synthetic_3d_gaussian_dataset.md`.

## 3. 현재 활성 작업 B — Uncertain Gaussian model foundations

Phase G proposal, model-only append adapter, occluded chart ownership foundation은 각각 구현·검증된 계약으로 유지한다. 이들은 visible-surface quality를 대신 증명하지 않으며, append 대상의 appearance/opacity와 downstream lifecycle은 여전히 명시적 차단 조건이다.

- 현재는 model-only 범위다.
- optimizer, trainer, renderer, checkpoint schema, global ranking/selection, conflict resolution 및 production integration은 시작하지 않는다.
- Gaussian append가 허용되려면 chart state와 safety eligibility를 포함한 상위 gate가 충족되어야 한다.

근거: `docs/worklogs/87_phase_g_uncertain_gaussian_proposal_foundation.md`, `docs/worklogs/88_uncertain_gaussian_append_adapter_foundation.md`, `docs/worklogs/96_occluded_chart_ownership_foundation.md`.

## 4. 명시적 비범위

다음은 현재 착수 금지다.

- Boundary-first isolated 결과를 기본 dispatcher 또는 production path에 연결하는 일
- optimizer/trainer/renderer/checkpoint 통합
- global chart ranking·selection 또는 conflict resolution
- 불완전한 false-hole evidence를 이용한 자동 topology 확정
- benchmark artifact만으로 visible surface 품질이 해결되었다고 선언하는 일

## 5. 현재 검증 상태와 알려진 위험

Boundary-first isolated 회귀는 계속 추가 중이다. `tests/test_trimmed_component_fitter.py`의 과거 `degenerate_fraction` 기대치 불일치 두 건은 hard degeneracy와 relative near-degeneracy를 분리해 해소했다. 전체 pytest의 최신 기준은 `536 passed, 1 skipped, 1 warning, 8 subtests passed`이며, 상세 근거는 `docs/worklogs/114_trimmed_component_jacobian_test_health.md`다. 새 Boundary-first 변경의 통과 근거와 정확한 수치는 최신 검증 작업로그에만 기록한다.

다음 작업자는 먼저 이 문서, `docs/worklogs/110_boundary_first_review_geometry_semantics_and_crossing_gate.md`, 그리고 관련 구현/테스트를 읽고 이어서 작업한다. 과거 방향의 세부 기록은 필요할 때 Git history로만 조회한다.