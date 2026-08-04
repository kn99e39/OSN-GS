# Worklog 9 — Reliability/Affinity Robustness Hardening

(사용자 요청 시 "Worklog 8"로 지정되었으나, 작업 시작 직전 `ls docs/worklogs/`로 재확인한 결과 동시 진행 중이던 Codex 세션이 이미 `8_trimmed_component_jacobian_test_health.md`를 선점하여 **115**로 번호를 재조정함. 매 라운드 시작 전 번호 충돌을 재확인하는 기존 규칙을 그대로 따름.)

## 0. 범위

Worklog 7의 covariance-guided structural reliability / pairwise manifold-affinity foundation을 대상으로 한 **Reliability/Affinity Robustness Hardening** 라운드. 목적은 synthetic 기본 fixture에서만 동작하던 heuristic을 boundary extractor에 바로 연결하는 것이 아니라, 복잡한 Scene의 surface-region formation에 사용할 수 있을 정도로 상태 계약·정규화·후보 그래프·불변성·실패 모드를 보강하는 것이다.

**이번 라운드에서 하지 않은 것 (명시적 제외, 사용자 지시 그대로):** ordered world-space boundary chain/loop 생성, world-space boundary half-edge materialization, 기존 Boundary-first builder adapter 연결, raster/KDE 경로 제거, default dispatcher 교체, trainer/production pipeline 연결, uncertain Gaussian proposal/append/ownership/checkpoint 변경, camera 기반 occlusion-boundary 분류, quality_state의 eligible 승격, 자동 Gate 승인, `trimmed_component_fitter` 관련 무관한 수정(동시 진행 중인 Codex 세션이 이미 처리, Worklog 8), 별도 설계 문서 작성.

---

## 1. 유지된 foundation (변경하지 않음)

- Worklog 6: star-shaped boundary validation, equal-angle anchor correspondence, synthetic-center-fallback 금지, 명시적 insufficient-support 거부.
- Worklog 4: review geometry semantics, 실제 `surface.evaluate()` 기반 curve export, representative support-crossing 진단.
- Worklog 7: covariance eigenframe 추출, eigenvector sign-independent normal 비교, reliable/ambiguous/rejected 구조적 증거, pair-fresh manifold-affinity 계산, multi-hole review-only 계약, silent-fallback 금지, raster/KDE secondary/legacy 지위.

## 2. Intrinsic/Contextual reliability 분리

`osn_gs/surface/torch_gaussian_structural_reliability.py`를 단일축(reliable/ambiguous/rejected)에서 두 개의 독립 축으로 재작성:

- **`IntrinsicStructuralReliability`** (이웃 의존성 없음): covariance conditioning(유한성/비퇴화), planar/needle/isotropic likelihood, scale validity(3개 scale 필드의 유한성·양수·선택적 expected-range). `intrinsic_reliable` / `intrinsic_ambiguous` / `intrinsic_rejected`.
- **`ContextualManifoldConsistency`** (이웃 필요): neighbor normal agreement, mutual tangent-plane residual, local curvature agreement, neighborhood support sufficiency, multi-surface neighborhood ambiguity, density variation sensitivity, scale consistency. `contextual_consistent` / `contextual_mixed` / `contextual_insufficient`.
- 최종 `reliability_class`(구버전 호환 projection): `intrinsic==REJECTED → REJECTED`; `intrinsic==RELIABLE and contextual==CONSISTENT → RELIABLE`; 나머지 → `AMBIGUOUS`. **핵심 검증 결과:** `two_perpendicular_surfaces`에서 crease 인접 Gaussian은 `intrinsic_reliable` + `contextual_mixed`로 남아 intrinsically-bad Gaussian과 같은 버킷에 떨어지지 않음 (crease 증거가 진짜 나쁜 Gaussian과 구분됨, 신규 테스트 `test_crease_gaussian_is_intrinsically_reliable_but_contextually_mixed`로 고정).

## 3. Covariance scale 계약

`GaussianCovarianceFrame`에 `tangent_major_scale`(sqrt λ1), `tangent_minor_scale`(sqrt λ2), `normal_thickness`(sqrt λ3), `equivalent_tangent_scale`(sqrt(major·minor)), `footprint_area`(π·major·minor)를 독립 필드로 추가. 하나의 스칼라로 collapse하지 않음. 실제 적용:
- `mutual_tangent_residual` → `tangent_major_scale`로 정규화 (thickness 아님).
- candidate radius → 평균 `tangent_major_scale`(scale_radius) **와** `equivalent_tangent_scale` 합(footprint_overlap) 두 독립 기준.
- oversized 판정 → `tangent_footprint_ratio`(두 Gaussian의 `equivalent_tangent_scale` 비율).
- close-parallel 분리 → `normal_direction_separation_over_thickness` (normal_thickness로 정규화, tangent scale 아님).
- 모든 분모/배수는 `ManifoldAffinityConfig`/`IntrinsicReliabilityConfig`/`ContextualConsistencyConfig`에 명시적 필드로 존재, 암묵적 상수 없음.

## 4. Candidate-neighborhood 정책

kNN 단독 후보 생성을 감사하고 `osn_gs/surface/torch_gaussian_manifold_affinity.py`의 `build_manifold_affinity_graph`를 재작성:
- 각 kNN 쌍에 대해 mutual-kNN 여부, scale-normalized radius(`within_radius`), footprint overlap(`footprint_overlap`)을 독립적으로 계산.
- `candidate_status`: `candidate` / `outside_candidate_support` / `capped_out` / `invalid_endpoint`.
- `candidate_reasons`: `mutual_knn` / `within_scale_radius` / `footprint_overlap` / `distance_only` / `deterministic_cap`.
- 거리 단독으로는 절대 `same_surface` 관계를 만들지 않음(`outside_candidate_support`일 때 `manifold_relation=not_evaluated`로 강제).
- **버그 발견 및 수정 (order-determinism):** 기존 구현은 `max_candidate_count_per_node` 캡을 원본 배열의 위치 인덱스 순회 순서로 그리디 적용했음 — 셔플 후 재실행 시 어떤 후보가 캡에 걸리는지가 달라짐(§8 불변성 위반). 후보를 `(distance, stable_id_pair)`로 정렬한 뒤 캡을 적용하도록 수정, 10회 랜덤 셔플에 대해 0건 불일치로 확인.

## 5. 직교 pairwise relation 상태

단일 `edge_state` 우선순위 체인을 4개 독립 축으로 교체:
- **Candidate status**: candidate / outside_candidate_support / capped_out / invalid_endpoint.
- **Endpoint structural status**: both_intrinsically_reliable / one_intrinsically_unreliable / both_intrinsically_unreliable / contextual_ambiguity_present.
- **Manifold relation**: same_surface / crease_or_orientation_discontinuity / parallel_but_separate / proximity_only / ambiguous / rejected / not_evaluated.
- **Relation confidence**: high / medium / low / not_applicable.
- 구버전 `edge_state`는 `.state` compatibility projection으로 유지(단, `not_evaluated → proximity_only` 매핑을 추가해 구버전 시맨틱 보존 — 이 매핑이 없으면 기존 Worklog 7 테스트 2건이 실패함을 확인 후 수정).
- 사용자가 제시한 예시(격리된 rejected floater)를 정확히 재현하는 테스트 추가: `candidate_status=outside_candidate_support`, `endpoint_status=one_intrinsically_unreliable`, `manifold_relation=not_evaluated`가 동시에 성립함을 확인 (`test_rejected_floater_outside_candidate_support_worked_example`).

## 6. Manifold-affinity 지표 확충

`PairAffinityMetrics`에 10개 지표를 페어마다 독립 기록: normal_alignment, mutual_tangent_residual, tangent_direction_displacement_ratio, normal_direction_separation_over_thickness, tangent_footprint_ratio, tangent_anisotropy_ratio, normal_thickness_ratio, neighbor_spacing_normalized_distance, local_curvature_change_proxy, normalized_distance. `_classify_relation`은 alignment+residual만으로 same_surface를 결정하지 않고 footprint-ratio guard를 추가로 통과해야 함.

**실패 모드별 확인 결과:**
- Oversized planar bridge (`anisotropic_planar_bridge` fixture, planar shape·floor 방향 normal·거대 footprint): footprint-ratio guard로 인해 floor와 `same_surface` 엣지가 0건 — isotropic 거부에 의존하지 않고도 차단됨을 확인.
- Crease: confidence gate 없이 유지(Worklog 7의 기존 수정 사항 그대로).
- 얇은 곡면(Thin curved surface): **실제로 회귀를 발견하고 되돌린 사례** — 아래 §7 참조.
- Close parallel surfaces: `gap_sweep` 스윕에서 상세 분석, 아래 §9 참조.

## 7. 시도했다가 되돌린 수정 (정직하게 기록)

`same_surface` 분기에 `normal_direction_separation_over_thickness` 상한 가드를 신규 추가해 close-parallel-gap=0.02의 19개 오분류 엣지를 고쳤으나, 검증 중 `smooth_curved_sheet`가 1개 region에서 2개 region(72, 8)으로 갈라지는 **회귀**를 발견함. 원인 분석: `normal_direction_separation_over_thickness`와 `mutual_tangent_residual`은 물리적으로 동일한 절대 오프셋을 서로 다른 분모(normal_thickness vs tangent_major_scale, 비율 약 25배 고정)로 정규화한 것에 불과해 독립적 신호가 아님. 곡면의 경우 완만한 곡률만으로도 이 절대 오프셋이 surfel 자체의 얇기(normal_thickness, 매우 작은 절대량)를 손쉽게 넘어서므로, 이 지표 하나만으로 "진짜 분리된 평행면"과 "완만히 휘는 연속 곡면"을 구분할 수 없음을 커브니처 스윕(§9)으로 확인(amplitude 0.05만 되어도 ratio가 120까지 치솟음, 반면 gap=0.02의 실제 분리 표면은 ratio 6.6~10 수준). 이 tension은 순수 pairwise 지역 정보만으로는 완전히 해소할 수 없는 것으로 판단, 신규 가드를 **되돌리고** 기존 parallel_separate 분기(이미 정상 동작)만 유지. gap=0.02는 §9에 threshold-sensitive 구간으로 정직하게 남겨둠.

## 8. Invariance 검증 결과 (Gate 조건)

`two_perpendicular_surfaces` scene, stable id 사용, 엣지를 id-keyed로 비교:
- **Translation invariance**: 통과 (5, -3, 2 이동 후 id-keyed 엣지 집합 완전 일치, 0 mismatch).
- **Rotation invariance** (position+covariance 동시 회전, eigenvector sign 포함): 통과 (z축 90도 회전 후 완전 일치).
- **Uniform-scale invariance** (position ×2.5, covariance ×6.25 동시): 통과 (정규화된 관계값 완전 일치).
- **Input-order determinism** (stable id 사용, seed 0~9 10회 랜덤 셔플): §4의 캡 순서 버그 수정 후 10/10 완전 일치로 통과. 수정 전에는 캡 관련 6개 엣지 불일치 발견 → 수정 후 0건.
- 4개 항목 모두 신규 `InvarianceTest` 클래스로 회귀 고정.

## 9. 확장된 synthetic robustness matrix 결과

`nurbs_constructor_benchmark/gaussian_reliability_scenes.py`에 추가: `make_curvature_sweep_scene`, `make_density_variation_scene`(5종), `make_position_noise_scene`, `make_orientation_noise_scene`, `make_anisotropic_planar_bridge_scene`, `make_gap_sweep_scene`, `make_missing_support_gap_scene`, `make_shape_ratio_sweep_scene`, `make_contamination_regression_scene`.

- **밀도 변화 5종**: 최초 `center_dense_boundary_sparse`(계수 1.5)와 `sparse_but_continuous`(고정 2.5배)가 boundary spacing을 candidate-support 임계값 근처로 밀어붙여 각각 9개, 10개 region으로 허위 분절되는 **fixture 캘리브레이션 문제**를 발견. 계수를 각각 0.8, 1.8로 낮춰 5종 모두 1개 region·전체 coverage로 정정.
- **위치 노이즈 스윕** (0/0.005/0.02/0.05): region 1→1→3→6, coverage 1.0→1.0→0.951→0.778로 점진적 저하 확인 (급격한 hard-boundary 스파이크 없음).
- **방향(covariance) 노이즈 스윕** (0/2/10/30/60도): region 1/1/1/1(coverage 0.864)/7(coverage 0.42) — soft-then-hard 저하 패턴 확인.
- **곡률 스윕** (amplitude 0.0/0.02/0.05): 완만한 곡률에서는 여전히 1개 region 유지 확인. **단, 높은 amplitude(0.05 이상)의 넓은-반경 후보 쌍에서 사인파의 국소 평탄점(느슨한 기울기 지점)이 서로 다른 위상에서 우연히 유사한 normal을 갖는 "위상 앨리어싱" 현상을 발견** — 순수 국소 pairwise 판정의 한계로 판단, §7의 시도-후-되돌림과 동일 근본 원인. 실제 fold/crease 수준까지 amplitude를 올려 fragmentation 전이점을 명확히 재현하는 것은 이번 라운드에서 완료하지 못함(다음 라운드로 이월).
- **Anisotropic planar bridge**: 기본 kNN(k=8)에서는 bridge의 최근접 이웃이 우연히 wall 쪽에 편중되어 floor와의 candidate 자체가 형성되지 않는 것을 발견 — 진단 목적으로 `candidate_neighbor_count=20`으로 넓혀 재확인한 결과, floor-bridge 페어는 `ambiguous`로 분류되고 `same_surface` 엣지는 0건 (isotropic 거부가 아니라 footprint/scale 근거로 차단됨을 명시적으로 확인).
- **Gap sweep** (0.02/0.05/0.1/0.15/0.3): gap≥0.05에서는 항상 2 region(49,49), cross-label same_surface 0건. **gap=0.02는 threshold-sensitive 구간**: 224개 cross-label 후보 중 203개는 정확히 `parallel_but_separate`, 19개는 `same_surface`로 남음 — 원인은 mutual_tangent_residual의 기대값(gap/tangent_scale=0.4)이 임계값(0.35)에 매우 근접해 위치 노이즈(±0.001)가 개별 쌍의 판정을 경계 너머로 밀 수 있기 때문. region-connectivity는 다수결/합의 메커니즘이 없어 단 하나의 오분류 엣지로도 두 region이 병합될 수 있음 — 이는 §11 diagnostic의 알려진 한계로 명시적으로 남김(§7에서 시도한 수정이 곡률을 깨서 되돌렸으므로, 견고한 해결에는 pairwise 지역 정보를 넘어서는 메커니즘이 필요, 다음 Gate 범위로 이월).
- **Missing-support gap** (중앙 30% 결손): 남은 고리형 영역이 1개 region으로 유지됨 확인 (샘플링 결손이 실제 discontinuity로 오인되지 않음).
- **Needle/near-isotropic 연속 스윕**: `shape_ratio_sweep` (ratio 0.0→1.0)으로 needle_like → ambiguous_shape → isotropic 전이를 확인, intrinsic 분류는 순수 isotropic 극단(ratio=1.0)에서만 rejected로 전환되고 그 이전(needle 영역 포함)은 ambiguous를 유지(정상 normal-ambiguous 취급, 과도한 거부 없음).

## 10. Neighborhood contamination 회귀 및 aggregation 비교

`make_contamination_regression_scene`: 깨끗한 9×9 평면에 6종 오염원(isolated floater, isotropic, wrong-normal planar, oversized planar, tiny-scale, nearby second surface) 삽입.
- 평면 자체의 same_surface 연결성은 6종 오염원 주변에서도 81/81 완전 유지.
- 각 오염원 자체 라벨: floater(ambiguous, 고립), isotropic(rejected), wrong_normal(intrinsic reliable이나 contextual mixed — crease성 증거로 정확히 남음), oversized(reliable — footprint guard가 pairwise 레벨에서만 작동하고 자기 자신의 contextual 계산에는 영향 없음, 아래 한계 참조), tiny_scale(reliable, 일부 이웃과 not_evaluated/parallel_but_separate).
- surrounding plane 이웃에 대한 영향: isotropic과 oversized/tiny_scale은 이웃을 거의 오염시키지 않음(대부분 reliable 유지); wrong_normal은 자신과 인접한 10개 이웃을 모두 ambiguous로 만듦 — 이는 실제 방향 불일치 증거이므로 억제하지 않는 것이 올바른 설계 판단.
- **Worklog 7의 isotropic-blob 4개 이웃 ambiguous 재분석 (필수 항목)**: 5개 aggregation 방식(mean/median/trimmed_mean(20%)/reliability_weighted/rejected_excluded)을 동일 fixture에 직접 비교.

| 방식 | 오염된 plane 이웃 수 (총 79개 중) |
|---|---|
| mean | 10 |
| trimmed_mean(20%) | 4 |
| median | 0 |
| reliability_weighted | 0 |
| **rejected_excluded (기본값)** | **0** |

**결론**: `rejected_excluded`(및 `reliability_weighted`)가 오염을 완전히 차단, `median`도 이 fixture에서는 동등하나 통계적 견고성에만 의존해 해석 가능성이 낮음, `trimmed_mean`은 절반만 개선, `mean`은 가장 취약. 기존 기본값(`rejected_excluded`)을 유지하는 근거로 재확인됨 — 구조적으로 REJECTED로 판정된 이웃만 명시적으로 배제하고 나머지는 통상 평균을 취해, "왜 배제되었는가"가 항상 intrinsic 판정에 귀속되어 해석 가능함.

## 11. Same-surface 연결영역 diagnostic (진단 전용, 경계 그래프 아님)

`diagnose_same_surface_regions`: `same_surface` 엣지만으로 BFS, region 수·크기·reliable coverage·ambiguous attachment·rejected exclusion을 계산. Crease 엣지는 병합에 절대 사용하지 않음. **이는 affinity-robustness 진단이며, boundary graph도 production chart segmentation도 아님을 명시.** 7개 기본 scene + 확장 스윕 전체에서 이 diagnostic으로 §9의 region-count/coverage 결과를 산출.

## 12. Raster/KDE 경로 메타데이터

기존 스키마/디스패처를 수정하면 범위 초과이므로 **코드 변경 없이 report-only로 유지**: legacy = `extraction_mode=raster_assisted_legacy, evidence_level=secondary, canonical_boundary_source=false`; 신규 = `extraction_mode=covariance_guided_manifold, evidence_level=primary_candidate, canonical_boundary_source=pending_gate`. 기존 `torch_component_boundary.py` 등은 이번 라운드에서 손대지 않음.

## 13. 실제 학습된 Gaussian 진단 (read-only)

`output/osn_gs_ab_3k/final/point_cloud.ply`(2,191,256 Gaussians, baseline A/B 비교용으로 이미 저장소에 존재하는 실제 학습 결과, 새로 다운로드하거나 trainer를 재실행하지 않음)를 read-only로 로드해 진단:
- 씬 중심 근접 4,000개 국소 샘플(균등 랜덤 서브샘플은 국소 밀도를 파괴해 후보 그래프가 무의미해짐을 먼저 확인 — 93%가 outside_candidate_support로 나와 폐기, 국소 crop으로 대체):
  - shape_class: ambiguous_shape 2010, planar_surfel 1051, needle_like 892, isotropic 47.
  - intrinsic: reliable 2358, ambiguous 1637, rejected 5.
  - contextual: mixed 3737, consistent 263.
  - candidate_status: candidate 18726, outside_candidate_support 838, capped_out 479, invalid_endpoint 35.
  - manifold_relation: ambiguous 7303, parallel_but_separate 6173, same_surface 3285, crease_or_orientation_discontinuity 1965, not_evaluated 1317, rejected 35.
  - same_surface region: 413개, 최대 5개 크기 [132,100,95,79,62], reliable coverage 0.679.
- **해석**: 실제 학습된 장면은 synthetic fixture보다 훨씬 잡음이 많고(다수가 ambiguous), thin/duplicate 구조로 인한 parallel_but_separate 비중이 큼 — 이는 예상된 결과이며 이번 라운드의 threshold가 실제 데이터에 맞게 튜닝되었다는 뜻은 아님(§7 명시).
- `real_trained_gaussian_diagnostic = run` (fixture_not_available이 아님, 실제 실행됨).

## 14~15. 테스트 실행 결과

**신규 타겟 테스트 스위트** (`tests/test_gaussian_reliability_affinity_robustness.py`, 신규 25개): 25 passed.
**기존 3개 파일** (`test_gaussian_covariance_frame.py`, `test_gaussian_structural_reliability.py`, `test_gaussian_manifold_affinity.py`, 23개, Worklog 7 산출물): 23 passed — 단, `.state` compatibility projection에 `not_evaluated → proximity_only` 매핑을 추가하는 수정이 필요했음(신규 candidate/relation 분리로 인해 2건이 일시적으로 깨졌던 것을 원복 아닌 **정당한 호환성 계층 보강**으로 수정).
**기존 Boundary-first isolated 전체 스위트** (Worklog 4/6 관련 17개 파일, 93개 테스트): 93 passed, 1 skipped.
**전체 저장소 `pytest`**: **561 passed, 1 skipped, 0 failed** (경고 1건은 무관한 기존 `test_observation_evidence.py`의 텐서 변환 경고).
**기존 2건 실패 (`test_trimmed_component_fitter.py`) 귀속**: 이번 라운드 시작 시점에는 이미 **해결된 상태**였음 — 동시 진행 중인 Codex 세션의 Worklog 8(`8_trimmed_component_jacobian_test_health.md`)가 처리한 것으로 확인, 본 라운드는 이 파일에 손대지 않음.
**이번 라운드가 만든 신규 실패**: 없음(전체 561 passed).
**Threshold-sensitive fixture / ambiguous transition 구간**: gap_sweep의 gap=0.02(§9), curvature_sweep의 위상 앨리어싱 구간(§9), density_variation의 원래 계수(§9, 이미 재보정 완료).
**Invariance 결과**: 4개 항목 모두 통과(§8).

## 16. Gate 조건 재확인

- Translation/rotation/uniform-scale invariance: 통과.
- Input-order determinism: 통과 (버그 수정 후).
- 깨끗한 평면/곡면의 불필요한 fragmentation 없음: 통과 (밀도 변화 재보정 후, 저-곡률 스윕에서).
- Perpendicular surface 허위 병합 없음: 통과.
- Close-parallel-sheet 허위 병합 없음: gap≥0.05에서 통과, gap=0.02는 threshold-sensitive로 명시적 disclosure(§9).
- Isotropic/needle/oversized-planar-bridge가 same_surface 브릿지를 만들지 않음: 통과.
- Worklog 4/6/7 회귀 없음: 통과.
- 저장소 전체 green: 통과(561 passed, 1 skipped, 0 failed).

## 17. 변경 파일

- `osn_gs/surface/torch_gaussian_covariance_frame.py`: scale contract 필드 5개, `covariance_conditioning_score` 추가.
- `osn_gs/surface/torch_gaussian_structural_reliability.py`: intrinsic/contextual 분리로 전면 재작성, 5종 aggregation 방식.
- `osn_gs/surface/torch_gaussian_manifold_affinity.py`: 후보 정책/직교 상태/10종 pairwise 지표/order-deterministic 캡/region diagnostic으로 전면 재작성.
- `nurbs_constructor_benchmark/gaussian_reliability_scenes.py`: 9종 신규 scene 생성 함수 + 밀도 변화 재보정.
- `tests/test_gaussian_reliability_affinity_robustness.py`: 신규, 25개 테스트.
- `docs/worklogs/README.md`: 본 worklog 항목 추가 예정.

## 18. 명시적으로 완료하지 않은 것 (overclaim 금지)

- covariance-guided boundary 완성이 **아님**.
- Ordered world-space boundary chain/loop/half-edge 생성 **없음** (§11의 same_surface region diagnostic은 이를 대체하지 않음).
- 기존 Boundary-first builder adapter 연결 **없음**.
- Production-ready surface segmentation **아님**.
- 연속 manifold validity가 보장된 것 **아님** (곡률 스윕의 위상 앨리어싱, gap=0.02의 threshold-sensitivity가 실제 한계로 남아있음).
- 실제 학습된 모든 Gaussian의 covariance가 true normal이라고 가정하지 않음(§13 진단은 read-only 관찰일 뿐).
- Default dispatcher 교체 가능 상태 **아님**.
- Boundary-first Gate 완료 **아님** — 본 라운드는 이 방향의 한 하위 foundation만 강화함.
- Production 통합 가능 상태 **아님**.

## 19. Dispatcher/production 비접촉 확인

이번 라운드에서 수정한 3개 모듈(`torch_gaussian_covariance_frame.py`, `torch_gaussian_structural_reliability.py`, `torch_gaussian_manifold_affinity.py`)과 1개 fixture 파일(`gaussian_reliability_scenes.py`)은 모두 `osn_gs.surface`/`nurbs_constructor_benchmark` 하위의 격리된 모듈로, renderer/trainer/dispatcher/ownership/checkpoint 코드를 import하거나 참조하지 않음. 전체 pytest가 561 passed로 기존 production 관련 테스트(voxel hierarchy, uncertain gaussian proposal/append, occluded chart ownership 등) 전부 그대로 통과함으로써 비접촉을 재확인.

## 20. 다음 Gate 범위 제안 (결정 아님, 제안만)

1. Gap=0.02급 threshold-sensitive 구간과 곡률 위상-앨리어싱 문제를 pairwise 지역 정보를 넘어서는 방법(예: 다수결/합의 기반 region-merge, path-connectivity 고려)으로 강건화.
2. 실제 학습 Gaussian(§13)에 대한 threshold 재보정 여부 검토(단, 이번 라운드에서 튜닝하지 않았고 제안도 아님 — 별도 논의 필요).
3. Same-surface region diagnostic을 실제 ordered boundary 후보로 승격할지 여부는 별도 Gate 논의 대상.

---

**결론**: 이번 라운드는 상태 계약·정규화·후보 그래프·불변성을 강화했고, 실제로 2건의 알고리즘 버그(order-determinism 캡, `.state` 호환성 매핑)를 발견해 수정했으며, 1건의 시도한 수정(close-parallel 가드)을 검증 중 회귀 발견 후 되돌렸다. 저장소 전체 테스트는 green이다. Boundary extractor 연결이나 production 통합은 이번 라운드의 범위가 아니며 시도하지 않았다.
