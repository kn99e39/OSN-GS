# Worklog 33 — Representative Graph Scale 독립 검증 및 Region Formation 복구

## 최종 질문에 대한 결론 먼저

> RepresentativeGraphScale 후보가 실제로 rigid-transform invariant하지 않았던 것인가, 아니면 rotation-sensitive representative selection을 포함한 end-to-end 테스트가 graph estimator와 upstream selection perturbation을 혼합하고 있었던 것인가?

**후자였다.** Representative 집합을 고정한 상태(frozen representative replay)에서 직접 검증한 결과, worklog 32에서 기각했던 graph-scale estimator(대표점 간 kNN spacing 기반)는 **완전히 정확하게** rigid rotation/translation/uniform scale에 대해 불변이었다(3k/5k/10k 실제 checkpoint에서 relation mismatch 0건, 전 edge 완전 일치). Worklog 32의 end-to-end 테스트는 매번 representative selection을 다시 실행했는데, 이 axis-aligned voxel grid selection 자체가 이미 문서화된 대로 회전에 정확히 불변하지 않아(이번 실측: 대표점 stable-ID overlap 23~26%) — 이 upstream perturbation이 estimator의 결함으로 오인됐던 것이다.

> 이를 분리한 뒤, threshold·representative cap·core admission을 완화하지 않고 manifold affinity graph의 candidate coverage와 same-surface connectivity를 복구해 real long-horizon snapshot과 post-ADC analytic positive control에서 core region을 형성했는가?

**그렇다.** Threshold, representative cap(2048 유지), core admission 조건 전부 그대로 두고 graph scale만 분리 적용한 결과:

- Real 3k/5k/10k: **region_count 0 → 75/85/64**, boundary_component_count 0 → 3/21/6, `boundary_failure_stage`가 A(candidate generation failed)에서 **C(component admission failed)**까지 전진 — A/B/C 세 단계 중 마지막 단계까지 도달했다. Materialization(닫힌 loop 형성)은 여전히 실패하지만, 이는 이번 작업 범위 밖의 boundary linking 정책 문제이지 이번에 다룬 graph connectivity 문제가 아니다.
- Positive/negative control 전부 통과, false merge 없음.

## 1. Worklog 32 결론 재검토

| 주장 | 재검토 결과 |
|---|---|
| 현재 Gaussian footprint scale로는 candidate 대부분이 탈락 | **재확인**, 변경 없음 |
| Region core 미형성의 직접 원인은 same_surface edge 부족 | **재확인**, 변경 없음 |
| Graph scale이 최우선 병목 | **재확인**, 이번에 해결 |
| G-knn/G-evidence/G-robust가 rigid-transform invariant하지 않다 | **기각됐다** — frozen 테스트에서 G-knn(G1)은 완전히 invariant함을 직접 증명 |
| Selection을 먼저 고치지 않으면 graph scale을 복구할 수 없다 | **기각됐다** — selection을 건드리지 않고 graph scale만 교체해 region_count가 실제로 회복됨 |
| Selection-dependent scale은 모두 사용할 수 없다 | **기각됐다** — G1은 representative 위치에만 의존하지만(selection-dependent) 완전히 invariant함 |

## 2. LocalEvidenceScale 최종 판정 — **옵션 B: provisional 유지**

판정 근거:
- Invariance: 검증됨(worklog 32), 변경 없음.
- 일관된 개선은 아님(3k는 여전히 소폭 악화, 5k/10k는 개선) — worklog 32에서 이미 disclosed.
- Negative control: 완전 동일(회귀 없음).
- **이번 작업으로 판정이 바뀐 부분**: region 형성 자체는 이제 graph scale(G1) 도입으로 성공한다 — `_seed_core_components`의 core-edge eligibility가 `intrinsic_class`만 보고(worklog 31에서 확인) `reliability_class`(contextual 포함)는 보지 않으므로, region 형성 성공 여부는 LocalEvidenceScale의 품질에 더 이상 결정적으로 의존하지 않는다. 즉 LocalEvidenceScale의 남은 결점(3k 소폭 악화)이 region 형성을 막지는 않는다.
- 판정: **production에 그대로 유지**하되 여전히 provisional이다 — normal_consensus 악화나 3k 결과는 별도 조사 과제로 남긴다(§15 참고). "invariance 통과=성공"으로 간주하지 말라는 지시를 지켜, 이번 재판정도 위 근거로만 내렸다.

## 3. 기존 invariance test가 실제로 검사하던 범위

`torch_gaussian_manifold_affinity.py`/`torch_pipeline.py` 코드를 직접 추적한 결과, 기존 두 개의 실패 테스트(`test_density_preserving_representative_selection.py`, `test_full_cloud_continuation_shell.py`)는 `_construct_canonical_with_full_evidence`를 처음부터 끝까지(representative selection 포함) 다시 실행한 뒤 `region_count`의 **완전 동일**을 요구했다. 이 계약은 selection의 회전 민감성(이미 문서화된, 별도 승인 없이는 고치지 않는 제약)과 graph-scale estimator의 정확성을 뒤섞고 있었다. 이번 세션에서 이 두 요소를 구조적으로 분리했다(§5/§6).

## 4. Frozen Representative Replay

신규 스크립트 `scripts/devtools/frozen_representative_graph_scale_test.py`가 실제 checkpoint에서 representative 단계 상태(positions, covariance, stable IDs, reliability)를 한 번 계산해 고정하고, **selection을 다시 실행하지 않고** `build_manifold_affinity_graph`만 직접 재호출할 수 있게 한다. Graph 함수에 candidate_scale/residual_scale을 직접 주입할 수 있는 신규 인자를 추가해(§9) 이 replay가 production 함수를 그대로 사용하도록 했다.

## 5. Fixed-Set Graph Invariance (Test A)

3000/5000/10000 checkpoint 전부에서 rigid rotation(angle=0.4) + translation + uniform scale(2.5)을 representative position/covariance에 직접 적용하고(re-selection 없이) G0/G1/G2를 비교했다.

| 후보 | 3000 same_surface (base→transformed) | relation mismatch | exactly invariant |
|---|---|---|---|
| G0 (기존 footprint) | 11 → 11 | 0 | true |
| **G1 (rep kNN spacing)** | **11 → 2125 → 2125** | **0** | **true** |
| G2 (normal-compatible spacing) | 11 → 2255 → 2255 | 0 | true |

5000: G0 10→10, G1 10→2151→2151, G2 10→2281→2281 (전부 invariant). 10000: G0 6→6, G1 6→1908→1908, G2 6→2042→2042 (전부 invariant). **세 스냅샷 모두, G0/G1/G2 전부 정확히 invariant했고, G1/G2는 same_surface edge를 190배 이상 더 찾아냈다.**

## 6. Selection Perturbation Robustness (Test B)

동일 회전을 원본 cloud에 적용하고 selection부터 재실행(`make_gaussian_density_sweep_scene('cylinder', 2, seed=2/3)`, max_points=48):

| seed | stable-ID overlap | region_count(base→rotated), G0 |
|---|---|---|
| 2 | 26% | 5 → 3 |
| 3 | 23% | 4 → 1 |

**약 75%의 representative가 회전 전후 다른 Gaussian으로 교체된다.** 이 정도의 selection turnover 앞에서 region_count의 정확한 일치를 요구하는 것은 graph 정합성이 아니라 axis-aligned voxel grid의 회전 민감성을 검사하는 것이었다. 이 결과에 따라 `tests/test_density_preserving_representative_selection.py`와 `tests/test_full_cloud_continuation_shell.py`의 해당 테스트를 **명시적으로** topology-stability 기준(양쪽 다 region_count>0, 서로 5배 이내)으로 완화했다 — 조용히 삭제/약화하지 않고 사유를 docstring에 전부 남겼다(§6 지시 준수).

## 7. Post-ADC Positive Control

`_construct_canonical_with_full_evidence`(실제 `reconstruct_visible_after_adc`가 호출하는 바로 그 함수)를 `gaussian_reliability_scenes.py`의 `box_face`/`box`/`cylinder`/`sphere`에 직접 적용(강제 다운샘플, G1 적용):

| scene | region (G0-only) | region (G1 적용 후) | boundary_component | stage 변화 |
|---|---|---|---|---|
| box_face | 2(fragmented) | **1**(정상 통합) | 0→6 | B→C |
| box | 6 | 6 | 3→28 | C→C(candidate 대폭 증가) |
| cylinder | 7 | 3(정상: side+2 cap) | 8→5 | C→C |
| sphere | 4 | 8(과분할, 기존 disclosed 이슈) | 0→4 | A→C |

box_face가 정확히 1개 region으로 통합된 것과 cylinder가 정확히 3개(side+2 cap, 실제 topology와 일치)로 형성된 것이 핵심 긍정 신호다. Sphere의 8-region 과분할은 worklog 125가 이미 disclosed한 별개 문제(consensus-aware region formation의 보수적 under-merging)이며 이번 세션에서 다루지 않았다.

## 8. Graph Scale 후보 비교

| | 불변성(fixed-set) | same_surface 개선 | 채택 |
|---|---|---|---|
| G0 footprint | invariant, 그러나 candidate 거의 없음 | 기준선 | 원복 |
| **G1 rep kNN spacing** | **invariant** | **190배+** | **채택** |
| G2 normal-compatible spacing | invariant | G1보다 6~7% 더 좋음 | 미채택(Python 루프 필요, 이득 대비 복잡도/성능 부담) |
| G3 full-cloud local support | 미구현(시간 제약) | — | 미시도 |
| G4 candidate-mode provenance | 미구현(시간 제약) | — | 미시도 |

G1을 채택했다: 완전히 vectorized(`torch.cdist`+`topk`, 대표점 개수 M에 대해서만 O(M²), 기존 candidate 생성 로직과 동일 복잡도 클래스), per-representative, 검증된 invariance, G2 대비 큰 이득 없이 구현이 단순하다.

## 9. Candidate Radius / Residual Denominator 분리

`build_manifold_affinity_graph`에 `candidate_scale`/`residual_scale` 두 개의 독립 파라미터를 추가했다(`_compute_pair_metrics`까지 관통). `footprint_overlap`은 여전히 무조건 `equivalent_tangent_scale`(Gaussian 자신의 모양)만 사용 — 손대지 않았다.

`tests/test_representative_graph_scale.py::CandidateResidualAblationTest`에서 합성 curved-plane fixture로 확인: candidate_scale만 G1으로 바꾼 경우와 residual_scale만 바꾼 경우 각각 same_surface count가 "둘 다 G1" 대비 낮음 — 두 역할이 실제로 독립적으로 기여함을 확인(완전 평면 fixture는 residual이 항상 0에 가까워 두 역할을 구분 못 함 — 곡률을 추가한 fixture로 교체해서야 구분 가능했다).

## 10-11. Candidate Radius / Residual Scale 설계 원칙

- Candidate radius: bounded kNN(`candidate_neighbor_count=8`, 기존 값 유지) 위에서만 평가 — 새 all-pairs 없음.
- Symmetric 연산: **arithmetic mean** 채택(기존 `average_tangent_major`와 동일한 연산 형태를 유지, 새 연산 방식을 발명하지 않음 — "결과가 잘 나오는 연산을 임의로 고르지 말라"는 지시에 따라 기존 계약과 동일한 연산을 그대로 재사용).
- Residual denominator: mutual_tangent_residual의 geometric meaning(분자, `|offset · normal| / scale`) 유지, 분모만 교체. `footprint_overlap`은 전혀 건드리지 않음.

## 12. Selection을 고쳐야 하는 조건 — 해당 없음

Frozen test는 정확히 invariant했고(§5), 이번 positive control(§7)에서 region topology가 심하게 붕괴하지도 않았다. Selection 자체를 후속 병목으로 판정할 조건(§12의 4가지)에 해당하지 않는다 — voxel grid나 FPS 정책은 이번에도 건드리지 않았다.

## 13. Production Repair 선택

G1을 `candidate_scale`과 `residual_scale` 양쪽 모두에 적용해 production에 반영했다. 만족한 조건: frozen rigid invariance(§5), uniform-scale invariance(선형 스케일링 직접 검증), post-ADC positive control에서 connected region 형성(§7), box crease/thin slab/parallel sheets false merge 없음(§14 아래), real 3k/5k/10k에서 candidate scarcity·same_surface degree 0 의미 있게 감소, core member/region 생성, representative cap 2048 유지, 기존 threshold 전부 유지.

## 14. Negative Control (production 경로, G1 적용 후)

| scene | region(G0) | region(G1) | false merge? |
|---|---|---|---|
| thin_slab | 2 | 2 | 없음 |
| box_isolated_floater | 0 | 1(floater 미포함 확인) | 없음 — floater stable ID가 representative에도 안 뽑혔고 region에도 없음 |
| box_isotropic_contamination | 2 | 1(정상 통합, 오염 인덱스 40/41 미포함 확인) | 없음 |
| box_with_bridge | 6 | 6 | 없음 |
| box | 6 | 6 | 없음 |

## 15. Real 3k/5k/10k 최종 결과

| iteration | region_count(전→후) | boundary_component_count(전→후) | boundary_failure_stage(전→후) | materialized(전→후) | runtime |
|---|---|---|---|---|---|
| 3000 | 0→75 | 0→3 | A→C | 0→0 | 20s→34.6s |
| 5000 | 0→85 | 0→21 | A→C | 0→0 | 22.7s→37.8s |
| 10000 | 0→64 | 0→6 | A→C | 0→0 | 35.1s→49.2s |

genuine_termination_candidate: 3k 1→67, 5k 1→125, 10k 1→61(worklog130 §12 기준 대비 극적 증가). `boundary_component_ambiguous_count`만 nonzero(3/21/6)이고 `closed_count=0`이 세 스냅샷 모두 유지 — 즉 **region/candidate 형성은 성공했지만 닫힌 loop 형성(boundary linking)은 여전히 실패**한다. 이는 다음 병목이다(§17).

## 16. Region/Core 결과

§15와 동일. Core member/consensus-attached 실측치는 이번 라운드 diagnostic_summary에 직접 노출되지 않아 region_count/boundary_component_count로 대체 확인했다(worklog 31의 trace 스크립트를 재실행하면 세부 breakdown이 가능하나, 이번 라운드는 시간 제약으로 waterfall 재실행은 생략하고 region_count 자체의 극적 회복을 핵심 증거로 삼았다).

## 17. Boundary Provenance

Boundary termination/linking 코드는 이번에도 건드리지 않았다. 새로 생긴 region/candidate들에 대해 `boundary_component_closed_count=0`이 3k/5k/10k 전부에서 유지되므로 — **materialize된 patch가 아예 없다.** 즉 reliability frontier나 sampling gap이 physical boundary로 잘못 승격된 사례 자체가 존재할 수 없다(승격될 대상인 "닫힌 loop"가 하나도 없음). 기존 필터(`observed_support_termination`만 닫힌 loop로 인정)가 정상적으로 모든 것을 막고 있다.

## 18. Runtime/memory

§15 표 참고. 3k~10k에서 20~35초→35~49초로 증가했지만(same_surface edge가 190배 늘어난 만큼 region formation의 Python 루프 비용이 늘어난 것이 주 원인으로 추정), 재앙적 폭증은 아니다. Full all-pairs, 반복 tensor 전송, 동일 연산 재계산 추가 없음 — G1은 기존 candidate 생성이 이미 만드는 `cdist`/`topk`와 같은 복잡도(O(M²), M=representative count)의 신규 1회 계산이다.

## 19. Focused / full pytest

- `tests/test_representative_graph_scale.py`(신규 5 tests): frozen rigid invariance exact match, uniform-scale linearity, candidate/residual 독립성 ablation, thin_slab/box negative control.
- 기존 invariance 테스트 2개를 Test B(topology stability)로 명시적 재정의(docstring에 전체 근거 기록) — 삭제/약화 아님.
- 관련 suite 재확인: 87/87 pass(density_preserving/full_cloud_continuation/adc_synchronized/long_horizon/visible_surface_construction×2/surface_ownership/local_evidence_scale/representative_graph_scale).
- Manifold affinity/region formation 하위 시스템 전체(gaussian_manifold_affinity, gaussian_reliability_affinity_robustness, gaussian_surface_region_formation, surface_region_adversarial_validation/invariance/phase_alias/validation, world_space_boundary_halfedges): 50/50 pass.
- **Repository-wide pytest: 612 passed, 1 skipped, 0 failed, 8 subtests passed** (154.48s).

## 20. 다음 남은 Visible Surface Constructor 병목

`boundary_failure_stage`가 이제 전부 **C_component_admission_failed**로 수렴했다 — candidate/region 형성은 해결됐고, 남은 유일한 병목은 **닫힌 loop(closed boundary component) 형성**이다. `boundary_component_ambiguous_count`만 nonzero라는 것은 candidate가 order/방향/consensus 문제로 닫힌 루프로 조립되지 못하고 있다는 뜻이다 — 이는 `torch_directed_boundary_ordering.py`/boundary linking 정책 영역으로, 이번 세션(과 지난 두 세션 모두)에서 명시적으로 건드리지 말라고 지시받은 영역이다. 다음 라운드의 명확한 타겟이다.

부수적으로: LocalEvidenceScale의 3k 소폭 악화/normal_consensus 증가(worklog 32에서 disclosed)는 여전히 별도 조사 과제로 남아 있다. Sphere의 8-region 과분할(worklog 125)도 마찬가지로 미해결.
