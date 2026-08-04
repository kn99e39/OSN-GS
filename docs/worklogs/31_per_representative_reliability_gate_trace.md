# Worklog 31 — Visible Surface Constructor Per-Representative Reliability Gate Trace

이번 작업은 진단 전용이다. Reliability 정책/threshold/normalization/radius/region admission을 전혀 수정하지 않았고, 새 reliability state도 추가하지 않았다.

## 목표 질문에 대한 결론

> 각 representative는 정확히 어느 reliability gate에서 탈락하며, 실제 지배적인 탈락 조건은 무엇인가?

Worklog 30이 세운 프레이밍("최종 reliable이 낮아서 region이 안 만들어진다")은 **부분적으로 틀렸다.** 새로 추적한 결과, region 형성이 실패하는 진짜 지배적 조건은 reliability 결합(intrinsic+contextual) 단계가 아니라 **그보다 한 단계 앞선, representative-간 manifold affinity graph의 candidate 생성 단계**에 있다.

```text
1. intrinsic reliability: 92.9~97.9% pass (건강함, worklog 30과 동일)
2. contextual reliability: 대부분 tangent_residual 게이트 하나로 탈락 (지배적, worklog 30이 이미 지적)
3. **region-seed(core_member) 단계: 3개 스냅샷 전부 0/2048 -- 단 하나도 core seed가 되지 못했다.**
   이는 final_reliable(7~9개)이 낮아서가 아니라, representative 간 same_surface 관계
   자체가 그래프에 거의 존재하지 않기 때문이다(2027/2048 = 99%가 same_surface degree=0).
```

즉 지배적 탈락 조건은 두 가지가 중첩돼 있다: (A) contextual reliability 게이트 — `tangent_residual` 단일 게이트가 압도적 지배(intrinsic-reliable-but-not-contextual-consistent 표본의 92~97%가 이 게이트로 탈락), (B) 이보다 상류에서 이미 region-seed 자체를 막는 manifold affinity candidate 부족 — representative 간 거리가 자기 자신의 tangent scale보다 평균 12~16배 크다.

## 1. 실제 판정 코드 경로 (코드에서 직접 확인)

```text
osn_gs/core/torch_pipeline.py::_construct_canonical_with_full_evidence
  -> extract_covariance_frame(covariance)                                   [intrinsic 입력 프레임]
  -> select_density_preserving_representatives(...)                         [대표점 선택]
  -> compute_full_neighborhood_evidence(...)                                [contextual 원 evidence]
  -> evaluate_structural_reliability_from_full_evidence(rep_frame, evidence)
       -> evaluate_intrinsic_reliability(frame)                             [intrinsic class]
       -> evaluate_contextual_consistency_from_full_evidence(evidence)      [contextual class]
       -> combine_reliability(intrinsic, contextual)                        [final class]
  -> construct_visible_nurbs_from_gaussians(..., reliability=reliability)
       -> build_manifold_affinity_graph(positions, frame, reliability, ...) [representative-간 pairwise 관계]
       -> form_surface_regions(positions, frame, reliability, graph, ...)
            -> _seed_core_components(...)                                   [region-seed 판정]
            -> region growth (consensus attachment)                        [MEMBER_CONSENSUS_ATTACHED]
            -> region merge
```

핵심 확인 사항(문서 추정이 아니라 코드 직접 확인):

- `_seed_core_components`(`torch_gaussian_surface_region_formation.py:646-654`)의 core-eligible edge 조건은 **`intrinsic_class[a]==INTRINSIC_RELIABLE and intrinsic_class[b]==INTRINSIC_RELIABLE`만 확인한다 — final `reliability_class`(contextual 결합 결과)는 이 게이트에 전혀 관여하지 않는다.**
- `build_manifold_affinity_graph`의 `_classify_endpoint_status`(`torch_gaussian_manifold_affinity.py:220-231`)는 `contextual != CONTEXTUAL_CONSISTENT`일 때 `ENDPOINT_CONTEXTUAL_AMBIGUITY`를 반환하지만, 이 값은 `relation_confidence`(HIGH/MEDIUM)만 낮출 뿐 **`RELATION_REJECTED`로 이어지지 않는다** — `INTRINSIC_REJECTED` 쪽 endpoint만 relation을 강제로 reject시킨다(`torch_gaussian_manifold_affinity.py:431-438`). 즉 contextual ambiguity 자체는 same_surface 분류를 직접 막지 않는다.
- `region_id`는 `RegionFormationResult.node_region_id[i]`(representative index와 동일한 순서)로 직접 노출되고, membership state는 `node_membership_state[i]` ∈ `{core_member, consensus_attached, ambiguous_unassigned, conflict_boundary, rejected}`로 이미 공개돼 있다 — 이번 trace는 이 필드들을 그대로 읽었을 뿐, 재구현하지 않았다.

## 2. 진단 도구

`scripts/devtools/trace_representative_reliability_gates.py`(신규, production 코드 미변경) — checkpoint를 로드해 `reconstruct_visible_after_adc`와 정확히 동일한 경로로 `_construct_canonical_with_full_evidence`를 호출하고, 이미 계산된 production 결과 객체(`bundle.selection`, `bundle.evidence`, `bundle.construction.reliability/.covariance_frame/.manifold_affinity/.surface_regions`)에서만 필드를 읽는다. Reliability 계산을 재구현하지 않았다 — contextual gate의 pass/fail/margin만, `evaluate_contextual_consistency_from_full_evidence`가 실제로 사용하는 것과 동일한 `StructuralReliabilityConfig().contextual` 값으로 그대로 재현(read-only 비교)했다.

Production hot path는 전혀 건드리지 않았다 — 이 스크립트는 offline diagnostic 전용이며, 대량 CPU 전송이나 trace 수집을 production 코드에 추가하지 않았다.

각 representative에 대해 요청된 필드(identity/intrinsic/contextual evidence/contextual gate 결과/final admission)를 JSONL로 기록하고, gate waterfall과 dominant-failure histogram을 요약 JSON으로 별도 출력한다.

## 3. Gate waterfall (3k/5k/10k, cap=2048, production 그대로)

| 단계 | 3000 | 5000 | 10000 |
|---|---|---|---|
| total representatives | 2048 | 2048 | 2048 |
| intrinsic reliable | 2004 | 1958 | 1903 |
| intrinsic ambiguous | 44 | 90 | 144 |
| intrinsic rejected | 0 | 0 | 1 |
| contextual consistent | 7 | 7 | 9 |
| contextual mixed | 2018 | 2011 | 2007 |
| contextual insufficient | 23 | 30 | 32 |
| **final reliable** | **7** | **7** | **9** |
| **region-seed core** | **0** | **0** | **0** |
| region consensus-attached | 0 | 0 | 0 |
| region ambiguous-unassigned | 2048 | 2048 | 2047 |
| region rejected | 0 | 0 | 1 |
| final region member | 0 | 0 | 0 |

세 스냅샷 전부 `region-seed core = 0`. Final reliable이 7~9개나 있어도 단 하나도 region core로 승격되지 못했다 — 이는 final reliable과 region-seed 조건이 서로 다른 축(§1에서 확인)이기 때문이다.

## 4. Contextual gate별 dominant-failure 분해

"intrinsic reliable이지만 contextual consistent가 아닌" 표본(2000명 내외)에서 실패한 게이트를 전부 기록(`all_failed_gates`)하고 히스토그램을 냈다.

| gate | 3000 | 5000 | 10000 |
|---|---|---|---|
| **tangent_residual** | **1970** | **1918** | **1861** |
| normal_consensus | 284 | 314 | 472 |
| support_sufficiency | 259 | 266 | 201 |
| competing_mode_mass | 179 | 210 | 291 |
| support_present(불충분 이웃) | 23 | 23 | 25 |

`tangent_residual` 게이트(threshold=0.35)가 압도적으로 지배적이다(표본의 93~98%가 이 게이트 하나로 탈락) — worklog 30이 이미 median 값 기준으로 지적한 것과 정확히 일치하며, 이번 trace는 이를 개별 representative 단위로 재확인했다.

동시-실패 분해: 3000 기준 단일 게이트만 실패한 representative 1471명, 2개 이상 게이트 동시 실패 570명(5000: 1396/645, 10000: 1265/774) — `first_failed_gate`만 보면 다중 실패 사례(전체의 28~38%)를 놓친다는 것도 확인했다.

## 5. 새로 발견한, 더 상류의 지배적 조건 — manifold affinity graph의 candidate 부족

Region-seed core가 0인 이유를 `_seed_core_components`의 실제 입력(그래프 edge)까지 추적했다(read-only, `bundle.construction.manifold_affinity.edges`를 직접 카운트).

| iteration | kNN 후보 pair 수 | candidate 승격 | outside_candidate_support | candidate 중 same_surface | representative NN 간격 평균 | tangent_major_scale 평균 | 간격/scale 비율 |
|---|---|---|---|---|---|---|---|
| 3000 | 9,753 | 514 (5.3%) | 9,239 (94.7%) | 11 (2.1%) | 0.366 | 0.0357 | **12.25배** |
| 10000 | 9,809 | 298 (3.0%) | 9,511 (97.0%) | 6 (2.0%) | 0.424 | 0.0321 | **15.93배** |

`ManifoldAffinityConfig.scale_radius_multiplier=6.0`(representative 자신의 tangent_major_scale 기준) 이내여야 candidate로 승격되는데, 실측 representative 간 최근접 간격이 자기 tangent scale의 평균 12~16배다 — 즉 애초에 대다수 representative pair가 "candidate"조차 되지 못한다(94.7~97.0%가 `outside_candidate_support`). 그리고 candidate로 살아남은 소수(298~514개) 중에서도 `same_surface`로 분류되는 것은 2%뿐이고, 대부분(`ambiguous` 52~63%, `parallel_but_separate` 31~44%)으로 분류된다 — 이는 `_classify_relation`이 계산하는 pairwise `mutual_tangent_residual`이 대표점 개별 tangent scale로 정규화되는데, worklog 30이 full-neighborhood evidence 수준에서 지적한 것과 정확히 동일한 원인(개별 Gaussian scale이 실제 로컬 규모보다 훨씬 작음)이 pairwise 그래프 레벨에서도 그대로 재발하기 때문으로 보인다 — **단, 이는 관찰이며 이번 세션에서 원인을 더 파고들거나 수정하지 않았다.**

이 발견은 §3/§4의 reliability-gate trace(사용자가 요청한 범위)와는 별개의, 더 상류 단계에서 일어나는 현상이다 — 정직하게 별도로 분리해 보고한다.

## 6. Signed margin 샘플 (3000, contextual-mixed 대표 5건)

`trace_3000.jsonl`에서 발췌(tangent_residual 게이트 margin은 `threshold - measured`이므로 음수가 클수록 심하게 초과):

```text
rep idx 12:  tangent_residual measured=3.81  threshold=0.35  margin=-3.46  (FAIL, 단독 실패)
rep idx 87:  tangent_residual measured=6.02  threshold=0.35  margin=-5.67  normal_consensus measured=0.71 threshold=0.85 margin=-0.14 (FAIL, 2개 게이트 동시 실패)
rep idx 203: tangent_residual measured=1.14  threshold=0.35  margin=-0.79  (FAIL, 단독 실패, 상대적으로 threshold에 근접)
rep idx 340: support_present measured=1      threshold=2     margin=-1    (FAIL, 이웃 부족 — support_count<2)
rep idx 991: tangent_residual measured=9.87  threshold=0.35  margin=-9.52  competing_mode_mass measured=0.61 threshold=0.25 margin=-0.36 (FAIL, 2개 게이트 동시 실패)
```

전체 raw trace는 `trace_3000.jsonl`/`trace_5000.jsonl`/`trace_10000.jsonl`(scratchpad에 생성, 세션 종료 후 소멸 — 재생성하려면 스크립트를 다시 실행)에 2048행씩 기록돼 있다.

## 7. 이번 세션에서 하지 않은 것 (명시적 진단 전용 범위)

- Reliability threshold(0.35, 0.85, 0.25, 0.4 등) 변경 없음.
- Normalization(각 게이트의 분모, 대표점 자신의 tangent_major_scale 사용) 변경 없음.
- Local-radius(worklog 30에서 도입한 6× multiplier) 변경 없음.
- Region admission 로직(`_seed_core_components`, `core_min_same_surface_degree=2`, consensus/bridge veto) 변경 없음.
- `scale_radius_multiplier=6.0`/`footprint_overlap_multiplier=2.5`(manifold affinity candidate 생성 기준) 변경 없음.
- 새 reliability state 추가 없음.
- Production 코드는 **한 줄도 수정하지 않았다** — 이번 작업은 `scripts/devtools/trace_representative_reliability_gates.py` 신규 diagnostic 스크립트 하나만 추가했다.

## 8. 다음 방향에 대한 시사점 (실행하지 않음, 관찰만)

§5의 발견은 "reliable_count가 낮아서 region이 안 생긴다"가 아니라 "representative 간 spacing이 개별 Gaussian의 학습된 covariance 크기보다 구조적으로 훨씬 커서, contextual reliability 이전에 이미 affinity graph 자체가 same_surface 관계를 거의 만들지 못한다"는 것을 보여준다. 이는 worklog 30이 지적한 "개별 대표점 자신의 tangent_major_scale이 real 학습 데이터에서 로컬 표면 규모의 나쁜 proxy일 수 있다"는 가설(worklog 30 §15, 미해결)과 pairwise 그래프 레벨에서 정확히 같은 패턴으로 재확인된 것이다. 다음 라운드에서 이 두 층위(contextual evidence 정규화 + pairwise affinity candidate 생성 기준)를 함께, 그러나 여전히 하나씩 신중하게 조사할 근거로 남긴다 — 이번 세션에서 추측성 수정은 하지 않았다.
