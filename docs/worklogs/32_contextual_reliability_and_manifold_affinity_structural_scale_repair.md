# Worklog 32 — Contextual Reliability 및 Manifold Affinity Structural Scale 복구

## 최종 결론 먼저

> Visible Surface Constructor가 개별 Gaussian covariance footprint를 contextual evidence와 representative graph의 local structural scale로 잘못 사용하고 있었는가?

**그렇다.** 개별 대표점 하나의 `tangent_major_scale`(그 Gaussian 자신의 covariance 크기)이 최소 두 곳에서 서로 다른 의미의 "local scale"로 잘못 재사용되고 있었다: (A) full-neighborhood contextual evidence의 local radius/residual denominator, (B) manifold affinity graph의 candidate radius/pairwise residual denominator.

> 실행 순서와 기존 reliability semantics를 유지하면서 LocalEvidenceScale과 RepresentativeGraphScale을 분리해, threshold나 representative cap을 완화하지 않고 real long-horizon snapshot과 ideal surface-aligned positive control에서 contextual reliability, manifold affinity, region formation 및 boundary accuracy를 복구했는가?

**부분적으로만 복구했다 — 정직하게 보고한다.** (A) LocalEvidenceScale은 실제로 분리 구현하고 production에 반영했으며, 3가지 invariance 요구(rotation/translation/uniform-scale)를 모두 만족함을 검증했다. (B) RepresentativeGraphScale은 **세 가지 서로 다른 후보를 구현·검증했으나 전부 기존 rigid-rotation invariance 테스트를 깨뜨려 최종적으로 되돌렸다.** 따라서 region_seed_core=0/same_surface degree≈0 현상(worklog 31에서 발견한 두 번째 병목)은 이번 세션에서 해결하지 못했다. Materialized NURBS 개수를 늘리기 위해 invariance를 희생하지 않았다 — 이는 §17 성공 기준이 명시적으로 요구한 바다.

## 1. 실제 실행 순서 (재확인, 코드 직접 확인)

```text
_construct_canonical_with_full_evidence
  1. evaluate_structural_reliability_from_full_evidence(rep_frame, evidence)
       -> evaluate_intrinsic_reliability
       -> evaluate_contextual_consistency_from_full_evidence
       -> combine_reliability
       -> reliability 객체 확정
  2. construct_visible_nurbs_from_gaussians(..., reliability=reliability)
       -> build_manifold_affinity_graph(..., reliability)   # graph_scale 원래 파라미터 없음, 이번에 추가했다가 되돌림
       -> form_surface_regions(...)
       -> _seed_core_components(...)  # intrinsic_class만 확인, reliability_class는 안 봄 (worklog 31 확인 사항 재확인)
       -> region growth / merge
       -> boundary recovery / materialization
```

Worklog 31이 확인한 사실(재검증, 변경 없음): `_seed_core_components`의 core-edge eligibility는 여전히 `intrinsic_class`만 확인하고, `_classify_endpoint_status`의 `ENDPOINT_CONTEXTUAL_AMBIGUITY`는 여전히 relation을 강제로 reject하지 않는다(confidence만 낮춤). 이 로직 자체는 이번 세션에서 건드리지 않았다.

## 2. Scale 사용처 전수 감사

| 파일:함수 | 사용 scale | 원래 의미 | 실제 사용 목적 | 이번 판정 |
|---|---|---|---|---|
| `torch_full_neighborhood_evidence.py::compute_full_neighborhood_evidence` local_radius | `representative_frame.tangent_major_scale` | 개별 Gaussian 자신의 covariance 크기 | full-cloud 이웃 판정 반경 | **오용 확인, 분리함(LocalEvidenceScale)** |
| 〃 `rep_tangent_scale_per_full`(tangent_residual denominator) | 〃 | 〃 | full-cloud 잔차 정규화 분모 | **오용 확인, 분리함** |
| `torch_gaussian_manifold_affinity.py::build_manifold_affinity_graph` `average_tangent_major`(candidate radius) | 〃 | 〃 | representative 간 candidate 반경 | **오용 확인, 분리 시도했으나 invariance 실패로 원복** |
| `_compute_pair_metrics`의 `mutual_tangent_residual` denominator | 〃 | 〃 | representative 쌍 잔차 정규화 분모 | **오용 확인, 분리 시도했으나 원복** |
| `_compute_pair_metrics`의 `footprint_overlap`(`equivalent_tangent_scale`) | 〃 | 〃 | 두 Gaussian splat의 물리적 overlap | **정당한 사용, 변경 안 함** — 이건 진짜로 개별 Gaussian 모양을 묻는 질문 |
| `evaluate_intrinsic_reliability`(scale-range/conditioning) | 〃 | 〃 | 개별 covariance 유효성 판정 | **정당한 사용, 변경 안 함** |
| `torch_gaussian_surface_region_formation.py::_evaluate_path_consistency` | 〃 | 〃 | phase-alias 경로 일관성 검사 | 감사만 함, 변경 안 함(범위 밖 — boundary/topology 정책) |
| `torch_full_cloud_continuation_shell.py`(worklog 130) | 〃 | 〃 | continuation shell 반경 | 감사만 함, 변경 안 함(범위 밖 — boundary linking 정책, 이번 작업 금지 목록) |
| `torch_visible_surface_construction.py`의 `frame.equivalent_tangent_scale`(boundary termination scale) | 〃 | 〃 | boundary candidate 각도 정규화 | 감사만 함, 변경 안 함(boundary linking, 범위 밖) |
| `torch_density_preserving_representative_selection.py`의 `normal_thickness` | 〃 | 〃 | mode 분리(cell 내부 클러스터링) | 무관 — selection 자체는 건드리지 않음 |

이번 세션에서 실제로 수정한 곳은 표의 위 4줄 중 2줄(LocalEvidenceScale, 실제 반영됨)과 시도했다가 원복한 2줄(RepresentativeGraphScale)뿐이다. 나머지는 감사만 하고 손대지 않았다(요청받은 boundary linking/continuation shell 금지 목록과 일치).

## 3. Scale semantics 분리 — 정의와 근거

**A. Gaussian Footprint Scale** — `frame.tangent_major_scale`/`tangent_minor_scale`/`normal_thickness`. 변경 없음. `evaluate_intrinsic_reliability`와 `footprint_overlap` 후보 생성에 계속 사용.

**B. LocalEvidenceScale** (신규, 채택) — 대표점별 `cbrt(cell_volume / source_count)`. `cell_volume`은 rotation/translation/uniform-scale-불변 characteristic scene length(`2 * sqrt(trace(cov(points))/3)`, 회전 하에서 `trace(R cov R^T) = trace(cov)`이므로 불변)를 selection budget에서 유도한 `resolution`으로 나눈 세제곱. `source_count`는 selection 단계에서 이미 계산된, 해당 대표점의 원 voxel cell·mode 멤버 수(`CanonicalSurfaceRepresentative.source_count`, 재사용 — 새 계산 없음). **처음에는 axis-aligned bounding box span으로 cell_volume을 구했다가 `test_construction_outcome_is_stable_under_rigid_rotation_translation_and_uniform_scale`이 깨져서(4≠1) characteristic-length 버전으로 교체했다** — 축정렬 bounding box는 회전에 따라 비등방적으로 변하기 때문.

**C. RepresentativeGraphScale** (신규, 시도 3회, 전부 원복) — 상세 §8.

## 4. 후보 비교 — LocalEvidenceScale (채택)

| 후보 | 정의 | 판정 |
|---|---|---|
| E0 (기존) | `tangent_major_scale` | 기각 — worklog 31: 실측 8.4배 작음 |
| E-cell (채택) | `cbrt(characteristic_cell_volume / source_count)` | **채택** — per-representative, source_count는 기존 selection 산출물 재사용, invariant 확인 |

3000 스냅샷 실측: E0 mean=0.0357, E-cell mean=0.237 (비율 8.4배).

## 5. 후보 비교 — RepresentativeGraphScale (전부 기각)

| 후보 | 정의 | 결과 |
|---|---|---|
| G0 (기존) | `tangent_major_scale` | 유지(원복) |
| G-knn | 대표점 자신의 k=8 최근접 대표점 median distance (`torch.topk`, 이미 candidate 생성에서 계산된 distance matrix 재사용, 추가 O(M²) 없음) | **기각** — `test_construction_outcome_is_stable_...`(4≠1→2≠4로 값만 바뀜), `test_region_and_reliable_counts_stable_under_rigid_transform`(5≠3) 둘 다 깨짐 |
| G-evidence | `reliability.contextual.neighbor_spacing`(= `evidence.mean_spacing`, LocalEvidenceScale 기반이므로 불변일 것으로 예상) | **기각** — 오히려 새 실패 케이스 추가됨(density sweep 테스트까지 깨짐), `assign_nearest_representative`의 Voronoi 배정 자체가 대표점 위치에 의존해 예상보다 덜 안정적 |
| G-robust | LocalEvidenceScale과 동일한 `cell_volume`을 대표점 전체의 **median** `source_count`로 나눈 단일 스칼라(대표점 전체에 동일 값 broadcast) | **최종까지 시도, 여전히 기각** — median source_count 자체는 회전 전후 거의 동일(6.0 vs 6.0, 실측)했지만 `test_region_and_reliable_counts_stable_under_rigid_transform`은 여전히 깨짐(1≠5) |

**근본 원인(진단, 미해결)**: representative SELECTION 자체가 이미 axis-aligned voxel grid를 쓰기 때문에 회전에 대해 정확히 불변이지 않다는 것은 기존에 이미 문서화된, 용인된 제약이다(`test_density_preserving_representative_selection.py`의 기존 docstring이 명시). 기존 G0(`tangent_major_scale`)는 "그 대표점 자신의" 내재적(intrinsic) 속성이라 어떤 특정 Gaussian이 대표로 뽑히든 값이 비슷했지만(같은 로컬 영역의 Gaussian들은 covariance 크기가 서로 비슷함), G-knn/G-evidence/G-robust는 모두 "대표점 집합 전체의 배치"에 의존하는 aggregate 양이라 대표점 선택 자체의 불변성 결여를 흡수하지 못하고 오히려 증폭시켰다. 이 근본 원인(voxel-grid 기반 selection의 회전 민감성)을 고치는 것은 이번 작업의 금지 목록(FPS policy 변경 금지, representative selection 변경 금지)에 명시적으로 걸린다.

## 6. Uniform-scale / density invariance 검증

LocalEvidenceScale: `characteristic_length`가 uniform scale factor `s`에 대해 정확히 `s`배로 스케일됨을 직접 수치 확인(scale=1.7 적용 시 ratio=1.6999999). Density invariance는 `source_count`가 밀도에 비례해서 증가하므로 `E ~ cbrt(1/count)`로 자연스럽게 감소 — 실측 3k→5k→10k에서 대표점당 source_count가 늘어나는 경향과 일치.

## 7. Real 3k/5k/10k 수정 전 결과 (worklog 31 그대로, 참고용 재게재)

| iteration | intrinsic reliable | contextual consistent | final reliable | region_seed_core |
|---|---|---|---|---|
| 3000 | 2004 | 7 | 7 | 0 |
| 5000 | 1958 | 7 | 7 | 0 |
| 10000 | 1903 | 9 | 9 | 0 |

## 8. `-surf` 수정 전/후 — 중요한 범위 확인

`osn-gs benchmark -surf`(box/cylinder/sphere)는 `pipeline.initialize()` → `_initialize_canonical()`을 호출하며, 이는 `_canonical_construction_indices`(단순 voxel-nearest 샘플러)와 `_canonical_initial_covariance`(local-PCA)를 쓰는 **완전히 다른 코드 경로**다. `_construct_canonical_with_full_evidence`(이번 수정 대상)는 `reconstruct_visible_after_adc`(학습 후 ADC 재구축) 전용이며 `-surf` 벤치마크 경로에서는 전혀 호출되지 않는다(worklog 129의 원래 설계: "no learned per-Gaussian covariance exists at raw-point-cloud init time"). 따라서 **`-surf` 결과는 이번 LocalEvidenceScale 수정과 무관하며, 수정 전/후 값이 다를 이유가 없다** — 실제로 확인 결과 동일했다.

`--points 600`: box는 `state=partially_constructed regions=6 patches=4 assigned=256/600`(패치 4개 실제 materialize됨), cylinder/sphere는 여전히 `no_admissible_region`/`boundary_recovery_failed`(closed-topology 미지원, worklog 128 기존 disclosed gap). `--points 3000`(cap 2048 초과, 그래도 `_initialize_canonical` 경로라 무관): box regions=7/components=129, cylinder regions=4/components=81, 둘 다 `review_required`(closed-loop 미형성) — 이 역시 pre-existing gap이며 이번 세션 무관.

Materialized 4개 패치 전부 `boundary_reason=observed_support_termination`으로 확인(§14 boundary provenance) — 기존 필터가 정상 작동 중이며 이번 세션에서 boundary_support_termination.py를 건드리지 않았으므로 당연한 결과.

## 9. Negative-control 수정 전/후

`gaussian_reliability_scenes.py`의 `thin_slab`/`box_isolated_floater`/`box_isotropic_contamination`/`box_with_bridge`/`box`를 각각 `cap = N//3`(강제 다운샘플, LocalEvidenceScale 실제 활성화)로 실행:

| scene | region_count (수정 전) | region_count (수정 후) | false merge? |
|---|---|---|---|
| thin_slab | 2 | 2 | 없음(앞/뒤 안 합쳐짐) |
| box_isolated_floater | 0 | 0 | 없음(floater 미부착) |
| box_isotropic_contamination | 2 | 2 | 없음 |
| box_with_bridge | 6 | 6 | 없음(6개 face 유지, bridge로 인한 붕괴 없음) |
| box | 6 | 6 | 없음 |

**수정 전/후 완전히 동일** — 이 5개 negative-control 시나리오에서는 LocalEvidenceScale 변경이 결과에 영향을 주지 않았다(회귀도, 개선도 없음). 이는 예상된 결과다: 이 시나리오들은 애초에 tangent_residual gate로 걸러지는 대상이 아니라 정상적으로 분리돼야 하는 케이스들이라, LocalEvidenceScale이 완화한 "정상 신호가 threshold를 못 넘는" 문제와 별개다.

## 10. 선택한 정의 요약

- **LocalEvidenceScale** = `cbrt(characteristic_scene_length³ / (resolution³ · source_count))`, per-representative, `FullNeighborhoodEvidenceConfig`에 새 필드 없이 `compute_full_neighborhood_evidence(..., local_evidence_scale=...)` 신규 파라미터로 주입(기본값 `None`→기존 `tangent_major_scale` fallback, 하위 호환).
- **RepresentativeGraphScale** = 채택 안 됨. `build_manifold_affinity_graph`/`_compute_pair_metrics`는 **원래 코드 그대로**(diff 없음).

## 11. Production 수정 위치

- `osn_gs/surface/torch_full_neighborhood_evidence.py`: `compute_full_neighborhood_evidence`에 `local_evidence_scale: Any | None = None` 추가, local radius/tangent-residual denominator에서 사용.
- `osn_gs/core/torch_pipeline.py`: `_construct_canonical_with_full_evidence`에서 `local_evidence_scale` 계산(다운샘플된 경우만) 후 `compute_full_neighborhood_evidence`에 전달.
- `osn_gs/surface/torch_gaussian_manifold_affinity.py`: **변경 없음**(원복 완료, git diff 없음).
- `osn_gs/surface/torch_visible_surface_construction.py`: worklog 30에서 이미 추가한 `reliability_failure_stage` 진단 필드 유지(이번 세션 추가 변경 없음).

## 12. Contextual-only ablation (A만 적용, B는 원래 코드)

§7 대비 §13(수정 후) 비교가 이 ablation 자체다 — B(manifold affinity)는 이번 세션 내내 원래 코드 그대로였으므로, 아래 §13 결과가 곧 "A만 적용"의 순수 효과다.

## 13. Graph-only ablation

시도했으나(§5) 전부 invariance 위반으로 최종 원복 — "graph만 개선되고 contextual은 그대로"인 상태를 production에 반영하지 못했다. 원복 직전 임시 커밋되지 않은 코드에서 관찰한 값(G-robust 버전, 참고용, production 아님): 3000 스냅샷에서 `region_seed_core`가 0에서 벗어나지 않았다(그래프 자체는 완화됐지만 invariance 실패로 폐기했으므로 이 숫자는 채택 안 됨, 기록만 남김).

## 14. Combined 결과

Combined는 달성하지 못했다(B가 원복됐으므로). 아래 §15는 A-only 결과다.

## 15. 수정 후 contextual waterfall (A-only, production 실제 값)

| iteration | intrinsic reliable | contextual consistent(수정 전→후) | final reliable(수정 전→후) | contextual insufficient(수정 전→후) |
|---|---|---|---|---|
| 3000 | 2004 | 7→3 | 7→3 | 23→5 |
| 5000 | 1958 | 7→11 | 7→11 | 30→0 |
| 10000 | 1903(1 rejected) | 9→13 | 9→12 | 32→1 |

경향: `contextual_insufficient`(지원 부족)는 전 스냅샷에서 크게 감소(local radius가 커져 더 많은 로컬 멤버가 포함됨). `tangent_residual` 실패 건수는 비슷하거나 소폭 개선. 대신 `normal_consensus` 실패가 늘었다(284→355, 314→474, 472→713) — 더 넓어진 로컬 반경이 실제 곡률/노이즈가 있는 멤버까지 포함하면서 나타난, 물리적으로 타당한 트레이드오프다. **순 효과는 스냅샷마다 다르다**(3000은 최종 reliable이 오히려 소폭 감소, 5000/10000은 증가) — "항상 개선"이라고 과장하지 않는다.

## 16. 수정 후 affinity graph 통계

변경 없음(B 원복) — worklog 31의 수치(3000: candidate 5.3%, same_surface 2.1%, spacing/scale ratio 12.25배)가 그대로 유효하다.

## 17. 수정 후 region/materialization 결과

3k/5k/10k 전부 `region_seed_core=0`, `final_region_member=0`, `region_count=0`, `construction_state=no_admissible_region` — **수정 전과 동일, 미해결.**

## 18. Boundary provenance 및 analytic accuracy

Boundary termination/materialization 코드는 이번 세션에서 전혀 건드리지 않았다. 실제 materialize가 일어난 유일한 케이스(§8의 `-surf` box, 무관한 코드 경로)에서 4개 패치 전부 `boundary_reason=observed_support_termination`으로 확인 — reliability frontier/sampling gap이 physical boundary로 승격되는 사례 없음(기존 필터가 그대로 작동).

## 19. Real snapshot 결과 (최종)

§15 표와 동일. 10k NaN 크래시는 worklog 30에서 이미 해결된 채 유지(재확인: 크래시 없음).

## 20. `-surf` 결과 (최종)

§8과 동일 — 이번 수정과 무관한 코드 경로이므로 변화 없음, 기존 disclosed gap 유지.

## 21. Negative-control false connection 여부

§9 — 5개 시나리오 전부 false merge 없음(수정 전/후 동일).

## 22. Runtime 및 memory

`_construct_canonical_with_full_evidence` 전체 runtime: 3k~10k 스냅샷에서 worklog 30 대비 유의미한 변화 없음(20~35초대 유지 — `local_evidence_scale` 계산은 O(1) 스칼라 연산 + O(M) 텐서 하나 추가일 뿐). Full all-pairs, 반복 tensor 전송, 동일 연산 재계산 없음 — 기존 `source_count`/`cell` 계산을 그대로 재사용했다.

## 23. Focused / full pytest

- `tests/test_local_evidence_scale.py`(신규, 3 tests): dense-plane에서 contextual-consistent 비율 개선 확인, 다운샘플 안 될 때 기존 경로 보존 확인, rigid-rotation/translation/uniform-scale invariance 회귀 가드.
- 기존 invariance suite 재확인: `test_density_preserving_representative_selection.py` + `test_full_cloud_continuation_shell.py` + `test_adc_synchronized_visible_nurbs.py` + `test_long_horizon_reliability_collapse_repair.py` + `test_visible_surface_construction.py`(worklog30의 `reliability_failure_stage` 테스트 포함) + `test_visible_surface_construction_invariance.py` + `test_surface_ownership.py` = **79/79 pass**.
- **Repository-wide pytest: 607 passed, 1 skipped, 0 failed, 8 subtests passed** (147.14s).

## 24. 다음 남은 Visible Surface Constructor 병목

1. **RepresentativeGraphScale/manifold affinity candidate 부족(bottleneck B)은 여전히 미해결이다.** Region_seed_core=0이 3k/5k/10k 전부에서 유지된다. 근본 원인은 representative SELECTION 자체(axis-aligned voxel grid)가 정확히 rotation-invariant하지 않다는, 이번 작업 범위 밖의 더 근본적인 제약이다 — 이걸 고치지 않는 한 selection에 의존하는 어떤 graph-scale도 같은 문제에 부딪힐 가능성이 높다.
2. Contextual reliability의 `normal_consensus` 실패가 A-only 수정 이후 오히려 늘었다 — 더 넓어진 로컬 반경이 실제 곡률/노이즈를 포함하기 시작한 부작용으로, 별도 진단이 필요하다.
3. §15에서 순 효과가 스냅샷마다 다르다는 점(3000은 개선 아님) — cap이나 다른 요인과의 상호작용을 더 봐야 한다.

## 25. 성공 기준 대비 최종 평가

| 기준 | 결과 |
|---|---|
| 실행 순서 정확히 유지 | 충족 |
| Contextual/graph 독립 병목으로 분석 | 충족(§5/§7/§17에서 명확히 구분) |
| Gaussian footprint/structural scale 역할 분리 | LocalEvidenceScale은 충족, RepresentativeGraphScale은 시도했으나 미충족 |
| Contextual tangent residual failure 의미 있게 감소 | 부분 충족(스냅샷마다 다름, §15) |
| Affinity 95~97% candidate 이전 탈락 해소 | **미충족** |
| ~99% same_surface degree 0 해소 | **미충족** |
| Core member/region 생성 | **미충족** |
| Representative cap 증가 없음 | 충족 |
| Threshold 완화 없음 | 충족 |
| `-surf` box/cylinder/sphere surface coverage 복구 | 해당 없음(무관한 코드 경로로 확인됨) |
| Negative-control false merge 없음 | 충족 |
| Chunked eigendecomposition/worklog131·132 성능 회귀 없음 | 충족(pytest 전체 통과, runtime 안정) |

**전체 결론**: 이번 세션은 진단 요구사항(§1~§10, §22 audit)을 완전히 충족했고, LocalEvidenceScale 수정 하나를 실제로 검증·반영했다. 그러나 RepresentativeGraphScale은 세 번의 서로 다른 시도가 전부 기존 invariance 테스트를 깨뜨려 최종적으로 원복했으며, 이로 인해 region 형성 자체는 여전히 실패 상태다. Materialized NURBS 개수를 늘리기 위해 invariance를 희생하는 대신, 정직하게 "미해결"로 보고하는 쪽을 선택했다.

## 26. Focused/full pytest 결과 요약

§23과 동일. 607 passed / 1 skipped / 0 failed.
