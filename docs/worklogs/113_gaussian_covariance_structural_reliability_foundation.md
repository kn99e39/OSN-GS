# Worklog 113 — Covariance-guided structural reliability/manifold-affinity foundation

## 상태

진행 중(이번 라운드 범위 내에서 완료). 이번 작업은 isolated Boundary-first NURBS construction의 방향을 canonical architecture로 갱신하는 첫 단계다. 기존 Boundary-first support/control/NURBS foundation(Worklog 110·112)은 폐기하지 않았고, 그 앞단의 boundary evidence extraction(현재 KDE/raster/marching-squares 기반)을 장기적으로 대체할 covariance-guided structural reliability + pairwise manifold-affinity foundation을 새로 추가했다.

사용자 지시대로 사용자가 지정한 "Worklog 111"은 이미 concurrent 작업(`111_nurbs_construction_synthetic_3d_gaussian_dataset.md`)이 선점했으므로, 다음 사용 가능한 번호(113 — 112도 지난 라운드에서 이미 사용)로 기록한다. Worklog 110은 이번 라운드에서 수정하지 않았다(중복 기록 금지 지시 준수 — 새 내용은 전부 이 문서에만 기록).

## 1. 유지한 기존 Boundary-first foundation

다음은 이번 라운드에서 전혀 수정하지 않았다.

- Common Boundary-first construction contract(`outer_boundary + interior_boundary`/`outer_boundary + observed interior_anchor` 역할 계약, `torch_boundary_first_visible_builder.py`).
- Boundary/support provenance(observed outer/inner boundary, interior anchor, support curve provenance, ordering, orientation, nesting, correction history).
- Worklog 110의 review geometry semantics(`observed_evidence_points`/`resampled_observed_evidence`/`control_polygon`/`correspondence_chord`/`evaluated_curve` 5종 representation kind, `torch_boundary_review_geometry.py`).
- 실제 `surface.evaluate()` 기반 검증(`evaluate_iso_edge`/`evaluate_interior_iso_curve`).
- Worklog 110/112의 representative support-crossing 진단(`detect_support_curve_crossings`, `scope`/`not_checked_categories` 명시).
- Multi-hole `review_required`/`partition_materialization_required` 안전 경계(`torch_boundary_multi_loop.py`, `torch_boundary_planar_partition.py`).
- Silent fallback 금지 전체 목록(PCA rectangle, box-grid, trimmed fallback, synthetic inner loop/center, 강제 폐곡선화 등).

## 2. 교체 대상으로 지정한 장기 가정 (이번 라운드에서는 아직 실제 대체하지 않음)

다음 flow는 이번 라운드에서 실제로 걷어내지 않았다 — canonical 장기 경로가 아니라는 지위만 이 문서에서 명시했다.

```text
Gaussian component -> single local PCA UV -> KDE/raster support mask -> marching-squares contour -> boundary truth
```

이 경로(`torch_component_boundary.py` 등 기존 isolated 파일)는 이번 라운드에서 코드 변경 없이 그대로 유지된다. §9에 새 지위를 명시했지만, 실제 `extraction_mode`/`evidence_level`/`canonical_boundary_source` 메타데이터를 그 파일들에 삽입하는 작업은 **아직 하지 않았다** — 이번 라운드는 지위 선언과 대체 경로의 독립적 foundation만 다룬다.

## 3. 새 covariance frame contract (`osn_gs/surface/torch_gaussian_covariance_frame.py`)

- `extract_covariance_frame(covariance)`: `(N,3,3)` 대칭 covariance를 eigh로 분해하고 내림차순(`lambda1>=lambda2>=lambda3`)으로 정렬한다. `eigenvector(lambda3)`를 normal candidate, 나머지 둘을 tangent candidate로 사용한다.
- Shape 분류: `planar_surfel`(lambda2/lambda3 분리 큼, lambda1/lambda2 분리 작음), `needle_like`(lambda1/lambda2 분리 큼, lambda2/lambda3 분리 작음), `isotropic`(lambda3/lambda1이 threshold 이상), `ambiguous_shape`(나머지). Threshold는 configurable(`planarity_threshold=3.0`, `elongation_threshold=3.0`, `isotropy_threshold=0.6`)이며 canonical 최종값이 아니다.
- `orientation_insensitive_alignment(a, b) = abs(dot(a, b))` — eigenvector sign ambiguity를 반영해 모든 pairwise 비교에 사용한다. Eigenvector 부호를 physical outward normal로 간주하지 않는다.
- `covariance_from_scale_rotation(scale, quaternion)`: 3DGS 스타일 `R diag(s^2) R^T` 구성 — production `osn_gs/gaussian/torch_model.py`를 import하지 않고 격리된 테스트/fixture 편의를 위해 자체 구현했다.

## 4. Structural reliability contract (`osn_gs/surface/torch_gaussian_structural_reliability.py`)

Observed Gaussian을 `reliable_structural_evidence` / `ambiguous_structural_evidence` / `rejected_structural_evidence` 3범주로 분류한다. Renderer/trainer 의존성 없이 `positions`와 이미 추출된 covariance frame만 사용한다.

측정한 5개 독립 축(전부 provenance에 원인별로 보존, 단일 scalar로 압축하지 않음):

- **planarity_score**(§5.1): `SHAPE_PLANAR`가 아니면 0. Planar인 경우 lambda2/lambda3 분리 정도를 0~1로 정규화.
- **neighbor_normal_agreement**(§5.2): k-최근접 이웃과의 orientation-insensitive normal alignment 평균.
- **mutual_tangent_residual**(§5.3): 양방향(자신의 normal 기준, 이웃의 normal 기준) tangent-plane 이탈을 각자의 local scale로 정규화한 뒤 최댓값을 취해 평균 — 한쪽 방향만 쓰지 말라는 지시를 반영했다.
- **scale_consistency**(§5.4): 자신의 local scale(최대 eigenvalue의 sqrt)과 이웃 median의 log-ratio로 outlier(과대/과소 Gaussian)를 검출.
- **local_support_score**(§5.5): 정규화된 이웃 거리에 대한 구간선형 ramp(가까우면 1, 멀면 0) — 최초 구현은 지수감쇠였으나 깨끗한 평면(격자 간격 대비 surfel 반지름 비율)에서도 낮은 점수가 나오는 calibration 결함을 발견해 ramp 방식으로 교정했다(§7 상세).

분류 규칙: 5개 축 모두 threshold 통과 시 `reliable`; 5개 중 하나라도 hard-reject 조건(예: planarity 극히 낮음, mutual residual 극히 큼, local support 극히 낮음, scale 극히 불일치)이면 `rejected`; 그 외에는 `ambiguous`. `reasons` 필드에 실패한 개별 axis 이름을 전부 남긴다.

## 5. Pairwise manifold-affinity contract (`osn_gs/surface/torch_gaussian_manifold_affinity.py`)

- Candidate edge는 k-최근접 공간 이웃에서만 생성한다(순수 거리만으로 edge를 만들지 않음 — 아래 분류 기준을 항상 통과해야 한다).
- 각 candidate pair에 대해 **그 pair 전용으로 새로 계산한** normal alignment / 양방향 mutual tangent residual / scale-normalized distance를 사용한다 — §4의 "이웃 전체 평균" reliability 점수를 재사용하지 않는다. 이는 실제로 중요한 설계 결정이었다(§7 참고): 이웃-평균 reliability는 실제 crease 지점에서 오히려 낮아지는데(반대쪽 표면의 상충하는 normal이 평균을 오염시키므로), pair별 신선한 계산이라야 그 지점에서도 crease를 올바르게 검출할 수 있었다.
- Edge state 6종: `same_surface`, `crease_or_orientation_discontinuity`, `parallel_but_separate`, `proximity_only`, `ambiguous`, `rejected`. 하나의 boolean으로 압축하지 않았다.
- 두 endpoint 중 하나라도 `rejected` reliability면 edge는 무조건 `rejected`("unreliable_endpoint"). 그 외에는 pair별 fresh 지표로 위 6종 중 하나를 결정한다.

## 6. Edge state 결정 기준과 reason (요약)

| 우선순위 | 조건 | state |
| --- | --- | --- |
| 1 | 정규화 거리가 candidate 상한 초과 | `proximity_only` |
| 2 | 두 endpoint 중 하나라도 `rejected` | `rejected` ("unreliable_endpoint") |
| 3 | normal alignment 높음 + mutual residual 낮음 | `same_surface` |
| 4 | normal alignment 높음 + mutual residual 중간~높음(같은 방향인데 tangent plane이 벌어짐) | `parallel_but_separate` |
| 5 | normal alignment 낮음 + 거리 가까움 | `crease_or_orientation_discontinuity` |
| 6 | 그 외 | `ambiguous` |

각 edge는 `confidence`(`high`/`medium`, 두 endpoint의 reliability 조합에서 파생)도 별도로 기록하지만, **crease 분류 자체는 confidence로 게이팅하지 않는다** — §7에 그 이유를 기록했다.

## 7. Synthetic/benchmark scene 결과 (`nurbs_constructor_benchmark/gaussian_reliability_scenes.py`)

7개 scene(plane, two_perpendicular_surfaces, close_parallel_sheets, smooth_curved_sheet, isolated_floater, isotropic_blob, oversized_bridge)을 새로 만들었다. 각 scene은 raw point cloud가 아니라 `(positions, covariances)` 쌍을 직접 반환한다.

측정 결과:

- **plane**: 81개 Gaussian 전부 `reliable`, same_surface edge만 존재(경계 셀 4개는 `proximity_only`).
- **smooth_curved_sheet**: 81개 전부 `reliable`, same_surface edge만 존재 — 완만한 곡률에서도 인접 표본은 연결된다.
- **two_perpendicular_surfaces**: floor/wall 내부는 각각 완전히 분리된 same_surface 영역이며, floor-wall 교차 edge는 전부 `crease_or_orientation_discontinuity`/`ambiguous`/`rejected`뿐이고 **same_surface는 전혀 없다.**
- **close_parallel_sheets**: lower/upper 내부는 각각 same_surface로 연결되지만 lower-upper 교차 edge는 `rejected`/`parallel_but_separate`/`ambiguous`뿐이며 **same_surface가 전혀 없다.**
- **isolated_floater**: 평면 79개는 전부 reliable, floater 1개는 `rejected`(mutual_tangent_residual/local_support 동시 위반), floater와의 모든 edge는 `proximity_only`(candidate 거리 상한 초과).
- **isotropic_blob**: 평면에 심은 isotropic Gaussian 2개는 `rejected`, 그 주변 4개는 정상보다 낮은 `ambiguous`(이웃 normal agreement가 isotropic 이웃 때문에 저하 — 실제로 정보가 불확실하다는 것을 반영하는 정당한 결과), 나머지 75개는 `reliable`.
- **oversized_bridge**: floor-wall gap에 심은 초대형 등방 Gaussian은 `rejected`이며, floor/wall 어느 쪽과도 same_surface edge를 만들지 않는다 — **두 표면을 bridge하지 못한다.**

## 8. Unreliable Gaussian이 surface graph에 미치는 영향

- `close parallel sheets`/`isotropic_blob` 검증 중 실제 calibration 결함을 발견하고 고쳤다: 초기 `local_support_score` 공식(지수감쇠, 분모 3.0)은 깨끗한 평면조차 threshold(0.4) 미만으로 떨어뜨려 **모든 Gaussian이 `ambiguous`가 되는 버그**를 만들었다. 구간선형 ramp(`close_ratio=1.0`, `far_ratio=10.0`)로 교체해 해결했다 — 이번 라운드에서 실제로 잡은 버그다.
- `two_perpendicular_surfaces`의 실제 crease 지점 Gaussian들은 이웃-평균 기준 reliability가 `ambiguous`로 나온다(반대쪽 표면 이웃이 평균을 오염시키므로) — 이는 §5 설계 의도와 일치하는 결과이지만, 만약 crease edge 분류를 "양쪽 endpoint가 reliable이어야 함"으로 게이팅했다면 실제 crease가 전혀 검출되지 않았을 것이다(처음 구현에서 실제로 이 문제가 발생해 confidence 게이팅을 제거했다). 최종적으로 crease 분류는 endpoint가 `rejected`가 아니기만 하면 지리적 기준만으로 판정한다.
- `test_unreliable_gaussian_removal_does_not_needlessly_fragment_a_reliable_region`(BFS 연결성 검사)로, 고립된 floater 1개가 존재해도 평면의 나머지 79개 reliable Gaussian이 여전히 하나의 same_surface 연결 영역을 이룸을 확인했다.

## 9. Raster/KDE path의 새 지위

이번 라운드는 지위만 이 문서에 선언한다(§2 참고, 실제 코드 메타데이터 삽입은 미착수):

- 기존 경로: `extraction_mode=raster_assisted_legacy`, `evidence_level=secondary`, `canonical_boundary_source=false`.
- 새 경로: `extraction_mode=covariance_guided_manifold`, `evidence_level=primary_candidate`, `canonical_boundary_source=pending_gate`.

사용자 승인 전까지 실제 canonical/default로 승격하지 않는다. 새 모듈은 기존 default constructor(`nurbs_constructor_benchmark/boundary_first.py`)나 isolated builder(`torch_boundary_first_visible_builder.py`)에 연결하지 않았다(§11 지시 — experimental adapter interface 자체도 이번 라운드에서는 설계하지 않았다. §18 참고).

## 10. 변경 파일

신규(전부 isolated, 기존 파일 미수정):

- `osn_gs/surface/torch_gaussian_covariance_frame.py`
- `osn_gs/surface/torch_gaussian_structural_reliability.py`
- `osn_gs/surface/torch_gaussian_manifold_affinity.py`
- `nurbs_constructor_benchmark/gaussian_reliability_scenes.py`
- `tests/test_gaussian_covariance_frame.py`
- `tests/test_gaussian_structural_reliability.py`
- `tests/test_gaussian_manifold_affinity.py`

기존 Boundary-first isolated 파일(`torch_boundary_*`, `boundary_first_support_runner.py` 등)은 이번 라운드에서 전혀 수정하지 않았다.

## 11. Targeted tests

```text
tests/test_gaussian_covariance_frame.py        5 passed
tests/test_gaussian_structural_reliability.py  6 passed
tests/test_gaussian_manifold_affinity.py       12 passed
(기존 Boundary-first isolated suite, 무수정 회귀 확인)
tests/test_patch_boundary.py
tests/test_boundary_first_visible_builder.py
tests/test_boundary_first_support_runner.py
tests/test_boundary_review_geometry.py
tests/test_boundary_support_network.py
tests/test_boundary_constrained_surface.py
tests/test_boundary_central_cap.py
tests/test_boundary_surface_quality.py
tests/test_boundary_first_support_pipeline.py
tests/test_boundary_multi_loop.py
tests/test_boundary_planar_partition.py
tests/test_boundary_source_fidelity.py
tests/test_component_boundary.py
합계 85 passed (신규 23 + 기존 62)
```

## 12. Full pytest

```text
532 passed, 2 failed, 1 skipped, 1 warning, 8 subtests passed
```

(직전 라운드 대비 +23, 전부 신규 테스트. 회귀 없음.)

## 13. 기존 failures와 attribution

기존 실패 2건은 이번 변경과 무관하다: `tests/test_trimmed_component_fitter.py::test_fits_flat_plane_with_low_residual`, `::test_jacobian_metrics_detect_a_healthy_flat_fit` — 둘 다 `degenerate_fraction == 0.0` 기대치, 실측 약 0.0017361111 (Worklog 105부터 이어진 별도 attribution 대기 항목, 이번 세션 어떤 라운드에서도 원인 제공하지 않음).

## 14. 아직 미구현인 ordered world-space boundary graph

§10(Ordered boundary recovery)은 이번 라운드 범위 밖이다. `classify_node_boundary_status()`(§15E 경량 diagnostic)는 same_surface/crease edge로부터 **각 node의 개략적 상태**(interior_continuation/crease_boundary_candidate/observed_support_boundary_candidate/unresolved_boundary)만 분류하며, ordered closed loop/open chain/branched graph 복원이나 world-space half-edge 순서화는 전혀 하지 않는다.

## 15. 아직 미구현인 bidirectional boundary fidelity

Worklog 110/112에서 이미 남겨둔 항목이며 이번 라운드에서도 변경하지 않았다.

## 16. 아직 미구현인 false-hole hardening

Worklog 110/112에서 이미 남겨둔 항목이며 이번 라운드에서도 변경하지 않았다(기존 area-ratio 보정을 canonical로 승격하지 않는다는 제약도 그대로다).

## 17. dispatcher/production 비접촉 확인

`git status` 기준 이번 라운드에서 내가 만든 파일은 전부 신규(untracked)이며, 기존 추적 파일(`nurbs_constructor_benchmark/boundary_first.py`, `runner.py`, trainer, production pipeline, uncertain Gaussian proposal/append, ownership, checkpoint, renderer production 경로, `osn_gs/gaussian/torch_model.py`, `osn_gs/core/torch_pipeline.py`, 기존 Phase C–G 코드)는 어느 것도 열람 이상으로 수정하지 않았다. 자동 Gate 승인도 하지 않았다. 새 covariance-guided 경로는 default constructor에 연결되지 않았고 `canonical_boundary_source=pending_gate`로 명시했다.

*(참고: `osn_gs/gaussian/torch_model.py`, `osn_gs/core/torch_pipeline.py`, `nurbs_constructor_benchmark/runner.py`, `scenes.py`, `README.md`가 git status상 modified로 표시되는 것은 이번 세션의 concurrent 작업(사용자 또는 별도 Codex 세션)에 의한 것이며 내가 이번 라운드에서 수정한 것이 아니다.)*

## 18. 다음 사용자 Gate에서 판단할 구체적 범위

1. Anchor/edge score 결합식과 threshold를 실제 benchmark 데이터로 튜닝할지, 아니면 더 넓은 synthetic sweep(다양한 밀도/노이즈/곡률)으로 먼저 검증할지.
2. §11에서 언급한 "isolated adapter/experimental input path"(새 covariance-guided graph의 출력을 기존 common Boundary-first builder의 `outer_boundary + interior_boundary`/`interior_anchor` 역할로 변환하는 interface)를 다음 라운드에서 설계할지.
3. Ordered world-space boundary chain/loop 복원(§10) 착수 여부와 우선순위.
4. Camera/view 기반 occlusion boundary 분류(§8) 착수 여부(이번 라운드는 명시적으로 범위 밖으로 남겨둠).
5. §9에서 선언한 raster/KDE legacy 지위를 실제 `torch_component_boundary.py` 등 기존 코드에 메타데이터로 반영할지(이번 라운드는 지위 선언만 하고 실제 코드 삽입은 하지 않았다).

## 명시적으로 주장하지 않는 것

- Covariance-guided path가 production-ready라고 주장하지 않는다.
- 모든 Gaussian covariance가 true normal이라고 주장하지 않는다(observation status와 geometric reliability는 별개 축이며, isotropic/needle/rejected Gaussian은 신뢰할 수 있는 normal evidence로 쓰지 않는다).
- Boundary graph가 완성됐다고 주장하지 않는다(§14).
- Continuous manifold validity가 보장된다고 주장하지 않는다(pairwise affinity는 finite candidate-neighbor 진단이며 continuous non-intersection/curvature-continuity 증명이 아니다).
- Boundary-first Gate가 완료됐다고 주장하지 않는다.
- Default dispatcher를 교체할 수 있다고 주장하지 않는다.
- Production integration이 가능하다고 주장하지 않는다.

Repository-wide pytest는 여전히 green이 아니다(기존 무관 실패 2건 잔존). 이번 라운드로 어떤 Gate도 완료를 주장하지 않는다.
