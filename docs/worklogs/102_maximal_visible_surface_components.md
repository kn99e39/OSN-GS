# Worklog 102 — Maximal Visible Surface Components

## 상태

**완료 — 실측 있음. 실제 scene에서 명확히 부정적인 결과이며, 정직하게 보고한다.** Worklog 97-101의 "conservative 초기화 → bilateral 증명 후 merge" 철학을 전면 교체해, "완전한 관측-가시 evidence에서 시작 → 명시적 관측/occlusion/불연속 증거가 있을 때만 CUT"하는 새 아키텍처(Maximal Visible Connectivity)를 구현·실측했다. Canonical Phase-C observation 상태를 그대로 재사용했고, 곡면에서 직선 3D chord 오판 문제(§5)를 실제로 발견·수정했으며, 8개 합성 fixture 전부 통과했다. 그러나 **실제 scene에서는 최대 component 비율이 92.69%로, Worklog 96(74.70%)보다도 나쁘고 Worklog 98(94.51%)에 근접한 수준의 percolation이 재발했다** — patio와 hedge/배경이 다시 하나의 component로 합쳐졌다(직접 확인). 관측 증거(occlusion/free-space) 자체는 실제 scene에서 거의 발동하지 않았고(전체 1,505,170개 cut 중 3,065개, 0.2%), 대신 WL98에서 재사용한 geometric discontinuity/positional cut이 지배적으로(98.9%) 작동했음에도, 그 정도 cut 밀도로는 dense kNN 그래프의 percolation을 막기에 여전히 부족했다.

## 아키텍처

```
학습된 2DGS surfel (관측-가시 전체)
    -> local candidate connectivity (WL96-101과 완전히 동일한 candidate graph, 재사용)
    -> per-edge CUT 판정, 셋 다 명시적 증거 기반:
         1. 관측 모순(known free space / occluded domain) -- 신규
         2. supported visible-surface discontinuity (WL98 재사용, MERGE 증명 아님)
    -> CUT 안 된 edge의 connected component = Maximal Visible Surface Components
```

Connectivity가 기본값이고, CUT이 증명을 필요로 한다 — Worklog 97-101과 정반대 방향.

## 1. Canonical Phase-C 재사용 — 정확한 방식

`osn_gs/surface/torch_observation_evidence.py`는 **전혀 수정하지 않았다.** `STATUS_*`/`VIEW_STATUS_*` 상수, `ObservationEvidence`/`CameraViewEvidence` dataclass, `_project_points`를 그대로 import했다. `classify_world_samples` 자체는 (Phase E가 실제로 쓰는) 수백 개 candidate 규모를 위한 순수 Python 루프라서 이번 배치가 필요로 하는 수백만 edge 규모에서는 실행 불가능하다는 것을 확인했다 — 그래서 **동일한 per-view 3-way depth-epsilon 규칙**(순서·경계조건까지 동일)을 벡터화한 `_per_view_status_codes`를 신규 작성했고, `test_vectorized_per_view_classification_matches_canonical_classify_world_samples`로 `classify_world_samples`의 실제 출력과 **완전히 동일**함을 직접 검증했다(제2의 호환 안 되는 시스템을 만들지 말라는 지시를 이렇게 충족했다).

## 2. Visible/Occluded 역할 분리 — hard contract

`on_observed_surface`만 visible connectivity의 근거가 된다. `behind_first_observed_surface`(occluded_candidate) 도메인은 절대 visible surface로 채우지 않는다 — 벽 A | occluder | 벽 B 합성 fixture로 정확히 2개 component가 나옴을 확인(`test_wall_with_central_occluder_yields_two_visible_components`).

## 3. Known free space — 별도의 hard contradiction

`known_free_space`는 "거기 표면이 없다"는 뜻으로, occlusion(있지만 안 보임)과 의미가 다르다 — 별도의 `CUT_KNOWN_FREE_SPACE` 사유로 분리 기록했다. 배경이 gap 뒤로 실제로 관측되는 fixture로 검증(`test_known_free_space_gap_separates_two_visible_components`).

## 4. Unknown != Occluded — 증거 없음은 CUT 근거가 아니다

Occluder도 배경도 없는 gap(카메라가 그 지점에 대해 아무것도 렌더링하지 않음)은 `unobserved`로만 남고 **CUT을 유발하지 않는다** — `test_unobserved_gap_without_positive_evidence_does_not_force_a_cut`으로 고정. 관측 증거가 없으면 기본값(연결 유지)을 그대로 둔다.

## 5. 직선 3D chord를 쓰지 않는 정확한 관계 (지시 §5, 실제 버그 발견)

**최초 설계는 틀렸다**: 두 endpoint의 자기 깊이를 선형보간해 "기대 깊이"로 삼는 방법을 처음 문서화했는데, 이는 사실상 직선 3D chord의 깊이와 구조적으로 동일하다는 것을 코드를 작성하는 과정에서(어떤 fixture를 돌리기도 전에) 스스로 깨닫고 고쳤다 — 지시가 명시적으로 금지한 바로 그 실패 모드였다.

**최종 설계**: 카메라 자신의 2D 화면-공간 경로를 따라 실제 렌더링된 depth를 읽되, 판정은 "점 추정과의 비교"가 아니라 **RANGE 판정**이다:

    [min(depth_i, depth_j) - edge_length - depth_epsilon,
     max(depth_i, depth_j) + edge_length + depth_epsilon]

`edge_length`는 두 surfel 사이의 실제 3D 거리(이미 candidate graph가 갖고 있는 값, 새 값 아님) — 이 범위는 기하학적으로 **필연적인** 상한/하한이다: 두 surfel을 잇는 진짜 표면이 카메라 시선 방향으로 아무리 기울어 있어도 그 depth가 이 범위를 벗어날 수 없다. 범위보다 훨씬 가까운 관측은 진짜 전경 occluder, 훨씬 먼 관측은 free-space 증거이고, 범위 안이면 곡률이 얼마나 크든(강하게 휜 원통이든) 절대 벌점을 받지 않는다. `test_curved_surface_with_camera_does_not_falsely_trigger_occlusion_from_curvature`로 직접 검증했다.

## 6. WL97 fragment가 아니라 전체 evidence에서 시작

`torch_maximal_visible_connectivity.py::partition_maximal_visible_components`는 WL97의 114,420개 초기 region을 전혀 참조하지 않는다 — `build_candidate_graph`(WL96-101과 완전히 동일한 함수)로 얻은 전체 candidate graph 위에서 직접 CUT을 평가한다. Worklog 100(`WL100_BILATERAL_BASELINE` 뷰)은 비교 baseline으로만 그대로 재실행했다(모듈 수정 없음).

## 7-8. 연결 유지가 기본값 / bilateral 과반은 topology law가 아님

새 판정 순서는 지시 §7과 동일하다: 관측 모순 → occluded 분리 → supported discontinuity, 세 게이트 모두 NO면 연결 유지. `interface_smooth_majority_fraction`(0.5)은 이 모듈에 아예 존재하지 않는다 — WL98의 geometric discontinuity/positional-sheet CUT 로직(min-combine, `residual_mad_multiplier=3.0`, `parallel_sheet_normal_over_tangent_ratio=1.0` 전부 재사용, 새 threshold 0개)을 "MERGE 증명"이 아니라 "CUT 증거"로만 사용한다 — WL98이 원래 설계된 의미(편측 CUT 결정) 그대로다.

## 9. CUT 사유 — 정확히 5개, 겹칠 수 있음(multi-label)

    CUT_KNOWN_FREE_SPACE
    CUT_OCCLUDED_DOMAIN
    CUT_VISIBLE_GEOMETRIC_DISCONTINUITY   (WL98 재사용)
    CUT_POSITIONAL_SHEET_SEPARATION        (WL98 재사용)
    UNRESOLVED_OBSERVATION_CONFLICT        (같은 edge에 대해 한 카메라는 gap을, 다른 카메라는 완전 연속을 보고 — fail-safe CUT)

## 10. 합성 fixture — 8개 전부 통과

신규 focused 테스트 14개(`tests/test_maximal_visible_connectivity.py`), 전부 통과.

| Fixture | 기대 동작 | 실측 |
|---|---|---|
| A. 완전 가시 평면 | 1 component | ✅(관측 증거 있음/없음 둘 다) |
| B. 완전 가시 곡면(원통) | 1 component, 곡률이 occlusion으로 오판되지 않음 | ✅ |
| C. 벽 A + occluder + 벽 B | 2 component, `CUT_OCCLUDED_DOMAIN>0` | ✅ |
| D. 같은 곡면, occluded gap | 2 component (물리적 연속성 != visible 연결성) | ✅ |
| E. known free-space gap | 2 component, `CUT_KNOWN_FREE_SPACE>0` | ✅ |
| F. 평행 시트 | 분리 유지 | ✅ |
| G. 진짜 sharp discontinuity | 경계 보존 | ✅ |
| H. 증거 없는 unobserved gap | CUT 없음, 1 component | ✅ |

CPU fallback 렌더러의 "실제 depth-순서 occlusion 없음"이라는 기존에 문서화된 한계(`tests/test_observation_evidence.py`) 때문에, C/E fixture는 흐린 blend 대신 **정확한 z-buffer를 직접 구성**하는 `_procedural_evidence` 헬퍼로 만들었다(같은 `ObservationEvidence`/`CameraViewEvidence` dataclass를 그대로 채움, 새 시스템 아님) — 구현 중 pixel round/inset 경계 버그 2개를 실측으로 발견·수정했다. WL98 positional 공식 자체에서도 실제 부호 버그를 하나 발견했다(§11).

## 11. 발견한 실제 버그: WL98 positional-offset 공식의 부호 오류

WL98/99/100이 공유하는 `tangential_offset = (delta_x - normal_offset.unsqueeze(-1) * average_normal).norm(...)` 공식은 `normal_offset`을 **abs값**으로 사용해 부호 있는 투영을 빼는데, signed dot product가 음수인 경우 normal 성분을 상쇄하지 않고 오히려 2배로 만든다. 기존 fixture들은 normal/offset의 부호가 우연히 항상 양수로 맞춰져 있어 이 결함이 드러나지 않았다. 이번 배치의 합성 fixture(반대 부호 normal 관례)가 이를 실측으로 드러냈다 — **이 모듈 자신의 복사본만 signed 버전으로 수정**했고, WL100은 "그대로 보존" 지시에 따라 건드리지 않았다. 향후 WL98/99/100 자체를 다시 만질 일이 있다면 이 결함을 반드시 함께 정정해야 한다.

## 실제 scene 비교 (A. WL100 vs B. Maximal Visible Connectivity)

Checkpoint: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000`(1,190,469 surfel, WL96-101과 동일). 카메라: 161개 train camera 전체(`llffhold=8`), 2DGS surfel rasterizer로 직접 depth 렌더링.

| | G. WL100 baseline | F. Maximal Visible Connectivity(신규) |
|---|---:|---:|
| Component/subset 수 | 112,768 | 13,585 |
| **최대 component 비율** | **22.91%** | **92.69%** |
| Candidate edge | 6,016,599 | 6,016,599(동일) |
| Spatial edge | 5,132,180 | 5,132,180(동일) |
| CUT된 edge | — | 1,505,170 (29.3%) |
| 유지된 edge | — | 3,627,010 (70.7%) |
| Coverage identity | True | True |

CUT 사유별 분해: `CUT_VISIBLE_GEOMETRIC_DISCONTINUITY=364,010`, `CUT_POSITIONAL_SHEET_SEPARATION=1,254,613`(둘이 전체 cut의 98.9%), `CUT_KNOWN_FREE_SPACE=1,685`, `CUT_OCCLUDED_DOMAIN=1,380`, `UNRESOLVED_OBSERVATION_CONFLICT=18,107`(관측 기반 cut 3종 합계 21,172개, 전체 cut의 **1.4%**뿐). `observation_evaluated_edge_count=1,295,809`(spatial edge의 25.3%만 어떤 카메라로도 co-observed됨 — 나머지는 관측 증거 자체가 없어 기본값(연결 유지)으로 남는다).

## 9번 항목: 테이블 곡면 거동

`VISIBLE_DISCONTINUITY_CUT_VIEW`(WL98 재사용 geometric cut)는 테이블 상판·다리를 포함해 scene 전역에 걸쳐 광범위하게(노란색) 발동한다 — WL98 자신의 실제 scene 측정(`cut_residual=364,010`)과 **정확히 일치**해 재사용이 올바르게 이뤄졌음을 교차 확인했다. 그러나 `MAXIMAL_VISIBLE_SURFACE_COMPONENTS` 뷰에서 테이블은 더 이상 독립된 색으로 보이지 않는다 — WL97-100이 유지했던 "테이블은 분리된 coherent 영역"이라는 결과가 이번 배치에서는 **사라졌다**(테이블이 거대 component에 흡수됨).

## 10번 항목: 테이블/바닥 관계

테이블과 바닥(patio)이 이번 배치에서는 **같은 최대 component**에 속한다(직접 확인: `MAXIMAL_VISIBLE_SURFACE_COMPONENTS.ply`에서 두 surfel의 색이 동일).

## 11번 항목: hedge/배경 거동

Worklog 99/100이 반복적으로 겪은 "patio와 hedge가 같은 component가 되는가" 질문을 이번에도 직접 확인했다: Worklog 100의 patio-측/hedge-측 seed(`node 117922`, `node 711179`)가 이번 실행에서 **완전히 동일한 색**(`f_dc = [0.354, -0.815, -0.815]`)으로 나왔다 — **같은 최대 component에 속한다.** WL99/100이 어렵게 막았던 그 percolation이 이번 아키텍처에서 재발했다.

## 12번 항목: Visible component 통계

    visible_component_count = 13,585
    component size: min 1, median 1, mean 87.6, p95 16, max 1,103,500
    largest_component_surfel_fraction = 0.9269
    singleton_surfel_count = 8,735 (0.73%)

## 13번 항목: Coverage identity

`assigned == total_surfels`(1,190,469), `unassigned == 0`, `multiply_owned == 0` — 실제 scene·합성 fixture 전부 확인.

## 14번 항목: Review export

`scripts/devtools/maximal_visible_connectivity_export.py` → `output/osn_gs_maximal_visible_connectivity/`:

    A. ORIGINAL_2DGS_SCENE
    B. OBSERVATION_STATE_VIEW
    C. OCCLUDED_DOMAIN_BOUNDARY_VIEW
    D. KNOWN_FREE_SPACE_CONTRADICTION_VIEW
    E. VISIBLE_DISCONTINUITY_CUT_VIEW
    F. MAXIMAL_VISIBLE_SURFACE_COMPONENTS
    G. WL100_BILATERAL_BASELINE

PNG preview: `output/osn_gs_maximal_visible_connectivity/preview_png/`.

## 재현 명령

```
python scripts/devtools/maximal_visible_connectivity_export.py \
  --checkpoint output/arch_2dgs_coverage_first_surface/2dgs_run1/30000 \
  --out output/osn_gs_maximal_visible_connectivity \
  --device cuda \
  --source-path DATASET
```

## 15번 항목: 테스트

- 신규 focused: `tests/test_maximal_visible_connectivity.py` 14개, 전부 통과.
- 전체 회귀: **1184 passed, 1 skipped, 1 warning, 18 subtests passed in 254.85s**(Worklog 101의 1170에서 정확히 +14).

## 결론

이 배치는 지시된 완료 조건("실제로 관측된-가시 evidence의 maximal connectivity를 복원하면서, 관측 증거가 free/occluded/discontinuous라고 말하는 지점에서 정확히 끊는가")에 대해 명확한 답을 준다 — **아니오, 실제 scene에서는 아니다.** Canonical Phase-C 관측 상태를 정확히 재사용했고, 곡률-오판 문제(직선 chord/선형보간 depth 문제 둘 다)를 실측으로 발견·수정했으며, 8개 합성 계약을 전부 통과했음에도, 실제 scene에서는:

1. **관측 기반 CUT(occlusion/free-space)이 사실상 거의 발동하지 않았다**(전체 cut의 1.4%) — local candidate 이웃(2× local spacing 이내) 사이에는 실제 occluder/free-space gap이 드물다는 뜻이다.
2. WL98에서 재사용한 geometric discontinuity/positional cut은 광범위하게(29.3% of spatial edges) 발동했지만, 이 정도로도 dense kNN 그래프의 percolation을 막기에 **여전히 부족했다** — WL98이 이미 증명한 것과 동일한 실패 모드가 완전히 다른 architecture(merge-by-proof가 아닌 connectivity-by-default)에서도 재현됐다.
3. **최대 component 92.69%는 WL96(74.70%)보다 나쁘고 WL98(94.51%)에 근접한다.** Patio와 hedge/배경이 다시 하나의 component가 됐고, WL97-100이 유지했던 테이블의 독립성도 사라졌다.

Architecture 성공을 주장하지 않는다. Maximal-visible-connectivity 철학 자체의 관측 상태 재사용과 곡률-안전 설계는 유효하게 구현됐지만, 실제 scene에서 percolation을 막기에는 evidence 밀도(특히 로컬 스케일에서의 occlusion/free-space 증거 희소성)가 부족하다는 것이 이번 배치의 정직한 실측 결론이다.
