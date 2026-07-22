# Boundary-leaf eligibility 분류를 production `extract_component_boundary` 기본값으로 적용

작성일: 2026-07-22
상태: **적용 완료 (production 코드 변경).** 사용자 승인("그게 좋겠다. 적용하고 보고해볼래?")에 따라 진행.
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/45`, `46`, `47`, `48`

## 배경

worklog 48에서 `active_plus_uncertain` eligibility 마스크가 실제 NURBS refit에서도 baseline 대비 chamfer_rms 동등/개선, false_fill 대폭 감소를 end-to-end로 검증했다. 사용자가 이를 production에 적용하도록 승인했다.

## 구현

### 1. 진단 코드를 production 모듈로 승격

`nurbs_constructor_benchmark/boundary_bias_analysis.py`(진단 전용, worklog 45-48)에 있던 eligibility 분류 로직을 `osn_gs/surface/torch_boundary_eligibility.py`(신규, production)로 이동했다: `ACTIVE_OBSERVED`/`UNCERTAIN`/`INACTIVE`/`COMPLEX` 상수, `DEFAULT_ELIGIBILITY_THRESHOLDS`, `LeafBoundaryProvenance`/`LeafEligibilityResult`, `compute_leaf_boundary_provenance`, `compute_leaf_eligibility`, hull-clip 유틸(`_convex_hull_2d`/`_sutherland_hodgman_clip`/`_polygon_area_2d`), 그리고 신규 `build_eligibility_filtered_coarse_mask(component, hierarchy, points, frame, resolution, ...)` — worklog 47/48의 `active_plus_uncertain_plus_complex` 뷰(interior 및 COMPLEX 경계 leaf는 원본 polygon 유지, ACTIVE_OBSERVED만 hull-clip, INACTIVE만 제외)를 반환한다. `boundary_bias_analysis.py`는 이제 이 production 모듈에서 import해서 재사용하며, 자체 정의를 중복하지 않는다(drift 방지).

`active_plus_uncertain`이 아니라 `active_plus_uncertain_plus_complex`를 쓴 이유: worklog 48이 검증한 4개 씬 전부 COMPLEX leaf가 0개였다 — `active_plus_uncertain`과 `active_plus_uncertain_plus_complex`는 이 4개 씬에서 완전히 동일한 결과다. COMPLEX leaf가 실제로 존재하는(미검증) 씬에서 그 leaf의 기여가 갑자기 사라지는 것을 막기 위해, 더 보수적인(기존 동작을 그대로 보존하는) 쪽을 선택했다.

### 2. `extract_component_boundary`에 배선

`osn_gs/surface/torch_component_boundary.py`의 §2.1 coarse-support 구성에 `filter_boundary_leaf_eligibility: bool = True`(신규 기본값)를 추가했다. `coarse_mask`(진단용, `coarse_support_cells`/`false_fill_cells`에 쓰이는 필드)는 항상 미필터링 원본 union으로 유지되고, `coarse_mask_for_combine`(실제 `refined_mask` 계산에 쓰이는 값)만 eligibility 필터링된 마스크로 교체된다. `False`로 설정하면 worklog 45 이전의 정확한 원래 동작(순수 union, 분류 없음)을 재현한다 — 회귀 비교용.

### 3. 발견하고 고친 버그: gap-closing dilation이 개선 효과를 거의 지워버림

프로덕션에 배선한 직후 `planar_hole` 씬으로 확인했더니 `refined_mask` cell 수가 필터링 켬/끔 상관없이 완전히 동일(3059)했다 — 뭔가 잘못됐다는 신호였다. 원인: 기존 `coarse_gap_closing_cells=2`(reprojection seam을 메우기 위한 순수 dilation, erosion 없음)가 eligibility로 이미 좁혀진 마스크에도 똑같이 적용되고 있었다. Dilation은 erosion 없는 순수 팽창이라, hull-clip으로 좁힌 경계를 k=2칸만큼 다시 밖으로 부풀려서 개선을 거의 지워버렸다(측정: undilated 2605 → k=2 dilation 후 3051, 미필터링 baseline 3059와 거의 같아짐).

그래서 처음엔 eligibility 경로에서 dilation을 아예 껐다(k=0). 그런데 이게 **더 심각한 문제**를 드러냈다: 4개 annulus 씬 중 3개(`planar_hole`, `planar_hole_elliptical`, `planar_hole_density_gradient`)에서 `classify_boundary_result`가 `annulus`가 아니라 `disk_like`/`complex`로 바뀌었다 — hole이 통째로 사라지거나 outer loop가 여러 개로 쪼개졌다. 개별 boundary leaf를 독립적으로 hull-clip/제외하면서 hole을 둘러싼 leaf 고리(ring)의 래스터 연결성이 끊기고, hole의 배경이 exterior 배경과 합쳐진 것이다.

**해결**: `coarse_gap_closing_cells`(기존, 미필터링 경로 전용, 기본 2)와 별도로 `eligibility_gap_closing_cells`(신규, eligibility 경로 전용)를 도입하고, k=0/1/2를 4개 실제 annulus 씬 전체에 대해 실제 `build_annulus_chart` fit + `ground_truth_metrics`/`patch_union_metrics`로 스윕했다:

| k | topology 안전(4씬 중) | 비고 |
|---|---|---|
| 0 | 1/4 (`offcenter`만 안전) | ring 연결성 깨짐 — 채택 불가 |
| 1 | **4/4** | topology 안전 + 개선 효과 대부분 유지 |
| 2 | 4/4 | topology 안전하지만 개선 효과 대부분 상쇄 |

`eligibility_gap_closing_cells=1`을 최종 기본값으로 채택했다.

## 최종 검증 결과 (실제 production 경로, `extract_component_boundary` 기본값 그대로)

| 씬 | 지표 | old_default(필터 끔) | **new_default(필터 켬, k=1)** |
|---|---|---|---|
| planar_hole | chamfer_rms / false_fill / coverage | 0.00580 / 0.1669 / 0.9901 | 0.00569(-1.9%) / **0.1281(-23%)** / 0.9852 |
| planar_hole_offcenter | 〃 | 0.00648 / 0.3333 / 0.9857 | **0.00464(-28%)** / **0.1582(-53%)** / 0.9928 |
| planar_hole_elliptical | 〃 | 0.00497 / 0.1123 / 0.9917 | 0.00475(-4.4%) / 0.1113(-0.9%) / 0.9955 |
| planar_hole_density_gradient | 〃 | 0.00511 / 0.1661 / 0.9609 | 0.00528(**+3.3%**) / 0.1669(+0.5%) / 0.9495(**-1.2%**) |

4개 씬 모두 topology는 `annulus`로 유지된다(실제 CLI로도 확인: `osn-gs benchmark --constructor boundary_first --scenes planar_hole --points 600` → `boundary_first: components=1 topologies=['annulus']`, `chamfer_rms=0.005695`, 위 표와 일치). 3/4 씬에서 명확한 개선(false_fill 최대 -53%, chamfer_rms 최대 -28%), 1/4 씬(`density_gradient`)에서 작은 악화(chamfer +3.3%, coverage -1.2%). worklog 48의 (dilation 없이 측정한) 헤드라인 수치보다는 개선폭이 작지만 — hole topology 안전성을 지키기 위한 정당한 trade-off이며, 여전히 순net 개선이다.

## 코드 변경 목록

- `osn_gs/surface/torch_boundary_eligibility.py` (신규, production)
- `osn_gs/surface/torch_component_boundary.py`: `filter_boundary_leaf_eligibility`(기본 `True`), `eligibility_thresholds`, `eligibility_gap_closing_cells`(기본 `1`) 파라미터 추가, coarse-support 구성 로직 수정, `diagnostics`에 `filter_boundary_leaf_eligibility` 플래그 추가
- `nurbs_constructor_benchmark/boundary_bias_analysis.py`: 자체 정의 제거, production 모듈에서 import하도록 리팩터링 (동작 변경 없음)
- `tests/test_component_boundary.py`: `test_coarse_gap_closing_cells_zero_falls_back_to_raw_coarse_mask`에 `filter_boundary_leaf_eligibility=False` 추가(테스트 의도 격리 — 신규 기본값과 무관하게 순수 경로만 검증)
- `tests/test_annulus_chart.py`: `EligibilityFilteringTopologySafetyTest` 신규 (기본값이 topology를 지킨다는 회귀 테스트 + k=0이 실제로 topology를 깨뜨린다는 걸 문서화하는 회귀 테스트, `count=600`이 재현하는 최소 조건임을 직접 측정으로 확인)

`boundary_first.py`(construct_boundary_first)는 수정하지 않았다 — `extract_component_boundary`를 기본 인자로 호출하므로 새 기본값이 자동으로 적용된다.

## 검증

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"  # 133 passed, 1 skipped (신규 2개 테스트 포함)
.venv\Scripts\python.exe -m nurbs_constructor_benchmark.runner --constructor boundary_first --scenes planar_hole --points 600  # 실제 CLI, topology=annulus, chamfer_rms=0.005695 (검증 스크립트와 일치)
```

## 한계 / 후속 과제

- `eligibility_gap_closing_cells=1`은 이번 4개 annulus 씬에서 직접 측정으로 정한 값이며, 정밀 튜닝된 최종값은 아니다.
- 곡면(curved, non-annulus fallback) 컴포넌트에서 eligibility 필터링 + gap-closing 상호작용은 별도로 검증하지 않았다 — `sine`/`crescent` 등 곡면 reprojection-seam 시나리오에 대한 재검증은 후속 과제로 남는다.
- `planar_hole_density_gradient`의 소폭 악화(coverage -1.2%, chamfer +3.3%) 원인은 별도로 분해하지 않았다.
- `DEFAULT_ELIGIBILITY_THRESHOLDS`는 여전히 worklog 47에서 정한 첫 pass 값이다.
