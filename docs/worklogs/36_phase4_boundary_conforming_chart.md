# 36. Phase 4 경계 정합 차트 생성기

날짜: 2026-07-20
상태: 구현 및 검증 완료. 사용자의 명시적 승인 없이 Phase 5를 시작하지 않는다.
기준 문서: `../Urgent_Work/OSN_GS_Final_Boundary_First_NURBS_Direction.md`, Phase 4.

## 완료한 범위

- Phase 2의 outer/hole loop count를 사용한 topology routing(disk_like, annulus, multi_hole, complex, non_chartable)을 추가했다. 구현 위치는 osn_gs/surface/torch_chart_topology.py다.
- hole-area significance filter(MIN_HOLE_AREA_FRACTION = 0.02)를 추가했다. outer loop 면적의 2%보다 작은 hole은 annulus가 아니라 disk_like로 처리한다. 이 필터가 없으면 density_gradient의 17셀 density-threshold artifact가 8-slice O-grid로 잘못 라우팅되어 fit이 붕괴했다.
- annulus O-grid generator(osn_gs/surface/torch_annulus_chart.py)를 추가했다. outer loop 하나와 significant hole 하나를 가진 component는 기본 8개의 radial NURBS wedge로 표현한다. 각 wedge는 기존 IDW/LSQ/foot-point fitter를 재사용하며 Coons 방식의 polar-local UV(local_s/local_t) seed와 slice-boundary 간 shared radius를 사용한다.
- slice별 bounds, control grid, seam diagnostics, topology checks, boundary_anchor_max_error 및 실제 fitted NURBS에서 평가한 U/V iso-line polyline을 JSON provenance로 export한다.
- support_domain_metrics의 raster fragmentation artifact를 수정했다. 이제 patch_union_metrics의 adaptive-density rasterization(_patch_xy_mask)을 재사용한다.

## 발견 사항: hard C0 시도는 정확도 회귀를 일으켜 되돌림

중간 버전은 free LSQ fit 후 각 wedge의 boundary control-point column을 인접 slice와 동일하게 덮어써 C0 continuity를 강제했다. seam gap은 약 1e-7까지 줄었지만 planar_hole의 accuracy가 악화됐다.

| variant | chamfer_rms | GT-compared false-fill | mean seam gap |
|---|---:|---:|---:|
| free fit(Co​​ons seed만 사용) | 0.0058 | 0.167–0.180 | 0.005–0.012 |
| hard constraint, 2-point chord boundary | 0.0061 | 0.200 | ~0 |
| hard constraint, raster loop boundary | 0.0095 | 0.311 | ~0 |
| Phase 3 baseline(rectangle + trim) | 0.0080 | 0.200 | n/a |

경계를 강제로 고정하면 wedge fit이 데이터가 지지하는 위치에서 벗어났다. Phase 2 raster loop point를 강제 경계로 사용한 버전은 cell-center의 staircase quantization noise 때문에 더 악화됐다. 따라서 plan §4.5의 초기 C0, 후속 G1/C1 방향에 따라 현재 구현은 shared Coons-style UV seed를 사용하는 free fit으로 유지하고 seam은 측정만 한다. 사용하지 않는 helper와 hard-constraint 잔여 parameter도 제거했다.

## 검증

전체 test 86개를 통과했다.

- planar_hole: annulus, wedge 8개, Jacobian fold 0, mean seam gap 0.012, max 0.064, support conformality 1.000.
- Phase 3 대비 chamfer RMS는 0.0058 대 0.0080, GT 비교 false-fill은 0.167 대 0.200으로 개선됐다.
- union_hole_count=2로 보고되지만 두 번째는 1-cell seam-speckle artifact이며, GT와 일치하는 1113-cell hole과 tiny-hole diagnostics로 구분된다.
- fitted NURBS의 inner/outer iso-line(v=0/v=1) 반경은 0.292–0.306 및 0.880–0.924로 수치 검증했다.
- plane, sine, crease, close_parallel_sheets, density_gradient는 모두 Phase 3 trimmed fallback으로 라우팅되며 기존 수치와 동일하다.
- legacy/Stage 1/Stage 1-F production file은 committed baseline(71a4ae0)과 diff가 없다.

## 안전장치와 남은 작업

- benchmark 전용이며 trainer default나 legacy/Stage 1 비교 경로를 변경하지 않는다.
- 결정론적 seam placement는 angle zero를 사용한다. curvature/confidence 기반 배치와 G1/C1 continuity는 후속 과제다.
- 2% hole significance threshold는 planar_hole과 density_gradient 두 사례로 조정한 값이므로 추가 labeled case에서 재검토해야 한다.

## 게이트

Phase 4는 O-grid, topology routing, Jacobian, seam measurement, iso-line, trimmed-baseline comparison 조건을 충족한다. 사용자가 명시적으로 승인하기 전에는 Phase 5를 시작하지 않는다.

## 업데이트, 2026-07-20: 통합 OSN-GS 벤치마크로 통합

세 개의 phase별 script(component_report.py, phase3_report.py, phase4_report.py)를 제거했다. boundary-first pipeline은 legacy/voxel_patch_stage1과 동일한 unified benchmark entry point에서 세 번째 --constructor 선택지가 된다.

새 module nurbs_constructor_benchmark/boundary_first.py는 BoundaryFirstState를 만들고, 모든 constructor가 사용하는 동일한 score_state body(nurbs_constructor_benchmark/runner.py)로 평가한다. 하나의 report.json에 직접 비교 가능한 field를 남기며 component별 topology/seam diagnostics도 boundary_first key에 기록한다. renderer export는 동일한 output/NURBS_output/scene convention을 사용한다.

통합 전과 byte-for-byte 동일한 수치를 확인했다(planar_hole: chamfer=0.0058, false_fill=0.167, seam gap=0.012/0.064). osn_gs/core/torch_pipeline.py와 trainer는 수정하지 않았으며 boundary-first construction은 osn_gs/surface/*와 benchmark orchestration 안에서만 동작한다.
