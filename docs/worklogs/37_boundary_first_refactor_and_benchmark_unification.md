# 37. Boundary-First Refactor Pass + Unified Benchmark CLI

날짜: 2026-07-20
상태: 완료.
governing document: OSN_GS_Final_Boundary_First_NURBS_Direction.md.

Phase 4 완료 후 다음 두 가지 후속 요청을 처리했다(worklog 31 참조).

1. Phase 1–4 module 전반에서 강제 overwrite와 임시/ad-hoc code를 제거하는 refactoring pass
2. Phase 1–4 pipeline을 개별 phase script가 아닌 실제 osn-gs benchmark CLI에 통합

## 1. Refactoring pass

모든 Phase 1–4 module(torch_surface_components.py, torch_component_boundary.py, torch_trimmed_component_fitter.py, torch_chart_topology.py, torch_annulus_chart.py)과 benchmark report script를 재점검하여 forced overwrite, dead code, stale comment를 제거했다. torch_annulus_chart.py의 hard-C0 잔여 코드는 Phase 4 review에서 이미 제거되어 있었다. 다른 module에서는 문제를 발견하지 못했다.

실제로 발견된 문제는 phase4_report.py가 phase3_report.py의 private symbol(_PseudoState, _PseudoModel, _uv_support_payload)을 직접 import하고 있었다는 점이다. 이를 새 공용 module nurbs_constructor_benchmark/benchmark_common.py로 추출하고 public name(PseudoState, PseudoModel, uv_support_payload)을 사용하도록 두 script를 수정했다.

session 시작 시 modified-but-unreviewed로 표시된 OSN_GS_Final_Boundary_First_NURBS_Direction.md와 docs/README.md도 검토했다. 두 변경은 governing-adaptive-voxel-contract와 legacy-retirement end-state를 추가한 문서 변경이며 code 변경은 아니다.

동작 보존을 검증했다. 전체 test suite 86/86 통과했고, phase3_report와 phase4_report를 planar_hole에서 재실행한 결과 worklog 31의 수치와 byte-identical했다(chamfer=0.0058, false_fill=0.167, seam gap=0.012/0.064).

## 2. Unified benchmark CLI

사용자의 stale rectangle+trim NURBS render screenshot을 통해 실제 사용 명령인 osn-gs benchmark가 Phase 1–4 결과를 반영하지 않는다는 사실을 확인했다. runner.py의 --constructor는 TorchOSNGSPipeline을 통한 legacy/voxel_patch_stage1만 지원했고, Phase 1–4는 module path를 직접 알아야 실행할 수 있는 세 개의 별도 script로 남아 있었다.

runner.py에 세 번째 --constructor 선택지로 boundary_first를 추가했다.

- 새 nurbs_constructor_benchmark/boundary_first.py: Stage 1 hierarchy → Phase 1 components → Phase 2 boundary extraction → Phase 4 topology-routed chart generation을 실행한다. non-annulus topology는 Phase 3 trimmed-rectangle baseline으로 자동 fallback한다. 반환값은 duck-typed BoundaryFirstState(model.get_xyz/.cluster_ids/.surface_uv, surface_patches, 단일 combined surface 없음)다.
- runner.py의 evaluate_scene을 construction(TorchOSNGSPipeline.initialize())과 일반화된 score_state(scene, state, construction_seconds, export_dir)로 분리했다. evaluate_scene_boundary_first()는 construct_boundary_first() 후 동일한 score_state를 호출하므로 세 constructor가 동일한 기준으로 평가된다. 하나의 report.json에 비교 가능한 수치를 남기고 boundary_first key에 component_count와 component별 topology/seam diagnostics를 기록한다.
- AnnulusChartSlice에 uv field를 추가했다. 각 slice가 내부적으로 계산하던 fit 후 Gaussian UV를 외부에 공개하여 annulus patch residual도 trimmed-rectangle patch와 같은 방식으로 계산한다.
- boundary-first에는 unassigned point scoring에 사용하는 단일 coarse fallback state.surface가 없다. score_state는 unassigned point residual을 0으로 계산하는 convention을 사용하며, unassigned fraction은 별도 metric으로 보고한다.
- boundary_first.py에 write_point_cloud_ply()를 추가했다. trainer의 covariance/opacity initialization을 실행하지 않으므로 scene position/color와 placeholder opacity/scale/rotation을 사용하는 최소 renderer-compatible Gaussian PLY를 생성한다. export 경로는 다른 constructor와 동일한 output/NURBS_output/scene/{point_cloud.ply,nurbs_surface.json} convention이다.
- runner.py에 다음 --bf-* flag를 추가했다: --bf-normal-threshold-degrees, --bf-offset-threshold-ratio, --bf-boundary-resolution, --bf-density-threshold(기본 3.0), --bf-coarse-gap-closing-cells, --bf-annulus-segments.
- 기존 phase별 세 script를 삭제했다.

의도적으로 osn_gs/core/torch_pipeline.py나 trainer는 수정하지 않았다. boundary-first construction logic은 osn_gs/surface/*와 benchmark-side orchestration 안에만 둔다. 통합한 것은 benchmark CLI surface이며 trainer의 기본 constructor와 lifecycle은 최종 phase 후 명시적 승인을 받아 wiring한다(plan §10.1).

### 검증

검증 명령:

    .venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
    .venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor boundary_first --scenes planar_hole --output <dir>
    .venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor legacy --scenes plane --output <dir>

- 전체 test: 86개 통과.
- boundary_first의 planar_hole: chamfer_rms=0.005800, false_fill=0.167, mean/max seam gap=0.01227/0.06433, jacobian_fold=0. 통합 전 Phase 4 수치(worklog 31)와 동일하다.
- renderer export를 disk에서 확인했다(point_cloud.ply 600 vertices, 유효한 PLY header, nurbs_surface.json 8 patches).
- legacy의 plane도 기존 baseline과 일치했다(chamfer_rms=0.028743). evaluate_scene/score_state 분리 후에도 기존 경로가 변경되지 않았다.

## 교훈

새 construction phase는 완료 조건의 일부로 기존 osn-gs benchmark CLI(nurbs_constructor_benchmark/runner.py의 새 --constructor 선택지)에 반드시 연결해야 한다. 정확한 module path를 이미 아는 사용자만 실행할 수 있는 script는 사실상 보이지 않는 기능이며 최신 구현이 benchmark에 반영되지 않았다는 오해를 만든다. 이 원칙은 project memory(feedback_benchmark_cli_unification)에 상시 규칙으로 기록했다.
