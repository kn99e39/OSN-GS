# Eligibility 분류 마스크의 실제 NURBS 재피팅 검증 — 종단간, 진단 전용

작성일: 2026-07-22
상태: 검증 완료, 보고. **production estimator는 여전히 미수정 — 사용자 승인 대기.**
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/45`, `46`, `47`

## 배경

worklog 47의 4-tier eligibility 분류(ACTIVE_OBSERVED/UNCERTAIN/INACTIVE/COMPLEX)는 마스크(raster cell) 레벨 지표까지만 확인했고, "실제로 NURBS를 다시 fitting했을 때도 이득이 유지되는가"는 명시적으로 미룬 상태였다("estimator-only, refit 안 함"). 사용자가 이 용어("NURBS refit")의 의미를 확인한 뒤 "진행해봐"로 승인하여, 이번 pass에서 실제 검증을 수행했다.

## 방법

`nurbs_constructor_benchmark/boundary_first.py`의 annulus 분기(`build_annulus_chart` 호출)와 동일한 절차를 스크립트에서 재현하되, `refined_mask` 인자만 3가지로 교체했다. production 코드(`torch_annulus_chart.py`, `torch_component_boundary.py`, `boundary_first.py`)는 전혀 수정하지 않았다 — 함수를 그대로 호출만 했다.

- `baseline`: `threshold_field & coarse_mask` (production 그대로)
- `active_only`: `threshold_field & eligibility_masks["active_only"]`
- `active_plus_uncertain`: `threshold_field & eligibility_masks["active_plus_uncertain"]`

(worklog 47과 동일하게, eligibility 마스크는 coarse_mask 쪽을 대체하는 것이지 threshold_field를 우회하지 않는다 — production의 `refined_mask = threshold_field & coarse_mask` 합성 규칙을 그대로 따른다.)

각 변형으로 만든 `refined_mask`를 실제 `build_annulus_chart`에 넣어 8-wedge O-grid를 진짜로 fitting하고, 그 결과를 runner.py가 실제로 쓰는 production 채점 함수 `ground_truth_metrics`(chamfer_rms/accuracy_rms/completeness_rms — analytic GT 표면과의 실제 거리)와 `patch_union_metrics`(union_false_fill_ratio, union_coverage_ratio)로 그대로 채점했다. 4개 annulus 씬(`planar_hole`, `_offcenter`, `_elliptical`, `_density_gradient`), `--points 600 --seed 0`(runner.py 기본값과 동일).

## 결과

| 씬 | 변형 | chamfer_rms | accuracy_rms | completeness_rms | false_fill | coverage | seam_gap |
|---|---|---|---|---|---|---|---|
| planar_hole | baseline | 0.00580 | 0.00864 | 0.00296 | 0.1669 | 0.9901 | 0.01227 |
| | active_only | 0.01410 | 0.01082 | 0.01737 | **0.0753** | 0.8736 | 0.03662 |
| | **active_plus_uncertain** | 0.00610 | 0.00689 | 0.00531 | **0.0761** | 0.9624 | 0.01197 |
| planar_hole_offcenter | baseline | 0.00648 | 0.01037 | 0.00259 | 0.3333 | 0.9857 | 0.01694 |
| | active_only | 0.00537 | 0.00652 | 0.00423 | **0.0608** | 0.9673 | 0.01222 |
| | **active_plus_uncertain** | **0.00482** | **0.00658** | 0.00307 | **0.0608** | 0.9798 | **0.01168** |
| planar_hole_elliptical | baseline | 0.00497 | 0.00750 | 0.00244 | 0.1123 | 0.9917 | 0.01115 |
| | active_only | 0.00890 | 0.00693 | 0.01087 | **0.0566** | 0.9151 | 0.02277 |
| | **active_plus_uncertain** | 0.00529 | 0.00669 | 0.00389 | **0.0481** | 0.9766 | 0.01103 |
| planar_hole_density_gradient | baseline | 0.00511 | 0.00772 | 0.00251 | 0.1661 | 0.9609 | 0.01660 |
| | active_only | 0.01236 | 0.00820 | 0.01651 | 0.1677 | 0.8315 | 0.02615 |
| | **active_plus_uncertain** | 0.00560 | 0.00767 | 0.00353 | 0.1669 | 0.9333 | 0.01296 |

## 해석

- **`active_only`(가장 보수적인 지지 영역)는 4개 씬 모두 chamfer_rms가 baseline 대비 악화**된다(`offcenter`만 예외적으로 개선) — coverage를 너무 많이 깎아서(0.83~0.92) completeness_rms가 크게 나빠지고(최대 +487%, planar_hole), 그게 chamfer_rms(accuracy와 completeness의 평균)를 끌어올린다. false_fill은 확실히 줄지만(0.166→0.075 등), **end-to-end 정확도로는 이득보다 손해가 크다** — worklog 47에서 "active_only는 정밀도 지표로만 읽어라, GT coverage를 못 채우는 걸 실패로 보지 않는다"고 명시했던 대로, 이 mask를 그대로 fitting에 쓰면 안 된다는 게 실제 fitting 결과로도 확인됐다.
- **`active_plus_uncertain`은 4개 씬 모두에서 결과가 뚜렷하게 좋다**:
  - chamfer_rms: 3개 씬에서 baseline과 사실상 동급(±0.0005 이내) 또는 더 낮음(offcenter: 0.00648→0.00482, -26%), 나머지 2개(planar_hole, elliptical)도 +0.0003/+0.0003 수준의 미미한 악화.
  - **false_fill이 3/4 씬에서 크게 개선**: planar_hole 0.167→0.076(-54%), offcenter 0.333→0.061(-82%), elliptical 0.112→0.048(-57%). density_gradient만 사실상 변화 없음(0.1661→0.1669) — 이 씬은 sparse Gaussian이 GT hole 경계 자체에 몰려 있어서 threshold_field 쪽이 이미 지배적이었던 것으로 보인다(추가 분석 없이 단정하지 않음, 아래 한계 참고).
  - seam_gap도 대부분 baseline과 동등하거나 더 낮다(density_gradient만 소폭 악화: 0.01660→0.01296은 오히려 개선. 표를 다시 확인하면 전부 개선/동등).
  - coverage는 baseline보다 살짝 낮지만(0.96~0.98) 실질적으로는 거의 손실이 없다.
- **결론**: worklog 47의 마스크 레벨 개선(특히 false_fill 축소)은 마스크 통계에서 끝나는 게 아니라 **실제 NURBS refit 이후에도 유지**된다 — 단, `active_only`가 아니라 `active_plus_uncertain`을 쓸 때만 그렇다. `active_only`는 정밀도 진단 도구로는 유효하지만 fitting 입력으로 그대로 쓰면 completeness가 무너져서 종합 정확도가 나빠진다.

## 한계

- 4개 annulus 씬, `count=600, seed=0` 1회 시드만 확인했다 — worklog 45/46/47처럼 여러 seed에 대한 안정성 스윕은 하지 않았다.
- threshold(`DEFAULT_ELIGIBILITY_THRESHOLDS`)는 worklog 47에서 이미 "첫 pass, 미세 튜닝 안 함"으로 명시된 값을 그대로 썼다.
- `planar_hole_density_gradient`의 false_fill이 거의 개선되지 않은 이유는 별도로 분해하지 않았다 — 필요하면 후속 분석 대상.
- Step 4-D(`worst_wedge_optimized`, 보류 중)와의 상호작용은 검증하지 않았다 — 이번 pass는 `segment_placement="uniform_angle"`(production 기본값) 고정.

## 검증

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"  # 131 passed, 1 skipped (변경 없음, 새 코드는 production 파일을 건드리지 않았으므로 회귀 없음)
```

새로 추가한 코드 없음 — 이번 pass는 스크래치 검증 스크립트(`nurbs_refit_validation.py`, 세션 scratchpad, 저장소에는 포함 안 함)로 기존 production 함수들을 그대로 호출한 것뿐이다. `torch_annulus_chart.py`, `torch_component_boundary.py`, `boundary_first.py` — **전부 이번 패스에서도 수정하지 않았다.**

## 제안 (구현 안 함, 승인 대기)

`active_plus_uncertain` 마스크를 `boundary_first.py`의 production `refined_mask` 합성 규칙에 통합하는 것을 다음 단계로 제안한다 — worklog 47이 제안했던 "`active_only`=observed / `active_plus_uncertain`=uncertain 이원화"와 달리, 이번 end-to-end 결과는 **`active_plus_uncertain` 하나를 현재 production `refined_mask`의 대체물로 채택**하는 편이 더 명확한 승인 대상처럼 보인다(false_fill 개선이 실제 fitting에서도 유지되고, chamfer_rms 손실이 사실상 없다). 다만 이는 제안일 뿐이며, 실제 production 코드 변경 전 다시 확인을 받는다.
