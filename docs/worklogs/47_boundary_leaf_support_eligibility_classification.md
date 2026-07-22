# Boundary-leaf support ELIGIBILITY 분류(ACTIVE_OBSERVED/UNCERTAIN/INACTIVE/COMPLEX) — 분석 전용

작성일: 2026-07-21
상태: 분석·prototype 완료, 보고. **production estimator는 여전히 미수정 — 사용자 승인 대기.**
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/45`, `46`

## 배경

worklog 46의 convex hull clip이 `density_gradient`/`sparse_outer_rim`에서 "under-coverage"를 만든 것에 대해, 사용자가 문제를 재정의했다: 이건 hull이 실패한 게 아니라, **그 sparse Gaussian들이 애초에 observed surface를 증명하는지, 아니면 baseline 3DGS가 불확실한 공간에 남긴 sparse artifact인지** 구분 안 한 것이 문제라는 것. OSN-GS는 이미 certain/uncertain Gaussian을 구분하는 게 핵심 설계 원칙이라(`[[project_osn_gs_direction]]`), 이걸 boundary support에도 적용해야 한다. 목표를 "sparse boundary 복구"에서 "boundary evidence의 신뢰도 분류"로 바꿨다. `TODO.md`의 기존 로드맵("Priority 6 — Real-data Input Eligibility", `tau_high`/`tau_low` hysteresis 패턴)의 축소판이기도 하다.

## 1차 설계에 대한 재검토 (GPT 리뷰 반영)

초안을 구현하기 전에 사용자가 가져온 리뷰가 11가지 문제를 지적했고, 전부 반영했다:

1. **COMPLEX를 원본 plane-AABB polygon 그대로 observed mask에 넣으면 안 됨** → COMPLEX는 별도 `complex` mask로 분리, `active_only`/`active_plus_uncertain`에는 절대 안 들어가고 `active_plus_uncertain_plus_complex`에만 포함.
2. **"hysteresis"라는 용어가 부정확함** → 실제로는 상태 이력이 없는 정적 2-threshold 분류라서 문서 전체에서 "two-threshold ternary classification band"로 정정.
3. **`neighbor_active_ratio`가 순환 정의될 위험** → 새 분류 결과가 아니라 반드시 **기존 Phase 1 `STATE_ACTIVE`**만 사용하도록 고정(`neighbor_phase1_active_ratio`로 개명).
4. **flat 씬에서 전체 leaf가 boundary leaf로 잡히는 문제** → leaf별로 `is_root_boundary_leaf`(도메인 박스 경계에 닿음, z축 두께 0 때문에 항상 True), `is_inactive_neighbor_leaf`(진짜 inactive/empty 이웃과 접함), `is_cross_component_boundary_leaf`(다른 컴포넌트와 접함) 3종 provenance를 분리 기록. `is_hole_boundary_leaf`는 Phase 2 정보가 있어야 판단 가능해서 이번 pass에서는 계산 안 함(한계로 명시).
5. **spacing_ratio 분모 불명확** → `median_nn_spacing / sqrt(L_u * L_v)`(추천안)를 기본으로 쓰고, 길쭉한 leaf 대응을 위해 축별 `rho_u`, `rho_v`(2D 최근접 이웃 벡터를 축별로 분해)도 별도 기록.
6. **plane residual 정규화 필요** → `plane_residual_world`와 `plane_residual_normalized`(leaf 스케일로 정규화) 둘 다 기록.
7. **normal consistency sign ambiguity** → `|dot(n_i, n_j)|` 사용, `normal_neighbor_count`도 같이 기록해서 이웃 0~1개인 경우 과신 안 하게(0개면 "neutral" vote, 투표에서 제외).
8. **weighted-sum 대신 vote-trace** → `primary_spacing_class` + `plane_residual_vote`/`normal_consistency_vote`/`neighbor_continuity_vote` + `final_class` + `class_transition_reason`(사람이 읽을 수 있는 문자열)을 전부 기록. 숨겨진 weight 없음.
9. **INACTIVE 판정을 보수적으로** → sparsity 하나만으로는 절대 INACTIVE 안 됨. `inactive_candidate`(spacing 기준)이면서 secondary 신호(plane/normal/neighbor) 중 **2개 이상** 나쁠 때만 INACTIVE, 그 외엔 UNCERTAIN.
10. **GT coverage와 observed coverage를 동일 목표로 두지 않음** → `active_only`(정밀도 중심: false-fill/outward bias/evidence purity)와 `active_plus_uncertain`(회수 중심: physical coverage)을 별도 지표로 보고, `active_only`가 GT를 다 못 덮는 걸 실패로 안 봄.
11. **테스트를 강한 수치 assertion 대신 구조 계약으로** → classifier 단위 테스트(결정론적 규칙 정확성)와 mask 구조 불변식 테스트(ACTIVE∩UNCERTAIN=∅, INACTIVE는 어떤 mask에도 기여 안 함, COMPLEX가 active에 안 섞임)를 분리. scene 수치는 report성 정보로만 다룸.

## 구현

`nurbs_constructor_benchmark/boundary_bias_analysis.py`에 전부 추가(production 코드 미변경):
- `compute_leaf_boundary_provenance`, `compute_leaf_eligibility`(위 규칙 구현), `build_boundary_leaf_records_with_eligibility`, `rasterize_eligibility_masks`(4개 분리 mask + 3개 누적 view), `analyze_scene_with_eligibility`.
- worklog 46의 convex hull clip 자체는 **그대로 유지** — 다만 이제 `ACTIVE_OBSERVED`로 분류된 leaf에만 적용된다.

**구현 중 잡은 버그 2개** (실제로 검증하지 않았으면 몰랐을 것들):
1. `neighbor_phase1_active_ratio` 분모에 root-boundary contact(neighbor=None)가 섞여 들어가서 모든 leaf가 낮게 나오는 문제 — 실제 공간 이웃(`neighbor_id is not None`)만 분모로 쓰도록 수정.
2. 모든 leaf가 UNCERTAIN/INACTIVE가 돼서 mask가 완전히 비면 `_fill_nan_nearest`가 채울 값이 없어 전부 NaN을 반환하던 버그 — 완전히 빈 경우 반지름 0(지지 없음)으로 명시적 처리.

## 결과: 7개 analytic-GT 씬

| 씬 | 분류 (A/U/I/C) | active_only (signed/coverage/false_fill) | active+uncertain (signed/coverage/false_fill) |
|---|---|---|---|
| circle | 6/1/0/0 | -0.027 / 0.562 / 0.000 | -0.012 / 0.688 / 0.022 |
| ellipse | 6/1/0/0 | -0.009 / 0.792 / 0.030 | +0.006 / 0.868 / 0.061 |
| ellipse_high_ecc | 6/1/0/0 | +0.025 / 0.889 / 0.110 | +0.032 / 0.910 / 0.133 |
| ellipse_offcenter | 6/1/0/0 | -0.009 / 0.826 / 0.012 | +0.002 / 0.910 / 0.031 |
| **ellipse_density_gradient** | **2/8/0/0** | -0.199 / 0.271 / **0.006** | **-0.003** / 0.674 / 0.095 |
| **ellipse_sparse_outer_rim** | **1/13/0/0** | -0.265 / 0.104 / **0.000** | **+0.008** / 0.674 / 0.152 |
| ellipse_anisotropic | 8/2/0/0 | -0.018 / 0.667 / 0.024 | +0.009 / 0.806 / 0.084 |

**핵심 발견**: 이전에 worklog 46에서 문제였던 두 sparse 씬에서, 분류기가 실제로 대부분의 leaf를 UNCERTAIN으로 정확히 걸러낸다(density_gradient 8/10, sparse_outer_rim 13/14). 그 결과:
- `active_only`는 매우 보수적(coverage 낮음)이지만 **false_fill이 거의 0**(0.006, 0.000)이다 — "확신 없는 곳은 관측됐다고 안 우긴다"는 목표를 정확히 달성.
- `active_plus_uncertain`은 worklog 46의 단순 hull-clip(density_gradient signed=-0.037/coverage=0.521, sparse_outer_rim signed=-0.051/coverage=0.458)보다 **더 나은 coverage(0.674)와 거의 0에 가까운 signed bias(-0.003, +0.008)**를 회복한다 — 이진 판정 하나로 타협하지 않고 두 개의 다른 신뢰도 view를 따로 제공한 게 실제로 이득이 됐다.
- 어떤 씬에서도 INACTIVE가 나오지 않았다(전부 ACTIVE_OBSERVED 아니면 UNCERTAIN) — 요청대로 INACTIVE 판정이 보수적으로 작동한다는 뜻이지만, 동시에 이 pass의 실제 데이터로는 INACTIVE 분기가 통합 테스트로 한 번도 실행 안 됐다는 뜻이기도 하다(단위 테스트로는 별도 검증함).
- `ellipse_high_ecc`(가늘고 긴 타원)는 여전히 가장 어려운 케이스다(active_only false_fill=0.110) — worklog 45/46에서도 계속 나타난 패턴과 일치.

## 결과: 기존 4개 annulus 씬 (estimator-only, refit 안 함)

| 씬 | 분류 (A/U/I/C) | refined cells (원본 / active_only / active+uncertain) |
|---|---|---|
| planar_hole | 6/4/0/0 | 3059 / 1943 / 2605 |
| planar_hole_offcenter | 6/1/0/0 | 3271 / 2599 / 2724 |
| planar_hole_elliptical | 6/4/0/0 | 2946 / 1891 / 2581 |
| planar_hole_density_gradient | 6/4/0/0 | 2946 / 1810 / 2510 |

analytic 씬과 같은 패턴 — 전부 ACTIVE_OBSERVED/UNCERTAIN으로만 나뉘고 INACTIVE/COMPLEX는 없다. `active_plus_uncertain`이 원본과 `active_only` 사이에서 합리적인 중간값을 만든다.

## 제안 (구현 안 함, 승인 대기)

1. **candidate 2(occupancy-grid contour)는 sparse 복구용이 아니라, dense-but-non-convex한 ACTIVE_OBSERVED leaf에서 convex hull이 concave 영역을 과도하게 채우는 경우에만 평가**해야 한다는 재정의를 지켰다 — 이번 7개 씬에서는 이 문제가 뚜렷하게 나타나지 않아서 별도로 구현하지 않았다.
2. **candidate 3(density feature)는 이미 classifier의 입력(spacing_ratio)으로 흡수됐다** — 더 이상 "fallback"이 아니라 판정 근거 그 자체다.
3. threshold(`spacing_ratio_low/high` 등)는 첫 pass 값(0.35/0.70 등)이고, 아직 정밀 튜닝은 안 했다 — 이번 7+4개 씬에서 합리적으로 작동하는 걸 확인했지만 "최종 확정값"은 아니다.
4. 다음 단계 제안: **production에 적용할 때는 `active_only`를 observed support로, `active_plus_uncertain`을 uncertain support(occluded-surface 추론 단계가 열리면 그쪽의 입력 후보)로 이원화하는 방향이 OSN-GS의 기존 certain/uncertain 원칙과 가장 잘 맞는다.**

## 검증

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"  # 131 passed, 1 skipped
```

새 테스트 9개: classifier 결정론적 규칙 테스트(dense→ACTIVE, sparse+무모순→UNCERTAIN, sparse+다중모순→INACTIVE, COMPLEX 무조건 우선, normal sign-ambiguity, 반복 호출 결정성), mask 구조 불변식 테스트(INACTIVE는 어떤 mask에도 없음, COMPLEX는 active에서 배제되지만 전체 union에는 포함, ACTIVE/UNCERTAIN이 각자의 mask에 정확히 분리).

`torch_voxel_hierarchy.py`, `torch_component_boundary.py`, `torch_boundary_refinement.py`, `torch_annulus_chart.py` — **전부 이번 패스에서도 수정하지 않았다.**
