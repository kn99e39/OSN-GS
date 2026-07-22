# Phase 4 하드닝, Step 1-3: seam/Jacobian/boundary 진단 지표 + 다중 씬 기준선

작성일: 2026-07-21
상태: hardening plan의 Step 1-3 완료. 다음은 Step 4(저위험 seed 변경).
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md` (이번 hardening pass의 근거 문서. `OSN_GS_Final_Boundary_First_NURBS_Direction.md`의 Phase 5 게이트에는 영향 없음 — 여전히 별도의 명시적 승인 필요).

## 배경

렌더링된 Phase 4 annulus O-grid 출력에서 예상된 8-wedge 구조는 맞았지만, 한쪽에 눈에 띄는 crack과 radial iso-line 간격 불균일이 보였다. 사용자가 검토한 상세한 외부 리뷰를 근거로, 이 문제는 Phase 4 자체의 책임 범위로 규정됐다(Phase 5는 UV field를 "사용"할 뿐, 망가진 UV field를 고치는 역할이 아니다). 정확한 지표 정의(`‖Su×Sv‖`가 아닌 singular value, along-seam vs across-seam continuity 분리, one-directional이 아닌 symmetric boundary distance, global 고정이 아닌 propagated orientation reference 등)를 두 차례 사용자가 직접 교정한 뒤 이 hardening plan으로 승인받아 작업을 시작했다. 자세한 교정된 정의는 `OSN_GS_Phase4_Hardening_Plan.md` 참조.

## Step 1 — 새 진단 지표 추가 (동작 변화 없음)

`osn_gs/surface/torch_annulus_chart.py`:
- `_jacobian_diagnostics()`: `J=[Su Sv]`의 진짜 parameterization singular value `sigma_min <= sigma_max`를 `J^T J`의 closed-form eigenvalue로 계산 (기존의 `‖Su×Sv‖` 면적값만으로는 부족). condition number와, **슬라이스별로 자체 전파(propagate)되는** orientation reference(컴포넌트 전체에 대한 고정된 global 벡터가 아니라서, 실제로 휘어진 annulus에서도 오탐하지 않음)를 통해 `orientation_flip_count`/`near_degenerate_count`를 별개 조건으로 분리했다 — 기존 `jacobian_min <= 0` 체크는 norm이라 절대 음수가 될 수 없어서 사실상 degeneracy만 잡고 있었다.
- `_measure_seams()` 확장: along-seam(`Sv` tangent 각도)과 across-seam(`Su` cross-derivative 각도, normal 각도, derivative 크기 비율)을 별도 지표군으로 분리 유지했고, 이 O-grid 구성에서는 부호 보정이 필요 없다는 좌표 convention(`u`=tangential, `v`=radial)을 문서화했다.
- `_parameter_quality()`: 슬라이스별 iso-line spacing CV, directional stretch, anisotropy, orthogonality.
- `_boundary_conformance()`: chart의 inner/outer edge와 Phase 2가 실제로 관측한 support boundary loop(Phase 4 자신의 Coons-seed bin이 아니라) 사이의 symmetric(양방향) 거리 — chart edge가 한 점으로 collapse된 경우를 one-directional 거리로는 놓칠 수 있는데 이를 잡아낸다.
- 새 진단 지표는 모두 `chart_quality`라는 새 namespace로 들어가며(`nurbs_constructor_benchmark/boundary_first.py`의 per-component payload), GT 비교용 `ground_truth` namespace와는 절대 섞지 않았다. `jacobian_min`->`min_area_jacobian`, `jacobian_fold_count`->`near_degenerate_slice_count`, `boundary_anchor_max_error`->`seed_boundary_anchor_error`로 실제 측정 내용에 맞게 개명했다.

동작 변화 없음 확인: `planar_hole`의 chamfer_rms/false_fill 불변(0.005800/0.167), 전체 테스트 86/86 통과.

## 즉시 발견된 근본 원인: O-grid 내부 극점 퇴화

새 지표를 `planar_hole`(seed=0)에 켜자마자 orientation-flip 샘플 5개가 나타났는데, 전부 seam `3->4`/`4->5`에만 몰려 있었다. 각 슬라이스의 전체 UV grid에 대해 in-plane Jacobian 부호를 매핑해보니(`planar_hole`은 완전 평면이라 z=0이므로 이 매핑이 유효하다) flip이 정확히 `(u~=0, v~=0)` 코너 — 구멍(inner boundary)에 가장 가깝고 동시에 seam(u=0)에도 맞닿은 지점 — 에서만, 그것도 4/5번 슬라이스에서만 발생했다(나머지 6개는 `min_jacobian_singular_value` 0.17~0.20, `max_jacobian_condition` 2.9~7.7로 건강한 반면 4/5번은 각각 0.010~0.012, 46~66).

근본 원인: 반지름 `r`에서의 실제 접선 방향 호 길이는 `r × angle_step`이라서, 안쪽 경계에 가까울수록 wedge가 가질 수 있는 실제 원주가 0에 가깝게 줄어드는데 반지름 방향 폭은 그대로다. 그래서 `Su`가 그 코너에서 데이터와 무관하게 자연히 작아지고, 그 코너의 sparse한 점 분포에 있는 약간의 noise에도 local parameterization의 부호가 매우 민감해진다. 반지름 방향 control point가 4개(`degree_v=1`)뿐인 상태에서 실제(정확한 원이 아닌) 데이터로 free LSQ fit을 하면 그 코너에서 국소적으로 self-intersect(접힘)가 종종 발생한다.

seed 0에 국한된 우연이 아니라 구조적 문제임을 확인: 테스트한 6개 seed 중 3개(0, 2, 5)에서 최소 한 슬라이스가 flip됐고, 모든 flip이 예외 없이 `(u~=0, v~=0)` inner corner에 집중됐다. 이는 플랜에 이미 있던 "inner boundary 근처 radial iso-line crowding" 이슈의 극단적 사례이지, 별개의 새 버그가 아니다.

## Step 2 — 지표 검증 (`tests/test_annulus_chart.py`에 13개 테스트 추가)

새 지표가 실제로 실패 상황에 반응하는지(단순히 존재하는 것만으로는 부족하다는 플랜의 요구사항에 따라) 검증했다.
- `JacobianDiagnosticsUnitTest`: 손으로 만든 건강한 flat grid(모든 지표가 정확히 1.0/깨끗함), 반지름 방향이 collapse된 grid(`near_degenerate_count > 0`), "bowtie" 형태로 뒤틀린 grid(면적은 건강하면서 `orientation_flip_count > 0` — degeneracy와 orientation-reversal이 올바르게 구분됨을 증명).
- `SeamMetricsUnitTest`: 완전 일치, 순수 평행이동(gap만 변하고 각도는 그대로 — 독립성 증명), tangent reversal(~180°), mirror된 슬라이스(normal과 tangent가 동시에 flip — 평면 surface에서는 이 둘이 구조적으로 결합돼 있음을 문서화), 마지막-처음 순환 seam closure.
- `BoundaryConformanceUnitTest`: 완전 일치, 균일 offset, 그리고 플랜 리뷰에서 사용자가 지적했던 시나리오를 직접 재현한 **collapse된 edge** 케이스: one-directional 거리만 쓰면 0.0(완벽해 보임)이 나오지만, symmetric 방향은 collapse를 정확히 잡아낸다(`reference_to_edge_mean > 0.1`, `coverage_ratio < 0.2`).
- `AnnulusOGridChartTest`: "known bad seed"(test 자체 `_annulus` fixture의 `seed=14`)로 8개의 orientation-flip 샘플을 재현하는 regression/detection guard 추가 — Step 1에서 발견한 실제 실패 상황을 새 지표가 잡아낸다는 것을 증명(합성 구성만이 아니라).

전체 테스트: 99/99 통과 (기존 86 + 신규 13).

## Step 3 — 다중 씬 기준선 (신규 씬 4개)

`nurbs_constructor_benchmark/scenes.py`/`support_domains.py`에 `planar_hole_offcenter`(구멍이 원점에서 벗어남), `planar_hole_elliptical`(타원형 inner/outer boundary), `planar_hole_density_gradient`(같은 원형 annulus지만 안쪽에 점이 몰린 밀도), `curved_annulus`(annulus support 위에 sine 높이) 4개를 추가했다 — Step 6의 gate threshold가 `planar_hole` 하나에만 overfit되지 않도록 하기 위함이다.

| 씬 | orientation_flips | max_jacobian_condition | seam_normal_deg_mean | false_fill | outer conformance chamfer/coverage | inner conformance chamfer/coverage |
|---|---|---|---|---|---|---|
| planar_hole | 5 | 66.1 | 10.00 | 0.167 | 0.095 / 0.605 | 0.023 / 0.965 |
| planar_hole_offcenter | **20** | **190.3** | **17.50** | **0.333** | 0.060 / 0.795 | 0.019 / 1.000 |
| planar_hole_elliptical | 2 | 22.9 | 5.00 | 0.112 | 0.068 / 0.704 | 0.022 / 0.985 |
| planar_hole_density_gradient | 0 | 19.7 | 0.00 | 0.166 | 0.098 / 0.534 | 0.026 / 0.964 |

씬을 하나만 봤으면 알 수 없었을 결론 두 가지:
1. **구멍이 중심에서 벗어난 경우가 inner-corner Jacobian degeneracy의 최악 케이스다**(중심형 대비 flip 4배, condition number 3배, false-fill 2배) — 각도별 반지름 폭이 비대칭이라 일부 슬라이스의 inner corner가 훨씬 얇아지기 때문. Step 4의 수정은 `planar_hole`뿐 아니라 반드시 이 씬으로도 검증해야 한다.
2. **outer boundary conformance는 모든 씬에서 일관되게 나쁘다**(chamfer 0.06~0.10, coverage 0.53~0.80), inner는 일관되게 좋다(chamfer 0.02~0.03, coverage 0.96~1.0) — 특정 씬의 우연이 아니다. 각도별 원형 반지름 기반 Coons seed가 (seed가 암묵적으로 맞춰져 있던) 작고 대략 원형인 hole loop에는 잘 맞지만, 더 크고 모양이 다른 outer boundary에는 잘 안 맞는다는 것을 확인했다 — 이제 가설이 아니라 확정된 Step 4 타깃이다.

**이 플랜의 범위를 벗어나는 것으로 확인된 한계:** `curved_annulus`는 O-grid로 아예 라우팅되지 않는다 — Phase 1의 component builder가 이를 `disk_like` 컴포넌트 2개로 쪼개버리며, `--bf-normal-threshold-degrees`/`--bf-offset-threshold-ratio`를 바꿔도(60°/2.0으로 시도) 변화가 없었다 — 즉 이 split은 Phase 1의 leaf-merge threshold가 아니라 Stage 1의 voxel hierarchy 구성 단계에서 일어난다. 이는 Phase 1/Stage 1 hierarchy 범위이지 이번 annulus-chart hardening 범위가 아니다. 실질적 영향: Jacobian reference normal의 슬라이스별 propagation 설계(실제 곡률에서 오탐하지 않도록 의도적으로 설계한 부분)는 아키텍처상 타당하지만, 실제 휘어진 O-grid 케이스로는 아직 검증되지 못했다 — flat 씬들로만 검증됨. curved multi-loop component construction을 다음에 다루는 사람을 위한 open item으로 남긴다.

## 검증

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor boundary_first --scenes planar_hole planar_hole_offcenter planar_hole_elliptical planar_hole_density_gradient curved_annulus --output <dir>
```

- 테스트 99/99 통과.
- `planar_hole`의 chamfer_rms=0.005800, false_fill=0.167 — hardening 이전 baseline(worklog 36/37)과 동일. Step 1-3이 순수 진단/씬 추가일 뿐 기존 동작에 영향이 없음을 확인.

## 다음

Step 4: 저위험 seed 변경만(축별 arc-length reparameterization, seam-offset sweep, Hermite/derivative-aware Coons seed)을 하나씩 ablation하며 이번 baseline과 비교 — inner-corner degeneracy(특히 `planar_hole_offcenter`에서 최악)와 outer-boundary conformance gap을 우선 타깃으로 삼는다. 플랜에 따라 hard constraint로 continuity를 강제하지 않으며, seed 수준 변경으로 부족할 때만 Step 5(soft seam penalty)로 넘어간다.
