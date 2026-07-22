# Phase 2 외곽 경계 추정기 편향 분석 (진단 전용, 추정기 변경 없음)

작성일: 2026-07-21
상태: 분석 완료, 보고. **estimator 코드는 전혀 안 건드림 — 사용자 승인 대기.**
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/39/43`

## 배경

Step 4-D(`worst_wedge_optimized`)가 `planar_hole_elliptical`의 false_fill을 악화시킨 원인을 사용자가 분석한 결과: `phase2_boundary_conformance`(Phase 2 자체 추정치 대비)는 거의 안 바뀌었는데 GT 대비 `union_false_fill_ratio`는 악화됐다 — 즉 optimizer는 Phase 2의 outer 추정치에는 계속 충실했지만, 그 추정치 자체가 GT보다 바깥쪽으로 편향돼 있다는 것. 이건 worklog 39가 이미 "outer boundary conformance가 모든 씬에서 구조적으로 나쁘다"고 찾아놓고 근본 원인은 못 밝힌 문제다. 사용자 지시: **candidate selector를 더 만들기 전에, 이 편향이 파이프라인의 정확히 어느 단계에서 처음 발생하는지부터 분해·검증하라.** estimator 동작은 전혀 바꾸지 않고, 분석과 제안만 먼저 보고한다.

## 방법론

`nurbs_constructor_benchmark/boundary_bias_analysis.py`(신규, 진단 전용 모듈)를 만들어 실제 GT(원/타원, 반지름이 각도의 함수로 정확히 알려진 도형)를 가진 새 합성 씬 7개를 생성하고, Phase 1/Phase 2 파이프라인을 그대로 호출(코드 변경 없음)해서 7단계 각각에서 "각도별 반지름 프로파일"을 뽑아 GT와 비교했다.

- 씬: `boundary_bias_circle`, `_ellipse`, `_ellipse_high_ecc`, `_ellipse_offcenter`, `_ellipse_density_gradient`, `_ellipse_sparse_outer_rim`, `_ellipse_anisotropic` (`nurbs_constructor_benchmark/support_domains.py`/`scenes` 아님 — 이 분석 전용이라 `boundary_bias_analysis.py` 안에 가벼운 자체 씬 구조체로 생성, 기존 `SyntheticGaussianScene`의 오라클/패치라벨 등 불필요한 필드를 끌고 오지 않기 위한 의도적 단순화).
- 단계: (1) raw Gaussian, (2) voxel/coarse union, (3) KDE의 radial crossing(2D grid가 아니라 각 각도 방향으로 반지름을 걸어가며 threshold를 넘는 지점), (4) thresholded binary mask, (5) marching-squares raw contour, (6) resampled contour(신규 유틸리티, 진단 전용), (7) Phase 4가 실제로 쓰는 것과 같은 per-angle-window 반지름 lookup.
- 지표: signed distance, edge-to-GT/GT-to-edge/symmetric Chamfer/Hausdorff, enclosed area error, false-fill area, coverage, 8-sector 방향별 편향 — 전부 `tests/test_boundary_bias_analysis.py`에서 아는 편향을 주입한 케이스로 먼저 검증했다(예: `r_stage = r_gt + 0.05` 전 각도에 넣으면 정확히 0.05가 나오는지).
- **버그를 하나 잡고 넘어갔다**: 처음 구현에서 Phase 1/2가 만드는 정규화된 UV 좌표계(`[0,1]^2`, 축마다 span이 다를 수 있음)에서 각도/반지름을 계산했는데, 이러면 원/타원의 실제 모양 자체가 깨진다. 원(circle) 씬에서 signed bias가 -25%씩 나오는 명백히 이상한 결과로 처음 드러났고, 월드 좌표계로 전부 다시 계산하도록 고쳤다(각도/반지름은 `scene.center` 기준 월드 XY에서, KDE 평가만 그 순간에 UV로 변환). 이 회귀를 스모크 테스트(`signed_distance_mean`이 참 반지름의 30% 넘게 벗어나면 실패)로 고정해뒀다.

## 결과

### 1. 편향이 최초로 발생하는 단계: **Stage 2 (voxel/coarse union)**

7개 씬 전부에서 예외 없이, stage 1(raw 점)에서는 편향이 거의 없거나(원형 씬들) sparse 샘플링 아티팩트로 인한 음의 편향(안쪽으로, 예상된 현상)인 반면, **stage 2로 넘어가는 순간 뚜렷한 양의(바깥쪽) 편향이 나타나고, 이게 전체 파이프라인에서 가장 큰 편향이다:**

| 씬 | stage1 | **stage2 (최대)** | stage7 (최종) |
|---|---|---|---|
| circle | -0.016 | **+0.035** | +0.019 |
| ellipse | -0.004 | **+0.092** | +0.031 |
| ellipse_high_ecc | +0.020 | **+0.074** | +0.047 |
| ellipse_offcenter | -0.006 | **+0.073** | +0.022 |
| ellipse_density_gradient | -0.061 | **+0.040** | +0.015 |
| ellipse_sparse_outer_rim | -0.083 | **+0.033** | +0.010 |
| ellipse_anisotropic | -0.009 | **+0.085** | +0.022 |

7개 씬 전부에서 stage2가 최종(stage7)보다 1.5~3배 더 큰 편향을 가진다. 즉 **KDE/threshold 단계는 stage2가 만든 과대추정을 오히려 줄이는 방향으로 작동한다** — 지금까지 의심했던 "KDE bandwidth/threshold 튜닝이 문제"라는 가설과 반대다.

### 2. 주원인: **Phase 1의 voxel plane-AABB polygon union이지, KDE/threshold가 아니다**

Stage 2(`coarse_mask`)는 Phase 1이 각 voxel leaf의 plane-AABB 교차 폴리곤을 union한 것이다. 각 leaf voxel은 자신의 AABB 전체를 지지 영역으로 잡기 때문에, 점이 실제로 없는 voxel 셀 가장자리까지 폴리곤이 확장된다 — 이게 파이프라인에서 가장 큰 단일 편향원이다.

Ablation sweep 결과 (bandwidth multiplier, threshold 모두 매우 민감함을 확인):
- `density_bandwidth_multiplier`: 1.0→4.0으로 올리면 stage3 편향이 -0.64→+0.15(ellipse)로 뒤집힐 만큼 민감. **현재 기본값(2.0)은 거의 zero-bias 근처에 위치** — 이미 우연히 괜찮은 지점에 있다.
- `density_threshold`: 1.5→5.0으로 올리면 stage3이 +0.06→-0.06으로 뒤집힘. **현재 기본값(3.0)도 zero-bias 근처.**
- resolution(32→128): stage2/4/7 편향이 완만하게 증가(예: ellipse stage2 +0.077→+0.099) — 존재하지만 부차적인 효과.
- adaptive vs fixed bandwidth: fixed(전역 고정)로 바꾸면 stage3이 파국적으로 나빠짐(sparse_outer_rim에서 -0.578!) — **현재 프로덕션 기본값인 adaptive per-sample bandwidth가 명백히 옳은 선택이고, 이 축은 원인이 아니다.**

즉 원인 순위: **1) Phase 1 voxel union polygon(가장 큼) > 2) grid resolution의 부차적 효과 > 3) KDE bandwidth/threshold(현재 기본값 근처에서는 오히려 문제를 줄이고 있음, 원인이라기보다는 완화책에 가까움) > 4) resampling(stage5→6, 씬마다 부호가 오락가락 — 뚜렷한 방향성 없음).**

### 3. 씬별 패턴

- 편향의 절대 크기는 이심률/비대칭성이 큰 씬(`ellipse`, `ellipse_anisotropic`, `ellipse_offcenter`)에서 크고, 등방형인 `circle`에서 가장 작다. 단, `ellipse_high_ecc`는 예외적으로 stage2 편향이 중간 정도(+0.074)인데도 stage7까지 가장 많이 남는다(+0.047, 7개 중 최대) — 얇고 긴 형태가 KDE thresholding으로 잘 안 지워진다는 뜻.
- `ellipse_sparse_outer_rim`은 stage1의 raw 편향이 가장 크게 음수(-0.083, 경계 근처 샘플이 거의 없어서)이지만, stage2~7은 오히려 다른 씬보다 작은 편에 속한다 — voxel leaf 크기 자체가 점이 희박한 영역에서는 작아지는 경향과 관련된 것으로 보인다(추가 조사 필요, 결론 아님).
- 섹터별(8구간) 편향은 이번 리포트에서 개별 방향 패턴까지 전부 나열하진 않았지만 raw 데이터는 이미 계산돼 있다 — 특정 방향에 편향이 몰리는지는 재실행으로 바로 확인 가능(`compute_bias_metrics`의 `sector_bias`).

### 4. 가장 안정적인 추정기 후보 (측정 결과일 뿐, 아직 구현 안 함)

이번 분석에서 "이걸로 바꾸면 확실히 낫다"고 단정할 수 있는 단일 설정은 못 찾았다 — bandwidth/threshold 둘 다 이미 거의 zero-bias 지점 근처에 튜닝돼 있고(그러니까 우연이든 아니든 현재 기본값이 이미 상당히 합리적이다), 씬마다 최적 지점이 미세하게 다르다. 유일하게 명확한 결론은 **"stage 2(Phase 1의 voxel union)를 그대로 두고 stage 3/4(KDE/threshold)만 손보는 방식으로는 stage 2의 과대추정을 근본적으로 못 없앤다"**는 것 — KDE 임계값을 아무리 조여도 stage 2가 만든 초과분을 부분적으로만 상쇄할 뿐이다.

### 5. 기존 4개 annulus 씬(before/after)

이번 패스에서는 실제 production 파이프라인(`osn-gs benchmark`)에 새 설정을 적용해서 before/after를 만들지 않았다 — **이 지시(estimator 변경 전 사용자 승인)를 지키기 위해서다.** 대신 이번 7개 씬에서 나온 stage-level 패턴이 기존 `planar_hole`/`planar_hole_offcenter`/`planar_hole_elliptical`/`planar_hole_density_gradient`의 outer conformance 문제(worklog 39, symmetric_chamfer 0.06~0.10, coverage 0.53~0.80)와 정성적으로 일치한다: 이 4개 씬 모두 이번에 찾은 stage2 과대추정 메커니즘의 영향을 받는 구조(voxel 기반 Phase 1 → Phase 2)를 그대로 쓰기 때문이다. 정량적 before/after는 실제 fix 방향이 정해진 뒤 다음 패스에서 만드는 게 맞다.

### 6. Step 4-D 재평가 가능 여부

**아직은 아니다.** Step 4-D의 `worst_wedge_optimized`는 Phase 2가 만든 (편향된) outer 추정치를 입력으로 그대로 쓴다 — 그 추정치 자체가 안 고쳐지면 optimizer가 아무리 똑똑해져도 "편향된 목표"를 향해 최적화할 뿐이다. Phase 2의 outer 추정 방식(특히 stage 2의 voxel union 과대추정)을 실제로 개선한 뒤에야 Step 4-D를 다시 의미 있게 평가할 수 있다.

## 다음 단계 제안 (구현 안 함, 승인 대기)

- **가장 유력한 방향**: stage 2(voxel union)의 과대추정을 줄이는 것 — 예를 들어 각 voxel leaf의 plane-AABB 폴리곤 대신 leaf 내부의 실제 점 분포를 반영한 더 타이트한 경계(예: leaf-local convex hull이나 alpha-shape)를 쓰는 방향. 다만 이건 Phase 1(`torch_voxel_hierarchy.py`)을 건드리는 일이라 이번 annulus-chart 중심의 Step 4 작업보다 범위가 크다.
- 사용자가 "금지" 목록에 넣은 것들(GT 기반 scene-specific 보정, 고정 inward offset, morphology erosion)은 전부 지켰다 — 어떤 것도 시도하지 않았다.
- 다음 결정은 사용자에게 맡긴다: (a) Phase 1 voxel union 개선을 별도 작업으로 진행할지, (b) 이 발견을 문서화만 해두고 Step 4-D는 현 상태로 보류 유지할지, (c) 다른 우선순위로 이동할지.

## 검증

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"  # 116/116
```

새 테스트(`tests/test_boundary_bias_analysis.py`, 5개): 알려진 상수 offset 주입 시 정확히 그 값이 나오는지, 국소적 bulge가 해당 섹터에만 나타나는지, 완전 일치 시 0이 나오는지, 실제 파이프라인이 유한하고 합리적인 범위 내 값을 내는지(월드/UV 좌표계 버그의 회귀 방지) 검증.

estimator 코드(`torch_component_boundary.py`, `torch_boundary_refinement.py`, `torch_annulus_chart.py`)는 이번 패스에서 **전혀 수정하지 않았다.**
