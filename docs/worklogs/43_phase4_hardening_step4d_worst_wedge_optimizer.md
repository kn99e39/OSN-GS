# Phase 4 하드닝, Step 4-D: worst-wedge seam-angle 최적화기 — 지금까지 중 최선의 결과, 완전한 승리는 아님

작성일: 2026-07-21
상태: 구현·검증 완료. 아직 기본값 채택은 안 함 — 사용자 결정 대기.
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/40~42`

## 배경

Step 4-B(seam-offset sweep), Step 4-C(Hermite seed) 둘 다 고정된, 씬에 무관한 규칙으로는 결정적 승자를 못 만들었다. 이 결과 자체가 사용자가 가져온 2차 리뷰의 핵심 지적을 뒷받침했다: 진짜 목표는 "arc length를 균등하게" 하는 게 아니라, worst wedge의 inner-corner collapse와 aspect-ratio distortion을 억제하는 것이다. 사용자가 Step 4-D 진행을 명시적으로 확인해서 착수했다.

## 접근: 전역 DP가 아닌 로컬 좌표 하강법

정확한 전역 최적화(72개 정도의 각도 bin에 대한 cyclic partition DP)는 `O(bins^3 * segments)`로 8-wedge 문제치고 과도하다고 판단해서 기각했다. 대신, 기존 `uniform_angle`의 경계 각도에서 시작해서 매 pass(기본 3회)마다 각 경계 인덱스를 순회하며, 그 경계를 움직였을 때 영향받는 딱 두 개의 wedge(양옆)에 대해서만 다음 지표를 계산한다.

```
kappa_k = max(d_k, w_k_inner) / (min(d_k, w_k_inner) + eps)
```

`d_k`는 반지름 방향 폭, `w_k_inner ≈ r_inner(theta) * 각도폭`은 inner 경계에서의 접선 방향 물리적 폭(Step 1에서 찾은 근본 원인, `Su→0`를 직접 겨냥). 경계 위치 후보 몇 개(기본 9개)를 좁은 window 안에서 평가해서 `max(kappa_{k-1}, kappa_k)`를 최소화하는 위치로 이동한다. `build_annulus_chart`가 이미 `inner_boundary`/`outer_boundary` 계산에 쓰던 것과 같은 cell 기반 반지름 lookup을 그대로 재사용했다.

새 `segment_placement` 선택지 `"worst_wedge_optimized"`로 추가했고(기본값은 여전히 `uniform_angle`, byte-identical 확인됨), `uniform_angle`의 시작점 위에서 정제하는 방식이라 완전히 새로운 배치 규칙이 아니라 그 위에 얹는 refinement다.

## 검증

- 합성 테스트: 각도 0 근처에 인위적으로 좁은 inner corner를 만든 반지름 프로파일에서, 정확히 그 코너에 있던 경계가 실제로 눈에 띄게(0.15 rad 이상) 벗어나는지 확인 — 통과. 균일한 프로파일에서는 noise만으로 경계가 표류하지 않는지도 확인 — 통과.
- 전체 테스트 111/111 통과(108 + 신규 3개).
- 기본값(`uniform_angle`)은 여전히 byte-identical.

## A/B 테스트 결과 — 지금까지 중 최선, 완전한 승리는 아님

| 씬 | flips (base→4D) | cond_p95 (base→4D) | chamfer (base→4D) | false_fill (base→4D) |
|---|---|---|---|---|
| planar_hole | 5→3 | 3.38→3.27 | 0.005800→0.005911 (noise 수준) | 0.167→0.173 (noise 수준) |
| **planar_hole_offcenter (타깃)** | 20→**12** | 8.64→**6.54** | 0.006482→**0.006098 (개선)** | 0.333→**0.324 (개선)** |
| planar_hole_elliptical | 2→3 | 3.56→4.04 | 0.004968→0.004928 (거의 동일) | 0.112→**0.137 (악화, +22%)** |
| planar_hole_density_gradient | 0→0 | 2.73→2.73 | 완전 동일 | 완전 동일 |

**`planar_hole_offcenter`(Step 3에서 찾은 최악 케이스, Step 4-B/4-C 둘 다 못 고쳤던 씬)가 Jacobian 건강성과 정확도 지표를 동시에 개선한 첫 후보다.** flip 20→12(40% 감소), condition p95 8.64→6.54(24% 감소)이면서 chamfer/false_fill도 둘 다 좋아졌다 — 지금까지 시도한 모든 후보 중 유일하게 "타깃 씬에서 트레이드오프 없는 개선"을 보여준 케이스다. `planar_hole_density_gradient`는 optimizer가 아무것도 움직일 필요를 못 찾아서 완전히 그대로다.

유일한 대가는 `planar_hole_elliptical`의 false_fill이 0.112→0.137로 22% 상대 악화된 것이다. 확인해보니 outer boundary conformance(Step 3에서 찾은 별개의, 아직 안 고친 문제)는 이 optimizer로 인해 거의 안 바뀌었다(offcenter의 outer symmetric_chamfer=0.061, coverage=0.795 — baseline과 거의 동일) — 이 optimizer는 inner-corner/aspect-ratio 메커니즘을 겨냥한 것이지 outer conformance 문제는 원래 범위 밖이었고, 실제로도 그대로 남아있다.

## 결정 대기

Step 4-B/C와 달리 이번엔 "완전히 별로"도 아니고 "완전한 승리"도 아니다 — 우선순위 타깃 씬에서는 트레이드오프 없이 개선됐지만 별개 씬(elliptical)에서 false_fill 22% 악화라는 진짜 비용이 있다. 플랜의 엄격한 "모든 씬에서 회귀 없어야 채택" 기준으로 보면 아직 통과가 아니다. 기본값을 `uniform_angle`에서 `worst_wedge_optimized`로 바꿀지는 사용자 판단이 필요해서 아직 안 바꿨고, `--bf-annulus-segment-placement worst_wedge_optimized`로 계속 실험 가능한 상태로 남겨뒀다.

## 검증 커맨드

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor boundary_first --scenes planar_hole planar_hole_offcenter planar_hole_elliptical planar_hole_density_gradient --bf-annulus-segment-placement worst_wedge_optimized --output <dir>
```
