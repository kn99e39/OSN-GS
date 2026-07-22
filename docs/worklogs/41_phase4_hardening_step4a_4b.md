# Phase 4 하드닝, Step 4-A/4-B: 진단 무결성 패치 + seam-offset 탐색

작성일: 2026-07-21
상태: Step 4-A 완료(진단 무결성 패치). Step 4-B(seam-offset sweep) 시도했으나 결정적 승자 없음, 채택하지 않음.
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/40_phase4_hardening_step4_arc_length_rejected.md`

## 배경

`arc_length_outer` 후보가 기각된 뒤, 사용자가 그 결과에 대한 2차 외부(GPT) 리뷰를 가져왔다. 리뷰 대부분이 타당했고, 가장 안전성이 중요한 지적(reference-normal propagation이 raw normal의 flip을 가릴 수 있다는 우려, 3.1)은 코드를 직접 다시 읽고 기존 테스트 결과로 검증한 결과 **실제 버그는 아님**을 확인했다. 하지만 holonomy 체크의 부재, scale 정규화 부재, 이름 오류, objective 미정의, seam-offset sweep 제안 등 나머지 지적은 타당해서 Step 4를 4-A/4-B/4-C/4-D로 재설계했다(플랜 문서 "Step 4 revision" 섹션 참고).

## Step 4-A: 진단 무결성 패치

1. **이름 정정**: `segment_placement="arc_length_outer"` → `"outer_radius_weighted_segment_placement"`로 전면 개명(내부 헬퍼 `_equal_arc_length_boundary_angles` → `_outer_radius_weighted_boundary_angles` 포함). 이 옵션은 boundary curve의 sample correspondence를 바꾸는 게 아니라 seam 각도 자체(chart decomposition)를 재배치하는 것이므로, "reparameterization"이라는 표현이 오해를 부른다는 지적이 맞았다.
2. **Scale-normalized singular value 추가**: `min_jacobian_singular_value_normalized = sigma_min / (characteristic_length + eps)`, `characteristic_length`는 컴포넌트의 median radial width. 절대값과 함께 보고(대체하지 않음).
3. **Orientation holonomy check 추가 — 구현 중 스스로 잡은 버그**: 처음 구현은 각 슬라이스의 reference normal을 "이전(이미 보정된) 슬라이스"에 맞춰 순서대로 sign-align하며 걷는 방식이었는데, 이 방식은 구조적으로 **항상 일관되게 나올 수밖에 없다**(greedy correction이 모든 flip을 흡수하므로) — 즉 아무것도 검출하지 못하는 무의미한 체크였다. 이걸 검증하려고 새로 작성한 테스트(`test_single_flipped_slice...`)가 "flip을 넣었는데도 consistent로 나온다"는 실패로 즉시 잡아냈다. 올바른 구현으로 교체: 링을 따라 인접한 모든 쌍(닫힘 쌍 포함) `dot(reference_k, reference_{k+1})`의 부호를 모아 그 **곱**이 양수인지 확인하는 방식(cyclic 부호열의 표준 parity invariant)으로 바꿨다.
   - **의도적으로 명시한 한계**: 이 구현에서 reference normal은 사실상 하나의 축에 대한 ± 이진값이라, 고립된 flip 하나는 항상 진입/이탈 경계에서 짝수 개(2개)의 local disagreement를 만든다 — 즉 고립된 flip은 이 체크로는 절대 안 잡힌다(수학적으로 불가능). 이건 실패가 아니라 정직한 한계다: 고립된 flip은 이미 `orientation_flip_count`와 seam의 `seam_normal_angle_deg`가 잡고 있고, 이 holonomy 체크는 진짜 non-orientable한(위상적으로 꼬인) construction 버그를 잡기 위한 일반적 안전망으로 존재한다.
4. **Per-sample heatmap export 추가**: `collect_samples`(내부)/`collect_diagnostic_samples`(외부) 플래그, 기본 `False`로 기존 동작에 영향 없음.
5. **Scope 정정**: 플랜 문서에 "Phase 4 hardening은 near-planar annulus component에서만 검증됐다"를 명시하고, `curved_annulus`가 라우팅되지 않는 문제를 Phase 5 게이트 논의 전에 해결하거나 명시적으로 수용해야 할 open item으로 기록했다(Step 4 범위 밖).

검증: 새 단위 테스트 7개(`OrientationHolonomyUnitTest`, `ScaleNormalizedJacobianUnitTest`), 전체 106/106 통과. `planar_hole`/`planar_hole_offcenter` 재실행 결과 기존 baseline과 완전히 동일 — Step 4-A는 순수 진단 추가라는 목표를 지켰다.

## Step 4-B: Seam-offset 탐색 — 시도했으나 결정적 승자 없음

`build_annulus_chart`에 `seam_phase_offset` 파라미터(기본 0.0, `uniform_angle` 모드에서만 의미 있음)를 추가하고, 8개 wedge 폭은 그대로 유지한 채 전체 seam phase만 회전시키는 방식을 시도했다. wedge 개수/폭이 안 바뀌므로 `arc_length_outer`보다 구조적으로 훨씬 안전한 후보였다.

4개 baseline 씬 × 8개 offset(한 wedge 폭을 8등분)을 스윕했다. **처음엔 Jacobian 지표(flip count, near-degenerate, condition p95)만으로 lexicographic 선택**을 했는데, `offset=0.2945`가 승자로 나왔다 — 전체 flip이 27→7로, 특히 타깃이었던 `planar_hole_offcenter`가 20→6으로 줄었다.

**하지만 여기서 멈추지 않고 플랜에 명시된 실제 채택 기준(chamfer/false_fill 회귀 여부)까지 확인하니 다른 그림이 나왔다.** `offset=0.2945`에서 false_fill이 4개 씬 중 3개에서 악화됐다(`offcenter` 0.333→0.375, `elliptical` 0.112→0.135, `density_gradient` 0.166→0.192). 8개 offset 전체에 대해 chamfer/false_fill을 다시 계산해보니 일관된 패턴이 드러났다: **어떤 고정 offset도 4개 씬 전체에서 동시에 이기지 못한다.** 예를 들어 `offset=0.5890`은 `planar_hole`/`planar_hole_offcenter`의 false_fill을 개선하지만 `planar_hole_elliptical`(+38% 상대 악화)과 `planar_hole_density_gradient`(+21% 상대 악화)를 희생시킨다. `offset=0.6872`도 비슷하게 다른 쌍을 맞바꾼다.

**결론**: 전역적으로 고정된 seam phase 회전만으로는 부족하다 — Step 1에서 찾은 진짜 원인(특정 씬의 sparse/tight 영역이 seam이 어디 있든 상관없이 실제 데이터에 의존적으로 나타나는 문제)과 일치하는 결과다. 어떤 offset도 채택하지 않고 `seam_phase_offset`은 기본값 0.0을 유지, `--bf-seam-phase-offset`으로 여전히 실험 가능한 ablation 도구로 코드에 남겨뒀다. 이 결과 자체가 Step 4-D(데이터 기반 adaptive seam 배치)의 필요성을 뒷받침하는 증거지만, 4-D는 여전히 별도 승인 게이트로 남겨둔다.

## 전체 스윕 표 (참고용)

| offset | 전체 flip | offcenter false_fill | elliptical false_fill | density_gradient false_fill |
|---|---|---|---|---|
| 0.0000 (baseline) | 27 | 0.333 | 0.112 | 0.166 |
| 0.0982 | 40 | 0.375 | 0.118 | 0.182 |
| 0.1963 | 22 | 0.353 | 0.131 | 0.180 |
| 0.2945 | 7 | 0.375 | 0.135 | 0.192 |
| 0.3927 | 7 | 0.333 | 0.156 | 0.180 |
| 0.4909 | 16 | 0.336 | 0.175 | 0.196 |
| 0.5890 | 23 | 0.324 | 0.155 | 0.201 |
| 0.6872 | 24 | 0.326 | 0.133 | 0.181 |

## 검증 커맨드

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor boundary_first --scenes planar_hole planar_hole_offcenter planar_hole_elliptical planar_hole_density_gradient --bf-seam-phase-offset <offset> --output <dir>
```

## 다음

Step 4-C(Hermite/derivative-aware Coons seed, seam continuity 개선이 목표이지 inner-corner collapse 해결이 목표가 아님을 명시)로 진행하거나, 여기서 사용자와 다시 체크인한다.
