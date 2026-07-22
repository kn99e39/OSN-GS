# Worklog 53: Step 4-D 재평가 — profile 기반 캐노니컬 objective 구현 및 다중 시드 검증

날짜: 2026-07-22

상태: **구현 완료, 4씬 x 5시드 검증 완료. `profile_constrained`도 canonical default 자격 없음.** Production은 `uniform_angle` 유지, 선택자/재시도/폴백 경로 추가 없음. Phase 4는 여전히 open.

## 배경

worklog 52는 기존 `worst_wedge_optimized`(Step 4-D)를 production eligibility 기준의 4씬(`planar_hole`, `planar_hole_offcenter`, `planar_hole_elliptical`, `planar_hole_density_gradient`) x 5시드로 재평가해 "canonical default 후보 아님"으로 결론짓고, 그 "Canonical objective revision proposal" 절에서 다음을 제안했다(직접 구현/검증하지 않은 제안이었음):

1. Phase-2의 실제 hole/outer loop에서 robust local quantile 기반 반경 profile을 뽑을 것 (raw min/max 대신).
2. wedge 양끝뿐 아니라 내부 샘플에서도 물리적 inner tangential width/aspect를 평가할 것.
3. 각도폭 상한과 하한을 모두 강제하고, characteristic length 대비 최소 inner tangential width 조건을 둘 것.
4. wedge의 Gaussian count/profile coverage는 confidence 신호로만 쓰고, scene selector나 별도 fitting 경로로 쓰지 말 것.
5. Phase-2 geometry에서만 유도한 outer-loop conformance 항을 추가하되, runtime objective에 GT chamfer/false-fill을 쓰지 말 것(이들은 benchmark acceptance metric으로만 남긴다).

이번 세션에서 사용자 승인(`a) 진행해`) 하에 위 5개 항목을 그대로 구현하고, worklog 52와 동일한 프로토콜로 재검증했다.

## 구현

`osn_gs/surface/torch_annulus_chart.py`에 새 opt-in `segment_placement="profile_constrained"` 추가. Production 기본값(`uniform_angle`)은 완전히 byte-identical(기존 43개 테스트 전부 통과, 신규 4개 추가로 137/137 + skip 1 유지).

- `_robust_local_radius(...)`: 각도창 내 표본의 median(또는 지정 quantile)을 반환, 표본 부족 시 창을 넓힘 — worklog 52의 "raw min/max가 노이즈에 취약하다"는 root-cause 지적을 직접 반영. Phase-2 hole loop처럼 이미 경계 위에 있는 점 집합에는 median(quantile 0.5)을, outer loop가 명시적으로 없어 density-cell fallback을 쓸 때는 quantile 0.9로 "대략적 극값"을 근사.
- `_optimize_profile_constrained_seam_angles(...)`: `_optimize_worst_wedge_seam_angles`와 동일한 local coordinate-descent 뼈대를 재사용하되,
  - 각 후보 wedge를 5개 내부 샘플(양끝 포함)에서 평가(제안 항목 2),
  - 각 샘플의 inner/outer 반경을 그 샘플의 profile에서 직접 읽어 `kappa = max(d, w_inner)/min(d, w_inner)`를 계산,
  - `characteristic_length`(hole loop 0.1-quantile ~ outer profile 0.9-quantile로 사전 추정) 대비 `w_inner`가 floor 아래면 kappa에 soft penalty를 곱함(제안 항목 3 후반부),
  - 탐색 창 자체에 하한과 **상한**을 모두 둬서(`min/max_angular_width_fraction`) 이전 optimizer에 없던 상한을 추가(제안 항목 3 전반부 -- density_gradient에서 한 wedge가 링 대부분을 먹어버리는 실패를 겨냥),
  - outer 샘플 반경과 그 wedge의 Coons chord(양끝 outer 값의 선형보간) 사이 편차를 `characteristic_length`로 정규화해 conformance 항으로 더함(제안 항목 5). GT chamfer/false-fill은 objective에 전혀 참조하지 않음(둘 다 로컬 profile/Phase-2 geometry에서만 계산).
  - wedge 카운트/밀도는 오직 `_robust_local_radius`의 창 넓히기(=낮은 confidence일 때 더 넓게 스무딩)로만 반영되고, 별도 selector나 경로 분기는 없음(제안 항목 4).
- `build_annulus_chart`는 hole loop/outer loop(있으면)의 극좌표를 무조건 계산해두고(`profile_constrained`가 아닐 때는 그냥 버려짐, 저비용), `segment_placement="profile_constrained"`일 때만 새 optimizer를 호출.
- CLI: `--bf-annulus-segment-placement`에 `profile_constrained` 선택지 추가(`nurbs_constructor_benchmark/runner.py`). `boundary_first.py`는 기존 파라미터 그대로 통과시키므로 변경 없음.

## 검증

- 단위 테스트(`tests/test_annulus_chart.py`, `ProfileConstrainedOptimizerUnitTest`): `WorstWedgeOptimizerUnitTest`와 동일한 두 시나리오(좁은 inner corner에서 실제로 멀어지는지, uniform profile에서는 움직이지 않는지) + 신규 **상한 경계 검증**(density_gradient류 sparse/thin 구간을 흉내낸 합성 profile에서 어떤 wedge도 `max_angular_width_fraction`을 넘지 않는지 직접 확인 — 이전 optimizer에는 대응하는 안전장치가 없었던 지점).
- `AnnulusOGridChartTest`에 smoke test 1개 추가(실제 `_annulus()` 픽스처에서 유한하고 건강한 fit인지, partition invariant가 유지되는지).
- 전체 스위트: `python -m unittest discover -s tests -p "test_*.py"` → **137 passed, 1 skipped**(기존 133+1 baseline 대비 신규 4개 테스트만 추가, 회귀 없음).
- 4씬 x 5시드 실제 production 경로 A/B: `construct_boundary_first(...) -> score_state(...)`를 그대로 호출하는 스크래치 스크립트로 `uniform_angle`/`worst_wedge_optimized`/`profile_constrained` 세 가지를 같은 시드에서 비교(runner.py의 CLI 기본값과 동일한 `--bf-*` 파라미터, `count=600`).

## 결과 (5시드 평균, uniform_angle → worst_wedge_optimized → profile_constrained)

| scene | flips | cond p95 | chamfer_rms | false_fill |
|---|---:|---:|---:|---:|
| planar_hole | 3.8 → 4.2 → 4.2 | 3.76 → 3.56 → 4.12 | 0.00556 → 0.00528 → 0.00548 | 0.1363 → 0.1255 → 0.1230 |
| planar_hole_offcenter | 49.2 → 25.6 → 31.6 | 9.91 → 10.24 → 11.14 | 0.00614 → 0.00616 → 0.00621 | 0.3655 → 0.3908 → 0.3844 |
| planar_hole_elliptical | 2.8 → 1.0 → 5.2 | 4.35 → 3.63 → 5.24 | 0.00511 → 0.00521 → 0.00516 | 0.1281 → 0.1357 → 0.1447 |
| planar_hole_density_gradient | 33.0 → 80.8 → 60.0 | 14.22 → 56.28 → 95.63 | 0.01460 → 0.01288 → 0.01531 | 0.3511 → 0.3511 → 0.3511 |

**`profile_constrained`는 `worst_wedge_optimized` 대비도, `uniform_angle` 대비도 깔끔한 개선이 아니다.**

- `planar_hole_elliptical`: flip 평균이 `uniform_angle`(2.8)과 `worst_wedge_optimized`(1.0) 모두보다 나쁜 5.2로 악화.
- `planar_hole_density_gradient`: `worst_wedge_optimized`가 이미 `cond_p95`를 14.22→56.28로 악화시켰는데, `profile_constrained`는 이를 95.63으로 **더 악화**시킴. 원인은 seed=1 한 케이스(uniform=44, worst_wedge=254, profile_constrained=**451**)에 집중 — sparse한 구간에서 `_robust_local_radius`의 창 넓히기가 결국 넓은/전역에 가까운 window로 수렴하면서 profile 추정치 자체가 국소성을 잃고, 새 objective가 그 부정확한 profile을 근거로 seam을 오히려 더 불안정한 위치로 옮긴 것으로 보인다(코드 상 확인, 별도 root-cause 스크립트로 격리하지는 않음 — Phase 4 hardening 문서의 "탐색은 하되 새 발견을 추가 조사로 무한 확장하지 않는다"는 기존 관례를 따름).
- `planar_hole_offcenter` seed=0 단일 케이스: `uniform_angle`=6 flips, `worst_wedge_optimized`=1 flip인데 `profile_constrained`=**28 flips**로 크게 악화 — 이 시드에서는 이전 optimizer가 이미 거의 완벽하게 처리하고 있었는데 새 objective가 더 나쁜 seam 배치를 선택.
- 유일하게 명확히 나은 지점은 `planar_hole_offcenter` seed=1(기존에 알려진 최악 시드)로, `cond_p95`가 32.13(worst_wedge) → 21.44(profile_constrained)로 개선 — 하지만 이는 4씬 x 5시드 전체 표에서 하나의 셀일 뿐, 전체 평균(11.14 vs 10.24)은 여전히 `uniform_angle`보다 나쁘다.
- `false_fill`은 세 방식 모두에서 거의 변화가 없다(특히 `density_gradient`는 완전히 동일) — union 기반 raster 지표가 seam 위치보다 다른 요인(coverage)에 더 지배되기 때문으로 보이며, Jacobian/flip 악화가 이 지표에는 반영되지 않는다는 점 자체가 이 지표만으로 판단하면 안 된다는 걸 다시 확인시켜준다.

## 결론

worklog 52가 제안한 5개 설계 원칙(robust profile, 내부 샘플링, 상한+하한, 절대 스케일 floor, outer conformance 항)을 문자 그대로 구현했음에도, **seam 각도만 지역적으로 재배치하는 접근 자체가 근본 메커니즘(inner-corner 붕괴, sparse-region profile 불안정성)을 해결하지 못한다**는 것이 이번 검증의 실질적 결론이다. 이전 `worst_wedge_optimized`가 실패한 지점과 겹치기도 하고 겹치지 않기도 하는 새로운 실패 지점을 만들어냈을 뿐, 어느 한 방법도 4씬 전체에서 우위를 보이지 않는다. Proxy를 더 정교하게 다듬는 방향(Step 4-D의 반복)은 이번 결과로 보아 수익 체감에 도달한 것으로 판단된다.

**Production은 변경하지 않았다.** `profile_constrained`는 `--bf-annulus-segment-placement profile_constrained`로만 선택 가능한 opt-in ablation 도구로 남긴다(기존 `outer_radius_weighted_segment_placement`/`worst_wedge_optimized`와 동일한 대우).

## 남은 선택지 (사용자 결정 필요, 미구현)

1. `planar_hole_offcenter`와 `planar_hole_density_gradient`를 `planar_hole`처럼 "문서화된 예외"로 받아들이고 Phase 4를 이 조건부로 종료한다.
2. seam 각도 재배치가 아닌 다른 메커니즘(예: wedge별 `resolution_v`/degree를 국소적으로 늘리거나, 문제 wedge만 별도 knot refinement)을 시도한다 — 지금까지의 모든 Step 4 후보(4-B/4-C/4-D 두 버전)가 "seam 위치만 바꾸는" 접근이었다는 공통점이 있다.
3. Step 4-D 계열 탐색을 여기서 중단하고 Step 5(post-fit continuity check, soft seam regularization)로 넘어간다 — 단, 이는 플랜에서 가장 큰 아키텍처 변경으로 명시된 지점이라 별도 승인 필요.

## 검증 명령

- `python -m unittest discover -s tests -p "test_*.py"` → 137 passed, 1 skipped.
- 4씬 x 5시드 A/B: 스크래치 스크립트(레포 외부, 세션 scratchpad)로 `construct_boundary_first`/`score_state`를 직접 호출 — 커밋된 CLI/스크립트 변경 없음.
