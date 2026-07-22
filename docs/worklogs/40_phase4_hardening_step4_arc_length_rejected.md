# Phase 4 하드닝, Step 4 (1차): 동일 호길이 세그먼트 배치 — 시도 후 기각

작성일: 2026-07-21
상태: Step 4 진행 중. 첫 번째 후보(outer arc-length 기반 segment 배치)는 실측 후 기각. 다음 후보로 이어감.
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/39_phase4_hardening_step1_3.md`

## 배경

Step 3의 멀티 씬 baseline에서 `planar_hole_offcenter`(구멍이 중심에서 벗어난 씬)가 inner-corner Jacobian degeneracy의 최악 케이스로 확인됐다(중심형 대비 orientation flip 4배, condition number 3배, false-fill 2배). Step 4의 첫 번째 저위험 seed 변경 후보로, 플랜에 명시된 "arc-length reparameterization"을 시도했다.

## 구현

`osn_gs/surface/torch_annulus_chart.py`에 `_equal_arc_length_boundary_angles()`를 추가하고, `build_annulus_chart`에 opt-in 파라미터 `segment_placement`를 도입했다.

- `"uniform_angle"`(기본값, 기존 동작과 byte-identical): 구멍 중심 기준 8등분 균등 각도.
- `"arc_length_outer"`: refined mask의 outer 반지름 프로파일(이미 `inner_boundary`/`outer_boundary` 계산에 쓰이던 것과 같은 cell 데이터)로부터 각도별 outer 반지름 히스토그램을 만들고, `ds/dtheta ≈ r_outer(theta)`로 누적 arc length를 근사한 뒤, 이를 역산해서 outer boundary 기준 등호(等弧, equal arc length) 지점에 8개의 seam 각도를 배치.

동시에 seam 경계가 더 이상 전역 `angle_step` 상수가 아니라 슬라이스별로 다른 폭을 가질 수 있도록, 각도 wraparound 처리(`theta_hi <= theta_lo`일 때 `+two_pi`)와 point selection의 각도 비교 로직을 일반화했다. `"uniform_angle"` 모드에서는 정확히 동일한 수치가 나오도록 설계했고, 실제로 확인했다(아래 검증 참고).

`nurbs_constructor_benchmark/boundary_first.py`/`runner.py`에 `--bf-annulus-segment-placement {uniform_angle,arc_length_outer}` CLI 옵션을 추가해 벤치마크로 바로 A/B 비교할 수 있게 했다.

## 검증: 기본 경로(`uniform_angle`)는 완전히 동일

- 전체 테스트 99/99 통과.
- `planar_hole` 재실행: `mean_seam_gap=0.01227`, `orientation_flips=5`, `jacobian_cond_p95=3.38` — 리팩터링 이전과 완전히 동일한 수치. 대규모 코드 변경(전역 상수 → 슬라이스별 폭, wraparound 일반화)에도 불구하고 기본 경로는 byte-identical함을 확인했다.

## A/B 테스트 결과: `arc_length_outer` 기각

4개 baseline 씬 전체에 대해 실행한 결과, 개선은커녕 4개 중 3개에서 악화됐다.

| 씬 | flips (uniform → arc_length_outer) | jacobian_cond_p95 | seam_normal_deg_mean | false_fill |
|---|---|---|---|---|
| planar_hole | 5 → 9 | 3.38 → 4.72 | 10.0 → 5.0 | 0.167 → 0.154 |
| planar_hole_offcenter (원래 타깃) | 20 → 20 | 8.64 → 8.35 | 17.5 → 25.0 | 0.333 → 0.341 |
| planar_hole_elliptical | 2 → 3 | 3.56 → 3.70 | 5.0 → 5.0 | 0.112 → 0.112 |
| planar_hole_density_gradient | **0 → 20** | 19.7 → 6.67 | **0.0 → 25.0** | 0.166 → 0.199 |

정작 고치려던 `planar_hole_offcenter`는 flip 개수가 전혀 줄지 않았고(20→20), false_fill과 seam_normal_deg는 오히려 악화됐다. `planar_hole_density_gradient`는 flip이 0에서 20으로 급증하는 파국적 회귀가 나왔다.

**원인 분석**: outer 반지름만으로 arc length를 계산하다 보니 inner 경계의 기하 정보를 아예 반영하지 못해서, Step 1에서 찾은 진짜 원인(inner corner의 접선 방향 물리적 길이가 0에 가까워지는 문제)에는 손도 대지 못했다. 게다가 `density_gradient` 씬은 outer 쪽 커버리지가 sparse해서 outer 반지름 히스토그램 자체가 noisy했고, 그 결과 arc-length 누적함수가 불안정해지면서 segment 경계가 엉뚱한 곳에 배치됐다.

## 결정

플랜에 이미 확립된 원칙 — "그럴듯해 보여도 실측해서 정확도를 해치면 채택하지 않는다"(hard-C0 시도를 되돌렸던 것과 같은 원칙, 이번엔 hard constraint가 아니라 seed 선택에 적용) — 에 따라 `segment_placement` 기본값은 `"uniform_angle"`로 유지한다. `"arc_length_outer"`는 코드에서 삭제하지 않고 테스트되고 동작하는 ablation 도구로 남겨둔다(이전 hard-C0 비교가 코드 히스토리에 문서화된 채로 남겨졌던 것과 같은 방식).

## 다음

Step 4의 남은 후보: seam-offset sweep, Hermite/derivative-aware Coons seed. 이번 실패에서 얻은 새 아이디어도 후보에 추가한다 — outer 단독이 아니라 inner와 outer를 함께 고려하는 arc-length 배치(예: 평균, 혹은 국소 곡률이 더 급한 쪽을 기준으로 삼는 방식). outer 단독 버전은 애초에 이번에 고치려던 inner-corner 메커니즘을 전혀 다루지 못한다는 게 명확해졌기 때문이다.

## 검증 커맨드

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor boundary_first --scenes planar_hole planar_hole_offcenter planar_hole_elliptical planar_hole_density_gradient --bf-annulus-segment-placement arc_length_outer --output <dir>
```
