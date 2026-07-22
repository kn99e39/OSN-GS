# Phase 4 하드닝, Step 4-C: Hermite/도함수 인지 Coons 초기값 — 미미한 효과, 채택하지 않음

작성일: 2026-07-21
상태: Step 4-A/4-B/4-C 전부 시도 완료, 결정적 승자 없음. Step 4-D(chart-layout 최적화)는 범위가 더 크므로 사용자 확인 후 진행.
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/41_phase4_hardening_step4a_4b.md`

## 구현

`build_annulus_chart`에 `hermite_boundary_seed: bool = False`를 추가했다. 기존 Coons seed는 각 슬라이스 경계의 반지름 값(`inner_lo`/`inner_hi`/`outer_lo`/`outer_hi`)만 선형 보간했는데, 이는 값은 이웃 슬라이스와 정확히 일치하지만 그 값의 **기울기**(`d(radius)/d(local_s)`)는 슬라이스마다 독립적으로 계산되어 경계에서 불연속일 수 있었다. Hermite 버전은 각 경계 인덱스에서 central difference로 **공유되는** 기울기(`inner_slope`/`outer_slope`)를 한 번만 계산해서, 슬라이스 k의 `local_s=1` 경계와 슬라이스 k+1의 `local_s=0` 경계가 정확히 같은 기울기 값을 참조하도록 cubic Hermite 블렌드로 교체했다. 기본값(`False`)은 기존 선형 공식과 완전히 동일 — 새 테스트(`test_hermite_boundary_seed_default_off_is_unchanged`)로 확인했다.

명시적으로 범위를 한정했다: 이건 **seam continuity 개선용**이지, inner-corner collapse(진짜 원인, `‖Su‖→0`)를 고치는 게 아니다. seed가 아무리 매끄러워도 그 근처의 실제 물리적 접선 길이가 0에 가까워지는 문제 자체는 그대로다.

## A/B 테스트 결과

4개 baseline 씬 전체에 적용한 결과, 효과는 작고 엇갈렸다.

| 씬 | mean_seam_gap (base→hermite) | seam_tangent_deg_mean | flips | chamfer_rms |
|---|---|---|---|---|
| planar_hole | 0.01227→0.01234 | 3.40→3.40 | 5→5 | 0.005800→0.006180 (**악화**) |
| planar_hole_offcenter (타깃) | 0.01694→0.01562 (개선) | 6.24→5.41 (개선) | 20→17 (개선) | 0.006482→0.006673 (**악화**) |
| planar_hole_elliptical | 0.01115→0.01124 | 거의 동일 | 2→2 | 0.004968→0.005166 (**악화**) |
| planar_hole_density_gradient | 0.01660→0.01669 | 거의 동일 | 0→0 | 0.005113→0.005182 (**악화**) |

타깃이었던 `planar_hole_offcenter`의 seam 지표는 소폭 개선됐지만(flip 20→17, tangent 각도 6.24°→5.41° — 여전히 0과는 거리가 멀다), 나머지 3개 씬은 seam 지표가 거의 안 바뀌었다. 반면 **chamfer_rms는 4개 씬 전부에서 일관되게 소폭 악화**됐다(절대값 +0.0002~+0.0004, 상대로 몇 % 수준). 더 매끄러운 seed가 어디서든 free LSQ fit을 데이터에서 살짝 밀어내는 것으로 보이며, 그 대가로 얻는 seam 개선은 4개 씬 중 1개에서만 의미 있게 나타났다.

플랜에 명시된 "chamfer/false_fill을 회귀시키지 않아야 채택" 기준에 비춰보면, 넓지 않은 이득 대비 전 씬에 걸친(작더라도) 일관된 chamfer 악화는 통과 기준을 만족하지 못한다. 기본값은 `False`로 유지하고, `--bf-hermite-boundary-seed`로 실험 가능한 ablation 도구로 코드에 남겼다.

## Step 4 종합 결론

원래 플랜이 제안했던 세 가지 저위험 seed-level 후보(`outer_radius_weighted_segment_placement`, `seam_phase_offset`, `hermite_boundary_seed`)를 전부 시도했고, **어느 것도 뚜렷한 승자가 아니었다.** 세 개의 독립적인 메커니즘에서 공통으로 나온 이 결과 자체가 의미 있는 발견이다 — inner-corner degeneracy와 outer-boundary conformance gap은 씬에 무관한 고정된 seed 조정으로는 해결되지 않고, 데이터에 적응적인 접근(Step 4-D)이 남은 유력한 방향이라는 뜻이다. 다만 4-D는 플랜상 chart-layout 최적화(seed tweak이 아니라 seam 각도 자체를 constrained optimization으로 정하는 것)로 범위가 더 크기 때문에, 시작 전에 사용자 확인이 필요하다.

## 검증

- 전체 테스트 108/108 통과(기존 106 + Step 4-C 신규 2개: `test_hermite_boundary_seed_default_off_is_unchanged`, `test_hermite_boundary_seed_on_produces_finite_healthy_fit`).
- 기본값(`hermite_boundary_seed=False`) 재실행 결과 `planar_hole` 수치가 baseline과 완전히 동일함을 확인(chamfer=0.005800, false_fill=0.167).

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m osn_gs.cli benchmark --constructor boundary_first --scenes planar_hole planar_hole_offcenter planar_hole_elliptical planar_hole_density_gradient --bf-hermite-boundary-seed --output <dir>
```
