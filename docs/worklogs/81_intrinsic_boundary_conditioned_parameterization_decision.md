# Worklog 81: PCA-UV vs intrinsic boundary-conditioned parameterization — chart-viability 판정

## 목적

Worklog 80은 chart-domain coverage를 고쳤다(0/5→4/5 통과). 그러나 통과한 4개 region은 전부 `extrapolative`였고, 같은 evidence를 `pca_parameterize_points`(단일 전역 affine projection)로 UV화했을 때 UV-인접 삼각형의 21~36%가 3D normal 부호 불일치를 보였다. 이 배치는 그 원인이 (1) region은 유효한 단일 chart인데 PCA-UV가 부적절한 parameterization인지, 아니면 (2) evidence 자체가 단일 injective chart를 지지하지 않는지를 **하나의 constructor-level 결정**으로 닫는다.

상류 계약은 전부 유지했다: region formation/ownership, sparse topology 추상, worklog 80 dense parametric chart support, covariance_normal, full_evidence_spacing, worklog 77 predicate correction, chart-domain coverage 계약, physical/parametric boundary 구분, visible Gaussian photometric 학습.

## 구현

신규 `osn_gs/surface/torch_intrinsic_boundary_parameterization.py`: **boundary-conditioned discrete harmonic(Tutte) embedding**.

1. worklog 80이 만든 dense chart boundary(이미 검증된 순회 순서)를 **고정된 convex 2D domain**(unit circle 위, 원래 순서·상대 arc-length 비율 유지)으로 못박는다. hull도 PCA rectangle도 아니다 — boundary **위치**는 전부 worklog 80 산출물에서 오고, embedding 형태만 convex로 고정한다(disk topology에서 해가 well-posed·injective가 되기 위한 표준 Tutte 경계 조건).
2. interior evidence는 **3D 원시 좌표에서의 kNN 그래프**(local manifold relation, 전역 축 없음)를 얻는다. 경계에 닿지 못하는 interior 포인트를 위해, 각 interior 포인트에 자신의 최근접 boundary 정점 하나를 추가 연결한다(새 위치를 만들지 않고 기존 거리 정보만 사용).
3. 비경계 노드는 이웃 UV의 균등 평균(discrete Laplace, 고전 Tutte scheme)을 boundary Dirichlet 조건 아래 선형 시스템으로 푼다. interior 그래프가 경계에서 끊긴 연결성분을 가지면(`STATE_DISCONNECTED_GRAPH`) 잇지 않고 그대로 실패 처리한다.

`tests/test_intrinsic_boundary_parameterization.py` 12개: 평탄한 disc에서 정상 materialize, boundary UV가 원 순회 순서를 보존, interior 없이도 boundary만으로 materialize, 정점 3개 미만 시 `STATE_INSUFFICIENT_BOUNDARY`, injectivity(근접 중복 UV 없음) 직접 검증.

## 비교 실행

신규 `scripts/devtools/intrinsic_parameterization_replay.py`: worklog 80의 dense chart support를 그대로 재구성한 뒤, **동일한** `fit_torch_visible_surface_lsq`(6×6, degree 2) 호출에 `initial_uv`만 PCA-UV/intrinsic-UV로 바꿔 fitting한다 — parameterization 외 다른 변수는 통제된다. held-out은 각 UV 자신의 좌표계에서 K=4 checkerboard로 분리한다.

## 실측 (real baseline_compatible@2900, region 0/1/2/3 — worklog 80 coverage 통과분)

| reg | evid | bnd | PCA class | PCA p95 | PCA nbp | PCA fold% | INT class | INT p95 | INT nbp | INT fold% |
|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|
| 0 | 93 | 17 | extrapolative | 4.67 | 0.62 | 30.9 | unsafe_geometry | 9.50 | 0.43 | 70.2 |
| 1 | 519 | 84 | extrapolative | 19.29 | 0.53 | 36.0 | extrapolative | 10.21 | 0.39 | 69.5 |
| 2 | 510 | 92 | extrapolative | 17.80 | 0.70 | 21.5 | extrapolative | 21.04 | 0.42 | 68.7 |
| 3 | 92 | 20 | extrapolative | 5.84 | 0.70 | 21.1 | extrapolative | 6.89 | 0.63 | 47.4 |

`interior_outside_boundary_count`는 intrinsic 쪽에서 4개 region 전부 **0**이다(Tutte embedding이 설계대로 boundary loop 안에 모든 interior를 담는다는 것은 확인됨). Jacobian near-degenerate는 양쪽 다 0.

`knn_k`를 8→20으로 올려 재실행해도(그래프 희소성 artifact 배제) neighborhood preservation은 여전히 더 낮고(0.32~0.43) fold%는 여전히 더 높다(38~65%) — 그래프 해상도 문제가 아니다.

**intrinsic이 PCA-UV보다 전부 더 나쁘다.** neighborhood preservation은 4개 region 전부 하락(0.53~0.70→0.39~0.63), fold%는 전부 상승(21~36%→47~70%), UV 근접 충돌은 region 1/2에서 새로 대량 발생(2→79, 2→100). 유일하게 이긴 지표는 interior_outside_boundary_count(설계상 자명)와 held-out p95(1/2 region에서만 개선, 0/3은 악화).

## 원인 판별

PCA-UV·intrinsic 둘 다에서 나타나는 fold를 같은 원인으로 볼 수 있는지 직접 측정했다. region별 owned evidence(boundary+interior 전체)에 대해:

- **local PCA normal(k=10 kNN) vs region canonical normal의 각도 불일치**: median |dot| 0.688~0.870, `frac(|dot|<0.5)`(각도차 60도 초과 비율) **16.3~37.4%**.
- **canonical-normal 방향 두께 / tangent 방향 extent 비율**: 0.169~0.546.

즉 evidence 자신이 국소적으로도 하나의 매끄러운 tangent plane을 따르지 않는다 — 국소 normal의 16~37%가 canonical 방향과 60도 넘게 어긋나고, normal 방향 두께가 tangent extent의 17~55%에 달한다. 이것은 parameterization이 만드는 artifact가 아니라 **evidence 자체의 3D 형상**이다. Tutte embedding은 disk topology의 injective embedding을 수학적으로 보장하지만, 그 보장은 "국소 그래프가 하나의 평면형 삼각분할과 위상적으로 호환된다"는 전제 위에서만 성립한다 — 그 전제가 real evidence에서 깨져 있으므로, injectivity 보장이 있는 embedding조차 PCA-UV보다 더 심하게 접힌다(fold%가 오히려 상승).

## 판정

**B와 C의 경계** — 정확히는: **PCA-UV가 부적절한 parameterization이라는 가설(A)은 명확히 기각된다.** injectivity를 수학적으로 보장하는 대안(intrinsic Tutte embedding)으로 교체해도 fold%가 악화되므로, 실패는 parameterization 선택의 문제가 아니다.

남은 것은 B/C 중 하나다: **evidence 자신이 단일 injective chart를 지지하지 않는다** — 국소 normal 불일치(16~37%)와 두께비(17~55%)가 이를 직접 뒷받침한다. 그러나 worklog 80이 이미 확인했듯 `independent_chart_components`(2-core 서로소 성분)는 4개 region 전부 **1**이다 — 기존 accepted topology는 어떤 분할도 증명하지 못한다. 지시에 따라 **이번 배치에서 임의 다중 chart 분할은 적용하지 않는다.**

따라서 이번 배치의 constructor-level 결론은: **(B) 관측된 evidence/topology 조건 하에서는 다중 chart 분해가 필요할 수 있으나, 그 분해를 정당화할 topology 해상도가 현재 존재하지 않는다.** production PCA-UV는 교체하지 않는다(교체할 근거 있는 대안이 없다 — intrinsic이 실측상 더 나쁘다).

향후 다중 chart 분해를 정당화하려면 필요한 조건(이번 배치가 측정한 것):
- 국소 normal 불일치가 60도를 넘는 점들이 공간적으로 군집해 있는지(현재 미측정 — 개별 비율만 확인함),
- accepted topology의 해상도를 evidence 밀도까지 올렸을 때 2-core 서로소 성분이 실제로 2개 이상으로 갈리는지.

## 검증

`tests/test_intrinsic_boundary_parameterization.py` 12개 전부 통과. 관련 기존 테스트(`test_dense_parametric_chart_support.py`, `test_region_owned_full_evidence.py`, `test_single_chart_uv_validity.py`) 재실행 48개 전부 통과. **production parameterization 계약을 교체하지 않았으므로**(판정이 A가 아님) 지시대로 전체 regression은 실행하지 않았다. hull·PCA rectangle·bounding box·alpha shape·강제 폐쇄·gap bridging·region merge·shape-specific fallback·임의 다중 chart 분할은 도입하지 않았고, normal/connectivity scale/boundary predicate/NURBS capacity는 재검토하지 않았으며 visible Gaussian photometric 학습은 손대지 않았다.
