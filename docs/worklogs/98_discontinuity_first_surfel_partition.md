# Worklog 98 — Discontinuity-first Surfel Subset partition

## 상태

**완료 — 실측 있음, 그러나 결과는 부정적이다(정직하게 보고).** Worklog 97의 region-concentration 방식이 실제 scene에서 곡률 있는 테이블 옆면을 정상 normal 회전만으로 여러 조각으로 쪼갠다는 사용자 지적에 따라, union rule을 다시 한 번 완전히 교체했다. 합성 fixture(평면·원통형 밴드·크리즈·평행 시트)에서는 설계 의도대로 동작했지만, **실제 scene 재실측에서는 percolation(거대 subset 재발생)을 방지하지 못했다** — scene의 94.5%가 다시 하나의 subset으로 합쳐졌다. Architecture 성공/실패 최종 판단은 이 배치에서 내리지 않지만, 이 결과 자체는 부정적이며 그렇게 보고한다.

## 1. 정확한 로컬 이웃 정의

Worklog 96/97과 **완전히 동일**: kNN spatial adjacency(`neighbor_count=8`) + local-spacing gate(`spatial_connect_spacing_multiplier=2.0`, 각자의 kNN 거리 중앙값 기준). `build_candidate_graph()`를 그대로 재사용해 세 방법(WL96/WL97/신규) 모두 동일한 local candidate graph 위에서 동작한다. **이번 배치에서는 이 candidate graph를 만들 때 normal compatibility로 edge를 미리 걸러내지 않는다** — normal gradient는 진단 신호일 뿐 최종 boundary 판정이 아니라는 지시(§2)에 따라, `normal_compatibility_min_alignment`는 provenance 기록용으로만 config에 남아있고 실제 cut 판정에는 전혀 관여하지 않는다.

## 2. Normal-gradient 정규화 거리

`g_ij = theta_ij / spatial_distance_ij`는 §2가 요구한 진단량으로 기록은 하지만(`normal_gradient_magnitude`, 아래 §3의 shape operator norm으로 대체 구현), **이 값 자체를 boundary 판정에 쓰지 않는다.** 이유는 지시와 동일하다 — 원통형 곡면처럼 정상적으로 큰 normal gradient를 갖는 매끄러운 표면을 boundary로 오판하기 때문이다.

## 3. Local shape operator 추정

신규 `osn_gs/surface/torch_discontinuity_first_surfel_partition.py::_fit_shape_operators`.

각 surfel `i`의 자기 자신의 kNN k=8 이웃(candidate graph와 동일 k, 별도 파라미터 아님)에 대해, 미분관계 `Delta n ~= -S Delta x_T`를 만족하는 2×2 shape operator `S_i`를, `i`의 **자기 tangent plane 좌표계**(`tangent_axis_u`, `tangent_axis_v` — 학습된 surfel의 intrinsic 값, eigen-decomposition 아님)로 투영한 뒤 batched 최소제곱(정규방정식 `(X^T X + ridge*I) B = X^T Y`, 2×2 선형계를 N개 노드에 대해 동시에 배치 처리)으로 구한다. Normal 부호 모호성은 이웃 normal을 쿼리 normal 부호에 맞춰 로컬로 정렬한 뒤 차분을 취해 해소한다(전역 outward 방향 지정 없음). `ridge=1e-8`은 근사 특이 행렬(이웃이 거의 공선인 경우)에 대한 수치 안정화 전용이며 정상적으로 조건화된 fit을 편향시키지 않는다.

## 4. Smooth-surface residual

Edge `(i,j)`에 대해 양방향 모두 계산한다: `predicted_ij = -S_i @ Delta x_T,ij`(i의 fit으로 j쪽 변화 예측), `predicted_ji = -S_j @ Delta x_T,ji`(j의 fit으로 i쪽 변화 예측). 최종 residual은 **두 방향의 최솟값**(`min(residual_i->j, residual_j->i)`)이다.

**구현 중 발견한 실제 문제와 정정**: 처음에는 최댓값(§4가 예시로 든 "information from both directions"의 자연스러운 첫 해석)을 썼는데, 합성 크리즈 fixture에서 크리즈로부터 2 pitch 이내에 있는 **같은 평면 내부** edge까지 대량으로 잘못 잘렸다(53개 subset으로 과분열, 그중 111/281이 진짜 크리즈를 건너지 않는 edge). 원인: 크리즈 근처 노드는 자신의 kNN 이웃이 양쪽 평면에 걸쳐 있어 `S_i` fit 자체가 오염되고, 그 오염된 `S_i`가 **모든** 자신의 edge를 의심스럽게 만든다. **양쪽 다 실패해야만 자른다**(min)로 바꾸자 같은 fixture에서 과분열이 53→26개로, 잘못된 same-side cut이 111→20개로 줄었고 최종적으로 두 개의 지배적 subset(각각 132/144, 두 평면의 92%)으로 수렴했다 — 나머지 20개는 크리즈 바로 위 1~2 pitch 폭의 작은 파편으로, 유한한 kNN 이웃 스케일에서 기대할 수 있는 정직한 불확실성 폭이다.

## 5. Positional / parallel-sheet 분리 기준

Edge의 변위 `delta_x_ij`를 두 방향으로 분해한다: normal 방향 성분 `normal_offset = |delta_x . average_normal|`, 그 나머지(접평면 내) 성분 `tangential_offset = ||delta_x - normal_offset * average_normal||`. **`normal_offset > tangential_offset`이면(비율 1.0 초과) 자른다** — "접평면을 따라 이동한 것보다 접평면을 벗어난 쪽으로 더 많이 이동했다"는, scale-free하고 자기-정규화된(자기 자신의 두 직교 성분을 비교하므로 별도 spacing 참조 불필요) 기준이다.

**구현 중 발견한 실제 버그**: 처음에는 기존 candidate graph의 `spatial_connect_spacing_multiplier`(2.0)를 그대로 재사용해 `normal_offset / local_spacing > 2.0`이면 자르도록 설계했다(§5의 "재사용을 우선하라"는 기존 프로젝트 관례를 따른 선택). **이 설계는 구조적으로 절대 발동할 수 없는 결함이었다** — candidate graph 자체가 이미 `total_distance <= 2.0 * spacing`인 edge만 통과시키므로, `normal_offset`(총 변위의 한 성분)은 항상 `total_distance` 이하이고, 따라서 같은 2.0 배수를 절대 넘을 수 없다. 합성 평행-시트 fixture(gap=0.15, pitch=0.1, 동일 normal)로 직접 테스트해 이 결함을 실측으로 발견했고(`subset_count=1`, cut=0이어야 할 리 없는 상황), 위 tangential-vs-normal 비교로 교체해 해결했다(같은 fixture가 정확히 2개 subset으로 분리, 회귀 테스트로 고정).

## 6. Boundary 결정 규칙과 새 파라미터 개수

Edge는 다음 중 하나라도 해당하면 자른다:

    A. residual >= median(residual) + k_MAD * 1.4826 * MAD(residual)     [robust, 이번 replay 자체의 residual 분포에서 유도]
    B. normal_offset > tangential_offset                                   [비율 1.0, 자기-정규화]

**새로 도입한 자유 파라미터는 정확히 2개**, 둘 다 스윕하지 않은 원칙적 상수다:
- `residual_mad_multiplier = 3.0` — 표준 로버스트 통계 관례("3 MAD", 3-sigma 규칙의 MAD 아날로그). fence 자체는 이번 replay의 실제 residual 분포에서 유도되고, 배수만 사전 고정값이다.
- `parallel_sheet_normal_over_tangent_ratio = 1.0` — "접평면 이탈이 접평면 이동보다 크다"는 자연스러운 동률(parity) 값.

## 7. 결정론적 구성

Boundary loop 재구성이나 순서 있는 topology를 전혀 요구하지 않는다(§6) — cut은 그래프 edge 집합일 뿐이고, cut 이후 **남은 그래프의 단순 connected component**가 최종 subset이다(Worklog 96/97과 동일한 `_connected_component_roots` 재사용). Cutting은 노드를 절대 제거하지 않으므로 고립된 노드는 자동으로 자기 자신만의 singleton component가 된다 — Worklog 97처럼 별도의 ownership propagation 메커니즘이 전혀 필요 없다(구조적으로 더 단순하다).

## 8. 합성 fixture 검증 (곡률 계약, §7~§8)

신규 focused 테스트 16개, 전부 통과. 핵심 결과만 요약:

| Fixture | 기대 동작 | 실측 |
|---|---|---|
| 평면(flat sheet) | 1 subset, cut 0 | ✅ 1 subset, cut 0 |
| 원통형 밴드 180° 회전(정상 곡률) | 1 subset 유지, normal gradient는 큼(median > 1.0)이어도 cut 0 | ✅ 1 subset, cut 0, gradient median ≈ 2.0 |
| 1/4 원통(90° 회전) | 1 subset 유지 | ✅ 1 subset, cut 0 |
| 90° 크리즈(진짜 불연속) | 잘림, 두 지배적 subset(각 평면) | ✅ cross-seam cut이 boundary evidence의 >70%, top-2 subset이 전체의 92% |
| 평행 시트(같은 normal, gap=0.15) | normal이 같아도 분리 | ✅ 정확히 2개 subset, 전부 positional 기준으로 절단 |

**곡률 자체는 boundary가 아니라는 핵심 설계 목표는 합성 fixture에서 확실히 검증됐다** — 원통형 밴드는 180° 회전에도 단 하나의 edge도 자르지 않았다(WL97이라면 concentration이 무너져 여러 조각으로 쪼갰을 상황).

## 9. 실제 scene 재실측: WL97 vs 신규 discontinuity-first

**Checkpoint 정정 필요**: Worklog 96/97이 쓴 30k checkpoint(`2dgs_run1/30000`)가 이 배치 시작 시점에 디스크에서 사라져 있었다(사용자의 `output/confirmed/` 정리 과정에서 유실된 것으로 추정). 사용자에게 질의해 **동일 설정으로 재학습**했다 — 최종 1,190,469 surfel, held-out PSNR 28.226/SSIM 0.8994(원본 1,197,331/28.256/0.8997과 오차범위 내로 일치, 재현성 확인됨). 이하 수치는 이 재학습 checkpoint 기준이다.

| 지표 | F. WL97 (region-concentration) | G. Discontinuity-first(신규) |
|---|---:|---:|
| Subset 수 | 104,977 | **7,676** |
| **최대 subset 비율** | **20.84%** | **94.51%** |
| Singleton 비율 | (WL97 자체 정의 다름) | 58.17%(subset 기준), 0.375%(surfel 기준) |
| Spatial edge | 5,132,180 | 5,132,180(동일) |
| Boundary cut edge | (region-coherence rejected 562,875) | **931,606**(spatial의 18.15%) |
| Cut 원인 분해 | 해당 없음 | residual 364,010 / parallel-sheet 635,372 / 둘 다 67,776 |
| Kept edge | (accepted 3,986,975 중 일부만 최종 병합) | **4,200,574**(spatial의 81.85%) |
| Coverage identity | true | true |

**핵심 발견: discontinuity-first가 percolation을 다시 만들어냈다.** WL96(74.70%)·WL97(20.84%, 이번 재측정)보다 오히려 **훨씬 큰 94.51%짜리 거대 subset**이 나왔다. 원인을 실측으로 확인했다:

- Edge의 18.15%만 잘렸고 81.85%(4,200,574개)가 살아남았다 — kNN k=8 그래프에서 평균 차수 ~7이 유지된다.
- **개별 edge를 국소적으로 올바르게 분류하는 것과, 그 cut들이 실제로 그래프를 위상적으로 분리하는 것은 별개의 문제다.** WL97은 "성장 중인 region 전체"의 누적 상태를 매 병합마다 재검사해 병합을 원천 차단하므로 percolation이 구조적으로 불가능하지만, discontinuity-first는 각 edge를 독립적으로(그 edge 하나만 보고) 판정하므로, cut들이 장면 전체에 흩뿌려진 형태(SMOOTH_SURFACE_MODEL_RESIDUAL 뷰 참고 — 바닥·산울타리 전역에 걸쳐 noise성 고residual이 산발적으로 분포)라면 dense한 kNN 그래프는 그 틈을 우회해서 하나의 거대 backbone으로 계속 이어질 수 있다. 이것이 정확히 관측된 현상이다.
- 시각 검토(§10) 결과 **테이블(상판+다리)은 이제 바닥과 같은 거대 subset 색으로 합쳐져 시각적으로 구분되지 않는다** — 즉 원래 동기였던 "곡률만으로 조각나는 문제"는 이 checkpoint에서 확인상 재현되지 않지만(테이블이 더 이상 다리별로 여러 색으로 쪼개지지 않음), 그 대가로 테이블이 바닥·산울타리와 통째로 합쳐졌다.

## 10. Review export 경로 및 정성적 검토

`output/osn_gs_discontinuity_first_surfel_partition/`(전체 scene, crop 없음, Worklog 96/97과 동일 카메라·팔레트 계열):

| view | 관찰 |
|---|---|
| A. ORIGINAL_2DGS_SCENE | 원본 scene, 정상 |
| B. RAW_INTRINSIC_NORMAL | Worklog 97과 동일 |
| C. NORMAL_GRADIENT_MAGNITUDE | 진단 전용, boundary 판정에 미사용(§2) |
| D. SMOOTH_SURFACE_MODEL_RESIDUAL | **테이블(상판·다리)은 짙은 파랑(낮은 residual, 잘 설명됨)**, 바닥·산울타리는 **주황(높은 residual)**이 장면 전역에 산발적으로 분포 — 실제 noisy real-world geometry가 shape operator model을 자주 위반한다는 뜻 |
| E. DETECTED_DISCONTINUITY_BOUNDARY | 마찬가지로 산울타리·바닥 전역에 옅게 퍼진 절단 흔적, 테이블 부분은 낮음 |
| F. WL97_REGION_CONCENTRATION_PARTITION | 테이블 다리가 여러 색으로 쪼개짐(사용자가 지적한 원래 문제, 이 checkpoint에서도 재현됨) |
| G. DISCONTINUITY_FIRST_PARTITION | **테이블·바닥·산울타리 대부분이 같은 적갈색 거대 subset** — 원래 동기(곡률 보존)는 달성했지만 percolation 문제가 다른 형태로 재발 |

**질문에 대한 답 (§11)**: "곡률 있는 테이블 옆면이 정상 회전에도 끊기지 않고 유지되는가, 동시에 진짜 불연속은 여전히 잘리는가?" — **부분적으로 그렇다.** 테이블은 더 이상 다리별로 쪼개지지 않지만, 바닥·산울타리로부터도 분리되지 않아 "정확히 하나의 곡면 subset"이라는 목표에는 도달하지 못했다(전체 거대 덩어리의 일부가 됐을 뿐). "바닥과 산울타리가 WL96의 거대 연결 subset으로 되돌아가지 않는가?" — **되돌아갔다**, 오히려 더 커졌다(74.70%→94.51%).

## 11. Coverage identity 증명

`discontinuity_first_accounting()`의 `coverage_identity_holds=true` — 1,190,469개 전부 assigned, unassigned 0, multiply-owned 0. Ownership kind: `smooth_continuation_component` 1,186,004(99.63%), `fallback_all_local_edges_cut` 3,811(0.32%), `fallback_no_spatial_neighbor` 654(0.05%) — 어떤 surfel도 조용히 사라지지 않았다.

## 12. 재현 명령

```
python scripts/devtools/discontinuity_first_surfel_partition_export.py \
    --checkpoint output/arch_2dgs_coverage_first_surface/2dgs_run1/30000 \
    --out output/osn_gs_discontinuity_first_surfel_partition \
    --device cuda --source-path DATASET --images images_8
```

런타임: F(WL97) 75.8초 + G(discontinuity-first, kNN 두 번 — candidate graph용/shape operator fit용) 141.6초 + rendering, 총 약 4분(RTX 5080).

## 13. 검증

**Focused 테스트 16개 신규**(`tests/test_discontinuity_first_surfel_partition.py`), 전부 통과: 평면 1-subset·cut-0 / **원통형 밴드 180° 회전이 하나의 region으로 유지**(핵심 설계 목표) / 1/4 원통도 분열 없음 / **크리즈가 잘리고 두 지배적 subset으로 분리**(top-2가 전체의 85% 초과) / 크리즈 boundary evidence가 실제 교차 edge 위주(>70%) / **평행 시트가 normal이 같아도 분리** / **평행 시트 기준이 candidate gate 재사용 시 절대 발동하지 않는 회귀 버그 고정**(실제로 발견한 버그) / 부호 뒤집힌 normal이 동일 결과 / 결정론(반복 실행 동일 결과) / residual threshold가 문서화된 median+MAD 공식과 정확히 일치 / coverage 계약(모든 surfel 정확히 한 번 소유, 드롭·중복 없음) / 원본 텐서 불변 / 빈 입력·단일 입력도 coverage 정확 / **새 자유 파라미터가 정확히 2개**(`residual_mad_multiplier`, `parallel_sheet_normal_over_tangent_ratio`)임을 고정 / 이 모듈이 per-surfel normal을 유도하지 않음(AST) / boundary cut 원인이 residual/parallel-sheet로 분리 보고됨.

**전체 회귀**: `1129 passed, 1 skipped, 1 warning, 18 subtests passed in 248.38s` — Worklog 97 기준선(1113 passed)에서 정확히 신규 16개만큼 증가, 실패·회귀 없음.

## 결론 없음

이 worklog는 discontinuity-first가 최종 architecture로 채택될 수 있는지 판단하지 않는다 — 다만 **실측 결과는 명확히 부정적이다**: 합성 fixture에서는 곡률 보존과 불연속 절단 모두 설계 의도대로 동작했지만, 실제 scene에서는 개별 edge 단위 판정이 percolation을 막지 못해 Worklog 96보다도 더 큰 거대 subset(94.51%)을 만들었다. Worklog 97의 region-level global state 검사가 실제로는 percolation 방지에 필수적이었다는 뜻일 수 있다 — 다음 단계는 사용자가 이 결과를 검토한 뒤, discontinuity 신호와 region-level anti-chaining을 **결합**하는 방향(예: WL97의 region-level concentration 검사를 이 배치의 곡률-인식 residual로 교체)을 고려할지, 아니면 완전히 다른 접근을 취할지 결정하는 것이다. 이 worklog 자체는 그 결정을 내리지 않는다.
