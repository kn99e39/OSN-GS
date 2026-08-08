# Worklog 79: constructor-wide chart 실패 귀속과 chart-domain coverage 계약

## 목적

Worklog 78이 physical termination과 parametric chart frontier의 의미 구분을 복원한 뒤에도 real `baseline_compatible@2900`은 7개 region에서 5개 chart를 materialize하면서 valid 1 / 강한 extrapolative 4로 남았다. 이번 배치는 one-factor ablation으로 돌아가지 않고, **chart topology부터 NURBS fitting까지를 한 번의 constructor-wide pass로 함께 평가**해 남은 실패를 귀속하고, 근거가 분명한 constructor-level 결함이 있으면 같은 배치에서 고친다.

covariance_normal, full_evidence_spacing, worklog 77 discretization correction, dense physical connectivity certificate, region formation/ownership/visible Gaussian 학습, physical/parametric 의미 분리는 전부 유지했다.

## 1. Worklog 78 chart-frontier eligibility 계약 검증

**구조적으로는 계약이 지켜진다.** `construct_region_parametric_chart_boundaries`에서 boundary edge를 만드는 `adjacency`는 **오직 `region.internal_accepted_edge_ids`**로만 구성되고, halfedge candidate는 `reason_by_node`(노드→reason 라벨)로만 쓰인다. 즉 relation half-edge는 **edge를 만들 수 없고 라벨만 제공**하며, 모든 boundary edge는 기존 accepted topology에 존재하고 leftmost-turn walk와 `validate_simple_closed_loop`를 통과해야 한다. ambiguous relation 증거가 topology 지지 없이 chart boundary가 되는 경로는 없다.

실측(baseline_compatible@2900) 5개 eligible chart의 segment kind는 `physical_termination` 2 + `crease` 1(region 0,2,3) 또는 `crease` 3(region 1,6)이며, **`observation_frontier`/`partition_seam`으로 승격된 ambiguous 증거는 0건**이다. 따라서 worklog 78 변경이 ambiguous 증거를 조용히 유효 boundary로 만든 사례는 이번 데이터에 존재하지 않는다.

다만 **별개의 계약 공백**을 확인했다(아래 3절): eligibility는 "accepted topology가 simple cycle을 이루는가"만 보고, **그 chart domain이 자신이 fitting될 evidence를 실제로 담는지는 전혀 보지 않는다.** 3-node 삼각형은 자명하게 simple cycle이므로 무조건 eligible이 된다.

## 2. 7-region constructor 실패 매트릭스

| region | rep member | owned evidence | evid/member | chart node | domain 밖 evidence | boundary/evidence extent | UV nbp | held-out p95 | 귀속 | 지배 단계 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 3 | 93 | 31.0 | 3 | **95.7%** | 0.159 | 0.62 | 4.39 | `boundary_chart_extent_mismatch` | before parameterization |
| 1 | 3 | 519 | 173.0 | 3 | **91.3%** | 0.410 | 0.53 | 19.74 | `boundary_chart_extent_mismatch` | before parameterization |
| 2 | 4 | 510 | 127.5 | 3 | **91.8%** | 0.622 | 0.70 | 14.69 | `boundary_chart_extent_mismatch` | before parameterization |
| 3 | 3 | 92 | 30.7 | 3 | **89.1%** | 0.667 | 0.69 | 6.43 | `boundary_chart_extent_mismatch` | before parameterization |
| 4 | 7 | 1035 | 147.9 | — | — | — | — | — | `ambiguous_branching_topology` | no chart |
| 5 | 4 | 375 | 93.8 | — | — | — | — | — | `genuinely_open_or_unsupported_topology` | no chart |
| 6 | 3 | 902 | 300.7 | 3 | **99.8%** | 0.148 | 0.50 | 14.40 | `boundary_chart_extent_mismatch` | before parameterization |

Jacobian near-degenerate는 **전 chart 0건**, local fold fraction은 0.00~0.63%로 전부 안전하다. 즉 **fitting geometry는 건강하고, 실패는 fitting이나 parameterization이 아니라 그 이전 단계에서 발생한다.**

### Worklog 78 수치 정정

worklog 78은 region 3을 `valid_supported`(p95 정확히 4.0)로 보고했다. 이번 pass는 **결정론적 spatial holdout**으로 평가했고 region 3은 train p95 4.94 / **held-out p95 6.43** / full p95 4.46이다. 즉 worklog 78의 4.0은 **fitting이 이미 본 evidence 기준**이며, held-out 기준으로는 region 3도 extrapolative다. **held-out 기준 valid_supported는 5개 중 0개**로 정정한다.

## 3. 지배 원인

**5개 extrapolative chart 전부 단일 원인이다: chart domain이 자신이 fitting될 evidence를 담지 못한다.**

원인 사슬은 constructor 안에서 완결된다.

1. region의 **representative topology는 3~7 노드**이고 공간적으로 evidence extent의 **0.15~0.67배**만 차지한다.
2. ownership propagation(`_propagate_with_evidence_gating`)은 normal/residual만 게이팅하고 **in-plane 거리 상한이 없으므로**, 3개 representative짜리 region이 **93~1035개** full-cloud 점을 소유한다(evid/member 31~301배).
3. chart boundary는 1의 representative에서 만들어져 **3-node 삼각형**이 된다.
4. worklog 67 경로가 이 삼각형 boundary를 2의 전체 owned evidence와 **짝지어** NURBS를 fit한다.
5. 결과적으로 evidence의 **89.1~99.8%가 domain 바깥**이고, 그 바깥 전부가 외삽이다.

즉 **boundary(=representative scale)와 fitting 입력(=ownership scale)이 서로 다른 스케일인데, 이 짝지음이 타당한지 검사하는 곳이 없었다.**

**region 4/5(no chart)**는 region 자신의 accepted-edge graph 구조로 분류했다(강제 폐쇄·임의 분할 없음).

- **region 5**: degree 최소 1인 **pendant 노드**가 있고 2-core가 3 미만 → cycle을 담을 수 없는 **genuinely open/unsupported topology**. 다중 chart 문제가 아니다.
- **region 4**: 7 노드에 degree 2/4/5, 2-core가 단일 성분이고 cyclomatic number ≥ 2 → 여러 cycle이 **한 성분 안에 얽힌 ambiguous branching**. 독립된 2개 cycle로 깨끗이 분리되지 않으므로 "다중 chart 필요"로 단정할 근거도 없다. 따라서 `ambiguous_branching_topology`로 두고 fail-closed 유지.

## 4. 적용한 constructor 교정: chart-domain coverage 계약

기하를 만들어내지 않고 고칠 수 있는 구체적 결함은 **4번 단계의 무검증 짝지음**이다. `fit_region_owned_full_evidence_patch`에 계약을 추가했다.

```
outside = evidence_outside_chart_domain_fraction(boundary_points, full_evidence_points)
if outside > MAX_EVIDENCE_OUTSIDE_DOMAIN_FRACTION:  # 0.5
    -> state = "chart_domain_does_not_cover_evidence"  (fail-closed, fitting 이전)
```

containment는 **boundary loop 자신의 best-fit 평면**에서 측정한다(합집합에 다시 PCA를 맞추면 먼 evidence가 자기 판정 프레임을 회전시키므로). point-in-polygon은 기존 `interior_within_boundary`를 재사용했다.

**이것은 외삽을 줄이려는 threshold 튜닝이 아니다.** 실측 위반값이 82.6~99.6%로 0.5에서 한참 떨어져 있어 0.5~0.85 어느 값을 써도 판정이 동일하며(전용 테스트로 고정), 0.5는 "자기 evidence의 과반도 담지 못하는 chart는 그 evidence의 chart가 아니다"라는 **가장 약한 형태의 계약**으로 택했다. containment가 정의되지 않으면(boundary 3점 미만/evidence 0) `None`을 반환해 이 사유로는 실패시키지 않는다.

부수적으로 `tests/test_region_owned_full_evidence.py`의 기존 fixture가 **`grid[:4]`(모두 x=-0.5인 공선 4점, 면적 0인 "boundary")** 를 쓰고 있었음을 발견해 실제 perimeter loop으로 교정했다. 이 fixture는 정확히 이번에 고친 결함(면적 없는 domain에 전체 격자를 fitting하고 `materialized`로 보고)을 그대로 인코딩하고 있었다.

## 5. Before / after (real baseline_compatible@2900, 7 region 전체 재실행)

| 항목 | before | after |
|---|---|---|
| eligible parametric chart boundary | 5 | 5 (불변) |
| materialized parametric chart surface(representative scale) | 5 | 5 (불변) |
| materialized physical surface | 0 | 0 (불변) |
| region-owned full-evidence fit `materialized` | 5 | **0** |
| `valid_supported`(held-out 기준) | **0** (worklog 78 보고 1은 train 기준) | 0 |
| `extrapolative` | 5 | **0** |
| `chart_domain_does_not_cover_evidence` | — | **5** (82.6/94.8/93.2/82.6/99.6%) |
| no-chart region | 2 (region 4,5) | 2 (불변) |

chart boundary 자체와 representative-scale chart surface는 변하지 않는다 — 그 표면은 자기 3~4개 representative에 대해서는 여전히 타당하다. 바뀐 것은 **"이 chart가 region의 owned evidence를 대표한다"는 주장**이 이제 거짓일 때 typed fail-closed된다는 점이다.

## 6. 판정

**현재 visible NURBS constructor는 real 데이터에서 사용 가능하지 않으며, region/chart 표현 자체의 재설계가 필요하다.**

근거는 이번 pass가 한 번에 보여준 다음 사실이다.

- 5개 chart 전부 동일 원인(`boundary_chart_extent_mismatch`)이고, 실패 단계가 전부 **parameterization 이전**이다. UV 왜곡도 NURBS 모델 불일치도 지배 원인이 아니며 Jacobian/folding은 전부 안전하다 — **fitting을 아무리 개선해도 이 실패는 사라지지 않는다.**
- 원인은 국소 버그가 아니라 표현의 구조다: **topology를 담는 단위(3~7 representative)와 evidence를 담는 단위(93~1035 point)가 31~301배 어긋나 있고, 전자가 후자의 공간 범위를 0.15~0.67배만 덮는다.** worklog 79의 계약은 이 어긋남을 **정직하게 드러내 fail-closed**할 뿐, 없앨 수는 없다.
- 나머지 2개 region은 chart를 만들지도 못한다(하나는 pendant를 가진 open topology, 하나는 ambiguous branching).
- worklog 75(normal source)·76(scale)·77(predicate)이 각각 병목 후보를 제거했고, 이번 pass는 남은 실패가 **chart 표현 단위의 스케일 불일치**임을 constructor 전체 관점에서 확정한다.

재설계가 향해야 할 지점은 명확하다: chart domain과 그 domain이 대표하는 evidence가 **같은 스케일에서 정의**되어야 한다 — 즉 representative topology를 evidence 밀도까지 끌어올리든지, ownership을 chart domain 범위로 제한하든지, 또는 region당 다중 chart를 정당한 근거로 구성하든지. 이번 배치는 그중 어느 것도 임의로 선택하지 않는다(hull·PCA rectangle·bounding box·강제 분할·gap bridging·region merge·shape-specific fallback 미도입).

## 검증

`tests/test_region_owned_full_evidence.py`에 chart-domain coverage 계약 테스트 5개 추가(작은 삼각형 domain fail-closed, 실제 enclosing boundary는 여전히 materialize, 0.5~0.85 전 구간 판정 불변, 정의 불가 시 이 사유로 실패하지 않음). 변경 모듈을 소비하는 focused 스위트 전체 **84 passed, 2 subtests passed**. 신규 state는 기존 `RegionOwnedFullEvidenceFit.state`에 대한 가산이고 production 소비자 중 이 state로 분기하는 곳은 없으며(`!= "materialized"` 검사는 모두 fail-closed로 동작), 지시대로 full pytest는 실행하지 않았다.
