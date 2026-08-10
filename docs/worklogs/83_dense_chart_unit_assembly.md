# Worklog 83: chart-scale topology/assembly — worklog 82 micro-component 조립

## 목적

Worklog 82는 region-owned evidence 위에 evidence-scale surface-consistency micro-component를 만들어 처음으로 valid_supported chart 16개를 얻었지만, 364개 component 중 91%가 no_chart로 끝나는 심한 파편화(중앙값 3~6점)를 확인하고 판정 (B)로 닫았다. 이번 배치는 threshold/kNN/normal/residual 재조정 없이, worklog 82의 micro-component를 **보수적 atomic 증거**로 취급하고 그 위에 **chart-scale topology/assembly layer**를 신설한다 — 어떤 micro-component들이 aggregate 증거로 하나의 defensible parametric chart를 이루는지 결정한다.

상류 계약 유지: Region은 ownership container, worklog 82 same-surface 추출은 미변경 atomic 증거, sparse representative topology는 macro provenance, typed crease/frontier separator, worklog 80 dense chart support, worklog 79 coverage, 현재 PCA-UV/6×6 NURBS, visible Gaussian 학습 — 전부 그대로.

## 구현

신규 `osn_gs/surface/torch_dense_chart_unit_assembly.py`: **component-level adjacency는 단일 pointwise edge가 아니라 aggregate 증거로만 구성**한다.

- **Candidate gate**: 두 micro-component의 최근접 점 거리가 `2.5 × (두 component 자체 spacing 평균)` 이내일 때만 후보 — worklog 72 `_connect`의 2.5x multiplier를 그대로 재사용(신규 상수 아님). 이 gate를 통과 못하면 edge 자체를 기록하지 않는다(candidate 아님).
- **Typed crease veto**: 두 component의 최근접 점이 worklog 80의 이미 타입된 arc 중 서로 다른 arc에 속하면, 다른 신호가 아무리 좋아도 즉시 거부 — worklog 82와 동일한 veto를 component 스케일에 재적용.
- **세 개의 독립 aggregate 신호** (전부 기존 계약 재사용, 재조정 없음):
  1. **normal alignment 평균**: candidate gate 안의 모든 cross-component 점 쌍의 normal alignment 평균이 0.85 이상(worklog 82와 동일 threshold).
  2. **반복된 same_surface correspondence**: cross-component 점 쌍 중 worklog 82의 same_surface 기준(정렬≥0.85 AND mutual residual≤0.35, 같은 정의)을 개별적으로 만족하는 쌍의 개수가 3 이상.
  3. **observed support occupancy**: 두 component의 최근접 점을 잇는 합성 edge에 worklog 76 `measure_edge_support_occupancy`(지금까지 disclosure 전용이었던 함수, 이번에 처음 acceptance 신호로 사용)를 적용해 내부 빈 bin이 없어야 함 — 빈 구간을 건너뛰는 edge는 이 신호를 통과하지 못한다.
- **판정**: crease veto 없음 AND 3개 신호 중 **2개 이상** 통과해야 `ACCEPTED`. 통과 못하면 `AMBIGUOUS`로 명시 disclosure(병합도 폐기도 아님). candidate gate 자체를 통과 못하면 `NOT_CANDIDATE`(edge 미생성).
- Chart unit = ACCEPTED edge만으로 만든 component-level connected component. 짝이 없는 micro-component는 그대로 단일-component chart unit(worklog 82가 이미 내적으로 정합함을 검증한 것과 동일).
- **fit 품질은 이 모듈이 읽는 입력이 아니다** — NURBS/UV/held-out 어떤 것도 파라미터에 없음(전용 테스트로 함수 시그니처를 직접 검사해 고정).

`tests/test_dense_chart_unit_assembly.py` 7개: 가까운 두 coherent half가 실제로 1개 chart unit으로 합쳐짐, 멀리 떨어진 component는 candidate조차 안 됨, crease arc가 가까운 두 component 사이에서도 병합을 막음, 방향이 어긋난 가까운 component는 ambiguous로 남고 병합 안 됨, non_manifold로 표시된 component는 조립에서 제외됨, 함수 시그니처에 fit/NURBS/jacobian/p95/held_out 관련 입력이 전혀 없음(구조적 검증), worklog 82의 실제 출력이 이 모듈의 유효한 입력임(통합 검증, 조립 전후 evidence 총합 불변).

### 파이프라인 재구성

신규 `scripts/devtools/dense_chart_unit_assembly_replay.py`: region-owned evidence → worklog 82 micro-component(미변경) → **worklog 83 assembly(신규)** → chart-unit별 worklog 80 dense chart support(미변경) → worklog 79 coverage(미변경) → PCA-UV(worklog 81 확정, 미변경) → 6×6 NURBS(미변경) → held-out 평가. 7개 real region 전체 실행.

## 실측 (real baseline_compatible@2900, 7개 region 전체)

| reg | evid | micro | units | unresolved% | edge relations | chart classes |
|---:|---:|---:|---:|---:|---|---|
| 0 | 93 | 12 | 3 | 7.5% | accepted 10 / ambiguous 3 / crease 7 | valid 1, extrap 1, no_chart 1 |
| 1 | 519 | 53 | 24 | 10.6% | accepted 36 / ambiguous 30 | no_chart 21, extrap 3 |
| 2 | 510 | 41 | 22 | 3.5% | accepted 21 / ambiguous 8 / crease 12 | valid 1, extrap 2, no_chart 19 |
| 3 | 92 | 12 | 9 | 19.6% | accepted 3 / ambiguous 8 / crease 3 | extrap 1, no_chart 8 |
| 4 | 1035 | 117 | 56 | 13.4% | accepted 76 / ambiguous 113 | no_chart 56 |
| 5 | 375 | 19 | 7 | 6.7% | accepted 16 / ambiguous 10 | no_chart 7 |
| 6 | 902 | 110 | 57 | 17.3% | accepted 59 / ambiguous 115 | valid 2, extrap 2, unsafe 3, no_chart 50 |

**전 7-region 합계: micro-component 364개 → chart unit 178개(52% 감소). ALL-CHART-UNIT: valid_supported 4, extrapolative 9, unsafe_geometry 3, no_chart 162.**

no_chart 비율은 91.0%(162/178)로, worklog 82의 90.9%(331/364)와 **거의 변화가 없다** — 이는 예상 밖의, 중요한 신호다(아래 원인 판별 참고).

### assembly가 실제로 무엇을 합치는지

region당 evidence 대비 size≥4 chart-unit 후보로 회수된 비율: **region 0 89.2%, 1 86.7%, 2 93.1%, 3 75.0%, 4 81.5%, 5 92.8%, 6 74.8%** — worklog 82의 심한 파편화(중앙값 3~6점)와 달리, evidence의 대부분이 이제 chart 후보 규모(4점 이상)로 회수된다. 실제 병합 사례: region 6 unit 4는 micro-component 19개(2~10점씩)를 합쳐 239점짜리 단일 chart unit이 됐고, region 5 unit은 최대 237점까지 합쳐진다.

**gap bridging 여부를 직접 감사**: 7개 region 전체에서 ACCEPTED edge 219개 중 occupancy 신호가 False인 채로(즉 normal+correspondence만으로) 수락된 edge는 **0개**다 — 채택된 모든 병합은 occupancy 신호(빈 구간 없음)를 실제로 통과했다. correspondence 신호(개별 점 쌍이 point-level same_surface 기준을 만족하는 쌍의 개수 ≥3)는 후보 503쌍 중 465쌍이 0, 3 이상은 3쌍뿐이었다 — 즉 실전에서 채택된 edge의 압도적 다수(219개 중 216개)는 **normal alignment 평균 + occupancy** 두 신호로 통과했다. 이는 결함이 아니라 설계 의도와 일치한다: "aggregate 증거로, 단일 pointwise edge가 아니라" 판정하라는 지시대로, 개별 점 쌍은 대부분 point-level 엄격 기준을 통과하지 못해도(worklog 81/82가 이미 확인한 국소 잡음 때문) aggregate 정렬과 occupancy는 여전히 두 component가 같은 sheet의 일부라는 것을 지지한다.

## 원인 판별: no_chart 비율이 왜 그대로인가

병합 후에도 남은 no_chart 178개 중 162개의 원인:

| 원인 | 개수 |
|---|---:|
| `chart_unit_too_small`(병합 후에도 4점 미만) | 71 |
| `no_sparse_macro_topology_for_arc_typing`(worklog 80 arc-typing에 필요한 sparse macro node 부족) | 38 |
| `dense_chart_support_coverage_failed`(worklog 79 계약) | 27 |
| `dense_chart_support_self_intersecting` | 18 |
| `no_dense_boundary_support` | 8 |

**병목이 이동했다.** worklog 82에서는 병목이 "micro-component가 너무 작아서 chart가 안 됨"(파편화)이었다. 이번엔 evidence의 75~93%가 4점 이상 chart-unit 후보로 회수되는데도(파편화 자체는 해소), **worklog 80의 sparse macro-topology arc typing(38건)과 dense chart boundary 자체의 구성(coverage 실패 27건 + self-intersecting 18건, 합 45건)이 더 크고 복잡해진 assembled chart unit에서 안전하게 닫히지 못한다.** 즉 assembly 메커니즘 자체는 작동하지만, assembly가 만들어낸 더 큰 evidence 덩어리를 최종 chart로 완성하는 하류 단계(macro-topology 해상도, boundary 구성)가 그 규모를 감당하지 못한다.

valid_supported 개수는 worklog 82의 16개보다 **오히려 4개로 줄었다**: micro-component 단계에서 valid였던 작은 chart 다수가 assembly로 인해 이웃과 병합되면서 chart가 커졌고, 그 결과 PCA-UV+6×6 NURBS(미변경, 재검토 금지 대상)로 hold-out p95≤4.0을 만족시키기 어려워졌다(예: region 1 unit 1은 95점, p95 7.67; region 2 unit 4는 85점, p95 9.13). 반면 region 6 unit 3(74점)은 병합 후에도 valid_supported(p95 3.58)를 유지했다 — **큰 assembled chart도 유효할 수 있다는 것을 보여주는 실증 사례**지만 흔하지는 않다.

## 판정

**(B) local surface unit은 실재하고 aggregate 증거로 chart 스케일까지 조립 가능하지만(파편화는 해소됨), 그것을 최종 chart로 안전하게 완성할 만큼 관측 topology가 아직 충분하지 않다.**

세부적으로:

- **assembly 메커니즘 자체는 성공적이다.** evidence의 75~93%가 이제 4점 이상 chart-unit 후보로 회수되고, 최대 239점짜리 단일 unit까지 정당하게 병합됐으며, 채택된 219개 edge 전부가 occupancy 검증(빈 구간 없음)을 통과했다 — gap bridging은 발생하지 않았다. 이는 (C)(evidence가 근본적으로 boundary-first 가정에 부적합)를 다시 한번 기각한다.
- **그러나 (A)(Region→Chart를 그대로 canonical로 채택)도 지지되지 않는다.** no_chart 비율이 90.9%→91.0%로 사실상 그대로다. 병목이 "micro-component가 너무 작음"에서 **"assembled chart unit이 worklog 80의 sparse macro-topology arc-typing 해상도와 dense chart boundary 구성 능력을 넘어섬"**으로 이동했을 뿐이다. valid_supported도 16→4로 줄었는데, 이는 병합 자체의 결함이 아니라 더 커진 evidence 규모에서 기존 PCA-UV+6×6 fit(재검토 금지 대상)이 hold-out 기준을 만족시키기 어려워졌기 때문이다.
- region 4/5는 이번에도 valid chart를 하나도 만들지 못했다(각각 56/7개 unit 전부 no_chart) — worklog 80의 ambiguous branching/open topology 분류와 일치하며, 이번 라운드로도 바뀌지 않았다.

## 검증

`tests/test_dense_chart_unit_assembly.py` 신규 7개 전부 통과. 관련 기존 테스트(`test_dense_surface_consistency_components.py`, `test_dense_parametric_chart_support.py`, `test_region_owned_full_evidence.py`, `test_boundary_support_spacing.py`) 재실행 54개 전부 통과. **Region→Chart 계층을 canonical로 교체하지 않았으므로**(판정이 A가 아님) 지시대로 전체 regression은 실행하지 않았다. threshold/kNN/normal/residual/parameter ablation은 수행하지 않았고, 재사용한 값(0.85, 0.35, 2.5x, correspondence count 3)은 전부 기존 계약에서 그대로 가져왔다. hull·PCA rectangle·bounding box·alpha shape·강제 분할·gap bridging·region merge·shape-specific fallback은 도입하지 않았고, fit 품질을 병합 판단에 사용하지 않았음을 함수 시그니처 테스트로 고정했다. visible Gaussian photometric 학습과 상류 region ownership은 손대지 않았다.
