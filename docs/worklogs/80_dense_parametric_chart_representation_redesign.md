# Worklog 80: parametric chart 표현 재설계 — topology와 geometry 역할 분리

## 목적

Worklog 79는 real visible-surface 실패가 구조적임을 확정했다: sparse representative accepted cycle(3~7 노드)이 chart의 **기하적 boundary**로 재사용되는데, region이 소유한 evidence(93~1035점)는 훨씬 큰 공간 스케일에 있어 evidence의 89.1~99.8%가 chart domain 바깥에 남았다. 이번 배치는 진단을 더 하지 않고 **표현 자체를 재설계**한다 — topology 역할과 geometric chart support 역할을 분리한다.

상류 계약은 전부 유지했다: region formation/ownership, sparse representative accepted topology(= topology 추상), covariance_normal, full_evidence_spacing, worklog 77 predicate correction, physical-boundary connectivity, physical termination과 parametric chart boundary의 구분.

## 재설계

신규 `osn_gs/surface/torch_dense_parametric_chart_support.py`.

- **Topology 역할(sparse accepted cycle)**: perimeter의 **순환 순서**와 각 arc의 **typed frontier provenance**(physical_termination / crease / observation_frontier / partition_seam)만 제공한다. chart의 기하적 범위로는 **절대 쓰지 않는다.**
- **Geometry 역할(dense boundary-support candidate)**: worklog 77의 보정된 predicate(미변경)가 region-owned observed evidence에서 뽑은 후보가 기하를 제공한다. 모든 chart 정점이 **관측된 Gaussian**이다.

이 분리가 타당한 근거는 실측이다. real baseline_compatible@2900에서 dense candidate의 공간 범위 / owned evidence 범위 = **0.966~1.020**인 반면, 이들이 대체하는 representative는 **0.148~0.667**이다. 즉 dense candidate는 이미 evidence와 같은 스케일에 있다.

구성(전 과정 boundary-first, hull·PCA rectangle·bounding box·alpha shape·shape-specific fallback 일절 없음):

1. sparse cycle과 dense candidate를 region 고유 canonical tangent frame으로 투영.
2. 각 dense candidate를 가장 가까운 sparse **arc**에 귀속 — **topology가 chart 소속을 제약하는 지점**이며, candidate는 그 arc의 typed provenance를 상속한다.
3. arc 내부에서 단조 정렬(실데이터 산포로 polyline이 되돌아가지 않도록 binning). **bin 해상도는 candidate 자신이 차지하는 span에서 유도**한다(아래 실측 결함 참고).
4. sparse 순환 순서로 arc를 이어 붙인다. **representative 자신은 최종 loop의 정점이 아니다** — arc를 배치하고 유형을 부여할 뿐 chart를 경계 짓지 않는다.
5. 3D nonplanarity와 2D crossing을 분리한 `evaluate_closed_loop_geometry`(worklog 71)로 검증한 뒤, **fitting 이전에** worklog 79 chart-domain coverage 계약을 적용한다.

Dense support가 없으면 sparse polygon으로 **fallback하지 않고 fail-closed**한다. dense candidate가 없는 arc는 typed provenance를 유지하되 `evidence_backed=False`로 명시 기록해, loop이 갖지 않은 관측 지지를 주장하지 않게 했다.

**다중 chart**는 region 자신의 accepted topology가 증명할 때만 허용한다: 2-core(차수 ≤1 노드를 반복 제거)가 **서로소인 연결 성분 2개 이상**으로 갈릴 때만. 한 2-core 성분 안에 여러 cycle이 얽힌 경우는 ambiguous branching이며 **분할하지 않는다**.

### 구현 중 발견·수정한 자체 결함

첫 실행에서 region 6이 dense candidate 218개 중 **16개만** 유지하고 coverage 72.7%로 실패했다. 원인은 arc 내부 binning의 bin 개수를 **sparse chord 길이**에서 유도한 것 — sparse chord는 representative 스케일(evidence 범위의 0.15~0.67배) 객체인데 candidate는 evidence 스케일에 있으므로, 이 모듈이 없애려는 스케일 불일치를 binning 단계에서 그대로 재도입한 셈이다. bin 해상도를 **candidate 자신의 투영 span**에서 유도하도록 수정했고, region 6은 16→88 정점, coverage 72.7%→56.1%로, region 0은 6→17 정점, 59.1%→31.2%로 바뀌었다.

## 7개 region before/after (baseline_compatible@2900)

BEFORE = worklog 78/79 production(sparse cycle이 기하 boundary), AFTER = 재설계. 이후 downstream 체인(coverage → parameterization validity → 6×6 NURBS → held-out 평가)은 양쪽 동일하다.

| reg | evid | BEFORE 정점 | BEFORE domain 밖 | BEFORE 분류 | AFTER 정점 | AFTER domain 밖 | AFTER p95 | AFTER 분류 |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 0 | 93 | 3 | 88.2% | `chart_domain_does_not_cover_evidence` | **17** | **31.2%** | 9.12 | extrapolative |
| 1 | 519 | 3 | 94.8% | `chart_domain_does_not_cover_evidence` | **84** | **31.0%** | 19.92 | extrapolative |
| 2 | 510 | 3 | 92.9% | `chart_domain_does_not_cover_evidence` | **92** | **29.0%** | 18.68 | extrapolative |
| 3 | 92 | 3 | 82.6% | `chart_domain_does_not_cover_evidence` | **20** | **32.6%** | 5.49 | extrapolative |
| 4 | 1035 | 0 | — | no_chart(ambiguous branching) | 0 | — | — | no_chart |
| 5 | 375 | 0 | — | no_chart(open topology) | 0 | — | — | no_chart |
| 6 | 902 | 3 | 99.6% | `chart_domain_does_not_cover_evidence` | 88 | **56.1%** | — | no_chart(coverage 미달) |

| 집계 | BEFORE | AFTER |
|---|---|---|
| chart-domain coverage 통과 | **0 / 5** | **4 / 5** |
| chart boundary 정점 수 | 3, 3, 3, 3, 3 | 17, 84, 92, 20, 88 |
| domain 밖 evidence | 82.6~99.6% | **29.0~56.1%** |
| valid_supported | 0 | 0 |
| extrapolative | 0(전부 coverage 이전 탈락) | 4 |
| unresolved/no-chart | 7 중 2(+coverage 탈락 5) | 7 중 3 |
| Jacobian near-degenerate | 0 | **0** |
| local folding(fitted surface) | — | 0.0000~0.0045 |
| held-out p95 | 측정 불가(coverage 탈락) | 5.49~19.92 |

## 남은 실패의 위치가 바뀌었다

재설계는 **worklog 79가 지목한 구조적 결함을 실제로 해소한다**: chart domain이 이제 evidence와 같은 스케일에 있고(정점 3→17~92, domain 밖 89~99.8%→29~56%), 4개 region이 처음으로 coverage 계약을 통과한다. Jacobian degenerate 0, fitted-surface folding ≤0.45%로 기하도 안전하다.

그러나 통과한 4개는 전부 `extrapolative`이며, 이제 그 원인이 **분리되어 보인다**. region 자신의 evidence를 PCA-UV로 파라미터화했을 때:

| reg | neighborhood preservation | UV collision | **raw evidence triangle fold** |
|---:|---:|---:|---:|
| 0 | 0.62 | 2 | **30.9%** |
| 1 | 0.53 | 2 | **36.0%** |
| 2 | 0.70 | 2 | **21.5%** |
| 3 | 0.69 | 0 | **21.1%** |

**region evidence 자체가 단일 정규 chart가 아니다** — UV-인접 삼각형의 21~36%에서 3D normal 부호가 어긋난다. Worklog 69도 같은 것을 봤지만 그때는 extent 문제와 뒤섞여 있었다. extent 문제를 제거한 지금, 이것이 **분리된 지배 원인**으로 남는다.

그러면 다중 chart가 답인가? **기존 topology는 그것을 증명하지 못한다.** `independent_chart_components`(2-core의 서로소 성분)는 region 0/1/2/3/4/6에서 전부 **1**, region 5에서 **0**이다. 즉 어떤 region도 accepted topology 상 두 개의 독립 cycle로 갈리지 않는다. 근거 없이 분할하지 않는다는 제약에 따라 **다중 chart는 채택하지 않았다.**

region 4(1035점)는 2-core 단일 성분에 cyclomatic ≥2인 ambiguous branching, region 5(375점)는 pendant를 가진 open topology로, 둘 다 chart cycle 자체가 없어 fail-closed 유지다.

## 판정

**(3) 기존 region/evidence 표현은 evidence-backed parametric chart domain을 공급할 수 있지만, 유효한 *단일* chart domain은 공급하지 못한다.**

세부적으로:

- **재설계 자체는 viable하고 채택할 가치가 있다.** topology/geometry 역할 분리는 worklog 79의 구조적 결함을 실제로 해소했다 — coverage 통과 0→4, domain 밖 89~99.8%→29~56%, 정점 3→17~92, 전 구간 기하 안전. 이 부분은 명확한 개선이다.
- **그러나 real 데이터에서 사용 가능한 chart를 만들지는 못한다.** valid_supported는 여전히 **0**이다. 실패 원인이 "chart가 evidence를 담지 못함"(parameterization 이전)에서 **"evidence가 단일 정규 chart가 아님"(parameterization 단계)** 으로 이동했을 뿐이다.
- **다중 chart 표현이 필요한 region은 이번 데이터에서 증명되지 않았다.** 모든 region의 accepted topology가 단일 2-core 성분이므로, 다중 chart는 정당한 근거 없이는 구성할 수 없다. 즉 판정 (2)는 **현재 topology로는 지지되지 않는다.**

따라서 남은 병목은 chart 구성 알고리즘이 아니라 **region 자체가 단일 chart가 아니라는 사실**과 **기존 accepted topology가 그 분할을 증명할 해상도를 갖지 못한다는 것**이다. 이 두 가지를 동시에 해결하려면 region 형성 단계에서 chart 단위를 정의하거나(현재 계약상 변경 금지), topology 해상도를 evidence 밀도까지 올려야 한다 — 이번 배치는 둘 다 임의로 선택하지 않았다.

## 검증

신규 `tests/test_dense_parametric_chart_support.py` 14개(재설계의 핵심 주장인 "기하는 dense support에서 오고 representative는 정점이 아니다", sparse polygon 단독으로는 coverage 실패한다는 대조 검증, arc별 typed provenance 보존, dense support 없을 때 sparse fallback 금지, coverage 실패·2노드 미만·자기교차 fail-closed, 미지지 arc 명시 기록, 2-core 서로소 성분만 다중 chart로 인정). 재설계 계약이 완성됐으므로 지시대로 **full regression**을 실행했다(결과는 커밋 메시지 참조). hull·PCA rectangle·bounding box·alpha shape·강제 폐쇄·gap bridging·region merge·shape-specific fallback·ownership clipping은 도입하지 않았고, normal/scale/fitting capacity는 재검토하지 않았으며 visible Gaussian photometric 학습은 손대지 않았다.
