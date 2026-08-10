# Worklog 89 — full-region face → membership-incidence boundary 최종 go/no-go

## 상태

**완료 — 최종 NO-GO.** Worklog 88의 candidate-independent membership-cut 방향은 유지하되, 잘못된 global PCA-UV rotation과 chart-unit induced-subgraph의 largest outer-face 선택을 제거했다. 요청된 full-region observed-face incidence 계약으로 7개 real region을 다시 측정했으며, 이 결과로 boundary-first visible constructor 재설계를 종료한다.

## 수행 작업

- `torch_full_region_surface_face_topology.py`
  - Worklog 82의 full-region `same_surface` adjacency와 기존 covariance local normal/tangent frame만 사용한다.
  - 각 evidence vertex에서 실제 neighbor 3D 방향을 local tangent plane에 투영하고 local angle로 cyclic order를 만든다.
  - stable ID는 같은 angle/distance 및 순회 표현의 동률만 해소하며 adjacency나 연결 순서를 만들지 않는다.
  - chart-unit membership을 보기 전에 full-region directed half-edge orbit 전체에서 observed surface face와 edge별 face incidence를 복원한다.
- `torch_chart_unit_face_incidence_partition_boundary.py`
  - full-region face 중 모든 vertex가 unit member인 face를 unit-supported face로 삼는다.
  - unit face incidence 2인 edge는 chart interior, 1인 oriented half-edge는 chart-unit boundary로 분류한다.
  - 기존 physical termination/crease/observation frontier provenance는 보존한다. provenance가 없고 full-region face incidence가 2인 continuous-surface cut만 `partition_seam`으로 분류한다.
  - boundary half-edge의 in/out degree가 각각 정확히 1일 때만 모든 독립 loop를 복원한다. `outer_boundary`/기존 계약의 `interior_boundary` role을 모두 보존하며 open/branching/non-manifold는 fail-closed한다.
  - interior loop가 증명되면 현 untrimmed PCA-UV/6x6 NURBS가 이를 묵시적으로 버리지 못하게 materialization을 중단한다.
- 기존 `torch_chart_unit_topology_partition_boundary.py` 경로는 corrected 구현의 compatibility import로 바꿔 금지된 global PCA/induced largest-face 구현을 제거했다.
- Worklog 79 coverage → 기존 PCA-UV → 기존 6x6 NURBS → held-out 평가 체인은 변경하지 않았다. threshold, kNN, normal source, residual, NURBS, UV ablation과 새 boundary heuristic은 추가하지 않았다.

## 계약 검증

신규 테스트는 다음을 직접 확인한다.

- rigid rotation 뒤에도 local covariance frame rotation system의 full-face incidence가 동일하다.
- chart membership 이전에 full-region face 두 개와 공유 edge의 양면 incidence가 복원된다.
- full face 양쪽 중 한쪽만 unit-supported일 때 실제 graph edge가 `partition_seam`이 된다.
- candidate 0 unit도 candidate anchor 없이 unit face incidence까지 평가된다.
- 모든 독립 outer loop를 보존하고, membership cut이 만든 interior loop도 기존 role로 보존한 뒤 현 untrimmed domain에서는 fail-closed한다.
- 모든 boundary segment가 실제 full-region `same_surface` edge다.

Focused regression: **86 passed in 6.41s**. 최종 NO-GO 조건이므로 canonical 통합과 repository-wide regression은 실행하지 않았다.

## 7개 real region replay

Checkpoint `output/extent_ab/val64/baseline_compatible/2900`, cap 2048, evidence 3526점. 산출물은 `output/extent_ab/val89/chart_unit_face_incidence_partition_boundary_replay.json`이다.

| region | evidence | unit | coherent % | recoverable % | physical-only % | mixed % | seam-only % | valid % | extrap. % | unsafe % | unresolved % | held-out p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 93 | 3 | 92.47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 92.47 | - |
| 1 | 519 | 24 | 89.40 | 0.58 | 0.58 | 0 | 0 | 0 | 0 | 0.58 | 88.82 | 2.675 |
| 2 | 510 | 22 | 96.47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 96.47 | - |
| 3 | 92 | 9 | 80.43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 80.43 | - |
| 4 | 1035 | 56 | 86.57 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 86.57 | - |
| 5 | 375 | 7 | 93.33 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 93.33 | - |
| 6 | 902 | 57 | 82.71 | 0.33 | 0.33 | 0 | 0 | 0 | 0 | 0.33 | 82.37 | 0.576 |

Evidence-weighted 전체 결과:

- coherent chart-unit evidence: **3108/3526 = 88.15%**
- cut-boundary recoverable: **6/3526 = 0.170%**, coherent evidence 대비 **6/3108 = 0.193%**
- physical-only / mixed / seam-only: **6 / 0 / 0 evidence**
- valid_supported / extrapolative / unsafe / unresolved: **0 / 0 / 6 / 3102 evidence**
- recovered domain evidence-weighted held-out p95: **2.675 local-spacing units**
- 178 unit 중 증명된 boundary loop는 10개이며 모두 `outer_boundary`였다. real replay에서 `interior_boundary`는 없었고, 테스트에서는 보존/fail-closed 계약을 확인했다.
- unresolved unit 사유: full-region face topology non-manifold/open **167**, loop는 닫혔으나 coverage/occupancy가 거부 **8**, full face incidence 1인데 physical provenance가 없어 seam으로 오분류하지 않고 거부 **1**.

## candidate 0 unit의 실제 fate

Worklog 89의 **full-region physical provenance를 unit membership으로 조회하는 정의**에서는 candidate 0이 **38 unit / 124 evidence**다. 모두 candidate gate 없이 평가했으며, 35 unit/115 evidence는 full-face topology non-manifold/open, 2 unit/6 evidence는 coverage/occupancy 거부, 1 unit/3 evidence만 physical-only domain으로 회수됐으나 unsafe였다. seam-only 회수는 0이다.

사용자가 지목한 Worklog 87 본문의 64개는 저장 artifact로 재현되지 않는다. Worklog 87 artifact의 실제 per-unit candidate-0 집합은 **71 unit / 164 evidence**이며 Worklog 88에서도 이 기록 오류를 정정했다. 동일 region/unit index 71개를 이번 corrected replay에 전부 대응했으므로 본문의 임의 64개 subset도 누락되지 않는다.

| corrected full-face-incidence fate | unit | evidence |
|---|---:|---:|
| open/non-manifold full-face topology | 61 | 134 |
| Worklog 79 coverage 실패 | 4 | 12 |
| occupancy 실패 | 3 | 9 |
| full incidence 1이나 physical provenance 없음 — fail-closed | 1 | 3 |
| physical-only 회수, unsafe | 2 | 6 |

Worklog 87에서 candidate 0이던 71개 중 이번 full-region provenance 정의로도 candidate 0인 것은 26개/60 evidence이고, 45개/104 evidence는 full-region candidate provenance를 가진다. 이 분류 변화는 admission gate가 아니라 provenance 조회 범위 교정이며, 71개 전체는 동일 topology 경로로 평가됐다.

## 평가와 최종 판정

Worklog 88 결과는 최종 판정 근거로 supersede한다. corrected 방식은 candidate나 sparse boundary fragment 없이 full-region observed face topology에서 membership cut을 직접 계산하므로 요청된 first-class `partition_seam` 계약을 정확히 시험한다. 그러나 coherent evidence의 **99.807%**가 closed coverage-valid domain에 도달하지 못했고, 도달한 0.193%도 전부 unsafe였다. 안전한 mixed/seam-only domain과 `valid_supported`는 하나도 없다.

따라서 **최종 NO-GO**다. Region→Charts를 canonical production path로 통합하지 않으며, 현재 visible Gaussian evidence에 대한 boundary-first visible-constructor redesign을 종료한다. 이 결론 뒤에 다른 boundary 실험을 시작하지 않는다.

## 남은 위험

- full-region local rotation이 manifold observed-face incidence를 증명하지 못한 unit은 보수적으로 미해결 상태다. 이를 임의 closure나 geometry heuristic으로 낮추지 않는다.
- 현 PCA-UV/6x6 untrimmed NURBS는 증명된 interior boundary를 표현하는 계약이 없으므로 이를 버리지 않고 fail-closed한다.
- 실험 모듈은 canonical 경로에 연결되지 않았고 visible Gaussian training, region ownership, Worklog 82~84 assembly/coherence, Worklog 79 coverage는 변경되지 않았다.
