# Worklog 88: chart-unit topology partition boundary — historical verdict

**상태 정정:** 이 판정은 Worklog 89가 supersede한다. Candidate-independent membership-cut 방향은 유지되지만, 여기의 global PCA rotation/induced largest outer-face construction은 요청된 full-region face-incidence 계약이 아니므로 최종 근거가 아니다.

## 목적

Worklog 87 구현은 승인됐지만 최종 판정 근거는 기각됐다. 두 결함은 명확했다.

1. seam이 Worklog 77 boundary-support candidate/fragment를 anchor로 요구해 candidate 0 unit을 실제로 시도하지 않았다.
2. multi-fragment 연결 순서를 stable ID가 정해, topology evidence가 아닌 임의 chain 하나의 실패를 전체 가능성 부재로 잘못 해석했다.

이번 라운드는 Worklog 87 daisy-chain을 수정하거나 튜닝하지 않았다. 별도 모듈에서 chart-unit membership이 full region evidence-scale `same_surface` topology에 만드는 cut을 직접 parametric boundary로 복원했다.

## 구현

신규 `osn_gs/surface/torch_chart_unit_topology_partition_boundary.py`:

- Worklog 82 기본값(k=8, cap=12, normal alignment 0.85, mutual residual 0.35, typed crease veto)을 그대로 사용해 **full region adjacency를 region당 한 번** 만든다.
- Worklog 83 chart-unit membership으로 이 graph의 induced subgraph를 만든다. Worklog 77 candidate는 admission 조건이 아니라 candidate 수와 `physical_termination` provenance를 위한 진단 입력일 뿐이다.
- 기존 PCA-UV로 관측 정점을 투영한 뒤 각 정점의 실제 graph neighbor를 tangent angle로 cyclic ordering해 rotation system을 만든다. 모든 directed graph edge를 half-edge로 순회하고, 실제 graph edge만으로 닫힌 face cycle을 복원한다. 가장 큰 embedded outer face를 택하되 이는 후보 graph cycle 사이의 선택이며 hull/rectangle/forced closure가 아니다.
- stable ID는 동일 angle/동일 면적처럼 topology-equivalent한 경우의 tie-break와 cycle canonicalization에만 사용한다. stable ID로 adjacency나 연결 순서를 만들지 않는다.
- loop edge endpoint에 기존 Worklog 77 candidate 또는 sparse typed arc provenance가 있으면 `physical_termination`/`crease`/`observation_frontier`를 보존하고, 나머지는 `partition_seam`으로 명시한다. 따라서 physical-only, mixed, seam-only가 모두 같은 메커니즘의 결과다.
- 모든 boundary edge는 full-region `same_surface` graph의 실제 edge다. self-intersection, observed-support occupancy, Worklog 79 coverage를 순서대로 적용하며 하나라도 실패하면 fail-closed한다. fit 품질과 원하는 patch 수는 topology 선택에 사용하지 않는다.

전용 replay `scripts/devtools/chart_unit_topology_partition_boundary_replay.py`는 기존과 동일하게 Region ownership → Worklog 82 micro-component → Worklog 83 assembly → Worklog 84 coherence → 신규 membership-cut boundary → Worklog 79 coverage → PCA-UV/6×6 NURBS → held-out 평가를 7개 real region 전체에 적용한다.

## 단위 검증

신규 테스트 8개는 다음을 고정한다.

- concave cycle도 실제 graph edge만 따라 복원하며 hull로 넓히지 않는다.
- stable ID를 바꿔도 선택된 geometric edge set은 바뀌지 않는다.
- open path는 강제로 닫지 않는다.
- candidate 0인 3-node topology도 candidate gate에서 탈락하지 않고 closed-loop/coverage 단계까지 실제 평가된다.
- candidate admission을 강제로 0으로 만든 dense disc도 seam-only domain을 materialize한다.
- 2-node candidate-0 unit은 실제 평가 후 topology 부족으로 fail-closed한다.
- dense physical support typing과 모든 최종 segment의 full adjacency 소속을 확인한다.

## 7개 real region replay

Checkpoint: `output/extent_ab/val64/baseline_compatible/2900`, cap 2048, evidence 3526점.

| reg | evid | units | coherent% | cut recoverable% | physical-only% | mixed% | seam-only% | valid% | unsafe% | unresolved% | held-out p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 93 | 3 | 92.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 92.5 | - |
| 1 | 519 | 24 | 89.4 | 1.7 | 1.7 | 0.0 | 0.0 | 0.0 | 1.7 | 87.7 | 2.595 |
| 2 | 510 | 22 | 96.5 | 2.0 | 2.0 | 0.0 | 0.0 | 2.0 | 0.0 | 94.5 | 2.572 |
| 3 | 92 | 9 | 80.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 80.4 | - |
| 4 | 1035 | 56 | 86.6 | 1.9 | 1.6 | 0.0 | 0.3 | 1.6 | 0.3 | 84.6 | 3.754 |
| 5 | 375 | 7 | 93.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 93.3 | - |
| 6 | 902 | 57 | 82.7 | 0.4 | 0.4 | 0.0 | 0.0 | 0.4 | 0.0 | 82.3 | 1.885 |

전체 evidence-weighted 결과:

- coherent chart-unit evidence: **3108/3526 = 88.15%**
- cut-boundary recoverable: **43/3526 = 1.22%**, coherent evidence 대비 **1.38%**
- physical-only / mixed / seam-only: **40 / 0 / 3점 = 1.13% / 0% / 0.085%**
- valid_supported / extrapolative / unsafe / unresolved: **31 / 0 / 12 / 3065점 = 0.88% / 0% / 0.34% / 86.93%**
- recovered domain들의 evidence-weighted held-out p95: **3.754 local-spacing units**
- materialized unit은 8개로 Worklog 87과 총량이 동일하다. 신규 표현은 seam-only 1개를 실제로 만들었지만 전체 yield를 바꾸지 못했다.

## candidate 0 unit의 실제 fate와 Worklog 87 수치 정정

동일 checkpoint/assembly/Worklog 77 코드를 다시 계산하면 candidate 0 unit은 Worklog 87 본문의 64개가 아니라 **71개(164 evidence)**다. 저장된 Worklog 87 replay artifact도 71개이며, 전부 member 2~3개라 `extract_dense_boundary_support`의 `n < 4` 계약상 candidate 0이다. 따라서 “71개 중 64개가 candidate 0”이라는 Worklog 87 문장은 재현되지 않는 기록 오류다. 이번 replay는 더 큰 실제 집합 71개를 전부 시도했으므로 사용자가 지목한 64개를 누락하지 않는다.

| candidate-0 outcome | unit | evidence |
|---|---:|---:|
| induced edge 1개(2-node), closed topology 없음 | 49 | 98 |
| induced edge 2개(3-node open), closed topology 없음 | 12 | 36 |
| closed triangle은 찾았으나 Worklog 79 coverage 실패 | 5 | 15 |
| closed triangle은 찾았으나 occupancy 실패 | 3 | 9 |
| 회수: typed physical-only, unsafe | 1 | 3 |
| 회수: **seam-only**, unsafe | 1 | 3 |

즉 candidate 0은 모두 실제 평가됐다. **71개 중 2개(6 evidence)만 회수**, 69개는 topology/coverage/occupancy 근거로 fail-closed했다. 후보 0 evidence tested/recovered는 전체 evidence 대비 **4.65% / 0.17%**다. 이는 Worklog 87의 “candidate가 없으면 시도 불가” 논리를 폐기하면서도, candidate-free topology가 production-scale coverage를 제공하지 못함을 직접 확인한 결과다.

## 최종 판정

**NO-GO.** 이번 구현은 요구된 first-class `partition_seam` 계약을 만족한다. candidate 0 unit도 같은 topology path에서 평가되고 실제 seam-only domain도 생성됐으며, stable ID chain은 완전히 제거됐다. 그럼에도 coherent evidence의 **98.62%가 closed, coverage-valid chart-unit cut boundary를 만들지 못했다**. 안전한 `valid_supported`는 coherent evidence의 **1.00%**뿐이다.

따라서 실패는 더 이상 physical candidate prerequisite나 Worklog 87의 임의 daisy-chain에 귀속할 수 없다. full evidence-scale same-surface topology에서 chart-unit membership cut을 직접 사용해도 대다수 coherent evidence가 닫힌 parametric domain을 형성하지 못한다는 NO-GO 조건이 성립한다.

지시대로 Region→Charts canonical 통합은 하지 않았고, 이 결과 뒤에 다른 boundary 실험을 제안하거나 시작하지 않는다. visible-constructor redesign line은 여기서 종료한다.

## 검증

- 구현 전 repository-wide 기준선: `931 passed, 1 skipped, 1 warning, 18 subtests passed in 250.18s`
- 신규 전용 테스트: `8 passed`
- Worklog 79/82-87 관련 focused regression: `87 passed in 6.58s`
- 7-region CUDA replay 2회 동일 집계 확인, 최종 artifact: `output/extent_ab/val88/chart_unit_topology_partition_boundary_replay.json`
- NO-GO이므로 변경 후 full regression은 실행하지 않았다. production canonical path, PCA-UV/6×6 NURBS, visible Gaussian training, Region ownership, Worklog 82/83/84 계약은 변경하지 않았다.
