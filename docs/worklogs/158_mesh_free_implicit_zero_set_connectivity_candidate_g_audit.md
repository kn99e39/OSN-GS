# Worklog 158 — Mesh-Free Implicit Zero-Set Connectivity Contract 및 Candidate G 감사

## 상태

**완료 — `ZERO_SET_CONNECTIVITY_CONTRACT_GAP`**

## 수행 내용

- WL153 authoritative TSDF scalar field와 WL154/WL155 Gaussian Region/TSDF ownership, WL156 causal accounting, WL157 separation records를 frozen 입력으로 재사용했다.
- 전역 mesh를 만들지 않고 bounded dense block마다 `skimage.measure.marching_cubes(method="lewiner")`의 local triangle incidence만 읽었다. 보존한 node는 `(lower lattice vertex key, axis)` canonical zero-crossing entity ID이며, local geometry/face mesh는 즉시 폐기했다.
- triangle co-incidence만으로 Candidate G incidence graph를 만들고, all-eight-corner authority·linear zero-crossing·same-Region ownership을 유지했다. 6/18/26 adjacency는 diagnostic control로만 남겼다.
- exact lattice-edge 공유, corner-degenerate, 가까운 평행 zero surface, authoritative nonzero gap, missing authority gap, different-region contact, unowned/ambiguous contact을 합성 A–H 계약으로 검사했다.
- WL157 Region 0 `EDGE_TOUCH`/`CORNER_TOUCH` pair를 Candidate G zero-crossing entity와 재대조하고, 기존 6-face cross-component relation도 별도 재검토했다.
- real Region 0/2/5 incidence NPZ와 report를 생성했다. Region 0 ambiguity가 확인되어 Stop Condition A에 따라 Boundary First/WL139 replay 및 Candidate-G real-scene overlay는 실행하지 않았다. frozen canonical Gaussian pair는 PNG-only로 복사했다.

## 결과 및 수치

| Region | owned cells | zero-crossing nodes | incidence edges | Candidate G components | ambiguous cells | largest component fraction | outside-largest |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,133,747 | 4,085,954 | 10,570,501 | 89,330 | 59,509 | 84.0880% | 14.0127% |
| 2 | 976,227 | 1,200,178 | 3,190,142 | 22,796 | 19,781 | 78.4968% | 19.4625% |
| 5 | 738,742 | 1,041,683 | 2,629,406 | 19,887 | 15,217 | 81.2672% | 16.6720% |

`ambiguous cells`는 raw local zero-set corner-degenerate triangle뿐 아니라 한 cell에 복수 graph component가 귀속되는 경우를 중복 없이 accounting한 값이다. raw ambiguous incidence cell은 Region 0/2/5에서 각각 `10,772/3,594/3,507`개였다.

WL157 Region 0의 `EDGE_TOUCH` pair `24,763`개 중 `6,719`개는 실제 shared zero-crossing entity를 양쪽 cell에서 공유하여 `TOPOLOGICALLY_CONNECTED_ZERO_SET`으로 재분류됐다. 나머지 `18,044`개는 `MERELY_EDGE_OR_CORNER_NEAR`였다. `CORNER_TOUCH` `3,464`개는 모두 `MERELY_EDGE_OR_CORNER_NEAR`였고, exact corner zero는 별도 ambiguity 규칙으로 bridge하지 않았다. native 6-face cross-component relation은 `0`개였다.

WL157의 one-cell gap `30,059`개와 authoritative-but-not-zero-surface `29,147`개를 유지했고, Candidate G가 gap을 bridge하지 않음을 확인했다. Region ownership, TSDF field, zero-surface eligibility, native 6-face component, Boundary First/WL139 fitter는 변경하지 않았다.

## 합성 계약 평가

합성 A–H는 모두 통과했다.

- A: face-adjacent planar zero set은 하나의 incidence component로 연결됐다.
- B: diagonal cell pair는 generic 18-neighbor가 아니라 실제 shared lattice-edge entity가 양쪽에 있을 때만 연결됐다.
- C: exact corner-degenerate contact은 ambiguity/disconnection으로 보수 처리됐다.
- D: 가까운 parallel zero surfaces는 분리됐다.
- E/F/G/H: authoritative nonzero, missing authority, different Region, unowned/ambiguous state는 모두 bridge 금지 guard를 통과하지 못했다.

## 평가 및 판정

local zero-set incidence 정의 자체는 합성 예제와 ordinary crossing에 대해 deterministic하다. 그러나 real frozen field에서 세 Region 모두 exact corner-degenerate incidence와 multi-patch cell ambiguity가 남았다. 이 상태에서 cell 간 연결을 하나로 선택하면 scalar field가 결정하지 않는 topology를 발명하게 되므로 Candidate G를 production 후보로 승격하지 않았다.

따라서 최종 판정은 **`ZERO_SET_CONNECTIVITY_CONTRACT_GAP`**이다. 이는 count 감소만으로 성공을 주장하지 않는 판정이며, Candidate G의 Region 0/2/5 graph는 진단 산출물로만 보존된다. 조건부 Boundary First/WL139 replay와 Candidate-G common-world overlay는 Stop Condition A 때문에 실행하지 않았다.

## 검증

- W158 focused tests + W157 regression: **9 passed**.
- Candidate G synthetic A–H: **all_pass=true**.
- global mesh intermediate: **생성하지 않음**.
- W158 visualization output: PNG-only copy, output 내부 PPM **0개**.
- 모든 생성 visualization 하위 항목에 view 의미와 legend를 설명하는 README를 작성했다.
- 상세 JSON report: [`worklog_158_report.json`](../../output/158_mesh_free_implicit_zero_set_connectivity_candidate_g/worklog_158_report.json)
- Candidate G graph records: `candidate_g_region_000000.npz`, `candidate_g_region_000002.npz`, `candidate_g_region_000005.npz`

## 남은 위험 및 후속 조건

- exact zero corner의 tie-breaking/국소 topology 계약이 추가로 정해지기 전에는 real Candidate G를 연결성 개선 근거로 사용할 수 없다.
- Candidate G incidence가 해결되어도 zero-set topology가 physical sheet identity를 보장하지 않는 문제는 별도 계약이다.
- canonical Gaussian pair는 보존했지만, Stop Condition A로 인해 새 Candidate-G real-scene overlay는 의도적으로 만들지 않았다.
