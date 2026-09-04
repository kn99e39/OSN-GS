# Worklog 157 — Same-Region TSDF Component Separation Topology and Spatial-Provenance Audit

## 작업 수행

- WL153 field, WL154 Candidate F sample/association/ownership/component ID, WL155 Gaussian ID-region-status mapping, WL156 frontier report를 frozen input으로 읽었다. W153–W156 원본 배열과 production 경로는 수정하지 않았다.
- Region 0, control Region 2/5에서 native 6-face component를 재집계하고, 같은 Region component 사이의 exact lattice separation을 `FACE_TOUCH`, `EDGE_TOUCH`, `CORNER_TOUCH`, `ONE_CELL_AXIAL_GAP`, `OTHER_NEAR_GAP`, `REMOTE`로 분리했다. 각 non-largest component의 nearest same-Region separation record를 NPZ로 보존했다.
- 6/18/26 connectivity는 production 승격 없이 diagnostic graph로만 계산했다. one-cell axial gap의 intervening cell은 WL153 field completeness, zero-surface, sample ownership/Region state로 재분류했다.
- Region 0의 tiny component(`size <= 8`) 중 substantial same-Region component까지 `>16` grid-cell인 remote island을 추출하고, fixed real-scene/common-world camera에서 TSDF support와 nearest Gaussian ID를 함께 투영했다.
- W155에서 이미 검증된 canonical Gaussian visualization pair를 PNG-only로 복사해 W157에도 보존했다. pair는 동일 Gaussian row/geometry/조건을 유지하며, W157 진단 overlay와 섞지 않았다. 모든 출력 directory와 nested visualization directory에 UTF-8 README를 추가·확인했다.

## 결과

### 1. Dominant component와 controls

| Region | owned sample | native component | largest size / fraction | non-largest sample |
|---|---:|---:|---:|---:|
| 0 | 3,133,747 | 41,319 | 2,968,372 / 94.7228% | 165,375 / 5.2772% |
| 2 | 976,227 | 11,704 | 873,414 / 89.4683% | 102,813 / 10.5317% |
| 5 | 738,742 | 11,008 | 684,116 / 92.6055% | 54,626 / 7.3945% |

Region 0의 native component size는 singleton 20,408개, `size <= 8` 38,631개다. 따라서 component 개수는 매우 많지만 sample population은 largest component에 집중되어 있다. 이는 “모든 작은 island가 같은 정도로 구조적”이라는 해석을 지지하지 않는다.

### 2. Exact inter-component separation

Region 0의 non-largest 41,318개 component에 대해 nearest same-Region separation을 기록했다.

| category | component record 수 |
|---|---:|
| FACE_TOUCH | 0 |
| EDGE_TOUCH | 24,763 |
| CORNER_TOUCH | 3,464 |
| ONE_CELL_AXIAL_GAP | 2,579 |
| OTHER_NEAR_GAP | 10,442 |
| REMOTE | 70 |

이 카운트는 component pair 전체의 개수가 아니라 각 non-largest component에 선택된 nearest separation record의 분류다. 별도로 26-neighbor lattice에서 edge-touch pair 24,763개와 corner-touch pair 3,464개가 확인됐다. 같은 Region face-adjacent native failure는 0개이며, component accounting도 누락 없이 완료됐다.

### 3. 6/18/26 topology control

Region 0 component count는 6/18/26에서 각각 `41,319 / 19,713 / 16,079`였다. 6→26 diagnostic adjacency만으로 component count가 25,240개(61.1%) 감소하고, largest 밖 sample fraction은 `5.2772% → 2.1446%`로 감소했다. Region 2와 Region 5도 각각 `11,704 / 6,550 / 5,648`, `11,008 / 5,265 / 4,370`으로 같은 경향을 보였다.

이는 edge/corner digital adjacency가 native 6-face fragmentation의 실질 원인 중 하나임을 보여주지만, 26 graph를 production connectivity로 승격하거나 component를 merge하지 않았다.

### 4. True one-cell TSDF gap attribution

Region 0에서 one-cell axial gap instance는 30,059개였다.

- `NOT_AUTHORITATIVE`: 0개
- `AUTHORITATIVE_NOT_ZERO_SURFACE`: 29,147개, affected component 14,045개, affected sample 3,059,458개
- `ZERO_SURFACE_DIFFERENT_REGION`: 729개, affected component 662개
- `ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS`: 183개, affected component 186개
- `OTHER_EXISTING_CONTRACT_STATE`: 0개

gap record에 참여한 component union은 14,537개, 3,061,049 samples(Region 0 owned population의 97.6802%)다. 특히 대부분의 intervening cell이 authoritative이면서 zero-surface가 아니므로, W156의 generic frontier/field-starvation 비율을 component split 비율로 재사용할 수 없고, 실제 local TSDF separation evidence를 별도로 봐야 한다. 반대로 다른 Region 또는 unowned/ambiguous 상태의 gap은 local gap의 존재는 증명하지만 Region 0 내부 surface continuity의 직접 증거로 과해석하지 않았다.

### 5. Remote same-Region island과 common-world 위치

tiny를 `size <= 8`, substantial을 `size > 8`, remote를 nearest substantial same-Region component의 L∞ separation `>16` grid cells로 정의했다. Region 0 remote tiny island은 170개 component, 271 samples이며 non-largest sample의 약 0.16%다. 따라서 remote island이 fragmentation을 지배한다는 판정은 지지되지 않는다.

`DSC07960.JPG`, `DSC08003.JPG`, `DSC08043.JPG` matched common-world overlay에서는 remote mark가 tabletop 전체나 vase 접촉부에 단일 cluster로 모이지 않고, 주로 table leg/base, 하부 pavement/grass, 화면 주변의 lower/side geometry와 peripheral background에 희소하게 나타났다. 이는 시각적 위치 검토 결과이며, 좌표만으로 tabletop·leg·background semantic label을 자동 확정한 것은 아니다.

Gaussian spatial extent도 같은 결론과 호환된다. Region 0 Gaussian은 248,086개, W155 existing graph의 structural core는 237,256개, concentration은 0.925009이다. largest TSDF component nearest Gaussian은 179,303개, remote tiny island nearest Gaussian은 107개였다. mapping hash `06c9e1cbc730f06581895b32ad683e8822c7626eb3de9017fa8f83aaf0248bce`와 W154 join exactness를 함께 기록했다.

### 6. W156과의 reconciliation

W156의 historical architecture verdict `TSDF_FIELD_STARVATION_DOMINANT`와 generic frontier accounting은 보존했다. W157에서는 그것을 component split fraction으로 사용하지 않고, 같은 Region component 사이의 exact separation과 intervening field state를 다시 계산했다. 따라서 W157의 결과는 W156 verdict를 덮어쓰거나 되돌리는 것이 아니라, W156에서 열어 둔 topology/spatial provenance 원인을 세분화한 것이다.

## 평가 및 판정

- synthetic topology/state contracts: `all_pass = true`
- production connectivity repair, 18/26 promotion, merge, bridge, gap fill, dilation/smoothing, ownership 변경: 모두 수행하지 않음
- canonical Gaussian visualization contract: matched `Original Scene`/`Observed-Occluded` pair 보존, 1,190,469 rows 및 geometry 유지, W157 추가 view는 별도 diagnostic overlay
- architecture verdict: **`MIXED_COMPONENT_STRUCTURE`**

판정 근거는 (1) edge/corner digital adjacency가 6-face 분리를 크게 줄이는 정량 증거와 (2) authoritative local one-cell TSDF gap이 다수 존재하는 정량 증거가 동시에 성립하기 때문이다. remote island은 희소하며, mechanical implementation bug contract는 통과했다.

## 산출물

- [W157 report](../../output/157_same_region_tsdf_component_separation_topology_spatial_provenance/worklog_157_report.json)
- [component separation records](../../output/157_same_region_tsdf_component_separation_topology_spatial_provenance/component_separation_records.npz)
- [one-cell gap records](../../output/157_same_region_tsdf_component_separation_topology_spatial_provenance/one_cell_gap_records.npz)
- [remote island provenance](../../output/157_same_region_tsdf_component_separation_topology_spatial_provenance/remote_islands.json)
- [matched review views](../../output/157_same_region_tsdf_component_separation_topology_spatial_provenance/review_views/README.md)
- [Gaussian spatial extent views](../../output/157_same_region_tsdf_component_separation_topology_spatial_provenance/gaussian_spatial_extent/README.md)
- [mandatory Gaussian pair](../../output/157_same_region_tsdf_component_separation_topology_spatial_provenance/mandatory_gaussian_visualization_pair/README.md)

## 남은 위험

- remote island의 semantic identity는 현재 common-world 시각 검토 수준이며, event-level physical-sheet/voxel closure lineage가 없으므로 자동 semantic attribution으로 승격할 수 없다.
- 6/18/26 결과는 topology 원인 분해용 diagnostic control이다. 현재 architecture에서 18/26 connectivity를 채택하거나 local gap을 메우는 구현 결정은 보류한다.
- W153 closure lineage와 per-event surface element provenance가 회복되지 않는 한, `MIXED_COMPONENT_STRUCTURE`를 physical-sheet correctness 판정으로 확대하지 않는다.
