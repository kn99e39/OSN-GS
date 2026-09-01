# Worklog 144 — Per-view renderer surface correspondence 및 physical-sheet oracle audit

## 상태 및 목적

완료. Worklog 143에서 확인한 renderer `depth_median` event 의미론을 그대로 사용하고, WL141의 frozen polygon/camera/ROI를 변경하지 않은 격리 진단이다. 비교 단위를 `WL127 point -> 여러 camera depth`에서 `각 camera가 독립적으로 생성한 renderer median-event surface cloud -> 다른 camera cloud`로 바꿔, WL142의 same-point 불일치가 단순 sampling artifact인지 실제 physical-sheet misassociation인지 확인했다.

자동 `Surface Membership`, 새 support-selection heuristic, `mu` tuning, KNN membership, connected-component/region growing, NURBS, continuation, Occluded Surface, SH, Candidate B 변경은 수행하지 않았다.

## 구현 충실도

- 입력: WL127 raw Visible Surface, WL139의 frozen `h=0.012105485424399376`, `mu=0.03631645627319813`, WL141의 세 ROI 및 세 frozen camera/polygon, WL142/WL143 report, frozen checkpoint.
- 각 polygon의 모든 integer pixel 중 valid positive `depth_median`만 사용해 event cloud를 만들었다. 별도 cloud로 먼저 저장했고, 합친 cloud는 combined visualization에만 사용했다.
- world 위치는 WL143에서 검증한 camera/view-space depth reconstruction을 그대로 재사용했다. source pixel로의 x/y/depth round-trip assertion을 cloud 생성 중 수행했다.
- 국소 방향은 고정 `k=12` world-space local PCA로 계산했다. normal은 point 선택·제거·membership 판정에 사용하지 않았다. sign-invariant angle만 diagnostic distribution으로 기록했다.
- 3D PNG는 기존 physical-chart 좌표로 표시되지만, PLY/NPZ 및 pairwise metric은 canonical Gaussian Scene world XYZ를 사용한다. 3D 표시에서만 `max_points=16000` display thinning을 적용했고 metric/full PLY는 thinning하지 않았다.
- 모든 입력 path/hash와 frozen scale을 root/case report에 기록했다. historical WL141 `MASK_ONLY` row는 cloud attribution에만 사용했으며 cloud 또는 mask를 바꾸는 데 사용하지 않았다.

## Per-view renderer event cloud 결과

각 cloud의 `valid_polygon_pixel_count`와 `valid_median_event_count`는 다음과 같다.

| case | camera 1 | camera 2 | camera 3 |
|---|---:|---:|---:|
| tabletop top | `DSC08050.JPG`: 4,506 / 4,506 | `DSC08045.JPG`: 4,510 / 4,510 | `DSC08017.JPG`: 3,587 / 3,587 |
| curved table rim | `DSC08043.JPG`: 28,191 / 28,191 | `DSC07960.JPG`: 9,649 / 9,649 | `DSC08003.JPG`: 23,622 / 23,622 |
| paver ground | `DSC08081.JPG`: 10,627 / 10,627 | `DSC07968.JPG`: 2,182 / 2,182 | `DSC07960.JPG`: 4,641 / 4,641 |

각 cloud별 world-space bounding box, centroid, extent, pixel count, polygon hash, point hash, camera identity는 case report와 `per_view_event_clouds/<camera>/`에 저장했다.

## Pairwise surface agreement

아래 값은 reciprocal `A->B`와 `B->A` nearest-neighbour 거리의 concatenation에 대한 `median / p95`이다. threshold나 support selection으로 사용하지 않았다.

| case / pair | world distance | distance / h |
|---|---:|---:|
| tabletop 08050↔08045 | `3.192 / 3.512` | `263.72 / 290.11` |
| tabletop 08050↔08017 | `1.437 / 1.866` | `118.67 / 154.12` |
| tabletop 08045↔08017 | `2.384 / 2.866` | `196.95 / 236.71` |
| curved rim 08043↔07960 | `0.400 / 1.951` | `33.07 / 161.17` |
| curved rim 08043↔08003 | `0.022 / 0.399` | `1.84 / 32.97` |
| curved rim 07960↔08003 | `0.419 / 1.815` | `34.64 / 149.91` |
| paver 08081↔07968 | `4.520 / 6.487` | `373.39 / 535.85` |
| paver 08081↔07960 | `4.898 / 7.891` | `404.63 / 651.84` |
| paver 07968↔07960 | `0.759 / 9.113` | `62.70 / 752.74` |

전체 continuous distribution, A→B/B→A 분리 값, normalized distance 및 raw arrays는 `pairwise_surface_distance_distributions.npz`와 case report에 있다.

## Local differential agreement

고정 `k=12` local PCA normal의 sign-invariant nearest-cloud angle median/p95(도)는 다음과 같다.

| case / pair | A→B | B→A |
|---|---:|---:|
| tabletop 08050↔08045 | `28.22 / 42.35` | `20.33 / 63.75` |
| tabletop 08050↔08017 | `22.24 / 35.75` | `43.13 / 68.78` |
| tabletop 08045↔08017 | `73.33 / 88.91` | `15.36 / 77.55` |
| curved rim 08043↔07960 | `53.54 / 86.17` | `20.70 / 83.52` |
| curved rim 08043↔08003 | `21.18 / 81.75` | `18.63 / 75.59` |
| curved rim 07960↔08003 | `27.49 / 87.63` | `40.96 / 85.53` |
| paver 08081↔07968 | `15.96 / 72.60` | `80.01 / 88.16` |
| paver 08081↔07960 | `56.86 / 86.25` | `73.33 / 88.09` |
| paver 07968↔07960 | `13.79 / 43.85` | `23.32 / 82.44` |

고정 10-degree angle histogram도 export했으며, 이를 단일 angle gate로 사용하지 않았다.

## WL127 MASK_ONLY attribution

각 historical WL141 `MASK_ONLY` point에 대해 세 independent event cloud까지의 1/2/3순위 거리를 계산했다. `nearest / second / third`의 distance/h median은 다음과 같다.

| case | nearest | second | third |
|---|---:|---:|---:|
| tabletop top | `636.93` | `715.30` | `810.42` |
| curved table rim | `0.81` | `1.64` | `30.33` |
| paver ground | `2.17` | `33.17` | `111.08` |

이 수치는 exact historical point identity를 요구하지 않는 diagnostic attribution이다. row-ID hash와 full distance matrix는 `wl127_mask_only_attribution_distributions.npz`에 저장했다.

## Fixed h / mu overlap accounting

연속 nearest-other distance를 먼저 기록한 뒤, frozen `h`와 `mu`를 descriptive reference radius로만 사용했다. radius sweep와 new membership threshold는 없다.

- tabletop: 각 cloud의 `h` 및 `mu` 기준 `shared_by_neither`가 모두 `1.0`이었다.
- curved rim: `h` 기준 cloud별 `shared_by_another`는 `0.366`, `0.244`, `0.457`, `shared_by_both`는 `0.015`, `0.039`, `0.022`였다. `mu` 기준으로도 `shared_by_both`는 `0.047`, `0.091`, `0.067`에 그쳤다.
- paver: 각 cloud의 `h` 및 `mu` 기준 `shared_by_neither`가 모두 `1.0`이었다.

## Cross-view reprojection 및 human review

각 case에 대해 모든 source cloud를 세 target camera image에 visibility culling 없이 재투영했다. 따라서 source camera→source image와 source camera→두 other image 총 9개 raw overlay가 case별로 있다. target frozen polygon outline도 함께 표시했으며, raw PLY/3D view와 같은 source geometry를 유지했다.

직접 검토 결과:

- `tabletop_top_oracle`: camera별 event가 ground/tabletop/brace로 갈라지며 한 common tabletop sheet에 붙지 않는다.
- `curved_table_rim_oracle`: 일부 `DSC08043↔DSC08003` cloud가 가까운 부분은 있으나, `DSC07960`을 포함한 큰 영역은 front-side/depth-layer 및 tabletop 구조로 갈라진다. raw reprojection에서도 단일 rim으로 유지되지 않는다.
- `paver_ground_oracle`: camera별 event가 grass/tabletop/front-table 구조로 갈라지며 paver sheet로 일치하지 않는다.

따라서 세 case의 classification은 모두 다음으로 고정한다.

**C. SEMANTIC_MASK_MISASSOCIATION**

이는 한 scalar threshold만으로 정한 결과가 아니다. common-frame 3D cloud, source-to-all-target raw reprojection, pairwise continuous distance, local differential distribution을 함께 검토한 결과다.

## Architecture attribution

- **PROMOTED**: renderer-native per-view median-event cloud construction, independent cloud-to-cloud continuous proximity accounting, WL127 point-to-independent-cloud diagnostic attribution.
- **RETAINED**: WL141 polygon/camera/ROI/MASK_ONLY, WL139 `h`/`mu`, WL143 depth reconstruction convention.
- **REJECTED**: per-view proximity를 final membership으로 승격, KNN membership, percentage vote, connected component/region growing, NURBS/continuation/Occluded Surface.
- **OPEN**: automatic support/membership, physical-sheet identity의 publishable definition, final multi-view aggregation.

이번 결과는 `MASK_ONLY`가 실제 independent renderer observations를 의미하지 않는다는 쪽의 강한 정성·정량 증거를 제공하지만, 자동 Surface Membership을 설계하거나 canonical architecture를 완성했다는 뜻은 아니다.

## 검증 및 산출물

- focused tests: `19 passed in 1.82s`
- module `py_compile`: 통과
- 실제 고정 CUDA audit: `COMPLETED_DIAGNOSTIC_AUDIT`, `COMPLETED_CASE_REVIEW`, `failures=[]`
- 산출물: `output/per_view_renderer_surface_correspondence_physical_sheet_oracle_audit/`
- 주요 파일: root report, 각 case `case_report.json`, `per_view_event_clouds/`, `3d_review/`, `cross_view_reprojection_overlays/`, pairwise/attribution/overlap NPZ

canonical renderer/checkpoint, 161 camera calibration, WL127–143 historical artifact, Candidate B는 변경하지 않았다.
