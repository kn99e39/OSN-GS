# Worklog 91 — MULTILAYER_OR_VOLUMETRIC의 temporal + lineage 귀속

## 상태

**완료 — Decision A. true center-distribution multilayer가 압도적이며 checkpoint 전 구간에서 안정적이다.** Worklog 89 boundary algorithm, Worklog 82 relation threshold, NURBS fitting, visible Gaussian training은 모두 미변경이다. Worklog 90의 covariance-footprint 귀속 로직도 그대로 재사용했다(재구현하지 않음). 이번 배치는 Worklog 90이 `MULTILAYER_OR_VOLUMETRIC`으로 묶은 91.44% evidence 안에서, 그 원인이 (a) center 자체의 실제 다층 분포인지 (b) covariance frame(orientation/shape)만의 표현 불일치인지를 checkpoint 5개에 걸쳐 분리했다.

## 고정한 범위

- Worklog 89 full-region face topology, Worklog 82 kNN=8/cap=12/normal alignment=0.85/mutual residual=0.35 + typed veto, Worklog 79 coverage→PCA-UV→6×6 NURBS 체인, region ownership, visible Gaussian training은 모두 미변경이다.
- Worklog 90의 `attribute_failed_chart_unit_surface_topology`를 그대로 import해 재실행했다. `MULTILAYER_OR_VOLUMETRIC`으로 분류된 unit만 이번 measurement의 대상이다.
- 새 constructor, 새 edge/relation, 새 threshold는 추가하지 않았다. `torch_chart_unit_surface_topology_temporal_lineage.py`의 layer-gap 분리 규약(local median gap의 3배 및 point cloud extent의 2%)은 sweep 대상이 아닌 고정된 robust 분리 규약이다.
- clone/split 기원 분리를 위한 새 per-Gaussian 저장 필드는 추가하지 않았다. `stable_gaussian_ids`는 이미 checkpoint에 존재하는 값을 읽기만 했다.

## 측정 방법

### 1. CENTER_GEOMETRY_LAYERING (covariance 독립)

각 실패 unit의 member center만으로 SVD 평면을 적합하고(각 Gaussian 자신의 covariance eigenvector는 전혀 사용하지 않음), 그 법선에 투영한 signed offset을 정렬해 gap 기반 1-D 클러스터링으로 layer 개수를 센다. Local median gap의 3배 또는 point cloud 자체 extent의 2% 중 더 엄격한 기준을 넘는 지점만 layer 경계로 인정한다.

### 2. COVARIANCE_ONLY_AMBIGUITY

동일 unit에서 Worklog 90과 같은 공식으로 `layer_conflict` node mask(covariance normal/tangent/thickness 기반)를 재계산하고, center geometry가 single-sheet(`layer_count=1`)인데도 이 mask가 켜지는 node 비율만 covariance-only ambiguity로 센다. Center 자체가 이미 multilayer면 covariance conflict는 표현 문제가 아니라 실제 불일치이므로 이 범주에서 제외한다.

### 3. ADC_LINEAGE

`stable_gaussian_ids`는 clone/split마다 새 ID를 발급하고 재사용하지 않으므로(`osn_gs/gaussian/torch_density_control.py`), checkpoint 간 ID 집합 차이로 born/pruned를 정확히 복원할 수 있다. 다만 checkpoint에는 per-Gaussian parent-ID/birth-type이 저장되지 않아 clone과 split의 기원은 stable ID만으로 구분할 수 없다 — 이는 훈련 코드나 저장 포맷을 바꾸지 않고 얻을 수 있는 상한이다. 대신 training log의 `OSN-GS ADC: iteration=N ...` 줄에서 그 iteration까지의 누적 clone_parents/split_parents/split_children/pruned 카운터를 읽어(`_log_adc_line`, 기존 Worklog 63/65 replay와 동일한 파싱 규약) 각 checkpoint 구간에 대응시켰다.

### 4. TEMPORAL ONSET

`output/extent_ab/val64/baseline_compatible`의 5개 실사용 checkpoint(600, 2900, 3000, 3100, final)를 모두 재생했다. Worklog 89/90과 동일한 region-owned evidence 파이프라인(Worklog 82 relation, Worklog 83 assembly, Worklog 84 coherence, Worklog 89 boundary)을 그대로 통과시켰다.

### 5. VISIBILITY / DEPTH ORDERING

`final` checkpoint의 최대 dominant multilayer unit(region 3, unit 0, 1386 evidence)에서 member 수 기준 상위 두 layer(1378개 vs 3개)를 골라, 실제 COLMAP train camera 161개 전체에 대해 `OSNGaussianRasterizer.render`를 실행했다. 두 layer 모두 `radii > 0`으로 보이는 카메라만 세고, 그 카메라들에서 각 layer 중심의 `world_view_transform` 기준 view-space Z 차이를 측정했다.

## 검증

`tests/test_chart_unit_surface_topology_temporal_lineage.py` 5개는 단일-sheet grid, 명확히 분리된 두 sheet, 2-point 경계 조건, covariance-only ambiguity의 참/거짓 분기를 hand-built fixture로 검증한다. 전체 회귀는 **948 passed, 1 skipped**(293.9초)다. Worklog 79~90 관련 focused 58개도 별도로 통과했다.

## 5개 checkpoint 재생 결과

Checkpoint당 owned evidence와 cap=2048은 Worklog 89/90과 동일 조건이다. 산출물: `output/extent_ab/val91/chart_unit_surface_topology_temporal_lineage_replay.json`.

| iteration | Gaussian 수 | failed topology evidence | true-center multilayer | covariance-only | true-center 비율 | covariance-only 비율 | layer count(중앙값, 가중) | anisotropy(가중) | opacity(가중) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 600 | 138,766 | 0 | 0 | 0 | — | — | — | — | — |
| 2900 | 1,882,679 | 3,073 | 2,627 | 183 | **93.49%** | 6.51% | 8.77 | 1.943 | 0.616 |
| 3000 | 1,977,536 | 9,889 | 8,456 | 684 | **92.52%** | 7.48% | 7.03 | 2.287 | 0.617 |
| 3100 | 2,074,050 | 11,368 | 10,062 | 652 | **93.91%** | 6.09% | 6.67 | 2.352 | 0.522 |
| final | 1,685,549 | 6,497 | 5,872 | 251 | **95.90%** | 4.10% | 6.91 | 2.178 | 0.528 |

iteration 600은 densification 초기(ADC 시작 직후, clone/split 누적이 아직 적은 시점)라 실패 topology unit이 0개다. 2900부터 등장하는 실패 evidence는 3100까지 계속 증가(3073→11368)하다가, 3100→final 구간의 대규모 screen-size pruning(아래 lineage 참고)으로 6497까지 줄어든다. 그러나 **true-center multilayer 비율은 4개 checkpoint 전 구간에서 92.5%~95.9%로 사실상 일정**하다.

## Lineage (checkpoint 구간별 누적 ADC)

| 구간 | stable ID 신규 등장 | stable ID 소멸 | 구간 끝 누적 clone_parents | 구간 끝 누적 split_parents/children | 구간 끝 누적 pruned | dominant multilayer unit의 이번 구간 신규 member |
|---|---:|---:|---:|---:|---:|---:|
| 600→2900 | 1,765,108 | 21,195 | 91,044 | 3,901 / 7,802 | 88 | 239 |
| 2900→3000 | 98,846 | 3,989 | 92,548 | 4,044 / 8,088 | 78 | 22 |
| 3000→3100 | 100,636 | 4,122 | 14,404(*) | 1,459(*) / 2,918(*) | 404,364 | 49 |
| 3100→final | 17,266 | 405,767 | — | — | — | 6 |

(*) 3000→3100 구간의 로그 라인은 opacity reset 이후 카운터가 재기록된 값으로, `screen=233178`이 이 구간에서 처음 큰 값으로 나타난다 — Worklog 65가 이미 문서화한 screen-prune storm(3100 부근)과 시점이 일치한다. `final` checkpoint에는 대응하는 `OSN-GS ADC: iteration=final` 로그 라인이 없어 해당 구간의 세부 카운터는 `null`이지만, stable ID diff 자체(405,767 소멸)는 직접 관측값이다.

Dominant multilayer unit(최종적으로 region 3 / unit 0, 1386 evidence)의 member는 600→2900 구간에 239개가 새로 태어나 이미 dominant 후보의 핵심을 이루고, 이후 세 구간에서 22/49/6개씩만 추가된다 — 즉 이 경쟁 layer 구조의 대부분은 초기 densification 파동에서 이미 형성되고 이후 pruning에서도 살아남는다.

## Visibility / depth ordering

Region 3 / unit 0의 두 dominant layer(1378 vs 3 member)를 train camera 161개 전체로 렌더링한 결과, **14개 카메라(8.7%)**에서 두 layer 모두 `radii > 0`(화면에 보임)이었다. 그 14개 카메라에서 view-space depth 차이는 평균 0.277, 범위 [0.211, 0.387]이다 — 두 layer가 같은 화면 영역에서 서로 다른 깊이에 동시에 보이는 competing-depth 구조이며, 어느 한쪽만 항상 가려지는 것이 아니다.

## 결정

**Decision A.** true center-distribution multilayer가 checkpoint 2900부터 final까지 4개 시점 전부에서 MULTILAYER_OR_VOLUMETRIC evidence의 92.5%~95.9%를 차지하며, 이 비율은 evidence 총량이 3073→11368→6497로 크게 변하는 동안에도 거의 흔들리지 않는다. Covariance-only ambiguity(centers가 single-sheet인데 covariance frame만 불일치하는 경우)는 4.1%~7.5%로 항상 소수다. Lineage 분석은 이 competing-layer 구조 대부분이 초기 densification 파동(600→2900)에서 이미 형성되어 이후 pruning에서도 유지됨을 보인다. Visibility 분석은 dominant 사례가 view-conditioned가림이 아니라 여러 카메라에서 동시에 보이는 실제 competing-depth 구조임을 확인한다.

covariance frame이 surface normal 표현으로서 부적합하다는 B 가설(covariance-only ambiguity가 우세)은 기각한다. 다음 방향은 covariance-footprint/surfel graph로 production topology를 교체하는 것이 아니라, ADC/densification/pruning이 이 competing-layer 구조를 왜 만들고 유지하는지를 조사하는 것이다.
