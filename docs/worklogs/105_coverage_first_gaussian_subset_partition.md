# Worklog 105 — Coverage-first Gaussian Subset partition (신규 top-level architecture 1단계)

## 상태

**완료 — 이 배치는 architecture 성공/실패 판단을 내리지 않는다.** 사용자가 GAUSSIAN_SUBSET_PARTITION을 직접 시각적으로 검토한 뒤 다음 단계를 결정한다.

Worklog 95~104의 selection-first 파이프라인(구조/support predicate를 통과한 evidence만 component·chart·patch가 되는 구조)을 계속 개선하지 않고, top-level visible-surface construction architecture를 **coverage-first**로 교체하는 첫 구현 배치다. 이번 배치의 범위는 오직 다음 1단계뿐이다:

    전체 학습된 visible Gaussian scene
        -> 모든 Gaussian에 대한 surface-orientation 표현
        -> coverage를 보존하는, normal-coherent, spatially-connected Gaussian Subset 분할

Trustable Gaussian 추정, latent surface, NURBS는 이번 배치에서 **구현하지 않았다**. Worklog 95~104 구현·산출물은 삭제/덮어쓰기하지 않았고 각자의 스크립트로 그대로 재생 가능하다.

## 1. 유도된 tangent-plane / normal 표현의 정확한 정의

신규 `osn_gs/surface/torch_gaussian_surface_orientation.py`.

3DGS Gaussian은 `scaling`(log 도메인 3축)과 `rotation`(quaternion)을 저장한다. 그 covariance는

    Sigma = R diag(s^2) R^T,   s = exp(scaling)

이고, 따라서 `R`의 **열 j가 정확히 고윳값 `s_j^2`에 대응하는 Sigma의 고유벡터**다:

    Sigma (R e_j) = R diag(s^2) R^T R e_j = s_j^2 (R e_j)

즉 Gaussian parameterization은 이미 세 주축을 **정확히** 담고 있어 production 경로에서 eigen-decomposition이 필요 없다. 이 표현이 가진 유일한 모호성은 **축 순서**다 — `scale_0/1/2`에는 정렬 보장이 없어서 "어느 축이 normal인가"가 정의되지 않는다. `derive_surface_orientation_from_scale_rotation`은 이걸 `s^2` 내림차순 정렬(`lambda1 >= lambda2 >= lambda3`)로 결정론적으로 해소하고 다음과 같이 정의한다:

    surface_normal = axis(lambda3)      # 가장 얇은 주축
    tangent_axis_u = axis(lambda1)      # 가장 긴 in-plane 방향
    tangent_axis_v = surface_normal x tangent_axis_u   # 우수좌표계 완성

이는 기존 `torch_gaussian_covariance_frame.extract_covariance_frame`의 규약(내림차순 고윳값, `eigenvector(lambda3)`가 normal candidate)과 **동일**하므로 두 모듈이 조용히 다른 normal 정의를 쓰는 일이 없다. 임의 covariance 입력(fixture/합성 테스트/비-3DGS 입력)을 위한 `derive_surface_orientation_from_covariance`도 **같은** 정규화 함수(`_assemble`)를 통과한다 — 두 번째 normal 정의를 만들지 않기 위한 설계이며, 테스트로 두 경로가 일치함을 확인한다.

**부호 규약**: 주축에는 고유 부호가 없다. 재현성 게이지로만(물리적 outward 방향 의미 없음) "절댓값이 가장 큰 성분을 양수로, 동률은 가장 낮은 index"로 정규화한다. 전역 normal flipping은 하지 않는다. 유사도 비교는 항상 unsigned `|dot(n_i, n_j)|`(`unsigned_normal_alignment`, 기존 `orientation_insensitive_alignment`에 위임)를 쓴다.

**Volumetric 두께는 surface가 아니다**: `normal_thickness = sqrt(lambda3)`는 진단으로만 저장하고, 분할은 tangent plane/normal 방향만 소비한다.

**축 분리 가능성은 진단 전용**: `axis_separability`가 `well_defined` / `tangent_axes_degenerate`(lambda1≈lambda2, 평면 자체는 유효하나 in-plane 축이 미결정) / `normal_axis_degenerate`(lambda2≈lambda3, normal 방향 미결정) / `isotropic` / `non_finite`를 기록한다. 임계값 3.0은 기존 `extract_covariance_frame`의 `planarity_threshold`/`elongation_threshold` 기본값을 그대로 재사용했다. **이 라벨은 어떤 Gaussian도 표현에서 제외하지 않는다** — 모든 입력 row는 항상 정확히 하나의 출력 row를 만든다.

## 2. Gaussian Subset 분할 알고리즘

신규 `osn_gs/surface/torch_coverage_first_subset_partition.py`.

1. **kNN spatial adjacency**: 전 Gaussian에 대한 정확한 brute-force kNN(query row 기준 chunk 분할). 후보 순위는 `torch.cdist`(대규모에서 matmul 경로)가 매기지만, 반환되는 모든 거리는 gather한 좌표에서 **직접 재계산**해 임계값이 matmul 반올림값에 적용되지 않게 한다. Self 제외는 거리가 아니라 **row index** 기준이라, 완전히 겹친 Gaussian도 서로의 정상 이웃으로 남는다.
2. **Local spacing**: 각 Gaussian의 kNN 거리 **중앙값**. 평균/최대가 아니라 중앙값이라 먼 이웃 하나가 연결 허용 스케일을 부풀리지 못한다.
3. **Spatial edge 채택**: `dist(i,j) <= spatial_connect_spacing_multiplier * min(local_spacing_i, local_spacing_j)`. `min`을 쓰는 이유는 성긴 영역의 점이 조밀한 표면으로 건너뛰지 못하게 하기 위함이다.
4. **Normal compatibility**: `|dot(n_i, n_j)| >= normal_compatibility_min_alignment`. **spatial edge 위에서만** 평가한다 — 전역 normal clustering이 아니므로 멀리 떨어진 평행 벽이 합쳐지지 않는다.
5. **Connected component**: `accepted = spatial AND normal_compatible` 그래프의 연결 성분. Shiloach-Vishkin 스타일 hooking + full path compression으로 GPU에서 계산한다.
6. **결정론적 subset ID**: 크기 내림차순, 동률은 성분의 최소 member index 오름차순. 같은 입력은 같은 ID를 재현한다.
7. **Fallback ownership**: accepted edge가 하나도 없는 Gaussian은 **자기 자신만의 subset**을 갖는다. UNASSIGNED 집단을 만들지 않으며, `fallback_normal_incompatible_neighborhood`(공간 이웃은 있으나 전부 normal 비호환)와 `fallback_no_spatial_neighbor`(공간 이웃 자체가 없음)로 구분해 보고한다.

**Subset의 공간 연결성은 구성상 보장**되며(accepted-edge 그래프의 연결 성분이므로), `count_spatially_disconnected_subsets`가 성분을 독립적으로 재계산해 기계적으로 재검증한다.

### 구현 중 발견·수정한 실제 결함

첫 구현의 connected-component 솔버는 min-label을 **edge endpoint**에 scatter했다. 이 방식은 라벨을 라운드당 그래프 1-hop만 전진시켜 O(diameter) 라운드가 필요하고, 실제 scene(1,685,549 node / 3,955,593 accepted edge)에서 128 라운드 예산을 초과해 예외로 종료했다. **root index**(`max(root_u, root_v)`가 `min(root_u, root_v)`를 채택)에 hooking하도록 고치자 임의 길이의 성분 사슬이 한 라운드에서 해소된다. 검증: 1.7M-node 랜덤 index path(직경 1.7M)가 0.05초에 1개 성분으로 수렴하고, 중간 규모 랜덤 그래프에서 `scipy.sparse.csgraph.connected_components`와 성분 분할이 정확히 일치한다. 이 실패 모드에 대한 회귀 테스트를 추가했다.

## 3. 모든 heuristic parameter와 존재 이유

전부 `CoverageFirstPartitionConfig` 한 곳에 모여 있고, report JSON `partition.partition_parameters`에 그대로 기록된다. **렌더 결과를 보고 반복 조정한 값은 없고, 이번 배치에서 hyperparameter search를 하지 않았다.** 아래 한 설정이 primary result를 만든다.

| 파라미터 | 값 | 존재 이유 |
|---|---:|---|
| `neighbor_count` | 8 | 기존 `ManifoldAffinityConfig.candidate_neighbor_count`, `torch_latent_surface_tangent_frame_field.FIELD_NEIGHBOR_COUNT`와 동일 값 — "local neighbourhood"의 의미가 구/신 파이프라인 사이에서 바뀌지 않게 한다. |
| `spatial_connect_spacing_multiplier` | 2.0 | "국소 sampling pitch의 최대 2배" — 진짜 sampling 이웃과 빈 공간을 건너뛰는 bridge를 가르는 기준. 1.0(정상적인 sampling jitter까지 절단)보다 느슨하고, 저장소가 의도적으로 관대한 support radius에 쓰는 4.0~6.0보다 훨씬 타이트하다. |
| `normal_compatibility_min_alignment` | 0.85 (31.79°) | 기존 `ManifoldAffinityConfig.same_surface_min_normal_alignment` 값을 그대로 재사용 — "같은 surface orientation"의 의미를 코드베이스 전체에서 하나로 유지한다. |
| `normal_separation_ratio` / `tangent_separation_ratio` | 3.0 / 3.0 | 진단 라벨 전용. 기존 `extract_covariance_frame`의 `planarity_threshold`/`elongation_threshold` 기본값 재사용. **분할에는 전혀 쓰이지 않는다.** |
| `knn_chunk_size` | 0(auto) | 성능 전용. (chunk, N) 거리 행렬을 약 3GB 작업 집합에 맞춘다. 결과에 영향 없음. |
| `max_label_rounds` / `max_pointer_jumps` | 128 / 64 | 안전 상한. 초과 시 잘못된 분할을 조용히 반환하지 않고 예외를 던진다. |
| `VERY_SMALL_SUBSET_SIZE` | 8 | 진단 보고 버킷 전용. 분할을 바꾸지 않고, 이번 배치에서 어떤 acceptance threshold도 여기서 유도하지 않는다. |

## 4. 모든 입력 Gaussian이 정확히 하나의 owner를 갖는다는 회계 증명

`subset_ids`는 길이 N의 단일값 ownership map이므로 다중 소유는 구조적으로 불가능하다. 그 위에 **독립적으로 유도된 두 값을 원소 단위로 대조**한다 — `subset_ids`에서 다시 센 subset별 점유량과, connected-component labelling에서 나온 `subset_sizes`. 둘 중 어느 쪽이든 조용히 누락/중복하면 항등식이 깨진다.

| 지표 | 값 |
|---|---:|
| Model 전체 Gaussian | 1,685,549 |
| Visible Gaussian(= 분할 입력) | **1,685,549** |
| Uncertain Gaussian | 0 |
| assigned | 1,685,549 |
| unassigned | **0** |
| multiply-owned | **0** |
| subset_id 범위 이탈 | 0 |
| `sum(subset_sizes)` | 1,685,549 |
| `subset_sizes_match_ownership_map` | **true** |
| `coverage_identity_holds` | **true** |

분할 입력은 Worklog 95의 7개 region(evidence 7,774개)으로 제한하지 않았고, latent support / boundary evidence / topology 성공 / chart validity / continuous-support 소속 / NURBS eligibility / held-out validity 중 어느 것도 요구하지 않았다(`restricted_to_prior_regions=false`, `required_latent_support=false`). 정적 테스트(AST)로 두 신규 모듈이 latent-surface/chart/identifiability/NURBS/boundary/held-out 모듈을 import하지 않음을 강제한다.

## 5. Subset 개수와 크기 분포

**총 subset 수: 166,585.**

| 지표 | 값 |
|---|---:|
| min | 1 |
| median | 1 |
| mean | 10.12 |
| p95 | 10 |
| max | **559,541** (전체 Gaussian의 33.2%) |

상위 16개 subset 크기: 559541, 92799, 50720, 40929, 40206, 34760, 27283, 19953, 19114, 15080, 14392, 11111, 10975, 10779, 7619, 6012.

### Subset 크기 히스토그램 (subset 수 / 그 subset들이 소유한 Gaussian 수)

| 크기 상한 | subset 수 | Gaussian 수 |
|---:|---:|---:|
| 1 | 107,947 | 107,947 |
| 2 | 23,650 | 47,300 |
| 4 | 15,577 | 52,272 |
| 8 | 9,513 | 58,014 |
| 16 | 5,095 | 59,093 |
| 32 | 2,542 | 57,673 |
| 64 | 1,187 | 52,972 |
| 128 | 592 | 52,655 |
| 512 | 361 | 85,658 |
| 2,048 | 85 | 75,898 |
| 8,192 | 22 | 88,425 |
| 32,768 | 8 | 128,687 |
| 131,072 | 5 | 259,414 |
| (초과) | 1 | 559,541 |

두 방향을 모두 보고하는 이유는 서로 다른 질문에 답하기 때문이다 — "분할이 미세 조각에 지배되는가"(subset 수)와 "그 조각들이 실제로 scene의 의미 있는 몫을 차지하는가"(Gaussian 수). subset 수 기준으로는 미세 조각이 압도적이지만, Gaussian 수 기준으로는 상위 14개 subset이 전체의 절반 이상을 차지한다.

## 6. Disconnected / singleton / tiny subset 통계

| 지표 | 값 | 비율 |
|---|---:|---:|
| Spatially disconnected subset | **0** | 0% (연결성 계약대로) |
| Singleton subset(크기 1) | 107,947 | 전체 subset의 64.80% |
| Singleton이 소유한 Gaussian | 107,947 | 전체 Gaussian의 6.40% |
| Very small subset(크기 ≤ 8) | 156,687 | 전체 subset의 94.06% |
| Very small subset이 소유한 Gaussian | 265,533 | 전체 Gaussian의 15.75% |

Edge 통계:

| 지표 | 값 |
|---|---:|
| Candidate edge(kNN, 중복 제거) | 8,655,268 |
| Spatial edge(거리 기준 통과) | 7,344,950 |
| 거리 기준 탈락 | 1,310,318 |
| **Normal-compatibility cut edge** | **3,389,357** (spatial edge의 46.1%) |
| Accepted edge | 3,955,593 |

Local spacing: min 0.00191 / median 0.03459 / mean 0.04338 / p95 0.09924 / max 4.87173 (scene extent 12.31 기준).

축 분리 가능성(진단 전용, 분할에 미사용): `well_defined` 337,194 / `tangent_axes_degenerate` 444,827 / `normal_axis_degenerate` 551,428 / `isotropic` 352,100 / `non_finite` 0.

## 7. Fallback ownership 통계

| Ownership kind | Gaussian 수 | 비율 |
|---|---:|---:|
| `normal_coherent_component` | 1,577,602 | 93.60% |
| `fallback_normal_incompatible_neighborhood` | 107,215 | 6.36% |
| `fallback_no_spatial_neighbor` | 732 | 0.04% |
| **fallback 합계** | **107,947** | **6.40%** |

Fallback Gaussian은 전부 자기 자신만의 subset을 소유한다 — 어떤 Gaussian도 normal confidence가 낮다는 이유로, 이웃 evidence가 약하다는 이유로, 미래의 latent surface가 어렵다는 이유로 제외되지 않았다.

## 8. Review export 경로

출력 루트: `output/osn_gs_coverage_first_subset_partition/`. **모두 원본 scene 좌표계, 전체 scene, crop 없음.** 각 view 폴더는 WebRenderer 규약대로 `iteration_<N>` 아래에 `point_cloud.ply` 정확히 1개(+ view D만 `nurbs_surface.json` 1개)를 가지며, `render.ppm`은 규약을 건드리지 않도록 iteration 폴더 **바깥**에 둔다.

| view | 경로 | 내용 |
|---|---|---|
| A. ORIGINAL_SCENE | `original_scene/iteration_0000001/point_cloud.ply`<br>`original_scene/render.ppm` | 학습된 scene 그대로(원본 SH DC 색). PPM은 checkpoint의 SH degree 3으로 렌더 |
| B. NORMAL_ORIENTATION_VIEW | `normal_orientation_view/iteration_0000001/point_cloud.ply`<br>`normal_orientation_view/render.ppm` | 유도된 surface normal을 **unsigned**로 인코딩(`rgb = |n|`) — `n`과 `-n`이 동일하게 렌더된다. 시각적 일관성을 위한 전역 flipping 없음 |
| C. GAUSSIAN_SUBSET_PARTITION | `gaussian_subset_partition/iteration_0000001/point_cloud.ply`<br>`gaussian_subset_partition/render.ppm` | 모든 subset이 결정론적 고유 색. subset `i`는 어떤 실행에서도 같은 색 |
| D. SUBSET_BOUNDARY_VIEW | `subset_boundary_view/iteration_0000001/point_cloud.ply`<br>`subset_boundary_view/iteration_0000001/nurbs_surface.json`<br>`subset_boundary_view/render.ppm` | normal 비호환으로 잘린 adjacency edge 강조. PLY는 Gaussian별 **cut ratio**로 색을 램프하고, JSON은 실제 cut segment를 `base_curves`로 담는다 |
| 회계 | `partition_report.json` | 위 §4~7 수치 전체의 원본 |

Subset 색은 황금비 등 저불일치 상수로 hue/saturation/value를 해싱해 연속된 subset ID가 이웃 색으로 떨어지지 않게 한 결정론적 팔레트다. **색과 카메라는 구성상 고정이며 결과 품질을 보고 고르지 않았다.**

View D의 인코딩 정정: 처음에는 "cut edge에 하나라도 닿으면 강조"라는 이진 플래그로 만들었는데, 이 scene에서는 spatial edge의 46.1%가 잘려 거의 모든 Gaussian이 켜지면서 정보량이 0이 됐다. 그래서 Gaussian 자신의 spatial edge 중 잘린 비율(**cut ratio**)로 램프하도록 바꿨다 — 튜닝된 임계값이 아니라 원 통계를 그대로 끝에서 끝까지 매핑한 것이다. 실측 cut ratio: mean 0.4589 / median 0.4444 / p95 1.0, 완전 절단 Gaussian 107,215개, 전혀 잘리지 않은 Gaussian 204,996개. `base_curves`에 담은 cut segment는 3,389,357개 중 결정론적 균등 stride로 49,844개(cap 50,000) — 공간적 crop이 아니라 scene 전역에서 균등하게 뽑는다.

### render.ppm (사용자 추가 지시)

**실행 가능했고 실행했다.** 4개 view 전부 `render.ppm`을 생성한다. 카메라는 `TorchOSNGSTrainer._preview_camera`가 고르는 것과 **정확히 동일**하다 — eval split의 train camera를 이름순 정렬해 첫 번째(`DSC07957.JPG`, train 161 / held-out 24, llffhold=8, 1600×1036, downscale 3.2419). 그래서 이 PPM들은 checkpoint 자신의 `output/extent_ab/val64/baseline_compatible/final/render.ppm`과 픽셀 단위로 직접 비교 가능하다. 이미지 픽셀은 전혀 디코드하지 않고(PIL size probe만) 그 한 대의 카메라만 재구성하므로 데이터셋 전체 로딩이 필요 없다. 렌더 backend는 CUDA rasterizer(`installed package`)다.

### Gaussian별 그룹 RGB 분할 시각화 (사용자 추가 지시)

**실행 가능했고 실행했다.** view B/C/D는 각 Gaussian의 SH DC 항에 view 색을 넣고 상위 SH 밴드를 0으로, degree를 0으로 두어 **Gaussian 하나하나가 자기 그룹의 flat RGB로 렌더**되게 한다. position/scale/rotation/opacity는 항상 checkpoint 원본이라 이미지 안의 geometry와 coverage는 정확히 학습된 scene의 것이다. 같은 색이 PLY에도 동일하게 기록되므로 WebRenderer와 PPM이 일치한다. View A만 원본 SH를 유지해 실제 scene 외관 기준선으로 남긴다. 이 재색칠은 메모리 상의 model에만 적용되며 checkpoint 파일은 절대 쓰지 않는다.

## 9. 1 subset : 1 future NURBS 계약에 대한 기록(이번 배치에서 구현하지 않음)

§9의 complexity-driven refinement는 **구현하지 않았다.** 나중에 평가할 수 있도록 scale 통계만 기록한다: 가장 큰 subset이 559,541개 Gaussian(전체의 33.2%)을 소유하고, 상위 6개가 각각 27,000개를 넘는다. 즉 "임의로 거대한 초기 normal-connected component가 항상 하나의 subset으로 남아야 하는가"라는 §9의 질문은 이 scene에서 실제로 발생하며, 향후 단계에서 다뤄야 한다.

## 10. 재현 명령

```
python scripts/devtools/coverage_first_subset_partition_export.py \
    --checkpoint output/extent_ab/val64/baseline_compatible/final \
    --out output/osn_gs_coverage_first_subset_partition \
    --device cuda \
    --source-path DATASET
```

`--source-path`를 생략하면 카메라 intrinsic/extrinsic이 없으므로 `render.ppm`만 생략되고(그 사유가 report에 기록된다) 나머지 export는 그대로 생성된다. 런타임: 분할 141.9초, 전체 약 143초(RTX 5080).

**Checkpoint 주의**: 이 배치도 `output/extent_ab/val64/baseline_compatible/final`(iteration 3100, PSNR 20.1)을 썼다 — Worklog 94~104와 동일한 checkpoint이며, `docs/Urgent_Work/HANDOFF_2026-08-19.md` §0이 제기한 "이 checkpoint가 사용자가 의도한 scene인가"라는 미해결 질문은 여전히 미해결이다. 다만 이번 export의 ORIGINAL_SCENE 렌더에는 중앙 테이블과 화분이 명확히 존재한다 — 즉 "중앙 물체 Gaussian 누락"은 학습된 scene 자체의 문제가 아니라 Worklog 103이 다룬 region-owned evidence 7,774개라는 **하위 선택**의 문제였고, 이번 coverage-first 분할은 1,685,549개 전부를 구성상 포함한다.

## 11. 검증

**Focused 테스트 25개 신규 추가.**

`tests/test_gaussian_surface_orientation.py` (8개): 저장된 scale 순서와 무관하게 normal이 가장 얇은 축을 따름 / scale-rotation 경로와 covariance 경로가 하나의 normal 정의를 공유 / frame이 정규직교·우수좌표계 / 부호 게이지가 결정론적이고 quaternion double cover(q와 -q)에 불변 / unsigned alignment가 반대 부호 normal을 동일 취급 / degenerate·non-finite row도 정확히 하나의 row를 생성(coverage) / non-finite covariance가 covariance 경로에서 살아남음 / provenance ID·position이 변경 없이 전달됨.

`tests/test_coverage_first_subset_partition.py` (17개): 모든 Gaussian이 정확히 하나의 subset owner를 가짐 / 회계가 누락·다중소유 0을 보고 / 고립 Gaussian이 보고된 fallback으로 ownership 유지 / normal 비호환 이웃만 가진 Gaussian도 ownership 유지 / **부호 뒤집힌 동등 normal이 같은 표면을 분할하지 않음** / **멀리 떨어진 평행 표면이 합쳐지지 않음** / **강한 국소 normal 불연속이 분할 경계를 만듦** / **긴 얇은 표면이 하나의 subset으로 수렴(§2의 solver 회귀)** / 모든 subset이 partition 그래프 상에서 연결됨 / subset ID가 결정론적이고 크기 내림차순 / 원본 텐서 불변 / 유도 orientation이 end-to-end로 분할을 구동 / 히스토그램이 모든 subset·Gaussian을 회계 / 파라미터가 중앙집중·보고됨 / 빈 입력과 단일 Gaussian 입력도 coverage 정확 / **분할 로직이 Worklog 95 support·chart·NURBS validity에 의존하지 않음(AST)** / ownership kind가 모든 Gaussian을 정확히 한 번 덮음.

**전체 회귀 1회 실행**: `1152 passed, 1 skipped, 1 warning, 18 subtests passed in 376.95s` — Worklog 104 시점 기준선 1127 passed + 신규 25개와 정확히 일치하며, 기존 테스트 실패·회귀는 없다.

## 12. 완료 조건 충족 여부

> "전체 학습된 Gaussian scene이 결정론적·공간 연결적·normal-coherent Gaussian Subset으로 분할되고, 모든 Gaussian이 정확히 한 번 배정되며, 다운스트림 surface-validity predicate 때문에 사라지는 scene 영역이 없다."

기계적으로: `coverage_identity_holds=true`, unassigned 0, multiply-owned 0, spatially disconnected subset 0, 분할 입력 1,685,549 = visible Gaussian 전체, 다운스트림 predicate 의존 없음(AST 강제). 시각적으로: 위 4개 view가 전체 scene을 crop 없이 제공한다.

## 결론 없음

이 worklog는 coverage-first architecture가 성공인지 실패인지, over-segmentation 정도가 수용 가능한지, 가장 큰 subset을 나눠야 하는지, normal 임계값이 적절한지에 대해 **어떤 판단도 내리지 않는다.** 사용자가 GAUSSIAN_SUBSET_PARTITION을 직접 시각적으로 검토한 뒤에야 다음 단계(subset-local Trustable Gaussian 추정 → latent surface → 1 subset : 1 NURBS Patch)를 구현한다.
