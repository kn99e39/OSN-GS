# Worklog 119-2 — exact-semantics 성능 최적화

## Agent Interpretation of Intent

Worklog 119의 수학, topology, chart membership, 고정 8×4 degree-2 NURBS, 정규화, solve 횟수·순서, UV correction 및 Metric G/C 정의를 유지하면서 실제 wall-clock을 가능한 한 줄인다. 독립 chart batching, solve 병합, dtype/precision, renderer, 샘플 수는 변경하지 않는다. 동일 입력에서 완전히 같은 값을 다시 계산하는 구현만 제거하고, exact 비교가 실패한 후보는 적용하지 않는다.

## Implementation Fidelity Statement

- WL107/109 topology와 WL112 camera-blob membership 불변.
- NURBS resolution/degree/regularization/correction rounds/projection iterations/solve fallback 불변.
- 결과 reduction 순서 불변.
- 각 변경은 focused exact test 또는 기존 FIXED report exact 비교로 검증.
- projector best-distance 직접 반환은 수학적으로 같지만 chunk shape 차이로 bitwise residual 비교가 실패해 즉시 되돌림.

## 구현

### exact KNN 1회 재사용

build_candidate_graph가 계산한 exact (N, k) neighbor index를 opt-in 보존하고 apply_secondary_geometric_gate가 count, k, device가 모두 같을 때만 재사용한다.

- N=1,190,469, k=8
- 동일 brute-force KNN 2회에서 1회로 감소
- int64 index 약 76.2 MB
- camera/WL119 경로만 retain_neighbor_index=True
- k 불일치 시 기존 fresh KNN fallback

### NURBS와 projector 중복 제거

- evaluate와 LSQ assembly용 values-only basis 분리
- Gauss-Newton iteration 사이 동일 UV point/derivative 재사용
- regular UV grid, projection-grid basis, identity, second-difference penalty 캐시
- point와 normal을 한 derivative 평가에서 반환
- ARM A 최종 uv_footpoint를 Metric G에 재사용해 projector 1회 제거
- ARM B fixed-UV normal matrix/RHS는 1회 조립하되 seed-dependent solve는 기존처럼 2회

### chart-domain과 host serialization

- ViewPixelChartSamples에 raster row/col, stable blob order/offset 보존
- view별 중복 connected-component labeling 제거
- chart별 전체 valid-pixel/image mask 검색과 np.nonzero 2회 제거
- stable slice가 legacy boolean selection의 row-major order를 exact 보존
- residual 12개, control diff, smoothness 2개를 한 bulk CPU transfer로 수집
- chart별 audit clone/torch.equal은 focused test로 이동
- region lookup과 empty spread 판정을 CPU bulk 경로로 변경

## 폐기한 후보

projector 내부 best_dist를 Metric G에 반환하는 후보는 CPU iterations=6, chunk_size=41에서 기존 전체-batch residual과 bitwise 불일치했다. batch shape가 수치 실행에 영향을 주므로 해당 후보는 완전히 되돌렸다.

## 검증

최종 focused suite:

    .venvScriptspython.exe -m pytest -q tests	est_camera_induced_visible_adjacency.py tests	est_coverage_first_subset_partition.py tests	est_nurbs_surface.py tests	est_camera_observed_chart_domains.py tests	est_renderer_native_pixel_surface_chart.py tests	est_visible_nurbs_geometry_uv_control_correction.py

결과: 83 passed in 3.83s.

추가 exact 검증:

- fresh/reused KNN gate의 모든 tensor/scalar 동일
- k mismatch fallback
- legacy/optimized projector CPU/CUDA, iterations 0/1/3/6 UV 동일
- values-only/legacy basis point 동일
- ARM B normal-system 재사용 control grid CPU/CUDA 동일
- stable blob slice의 index/order/UV/XYZ/representative id 동일
- 기존 pixel-record legacy contract 유지

## 벤치마크

공통 조건:

- checkpoint: output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt
- DATASET, images_8, RTX 5080
- --max-views 8 --max-charts 512
- 비경합 GPU, nvidia-smi 1초 sampling
- 기준: 119-1 FIXED r3
- 최종: optimized r3

| 지표 | FIXED r3 | optimized r3 | 변화 |
|---|---:|---:|---:|
| 내부 total | 185.494초 | 93.006초 | 1.994배, 49.86% 감소 |
| chart loop | 43.087초 | 21.342초 | 2.019배, 50.47% 감소 |
| 처리량 | 11.883 charts/s | 23.990 charts/s | 2.019배 |
| chart GPU mean | 15.76% | 19.76% | +4.00%p |
| chart GPU median/p95 | 16% / 16% | 16% / 17% | 거의 동일 |
| chart active fraction | 100% | 100% | 동일 |
| 전체 GPU mean | 78.53% | 77.90% | 사실상 동일 |
| 전체 median/p95 | 100% / 100% | 100% / 100% | 동일 |
| peak memory | 9,164 MiB | 6,249 MiB | 측정상 감소 |

optimized 외부 process wall은 r3 95.689초, r1/r2 약 97.80초였다.

## profile 비교

동일 2-view/128-chart cProfile:

| 함수 | 기존 | 최종 |
|---|---:|---:|
| 전체 | 155.283초 | 79.201초 |
| _knn | 2회 / 141.312초 | 1회 / 70.678초 |
| projector | 528회 / 7.553초 | 400회 / 3.313초 |
| basis pair | 11,104회 / 5.346초 | 5,332회 / 2.542초 |
| evaluate | 3,296회 / 4.687초 | 928회 / 1.040초 |
| LSQ normal system | 544회 / 0.781초 | 408회 / 0.479초 |
| second-difference penalty | 544회 / 0.198초 | 2회 / 0.001초 |
| blob labeling | 2회 / 0.282초 | 1회 / 0.250초 |

profile 전체 1.961배 개선.

## 결과 동등성

FIXED r3와 optimized r3에서 accounting, synthetic contracts, topology replay, UV displacement, Metric G/C, pixel attribution, region results가 exact 동일했다.

geometry-source bounded sample은 torch.randperm 때문에 process마다 원래 달라진다. normal/position summary mean은 CUDA index_add/scatter의 process 간 미세 비결정성이 있으며 optimized r1/r2끼리도 달랐다. 주요 deterministic report section은 exact 동일하다.

## 평가와 최종 판정

GPU utilization 자체는 거의 그대로인데 total/chart throughput이 약 2배가 됐다. 개선은 GPU 포화가 아니라 동일 KNN/projection/basis/labeling/host extraction 중복 제거에서 왔다.

최종 profile 79.201초 중 남은 exact brute-force KNN 한 번이 70.678초(89.2%)다. chart GPU median은 여전히 16%다.

- 전체 남은 지배 병목: O(N²) exact KNN 1회
- chart 남은 원인: 작은 sequential projection/basis kernel의 CURRENT EXECUTION-GRANULARITY LIMIT
- solve-only batching은 주병목을 겨냥하지 않는다.
- 다음 대규모 개선은 exact spatial-neighbor architecture 또는 chart projection/basis batching이 필요하며 이번 범위 밖이다.