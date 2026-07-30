# Worklog 131 — Canonical Reconstruction GPU Synchronization 최적화

## 목표

실제 CUDA ADC visible reconstruction에서 CPU/GPU 동기화 병목을 계측하고, 결과 정책을 바꾸지 않는 범위에서 GPU batch 및 tensor 연산으로 이전한다.

## 계측 환경

- RTX 5080 16GB, 로컬 `DATASET`, 2 views, `image_downscale=8`, `train_resolution_scale=4`
- `canonical_construction_max_points=2048`, terminal detached reconstruction 1회
- CUDA stage 양끝에 synchronize를 둔 임시 profiler 사용

## 병목과 변경

`evaluate_intrinsic_reliability`는 full observed Gaussian 약 138,766개에 대해 CUDA scalar를 Python `if`로 하나씩 읽었다. 각 비교가 host synchronization을 발생시켜 intrinsic classification만 31.027초를 사용했다.

같은 판정 우선순위(conditioning → scale → isotropic → planar reliable → ambiguous)와 문자열/reason payload 계약을 유지한 채, 판정을 GPU boolean mask로 계산했다. 최종 class/reason tuple만 bulk CPU transfer 후 생성한다.

또한 density-preserving FPS는 bounded candidate set에서 CUDA pairwise distance를 한 번 캐시하고 GPU tensor에서 tie를 고른다. 안전 한도 256MiB를 넘는 경우에는 exact vector-norm fallback을 사용한다. full-neighborhood nearest assignment의 기본 CUDA batch는 4,096에서 16,384로 늘렸다. 이 값은 각 query row의 nearest argmin 결과를 바꾸지 않는 memory/performance knob다.

## 결과

| 항목 | 이전 | 이후 |
| --- | ---: | ---: |
| `evaluate_intrinsic_reliability` | 31.027s | 0.010s |
| representative selection | 4.925s | 3.874s |
| full evidence assignment/aggregation | 0.028s | 0.020s |
| canonical construction | 1.579s | 1.567s |
| 전체 detached reconstruction | 38.146s | 5.675s |

동일 input의 end-to-end speedup은 약 **6.7x**다. construction state와 failure 결과는 전후 모두 `boundary_recovery_failed`였고 materialized surface는 0으로 동일했다.

## GPU utilization과 발열 해석

200ms `nvidia-smi` 샘플에서 기존 병목 구간의 SM utilization은 대체로 0~15%, power는 32~74W(360W limit), 온도는 44~47°C였다. 높은 순간 utilization/clock 표시는 짧은 CUDA kernel 또는 메모리/launch 구간을 뜻하며, sustained tensor-core compute를 뜻하지 않는다. 원인은 대규모 GPU 연산 자체의 부족보다 Python scalar read가 반복적으로 CUDA를 host에 동기화한 것이었다.

최적화 후 주요 잔여 시간은 약 3.8초의 mode-aware representative selection이다. 이 단계는 stable-ID 순서와 online centroid를 사용하는 NumPy/Python greedy cell clustering이다. 완전 GPU화는 mode assignment의 tie/ordering contract를 별도로 보존해야 하므로, 이번 동기화 제거와 분리한 다음 최적화 작업으로 남긴다.

## 검증

- focused: `25 passed`
- repository-wide: `597 passed, 1 skipped, 1 warning, 8 subtests passed`

## 남은 위험

- 16,384-query batch는 기본 2,048 representative cap에서 약 128MiB distance workspace를 사용한다. 현재 16GB GPU에서는 충분히 안전했지만, 더 작은 GPU에서 memory pressure가 관측되면 performance-only fallback 정책을 추가 검토한다.
- mode-aware selection의 GPU-native 재구현은 selection 결과가 바뀌지 않음을 stable-ID shuffle과 real-scene snapshot으로 입증한 뒤에만 진행한다.