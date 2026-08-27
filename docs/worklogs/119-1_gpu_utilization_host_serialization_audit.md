# Worklog 119 GPU Utilization / Host Serialization Audit

상태: 진행 중

## Agent Interpretation of Intent

### DIRECTION

Worklog 119의 계산 결과와 실행 의미를 그대로 보존하면서, chart-fitting loop에서 제거 가능한 Python/host serialization만 식별하고 필요할 때 동일 bookkeeping을 유지하는 구현 세부사항만 바꾼다.

### PURPOSE

이미 적용된 `pixel_records` bulk serialization fix를 실제 bounded run에서 검증하고, WL119 및 WL118 상속 경로에 남은 유사한 serialization이 throughput을 실제로 제한하는지 측정한다. GPU utilization 수치 자체를 목표로 삼지 않고, throughput과 exact output preservation을 기준으로 판단한다.

### CENTRAL INTENT

관측된 낮은 GPU utilization을 `REMAINING IMPLEMENTATION BOTTLENECK`, `CURRENT EXECUTION-GRANULARITY LIMIT`, 또는 `MIXED`로 근거 있게 귀속한다. 많은 독립적인 소형 per-chart NURBS solve가 순차 실행되는 현재 모델의 한계를 구현 결함으로 오인하지 않는다.

### 동결해야 할 것

- WL107/109 topology와 WL112 camera-blob membership
- fixed 8x4, degree-2 NURBS control grid와 현재 regularization
- per-chart fitting, correction rounds, UV semantics, chart membership, sample population
- solve ordering, dtype/precision, renderer, G0/G1/G2 정의와 G0/G1/G2 accounting
- checkpoint, camera set, output semantics 및 실험적 비교 계약

### 허용되는 변경

측정된 hotspot에 한해 bulk tensor indexing/transfer, vectorized reduction, exact-order preallocation 등 산술 결과와 externally-visible bookkeeping을 동일하게 유지하는 implementation-only 변경만 허용한다. 테스트는 이 exact-equivalence 계약을 직접 보호한다.

### 금지되는 변경

independent chart solve batching/merging, solve ordering 변경, `torch.linalg.solve` 또는 fallback 변경, regularization·control-grid·degree·correction-round·UV·topology 변경, precision/TF32/mixed precision, custom CUDA/fusion/multi-stream, renderer 변경, sample 감소를 하지 않는다.

### 기존 full Worklog 119 실행 보호

handoff 식별자 `bjgreh4iu`에 해당하는 것으로 보이는 실행을 종료·재시작·signal·suspend하지 않고, 기존 output directory를 재사용하지 않는다. 감사 시작 시점에 동일 WL119 script를 실행하는 Python 프로세스가 두 개 보였고 RTX 5080 메모리 점유가 약 15.9 GiB였으므로, GPU benchmark는 해당 실행이 끝나고 GPU가 uncontended임을 확인한 뒤에만 수행한다.

### benchmark 경쟁 회피

benchmark는 활성 full run과 같은 GPU에서 동시에 실행하지 않는다. 별도 idle GPU가 없으므로, 현재는 source/test/profiler preparation만 진행하고 target GPU가 idle이 된 뒤 bounded A/B를 순차 실행한다. 기존 실행의 output directory `output/119_osn_gs_geometry_uv_control_correction`는 사용하지 않는다.

### Prompt-required decision과 Agent-introduced operational choice

#### PROMPT-REQUIRED DECISION

- known `pixel_records` fix를 exact equivalence와 실제 bounded performance로 평가한다.
- 동일 checkpoint/cameras/GPU/environment/code path의 OLD vs FIXED 비교를 우선한다.
- 모든 후보는 materiality를 먼저 확인하고 semantics-neutral일 때만 변경한다.
- 최종 귀속은 세 가지 분류 중 하나로 evidence와 함께 보고한다.

#### AGENT-INTRODUCED OPERATIONAL CHOICE

- bounded workload: `--max-views 8 --max-charts 512`; startup만 측정하지 않고 반복 chart-fitting을 포함시키기 위한 고정 선택이다.
- benchmark arm: main worktree가 아닌 `C:\tmp\wl119_gpu_audit`의 isolated source copies를 사용한다. OLD copy에는 known pre-fix pixel-record loop만 복원하고, FIXED copy에는 현재 구현을 유지한다.
- timing: `time.perf_counter()`로 process total 및 chart-loop 진입 직전/종료 직후를 기록하고, 두 arm에서 동일한 timing point를 사용한다.
- GPU sampling: benchmark 중 `nvidia-smi`의 1초 cadence 샘플을 사용한다. 평균·중앙값·p95·`utilization.gpu > 0` 비율·가능한 peak memory를 같은 interval로 계산한다.
- Python profiling: bounded profiler run에는 `cProfile`/`pstats`를 사용해 host function-level materiality를 확인하고, profiler overhead가 A/B 수치에 섞이지 않도록 performance A/B와 분리한다.
- CPU indication: 가능하면 Windows processor counter 또는 process CPU time을 보조 지표로 기록하되, GPU contention이 없는 조건에서만 해석한다.

## Implementation Fidelity Statement

현재까지 구현된 vectorized `pixel_records` 경로는 sampled column을 bulk `.numpy()`로 옮긴 뒤 기존 순서의 Python `zip()`으로 최대 200개/chart record를 구성한다. 이 감사에서 허용하는 변경은 해당 bookkeeping의 exact output을 유지하는 것뿐이며, NURBS fitting 수식·입력·solve 순서에는 손대지 않는다. full run이 활성인 동안에는 실행 환경과 산출물을 변경하지 않는다.

## 1. 시작 상태

- branch: `arch/2dgs-coverage-first-surface`
- HEAD: `b4da607c22a3f940064bf788c52c9180fc8de0d2`
- 기존 변경: Worklog 119 script/test/worklog untracked, 두 report JSON 삭제 상태. 이 감사는 기존 WL119 문서를 수정하지 않는다.
- 활성 상태: 동일 checkpoint와 `--source-path DATASET --images images_8`를 사용하는 Python 실행 두 개가 관찰됨. GPU benchmark는 보류한다.

## 2. 초기 소스 감사

- 현재 `pixel_records`는 scalar tensor indexing 대신 sampled columns의 bulk NumPy conversion 후 bounded `zip()`을 사용한다.
- `_chart_scalar_color`의 `for r in chart_records`는 chart-level 후처리이고, pixel population loop가 아니다. benchmark/profile에서 materiality를 확인한다.
- G0/G1/G2 per-view disagreement는 mask/index tensor 연산이며, hot path의 per-pixel Python extraction은 확인되지 않았다.
- representative-footprint spread는 `index_add_`/`scatter_reduce_`를 사용한다. WL118 상속 `accumulate_image_space_pairs`도 per-view tensor vectorization과 `index_add_` 회계를 사용한다.

나머지 측정 결과와 exact-equivalence 증거는 감사 진행에 따라 아래에 추가한다.

## 3. Known pixel-record fix exact equivalence

- 현재 구현은 sampled columns를 한 번에 NumPy로 옮긴 뒤 _build_pixel_records_vectorized에서 기존 record 순서와 Python scalar 변환을 유지한다.
- TestPixelRecordSerializationEquivalence를 포함한 Worklog 119 focused test는 14 passed다.
- 테스트 fixture는 record count/order, 모든 key, int/float의 Python 변환, rho branch 분류, None 처리, NaN 및 Inf를 old scalar reference와 비교했고 mismatch는 0건이었다.
- bounded OLD/FIXED r3 모두 512 charts와 42,998 pixel records를 기록했다.
- 두 report의 accounting, synthetic contracts, camera metadata, corrected UV A/B metric, pixel-level D attribution, UV displacement, region 결과는 exact equal이었다. geometry source 및 normal disagreement section의 차이는 별도 process의 stochastic bounded index sample에 의한 것이며 pixel-record path의 입력이나 출력이 아니다.

## 4. Benchmark protocol

보호된 full run이 끝난 뒤 GPU 0의 compute utilization 0%, memory 951 MiB 수준을 확인하고 측정했다. 기존 output directory output/119_osn_gs_geometry_uv_control_correction는 사용하지 않았다.

- checkpoint: output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt
- cameras: DATASET, images_8, 161 train cameras 중 앞 8개
- bound: --max-views 8 --max-charts 512
- GPU: NVIDIA GeForce RTX 5080, GPU 0
- arm: C:\tmp\wl119_gpu_utilization_audit_20260826 아래 isolated OLD/FIXED copies
- OLD: pre-fix sample_idx.tolist()와 per-field tensor indexing를 복원한 copy
- FIXED: 현재 bulk NumPy + named helper copy
- timing: 동일 chart-loop 진입/종료 지점의 time.perf_counter() 및 epoch marker
- GPU sampling: nvidia-smi query-gpu timestamp, utilization.gpu, memory.used; 1,000 ms cadence
- profiler: FIXED 2-view/128-chart cProfile 별도 실행. profiler 시간은 A/B 수치에 섞지 않았다.

재현 명령의 핵심은 다음과 같다.

    .venv\Scripts\python.exe C:\tmp\wl119_gpu_utilization_audit_20260826\old\visible_nurbs_geometry_uv_control_correction.py --checkpoint output\arch_2dgs_coverage_first_surface\2dgs_run1\30000\checkpoint.pt --out output\119_gpu_utilization_audit_old_20260826_r3 --device cuda --source-path DATASET --images images_8 --max-views 8 --max-charts 512

    .venv\Scripts\python.exe C:\tmp\wl119_gpu_utilization_audit_20260826\fixed\visible_nurbs_geometry_uv_control_correction.py --checkpoint output\arch_2dgs_coverage_first_surface\2dgs_run1\30000\checkpoint.pt --out output\119_gpu_utilization_audit_fixed_20260826_r3 --device cuda --source-path DATASET --images images_8 --max-views 8 --max-charts 512

## 5. OLD vs FIXED measured result

| 지표 | OLD | FIXED | 변화 |
|---|---:|---:|---:|
| total wall time (s) | 186.351 | 185.494 | -0.46% |
| chart-fitting loop (s) | 43.840 | 43.087 | -1.72% |
| charts/sec, chart loop | 11.679 | 11.883 | +1.75% |
| charts/sec, total | 2.748 | 2.760 | +0.46% |
| GPU util mean, full run | 78.35% | 78.53% | 사실상 동일 |
| GPU util median, full run | 100% | 100% | 동일 |
| GPU util p95, full run | 100% | 100% | 동일 |
| GPU active fraction, full run | 98.94% | 98.93% | 동일 |
| GPU util mean, chart loop | 15.21% | 15.76% | 낮은 수준 유지 |
| GPU util median, chart loop | 15% | 16% | 낮은 수준 유지 |
| GPU util p95, chart loop | 16% | 16% | 동일 |
| GPU active fraction, chart loop | 100% | 100% | 동일 |
| peak GPU memory (MiB) | 9,160 | 9,164 | 사실상 동일 |

full-run GPU 수치는 topology replay의 높은 GPU 사용량을 포함한다. chart loop는 nvidia-smi 1초 sample window로 각각 43/42 samples를 얻었고, 짧은 kernel gap 자체를 sample cadence보다 세밀하게 분해할 수 없다는 한계가 있다. 그럼에도 두 arm에서 평균 15~16%, p95 16%로 일관되게 낮았다.

CPU 보조 측정은 Windows Python launcher와 child를 command-line으로 함께 추적한 값이다. benchmark 중 누적 matching CPU time은 OLD 약 1.251 core-equivalent, FIXED 약 1.254 core-equivalent였으므로 fix 후 host saturation이 해소됐다고 볼 수 없다. launcher/child 집계 특성상 정밀 CPU utilization 수치가 아니라 one-core-plus orchestration indication으로만 사용한다.

## 6. Profiler 및 후보 hotspot 판정

FIXED 2-view/128-chart cProfile은 전체 155.283초, chart loop 11.020초였다.

- topology replay의 _knn cumulative time: 141.312초. 전체 profiler 시간의 약 91%로 chart fitting 전에 발생한다.
- chart fitting의 project_torch_points_to_nurbs: 7.553초, fit_torch_visible_surface_lsq: 4.501초. 이는 fitting/projection 계산이며 serialization bug로 분류하지 않았다.
- _build_pixel_records_vectorized: 128 calls, cumulative 0.015초. bulk transfer 이후 bounded record construction은 material bottleneck이 아니다.
- _chart_scalar_color: 4 calls, cumulative 0.007초. chart-level 후처리 loop로 유지했다.
- label_same_component_blobs와 build_view_chart_pixel_samples는 view당 한 번의 NumPy/scipy 및 tensor grouping 경로이며 per-pixel scalar extraction을 하지 않는다.
- G0/G1/G2 disagreement loop는 mask/index tensor 연산으로 확인됐고 per-pixel Python GPU-to-CPU synchronization은 발견되지 않았다.
- representative-footprint spread는 index_add_와 scatter_reduce_를 실제 사용한다. WL118 inherited accumulate_image_space_pairs도 tensor unique와 index_add_ vectorized accounting path다.

변경한 추가 구현은 _build_pixel_records_vectorized라는 이름 있는 helper 추출뿐이다. fitting mathematics, solve count/order, dtype/precision, topology, renderer, sample population 및 출력 의미는 변경하지 않았다.

## 7. 최종 귀속

판정: CURRENT EXECUTION-GRANULARITY LIMIT

known pixel-record serialization overhead는 제거됐고 chart throughput은 1.7% 개선됐다. 그러나 fixed arm에서도 GPU utilization은 15~16%에 머물렀고, CPU indication은 OLD와 동일했다. profiler와 A/B를 합치면 남은 chart-loop idle은 제거 가능한 대규모 host serialization보다, 최대 약 32 control-point unknown을 가진 많은 독립 per-chart LSQ solve가 순차적으로 실행되는 현재 execution granularity와 짧은 GPU kernel/launch gap에 주로 귀속된다.

full-run 평균 GPU utilization이 약 78%인 것은 topology replay의 _knn이 GPU를 사용하기 때문이며, chart-fitting low-utilization의 반증이 아니다.

## 8. 명시적으로 구현하지 않은 architecture opportunity

여러 chart solve를 크기별 bucket/padding 또는 batched solve로 묶으면 launch amortization을 실험할 수 있다. 그러나 이는 이번 배치에서 금지된 execution-model/architecture 변경이고, solve ordering·reduction order·bitwise 결과 보존 계약을 다시 설계해야 한다. 전체 chart size histogram도 현재 report에 없어 padding 손실을 정량화할 수 없다.

따라서 다른 agent가 제시한 1.5~2.5배는 다음 배치의 검증 가설로는 사용할 수 있지만, 이번 audit의 측정된 p와는 연결되지 않은 추정 범위다. 이번 결과에서 직접 확인된 known fix의 throughput 이득은 chart loop 약 1.7%, total 약 0.5%이며, 1.5~2.5배 wall-clock 개선을 기대치로 확정할 근거는 없다. 다음 배치에서 batching을 승인한다면 먼저 per-chart pixel count/solve time/kernel timeline histogram과 exact-output parity harness를 수집해야 한다.