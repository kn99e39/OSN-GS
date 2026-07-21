# 15. 학습 병목 감사

날짜: 2026-07-14

## 근거

notebook에서 완료된 10,000 iteration 실행은 현재의 분리 timing을 이미 기록한다.

    iteration=10000 ... render_loss=0.013s surface_loss=0.056s backward=0.123s
    optim=0.002s density=4.644s save=0.032s log=0.001s total=4.893s avg_iter=0.188s

과거 저장 출력은 surface_loss와 backward를 분리하지 않았으므로, 약 0.33초의 normal iteration 수치로 현재 patch-minibatch 경로를 평가할 수 없다.

## 병목

1. Surface loss는 계속해서 가장 큰 정상 상태의 OSN-GS 고유 비용이다.
   - 최대 8,192개의 certain Gaussian을 sample하고, active patch로 분류한 뒤 최대 16개의 rational patch를 평가하고 control grid를 거쳐 매 iteration backward한다.
   - 16-patch round-robin budget이 비용을 제한하지만 surface_loss=0.056초와 backward=0.123초에 대한 기여는 여전히 크다. renderer 비용이 아니라 의도적인 구조 비용이다.

2. iteration별 CUDA-to-CPU metric 추출이 hot path를 직렬화한다.
   - 각 view가 float(mse.detach().cpu())를 수행한다.
   - 매 iteration마다 float(total.detach().cpu())도 수행한다.
   - 이 연산은 CPU가 queue된 CUDA kernel을 기다리게 하고, uncertainty loss에도 새 CPU scalar를 통해 전달한다. metric은 progress cadence가 host 값을 요구할 때까지 device tensor로 유지해야 한다.

3. ADC가 지배적인 주기적 spike다.
   - 100 iteration마다 실행한다.
   - clone, split, prune은 모두 full Gaussian parameter tensor를 다시 만들며 Adam state 보존을 위해 parameter group별 moment tensor를 할당·복사한다.
   - 약 190k Gaussian에서 iteration 10,000의 density stage가 4.644초였다. 이는 CPU-only ADC가 아니라 GPU tensor allocation/copy와 report scalar 추출에 따른 GPU synchronization의 조합이다.

4. Surface maintenance는 별도의 주기적 global scan이다.
   - 1,000 iteration cadence에서 모든 certain Gaussian의 UV를 갱신하고 모든 patch의 residual을 평가한다.
   - local correction은 드물지만 failed patch에 대해 voxel rebuild를 유발한다. voxel normal은 chunked GPU cdist/SVD를 쓰지만 mixed-resolution adjacency와 connected-component label은 region bound/normal을 CPU로 넘기고 Python loop를 사용한다.
   - normal iteration에는 영향을 주지 않지만 1,000 iteration 경계에서 ADC 및 streaming과 겹칠 수 있다.

5. Full snapshot streaming은 설정된 checkpoint에서 비용이 크다.
   - STREAM_MAX_GAUSSIANS = 0이면 모든 Gaussian의 position, scaling, rotation, opacity, color를 GPU에서 CPU로 복사한다.
   - NURBS와 voxel payload도 모든 patch/control grid 및 voxel-region array를 복사한다.
   - iteration 10,000에서는 stream schedule과 final forced streaming이 겹쳐 같은 snapshot이 두 번 전송된다.

6. Timing instrumentation도 CUDA를 synchronize하지만, 100 iteration logging cadence에서만 수행된다.
   - stage별 수치를 얻는 데 필요하며 0.188초 평균의 원인은 아니다.

## 우선순위

1. hot-path host scalar 추출을 제거하고 MSE/loss aggregation을 CUDA에 유지한다. progress 또는 snapshot metadata가 필요할 때만 metric을 host에 materialize한다.
2. ADC interval마다 clone -> full rebuild -> split -> full rebuild -> prune -> full rebuild를 피하도록 ADC tensor/Adam-state 확장과 prune을 단일 shape transaction으로 재구성한다.
3. bounded pinned-memory copy 정책으로 snapshot capture를 학습과 분리하고, final snapshot 중복 전송을 제거한다.
4. 현재 persistent NURBS/voxel lifecycle은 유지한다. 구조 목표를 약화시키기보다 명시적 cadence에서만 maintenance를 최적화한다.

이 감사는 성능 동작을 변경하지 않았다.
