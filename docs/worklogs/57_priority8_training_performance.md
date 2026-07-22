# Priority 8 학습 성능 및 품질 격차 작업

날짜: 2026-07-22

## 수행 작업

- ADC의 clone, split-parent 제거, opacity/size prune를 후보 tensor에서 계산한 뒤 `replace_tensors()` 한 번으로 commit하도록 통합했다. 기존 Gaussian row의 gradient와 Adam state는 최종 keep index로 보존하고 새 child row는 0 state로 시작한다.
- full Gaussian snapshot은 CUDA tensor를 pinned CPU memory로 `non_blocking=True` 복사하고 CUDA event만 worker에 전달한다. worker queue 기본 크기는 2이며 `--stream_queue_size`로 조절한다.
- 동일 iteration snapshot을 이미 enqueue했다면 종료 시 forced snapshot을 다시 만들지 않는다. cache와 WebSocket은 동일한 JSON 문자열을 재사용한다.
- surface maintenance는 기본 16개 patch를 round-robin으로 검사한다. `--surface_maintenance_patch_budget 0`은 이전처럼 모든 patch를 검사한다. UV refresh와 support-mask refresh도 선택된 patch만 수행한다.
- 학습 view의 결정론적 순환을 seed 기반 epoch별 `randperm`으로 교체했다. 각 epoch에서 모든 view를 중복 없이 한 번씩 사용하며 같은 seed는 동일 순서를 재현한다.
- 노트북 Train 셀에 `STREAM_QUEUE_SIZE=2`, `OSN_SURFACE_MAINTENANCE_PATCH_BUDGET=16`을 추가하고 두 CLI에 같은 기본값을 연결했다.

## 결과

- ADC clone+split 회귀에서 shape transaction 호출 횟수는 1회다. 이전 경로는 clone append, split append, split-parent prune, final prune가 각각 full tensor/Adam rebuild를 유발할 수 있었다.
- 실제 CUDA pinned-copy 검사: destination `cpu`, `is_pinned=True`, source와 값 동일.
- DATASET에서 CUDA 2-iteration smoke를 수행했다. 138,766 Gaussians snapshot은 `00000002.json` 한 개만 생성됐고 NURBS payload를 포함한 JSON 검증을 통과했다. iteration 2 종료 시 duplicate final snapshot은 없었다.
- 전체 테스트 150개 통과, 1개 skip. 학습 회귀 테스트는 19개 통과했다.

## 평가

- 정상 iteration의 학습 수식, D-SSIM, NURBS loss, optimizer 순서는 바꾸지 않았다.
- ADC shape 변경은 기존 성장/prune 순서의 결과와 report 의미를 유지하면서 full parameter/Adam 재할당을 한 번으로 줄였다.
- maintenance 비용은 patch 수에 대해 bounded된다. 기본 16개보다 patch가 많으면 각 patch의 residual patience가 wall-clock iteration 기준으로 느리게 누적되지만, 검사를 생략하지 않고 round-robin으로 순환한다.
- view sampling은 의도적인 품질 경로 변경이다. 기존 순환 순서와 수치 동일성을 목표로 하지 않고 original 3DGS와 같은 without-replacement shuffle 특성을 복원한다.

## 남은 위험

- 기존 `output/osn_gs_scene/5000`은 OSN-GS 반해상도이고 `output/scene` baseline은 full resolution이라 품질 비교에 사용할 수 없다.
- 실제 품질 acceptance는 동일 dataset·동일 해상도·10k 조건의 OSN-GS/Graphdeco 재학습과 동일 holdout 평가가 필요하다.
- CUDA 2-iteration smoke는 pinned snapshot correctness를 검증했지만, 190k 이상 Gaussian에서 ADC spike 감소 폭은 다음 10k timing 로그로 정량화해야 한다.