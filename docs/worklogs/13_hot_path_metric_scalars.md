# 13. Hot-Path Metric Scalar Extraction Removed

날짜: 2026-07-15

## 배경

`docs/worklogs/11_training_bottleneck_audit.md`의 우선순위 1: "hot-path host scalar 추출을 제거하고 MSE/loss 집계를 CUDA에 유지, metric은 progress/snapshot이 host 값을 요구할 때만 materialize한다." 감사만 기록돼 있었고 실제 구현은 착수 전이었다.

## 작업 (`osn_gs/core/torch_trainer.py`, `train()` 루프)

- 카메라별 `image_loss_value += float(mse.detach().cpu())`를 device 텐서 누적(`mse_accum = mse_accum + mse.detach()`)으로 교체했다. 매 view마다 강제되던 GPU→CPU 동기화가 사라진다.
- `mean_mse`를 CPU float에서 다시 device 텐서로 되돌리던 `torch.as_tensor(image_loss_value / n, device=...)` 왕복을 제거하고 `mean_mse = mse_accum / num_cameras`로 device에서 직접 계산한다. 이 텐서가 그대로 `uncertain_confidence_loss`의 `residual_mse`로 들어간다(기존에도 grad 없는 텐서였으므로 수치 동작 동일).
- `state.last_loss`/`state.last_psnr`를 매 iteration `float(total.detach().cpu())`로 뽑던 것을, 새 헬퍼 `_needs_metric_scalars(iteration)`이 True일 때만 materialize하도록 바꿨다. 소비처는 progress 로그, stream snapshot metadata, 저장되는 metrics 파일 세 곳뿐이며, 각각 `_should_log_progress` / `_should_stream_iteration` / (`write_output_files` and `_should_save_iteration`)로 판정한다.
- `state.last_loss`/`last_psnr`는 여전히 float 필드로 유지(state dataclass·checkpoint 포맷 무변경). materialize 안 되는 iteration은 이전 값을 그대로 갖고 있으며, 모든 소비처는 자신이 읽는 iteration에서 반드시 새로 계산된 값을 본다.

## 결과 / 검증

- 마지막 iteration은 `_should_log_progress`가 항상 참(`iteration == iterations`)이라, 루프 종료 후의 강제 final stream/save가 읽는 `last_loss`/`last_psnr`는 항상 그 iteration에서 갓 materialize된 값이다.
- `tests/` 전체 26개 통과(학습 회귀 14개 포함).
- 6-iteration CPU smoke(progress 간격 3): iter 1/3/6에서 loss가 0.4153→0.4122→0.4078로 감소, psnr 5.503→5.600로 증가. 중간 iter 2/4/5는 scalar 추출을 건너뛰고, 최종 `last_loss`/`last_psnr`는 유한·정상값. materialize를 건너뛴 iteration이 최종값을 오염시키지 않음을 확인.

## 평가

NURBS/voxel 구조나 loss 정의는 전혀 바꾸지 않고, 매 iteration 발생하던 host-scalar 동기화(카메라당 1회 + iteration당 1회)만 제거했다. GPU 학습에서 이 동기화는 큐잉된 CUDA 커널을 CPU가 기다리게 해 hot path를 직렬화하던 원인이다(감사 문서 2번 항목).

## 남은 위험 / 후속

- 실제 CUDA 환경에서 iteration 평균 시간 개선 폭은 다음 학습 timing 로그(`surface_loss`/`backward` 분리 형식)로 재확인해야 한다. 이번 검증은 CPU smoke까지다.
- 감사 문서의 우선순위 2(ADC clone/split/prune의 단일 shape transaction화)와 3(snapshot capture 분리 + final snapshot 중복 전송 제거)은 여전히 미착수다. `_clamp_uncertain_confidence`의 매 iteration `is_uncertain.any()` 동기화도 Stage 1(uncertain 없음)에서 불필요하게 남아 있어 후속 정리 대상이다.
