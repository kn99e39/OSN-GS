# Worklog 67: ADC 후 empty_cache() 추가 + timing 로그 주기 10으로 변경 (검증 전, 조치만)

날짜: 2026-07-23

상태: **코드 변경 완료, 회귀 테스트만 확인. 실제 학습으로 스파이크 감소 검증은 아직 안 함(다음 지시 대기).**

## 배경

수정판(screen-size pruning 버그 fix) 10k A/B 학습 로그를 분석하던 중, iteration 7000대 이후(Gaussian 400만개대, VRAM 15.5~15.8GB/16.3GB 사용) 구간에서 `render_loss`/`backward`/`density` 타이밍이 불규칙하게 0.01s~24s를 오가는 현상을 발견했다. 특정 연산이 항상 느린 게 아니라 매번 다른 연산이 랜덤하게 튀는 패턴이라, 순수 계산량 문제가 아니라 **CUDA 캐싱 allocator가 VRAM 한계 근처에서 파편화되어 겪는 stall**로 추정했다.

베이스라인(`gaussian-splatting/scene/gaussian_model.py::densify_and_prune`, 497행)을 확인해보니 매 densification step(100 iteration마다) 마지막에 `torch.cuda.empty_cache()`를 명시적으로 호출하고 있었다. OSN-GS 코드베이스 전체를 grep해보면 `empty_cache`/`memory_allocated`/`memory_reserved` 호출이 단 한 군데도 없었다 — clone/split/prune마다 임시 큰 텐서를 여러 번 만들었다 버리는 구조(`_shape_transaction_candidates` → `_commit_shape_transaction`)인데도 해제된 메모리를 caching allocator가 계속 쥐고만 있는 상태였다.

## 변경 사항

1. **`osn_gs/core/torch_trainer.py`**: ADC(`apply_adaptive_density_control`) 실행 직후, `self.device == "cuda"`일 때 `torch.cuda.empty_cache()`를 호출하도록 추가. 베이스라인과 동일하게 매 ADC step(기본 100 iteration)마다 1회 실행된다.
2. **`timing_log_interval` 기본값을 100 → 10으로 변경** (`torch_trainer.py`의 `TorchTrainingConfig`, `scripts/train_osn_gs_torch.py`, `osn_gs/interop/colab_args.py` 3곳 동기화). 베이스라인의 tqdm 진행바 postfix 갱신 주기(`iteration % 10`)와 맞춘 것으로, 상세 상태 로그(`progress_log_interval`, ADC 로그)는 기존대로 100 유지하고 iteration별 소요 시간 로그만 더 촘촘하게(10배) 찍히도록 분리했다. 목적은 이번처럼 산발적으로 튀는 iteration을 더 잘 잡아내기 위함 — 기존엔 10000 iteration 중 100개 샘플뿐이라 "상위 1%"가 사실상 최댓값 1개와 다를 게 없었는데, 이제 1000개 샘플이 생겨 분포를 조금 더 의미 있게 볼 수 있다.

## 검증

- 전체 pytest: `205 passed, 1 skipped`. 회귀 없음.
- **실제 10k 학습으로 스파이크가 실제로 줄어드는지는 아직 검증하지 않았다.** 사용자 지시에 따라 이번엔 코드 변경 + 기록까지만 하고, 실행 검증은 별도 지시가 있을 때 진행한다.

## 다음 작업

지시가 오면: 이 상태로 실제 A/B(혹은 최소 스모크)를 재실행해 iteration 7000+ 구간의 `total` 타이밍 분포(특히 max/스파이크 빈도)가 이전 대비 개선됐는지 확인한다.
