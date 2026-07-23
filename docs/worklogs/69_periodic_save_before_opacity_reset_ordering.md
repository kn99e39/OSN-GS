# Worklog 69: periodic save가 opacity reset 직후를 캡처하던 순서 버그 수정

날짜: 2026-07-23

상태: **완료.**

## 배경

사용자가 이전 수정판 OSN-GS 10k 실행(`osn_gs_ab_10k_fix`)의 iteration 6000 `render.ppm`을 확인했는데 거의 새까맣게 나왔다. 변환해서 직접 보니 배경 실루엣만 희미하게 보이는 수준이었다.

## 원인

`torch_trainer.py`의 학습 루프 순서가 다음과 같았다:

1. ADC(clone/split/prune)
2. **opacity reset** (`opacity_reset_interval` 기본 3000 → iteration 3000/6000/9000마다 모든 Gaussian opacity를 0.01로 초기화)
3. `state.model.optimizer.step()`
4. periodic save(`save_interval` 기본 1000)

`save_interval`(1000)과 `opacity_reset_interval`(3000)이 겹치는 iteration(3000/6000/9000)에서, **opacity를 막 0.01로 초기화한 직후의 상태를 그대로 저장**해버렸다. 실제 학습이 망가진 게 아니라 저장 순서 때문에 하필 리셋 직후 순간을 스냅샷한 것이었다.

Baseline(`gaussian-splatting/train.py`)은 반대로 `scene.save(iteration)`(155-157행)이 densification/opacity reset(160-170행)보다 먼저 실행돼서 이 문제가 없다.

## 수정

`osn_gs/core/torch_trainer.py`의 `_train_loop`에서 stream snapshot + periodic save 블록을, `state.iteration` 대입/metric scalar 캡처 직후 · surface maintenance/ADC/opacity reset보다 **앞으로** 이동시켰다. Baseline의 "save-before-densify_and_prune" 순서와 동일해졌다. 기존 위치(옵티마이저 step 이후)에 있던 중복 블록은 제거했다.

이 변경으로 저장된 스냅샷은 이제 "이번 iteration의 구조적 변경(clone/split/prune/reset)과 옵티마이저 step 적용 전, 직전 iteration 종료 시점"의 상태를 반영한다 — baseline의 저장 시점 의미와 동일하다.

## 검증

- 전체 pytest: `205 passed, 1 skipped`. 회귀 없음.
- 실제 재학습으로 iteration 6000 render.ppm이 정상적으로(새까맣지 않게) 나오는지는 아직 확인 안 함 — 다음 A/B 재실행 때 확인.

## 참고

Worklog 68에서 추가한 baseline-parity 고정 카메라(`_preview_camera`, 이름순 정렬 첫 train camera)와는 별개 문제였다. 그쪽은 "어느 시점을 봐도 같은 카메라인가"였고, 이번 건 "그 iteration의 어느 순간을 캡처하는가"였다.
