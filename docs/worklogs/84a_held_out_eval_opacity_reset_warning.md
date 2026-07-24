# Worklog 84-A: Held-out 평가 opacity reset 경고 및 결과 메타데이터

날짜: 2026-07-24

상태: **완료. 최종 iteration이 opacity reset과 겹치는 held-out A/B 결과를 명시적으로 식별한다. 학습 순서와 모델 값은 변경하지 않았다.**

## 작업

- `final_iteration_opacity_reset_applies()`를 `osn_gs/eval/held_out_metrics.py`에 추가했다. trainer의 실제 조건(`reset_interval > 0`, `iteration < densify_until_iter`, interval 배수)을 순수 predicate로 동일하게 표현한다.
- `train.py`와 `scripts/train_osn_gs_torch.py`가 `--eval` 종료 후 이 조건이면 경고를 출력한다.
- `held_out_eval.json`에 `post_opacity_reset: true/false`를 기록한다. 후속 비교/집계가 reset 직후 수치를 정상 checkpoint 평가로 오인하지 않는다.
- `tests/test_held_out_metrics.py`에 3000/6000 iteration 양성 사례와 densify 종료·비배수·disabled reset 음성 사례를 추가했다.

## 결과와 평가

`--iterations 3000`처럼 기본 `opacity_reset_interval=3000`의 배수에서 `densify_until_iter` 이전에 끝나는 run은 기존처럼 post-training 모델을 평가하되, 콘솔과 JSON 양쪽에서 **reset 직후 평가**임을 표시한다. 따라서 학습 재현성이나 baseline parity 경로를 바꾸지 않으면서 잘못된 held-out 해석을 차단한다.

## 검증

- `python -B -m unittest tests.test_held_out_metrics` → `4 tests, OK`
- `python train.py --help`, `python scripts/train_osn_gs_torch.py --help` → 두 CLI import/argument 초기화 성공
- `git diff --check` → whitespace 오류 없음(CRLF 변환 경고만 기존 작업 트리에 표시)

## 남은 위험

- 이 변경은 pre-reset 모델을 보존하거나 복원하지 않는다. 비교 목적 run은 reset 배수가 아닌 종료 iteration을 선택하거나 저장 checkpoint를 명시적으로 평가해야 한다.
- Phase G Gaussian proposal 및 production integration은 Worklog 83의 승인 게이트 범위 밖이므로 착수하지 않았다.
