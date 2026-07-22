# 12. 표면 손실 패치 미니배치

날짜: 2026-07-14

## 관측

Notebook 저장 로그를 확인했다. iteration 900-1900에서 render_loss는 0.005-0.007초였지만 backward는 0.294-0.321초였고 평균 iteration은 약 0.33초였다. iteration 1000/2000의 surface maintenance는 약 2.2초 spike였지만 매 iteration 평균 병목은 아니었다.

## 원인

NURBS surface loss가 최대 8,192 Gaussian sample에 대해 108개 patch를 매 iteration 순회했다. 각 patch의 bool(mask.any())는 CUDA tensor를 Python bool로 전환해 GPU synchronization을 일으킬 수 있었고, patch 평가와 smoothness도 모든 patch에 반복 적용됐다. 기존 timing의 backward에는 surface loss forward 시간까지 포함되어 있었다.

## 작업

- surface_loss_patch_budget을 추가했다. 기본값은 16이며 0은 기존 full-patch 동작을 유지한다.
- patch ID를 round-robin 방식으로 선택해 108개 patch가 약 7 iteration마다 한 번씩 loss에 참여하도록 했다.
- bool(mask.any())를 제거하고 active patch 내부의 empty tensor operation과 index_copy 기반 anchor assembly로 변경했다.
- smoothness도 동일한 active patch subset에 적용한다.
- trainer timing에 surface_loss를 분리해 다음 학습에서 forward surface 비용과 backward 비용을 구분한다.
- notebook Train 셀에 OSN_SURFACE_LOSS_PATCH_BUDGET을 노출했다.

## 평가

NURBS와 voxel을 비활성화하거나 구조를 단순화하지 않고, NURBS 최적화 스케줄만 GPU 친화적으로 바꿨다. 모든 patch가 계속 학습되되 매 iteration의 Python loop, kernel dispatch, GPU synchronization 수가 patch 수에 비례해 증가하지 않도록 제한된다.

## 검증

- patch minibatch loss가 finite value와 selected control-grid gradient를 생성하는 회귀 테스트 추가.
- Torch smoke/regression test 16개 통과.
- smoke timing에서 surface_loss와 backward가 별도 항목으로 출력되는 것 확인.

## 남은 위험

- 실제 GPU scene에서 최종 절감 폭은 control-grid 해상도와 active patch의 Gaussian 분포에 따라 달라진다. 다음 학습 timing에서 surface_loss와 backward를 확인해야 한다.
- patch minibatch는 full-patch smoothness의 unbiased schedule이지만, 극소 patch의 갱신 빈도는 후속 density-aware sampling으로 개선할 수 있다.
