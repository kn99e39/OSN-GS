# 13. Surface Loss Runtime Audit

날짜: 2026-07-14

## 관측

Notebook에 저장된 10,000 iteration run은 iteration 900~2000에서 평균 0.319~0.334초였다.

- render_loss: 0.005~0.007초
- backward: 0.294~0.332초
- 일반 density: 약 0.010초
- 1,000 / 2,000 iteration spike: surface maintenance와 ADC

따라서 지속적인 병목은 rasterizer가 아니라 NURBS surface loss와 그 gradient다. 해당 과거 timing 형식에는 surface_loss 항목이 없으므로 현재 trainer의 분리 timing 및 patch minibatch 적용 전 run이다.

## 현재 경로

현재 trainer는 surface_loss_patch_budget 기본값 16을 사용한다.

- 108 patch 전체를 매 iteration 평가하지 않는다.
- 16 patch를 round-robin으로 순환하며 NURBS anchor/smoothness loss를 계산한다.
- 0을 지정할 때만 모든 patch를 매 iteration 평가한다.
- trainer는 surface_loss와 backward 시간을 별도 timing field로 기록한다.

Train entrypoint에 patch_budget startup log를 추가했다. 다음 run의 output에는 다음이 나타나야 한다.

OSN-GS surface loss: patch_budget=16 (0=all patches)

## 검증 기준

새 run에서 surface_loss timing field가 존재하고 patch_budget=16이면 이전 0.33초 로그와 같은 full-patch path가 아니다. 이후에도 surface_loss 또는 backward가 지속적으로 높으면, 해당 새 timing을 기준으로 topology batch evaluation까지 추가 최적화한다.
