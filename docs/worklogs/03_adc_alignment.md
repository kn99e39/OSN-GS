# 03. Original-3DGS-Aligned ADC

## 작업 내용

- 강제 상위 10% gradient fallback을 제거하고 실제 threshold와 tracked mask만 사용한다.
- densify start/end/interval, opacity reset, screen-size prune 시작 시점을 분리했다.
- Split offset을 부모 quaternion orientation으로 회전한다.
- Clone/split child가 UV, patch ID, uncertainty metadata를 상속한다.
- Opacity/screen/world prune 사유를 각각 로그에 기록한다.
- Exponential position learning-rate schedule을 연결했다.
- Uncertain-to-certain promotion은 추가하지 않았다.

## 결과

- ADC가 설정 임계값을 우회하지 않는다.
- 대량 prune 발생 시 원인별 개수를 확인할 수 있다.
- Gaussian shape 변경 시 기존 Adam row-state 보존 경로를 계속 사용한다.

## 평가

기존 구현보다 원본 3DGS 정책에 가깝다. CUDA 장기 학습에서 densification 분포와 opacity-reset 직후 품질 변화를 확인할 필요는 있지만, 긴 검증은 이번 단계에서 수행하지 않았다.
