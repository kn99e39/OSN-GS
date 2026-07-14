# 01. Surface State Preservation and Trainable NURBS

## 작업 내용

- Surface rebuild가 새 Gaussian model을 생성하던 경로를 제거했다.
- Rebuild는 voxel, base curve, visible NURBS만 갱신하고 기존 Gaussian parameter와 Adam/ADC 상태를 유지한다.
- Certain Gaussian에 PCA 기반 `(u, v)`와 voxel region ID를 저장한다.
- NURBS control grid와 rational weights에 별도 Adam optimizer를 연결했다.
- Gaussian-to-surface fitting loss와 curvature regularization을 실제 gradient graph에 연결했다.
- Surface fitting loss는 최대 8,192 Gaussian만 사용해 iteration 메모리를 제한한다.

## 결과

- `py_compile` 통과.
- Torch pipeline CPU smoke test 2개 통과.
- 1 iteration 학습에서 Gaussian optimizer와 surface optimizer가 함께 step한다.
- Surface rebuild 시 Gaussian model 교체와 optimizer 재생성이 더 이상 발생하지 않는다.

## 평가

기존의 가장 큰 상태 손실 문제는 제거됐다. NURBS는 이제 export-only 데이터가 아니라 학습되는 중간 표현이다. 현재 UV는 전역 PCA chart 기준이므로 복잡한 장면을 위한 voxel-boundary multi-patch 단계가 다음 작업으로 필요하다.

## 남은 위험

- Surface optimizer 상태는 아직 checkpoint resume에 포함되지 않는다.
- 전역 단일 chart는 접힘이나 분리된 표면을 표현하기 어렵다.
- 실제 CUDA 장기 학습의 수렴성은 이번 빠른 검증 범위에 포함하지 않았다.
