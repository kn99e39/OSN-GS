# Worklog 94 — Boundary-first 표면 품질 측정 게이트

## 상태

- 격리된 Boundary-first annulus 경로에 수치 품질 측정을 추가했다.
- 기존 dispatcher, trainer, renderer, Phase G append adapter는 변경하지 않았다.
- 기존 dispatcher 통합 준비도는 약 55%다.

## 수행 내용

- `torch_boundary_surface_quality.py`에 `measure_boundary_first_surface_quality()`를 추가했다.
- 측정 항목은 다음과 같다.
  - 구성된 degree-one patch의 경계 control sample과 NURBS 평가값의 최대/RMS 오차
  - closed multi-patch의 cyclic seam 최대 오차
  - 경계 corner에서의 최소 Jacobian norm
  - finite 여부와 patch 개수
- 이 측정은 diagnostic이며 eligibility, legacy dispatcher routing, production 경로를 변경하지 않는다.

## 검증

- `curved_annulus` isolated positive control에서 경계 sample 오차와 cyclic seam 오차가 `1e-6` 이내이고 최소 Jacobian norm이 양수임을 확인했다.
- 같은 입력의 payload가 결정적인지 확인했다.
- 표적 회귀: Boundary-first quality/recovery/pipeline 10 passed.
- 공유 작업 트리 전체 pytest: `386 passed, 27 failed, 1 skipped, 1 warning`.
  - 실패 27개는 모두 `tests/test_uncertain_gaussian_append_adapter.py`에 집중됐으며, 현재 별도 Agent가 다루는 append adapter 범위다.
  - Boundary-first 변경이 append adapter의 모델 초기화/transaction 실패를 수정하거나 우회하지 않았다.

## 해석과 한계

- 이 게이트는 현재 materialized patch가 자신이 채택한 support sample과 seam을 정확히 보존하는지 검증한다.
- coarse support sampling이 원본 관측 경계 전체를 얼마나 근사하는지는 별도 측정이다. 따라서 이 결과만으로 curved annulus의 전체 shape fidelity가 충분하다고 주장하지 않는다.

## 남은 위험과 다음 단계

- 원본 outer/hole polyline 대 surface 거리와 normal/curvature 오차를 측정하는 shape-fidelity gate가 필요하다.
- disk/비볼록/다중 루프의 interior support 정책이 필요하다.
- 그 뒤에만 feature-gated dispatcher 연결을 검토한다. append adapter 실패군의 수정은 이 작업 범위 밖이다.