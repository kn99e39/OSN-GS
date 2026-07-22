# 7. 밀도 적응형 복셀 NURBS

날짜: 2026-07-10

## 작업

- 고정 해상도 voxel을 coarse-to-fine 밀도 적응형 voxel로 교체했다.
- 초기 구성에서는 Gaussian 개수를 밀도로 사용하고, surface rebuild에서는 `opacity * clamp(median covariance volume / covariance volume, 0.1, cap)`을 밀도 가중치로 사용한다.
- occupied coarse cell의 밀도 quantile 이상인 영역만 설정된 depth까지 세분화한다.
- 서로 다른 level의 cell을 공통 finest-grid AABB로 표현해 face adjacency와 normal boundary 분리를 유지한다.
- patch 밀도와 boundary 비율을 이용해 제한된 NURBS control-point 예산을 patch별로 배분한다.
- voxel level, weighted density, bounds를 streaming/export payload에 포함했다.
- notebook Train 셀에 adaptive toggle, subdivision depth, density quantile, covariance weight cap을 추가했다.

## 결과

- 기본 coarse grid는 16이고 depth 1일 때 밀집 영역의 최대 해상도는 기존 32 grid와 같다.
- 희소 영역은 coarse cell로 남고 밀집 영역만 더 작은 NURBS 배치 영역을 만든다.
- `max_surface_control_points` 전역 한도는 계속 적용되어 NURBS scale 증가 시 메모리 폭증을 제한한다.
- 기존 Gaussian 모델은 surface rebuild 때 교체되지 않으며 uncertain-to-certain promotion도 추가하지 않았다.
- Torch smoke/regression test 12개 통과.

## 평가

복잡한 scene 전체를 하나의 균일 영역으로 뭉뚱그리는 문제를 줄이면서, Gaussian 밀집도와 covariance 크기가 실제 patch 세분화와 NURBS 표현력에 반영된다. topology 계산은 매 iteration이 아니라 기존 surface rebuild 시점에만 수행하므로 학습 루프의 상시 비용은 늘리지 않는다.

## 남은 위험

- rebuild 사이에 밀도가 quantile 경계에서 오갈 경우 patch ID topology가 바뀔 수 있다. 지속적인 patch ID matching 또는 hysteresis는 후속 안정화 항목이다.
- 현재 visibility EMA는 Gaussian state에 저장되지 않아 밀도식에 직접 포함하지 않았다.
- depth를 크게 올리거나 quantile을 낮추면 region 수와 normal 추정 비용이 빠르게 증가할 수 있다.
