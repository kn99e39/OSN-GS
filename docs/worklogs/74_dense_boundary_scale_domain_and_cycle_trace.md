# Worklog 74: Dense Boundary Scale Domain 및 Cycle Trace

## 측정

Worklog 72/73 production candidate/threshold/connectivity는 변경하지 않았다. `torch_dense_boundary_scale_diagnostics.py`가 진단 전용으로 (1) `no_candidate_within_local_scale` half-line의 방향상 최근접 dense candidate 거리, (2) dense candidate nearest-neighbor spacing, (3) distance-stage cycle edge의 normal/tangent/ambiguity/mutuality 생존을 기록한다.

후보 normal은 full-cloud covariance eigenframe의 `normal_candidate`이고 tangent은 `cross(normal, local missing-sector outward direction)`이다. 따라서 cycle-breaking tangent 판단은 covariance-derived Gaussian orientation에 직접 의존한다.

## baseline-compatible@2900

7개 region에서 boundary-support candidate spacing / full-evidence spacing median은 2.08--3.72배다. representative spacing, full-evidence spacing, candidate spacing은 report에서 분리했다.

`no_candidate` 1,068 half-line의 방향상 최근접 후보 normalized distance는 region median 3.89--4.79, p95 8.53--12.64다. bucket은 `<=2.5`: 0, `2.5--5`: 660, `5--10`: 366, `>10`: 42다. 따라서 68% 거리 실패를 genuine boundary absence로 단정할 수 없고, full-evidence spacing이 filtered sparse boundary-support connectivity에 부적합할 가능성이 강하다.

candidate angular largest-gap diagnostic은 19.6--72.0도이고 bbox span은 Worklog 73의 0.89--1.00으로, 후보가 작은 sector에만 국한된 coverage 실패도 지지되지 않는다.

distance-stage cycle edge 34개 중 16개 mutuality까지 생존, 13개 tangent incompatibility, 5개 ambiguity/nonreciprocity로 제거됐다. tangent 제거 edge는 distance ratio 0.88--2.48인데 tangent margin이 -0.01~-0.49인 사례가 주로 관측됐다. orientation/tangent은 scale mismatch와 독립된 cycle-destruction 병목이다.

## 판정

다음 production batch에서는 threshold를 즉시 바꾸지 않는다. 다만 별도 `boundary_support_spacing` contract 후보를 검증할 근거는 충분하다. 동시에 covariance-derived tangent이 cycle을 깨므로, 다음 실험은 explicit structural-normal/tangent A/B여야 한다. hull, bridge, forced closure, fitting 변경은 금지 유지.

## 검증

focused tests: `tests/test_dense_boundary_connectivity_diagnostics.py`, `tests/test_dense_boundary_scale_diagnostics.py` **2 passed**. Full pytest 미실행.
