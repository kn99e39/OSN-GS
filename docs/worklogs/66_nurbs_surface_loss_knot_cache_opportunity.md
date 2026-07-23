# Worklog 66: NURBS surface loss knot vector 캐시 구현

날짜: 2026-07-23

상태: **최소 범위 캐시 구현 및 전체 회귀 검증 완료.**

## 배경

베이스라인 A/B 진행 중 OSN-GS avg_iter가 baseline 대비 약 2.2~2.4배 느린 것을 확인했고, 그 중 NURBS 생성/최적화 파이프라인 쪽에 최적화 여지가 있는지 코드베이스를 검토했다.

## 발견

`osn_gs/surface/torch_nurbs.py::TorchNURBSSurface._basis_tables()`가 `evaluate()` 호출마다 `_clamped_knot_vector()`로 `knots_u`/`knots_v`를 매번 새로 계산한다. 하지만 이 knot vector는 `(control_grid의 shape, degree)`에만 의존하는 정적인 값이라, control point 좌표가 최적화로 바뀌어도 변하지 않는다. shape가 바뀌는 시점은 `surface_rebuild_interval`(기본 1000 iter)마다 도는 maintenance뿐이다.

그런데 `osn_gs/losses/torch_losses.py::nurbs_surface_loss()`(143-148행)는 매 iteration 활성 patch(`surface_loss_patch_budget=16`)마다 `patch.evaluate()`를 호출한다. 각 호출은 u/v knot 생성 뒤 Cox-de Boor basis와 surface point를 계산한다. 이 중 control-grid 좌표·weight·query UV와 무관한 knot vector 생성만 확실한 구조 불변 낭비다. 10000 iteration과 16 active patches 기준 축별 knot vector 생성은 약 32만 회다. 구현 전에는 이 비용이 `surface_loss`의 상당 부분일 가능성을 제기했지만, 아래 isolated benchmark 결과 실제 개선 폭은 CUDA forward 약 6.3%로 제한적이었다.

## 구현

`TorchNURBSSurface`에 생성자·repr·비교에서 제외되는 private cache field를 추가했다. 캐시 키는 control-grid U/V 크기, effective U/V degree, dtype, device이며, `_basis_tables()`가 처음 호출될 때 knot pair를 만들고 이후 재사용한다. control-grid 크기, degree, dtype, device가 달라지면 다음 평가에서 자동으로 무효화하고 다시 만든다.

Control point 좌표와 rational weight 갱신은 knot vector에 영향을 주지 않으므로 매 iteration surface optimizer step에서는 캐시를 유지한다. 체크포인트 형식, surface 생성자 호출부, 학습 수학은 변경하지 않았다.

부가적으로 확인한, 더 큰 리팩터링이 필요한 후보:
- `nurbs_surface_loss`의 patch별 Python for-loop 자체를 batch화(patch마다 control grid 크기가 15~512로 ragged라 padding+mask 방식 필요, 리스크/작업량 큼).
- `uncertain_anchor_loss`/`uncertain_confidence_loss`의 `is_uncertain.any()` GPU→CPU sync — 현재 실 데이터 학습에서 `uncertain=0`이라 우선순위 낮음.

## 검증 및 평가

- `tests/test_nurbs_surface.py`: 첫 평가에서만 두 축 knot를 생성하고, 반복 평가는 재사용하며, degree/control-grid shape 변경은 재생성하는지 검증했다. 캐시를 두 번 사용하는 loss에서도 control grid, weights, UV gradient가 모두 유한함을 확인했다.
- NURBS 집중 테스트: `15 passed`.
- training regression: `20 passed`.
- 전체 pytest: `204 passed, 1 skipped, 8 subtests passed`. 기존 `torch_nurbs.py` requires-grad scalar 변환 warning 1건 외 신규 warning/error는 없다.
- 16 patches, patch당 16×16 control grid, 512 UV, 60 rounds의 isolated forward microbenchmark에서 forced-uncached 대비 CPU `1.039x`, CUDA `1.063x`였다. CUDA 절감은 evaluate당 약 `0.0844 ms`이며 active patch 16개 기준 약 `1.35 ms/iteration`에 해당한다.

## 결론과 남은 위험

순수 캐싱으로 재계산 낭비를 제거했으며 정확도·autograd·checkpoint 계약은 유지됐다. 다만 이 측정은 full training A/B가 아닌 isolated forward benchmark다. 기존 `surface_loss`의 `0.03–0.06s` 전체를 설명하거나 해결하지는 않으며, Cox-de Boor basis recursion과 patch별 Python loop는 그대로 남아 있다.

Ragged patch batching 또는 UV basis cache는 입력/무효화/메모리 정책이 필요한 별도 리팩터링이다. 이번 간단 최적화 범위에는 포함하지 않았고 추가 승인 없이 확장하지 않는다.
