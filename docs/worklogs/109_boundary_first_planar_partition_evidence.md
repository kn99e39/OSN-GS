# Worklog 109 — Boundary-first planar partition evidence foundation

## 상태

진행 중. multi-loop NURBS materialization 전의 안전한 non-overlap evidence foundation을 추가했다.

## 수행 내용

- `torch_boundary_planar_partition.py`를 추가했다.
- 하나의 outer loop와 복수 hole에 대해 다음을 확인한다.
  - ordered loop 존재
  - hole nesting ownership
  - UV signed area degeneracy
  - outer/hole orientation 관계
  - outer boundary owner count = 1
- 유효한 role evidence가 있어도 chart를 생성하지 않고 `partition_materialization_required` review 상태를 반환한다.

## 검증

```text
1 test OK
- tests.test_boundary_planar_partition
```

## 경계

이 foundation은 multi-loop chart materialization이 아니다. non-overlapping planar-domain partition을 실제 patch network로 만드는 작업은 별도 범위로 남아 있으며, dispatcher/production은 변경하지 않았다.

## 2026-07-27 후속 — exporter review layer 구현

v6 artifact(선형 fan/wedge)를 정상 결과로 취급하지 않기 위해, 다음 단계로 예정돼 있던 exporter review layer를 이 격리 경로에 추가했다.

### 수행 내용

- `osn_gs/surface/torch_boundary_first_visible_builder.py`
  - `_materialize_boundary_role_network()`가 이제 `observed_outer_boundary`(항상), 그리고 역할에 따라 `observed_inner_boundary`(interior_boundary) 또는 `observed_interior_anchor`(interior_anchor)를 provenance에 JSON-safe list로 기록한다.
  - `interior_support_crosses_unobserved_region`으로 거부되는 unsupported 경우에도 관측 evidence는 그대로 남긴다.
  - 이 값은 raw 관측 loop/anchor이며, resampling되기 전 원본이다.
- `nurbs_constructor_benchmark/boundary_first_support_runner.py`
  - `_boundary_first_review_layers()`를 추가해 각 `visible_results` 항목에 `boundary_roles`, `correspondence`, `observed_outer_boundary`, `observed_inner_boundary`, `observed_interior_anchor`, `reconstructed_outer_boundary`, `reconstructed_inner_boundary`, `support_curves`, `patch_seams`를 붙인다.
  - closed support-network(annulus류) 경우 `support_curves`/`reconstructed_*_boundary`는 이미 materialize된 patch의 control grid에서 **손실 없이** 그대로 뽑아낸다 (`support_curves[index]`가 patch의 u=0 control row와 정확히 일치하므로 별도 재표본화가 필요 없다).
  - observed-anchor cubic fan(central cap) 경우 `support_curves`는 pole→boundary sample 2점 spoke이고, `reconstructed_outer_boundary`는 각 patch의 P0 control point를 순서대로 모은 것이다.
  - `_seam_payloads()`가 component-local seam index를 export 전역 patch id로 변환해 `patch_boundaries`를 채운다. 기존에 항상 `patch_boundaries=[]`였던 문제를 해결했다.
- `nurbs_constructor_benchmark/boundary_first.py::renderer_payload()`
  - 새 keyword-only 인자 `boundary_first_review`를 추가했다. 기본값 `None`이며 넘기지 않으면 payload에 키가 생기지 않아 legacy dispatcher(`construct_boundary_first`) 호출부와 `runner.py`, 기존 `test_patch_boundary.py` 호출은 전혀 영향받지 않는다.
  - isolated runner는 이 인자로 `nurbs_surface.json`에도 `boundary_first_review` 배열을 심는다. 렌더러가 아직 이 필드를 그리지 않아도 JSON에는 먼저 명시한다는 지시를 반영했다.

### 검증

```text
tests/test_boundary_first_visible_builder.py       3 passed (+observed_outer/inner_boundary 길이 검증 추가)
tests/test_boundary_first_support_runner.py         3 passed (+2 new: annulus review layer, plane anchor spoke review layer)
tests/test_boundary_support_network.py              4 passed
tests/test_boundary_constrained_surface.py
tests/test_boundary_central_cap.py
tests/test_boundary_surface_quality.py
tests/test_patch_boundary.py
tests/test_boundary_first_support_pipeline.py       총 26 passed (선택 묶음)
```

전체 pytest: `483 passed, 2 failed, 1 skipped, 1 warning, 8 subtests passed`. 실패 2건은 여전히 `tests/test_trimmed_component_fitter.py`의 `degenerate_fraction` strict-zero 기대치(실측 약 0.0017361111)이며 이번 변경과 무관한 기존 실패다.

### 남은 위험과 다음 작업

- support crossing gate는 아직 미구현이다. auto correspondence가 고른 support curve끼리 교차하는지 검출하지 않는다.
- support/boundary는 여전히 선형 보간(interior)과 cubic 순수 원주 방향(circumferential)뿐이며, observed sample 기반 curved LSQ fit은 아직 없다.
- bidirectional source-boundary fidelity(재구성→관측, 관측→재구성 양방향), false-hole persistence/raw-support/genuine-small-hole negative control은 이번 범위에 포함하지 않았다.
- v6/v7 artifact를 eligible/production-ready로 선언하지 않았다. dispatcher/production/trainer/uncertain Gaussian append/checkpoint/multi-hole materialization은 이번 작업에서 변경하지 않았다.