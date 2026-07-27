# Worklog 105 — Boundary-first 전체 회귀 checkpoint

## 상태

진행 중. isolated Boundary-first 변경 후 전체 pytest를 실행했다.

## 전체 pytest 결과

```text
475 passed, 3 failed, 1 skipped, 1 warning, 8 subtests passed
```

이번 작업 시작 전에 이미 관측된 실패 3건은 그대로다.

1. `tests/test_annulus_chart.py::AnnulusOGridChartTest::test_known_bad_seed_reproduces_inner_corner_degeneracy_under_independent_fit`
   - 기대 orientation flip sample 8, 실제 0.
2. `tests/test_trimmed_component_fitter.py::TrimmedComponentFitterTest::test_fits_flat_plane_with_low_residual`
   - 기대 degenerate_fraction 0.0, 실제 약 0.001736.
3. `tests/test_trimmed_component_fitter.py::TrimmedComponentFitterTest::test_jacobian_metrics_detect_a_healthy_flat_fit`
   - 기대 degenerate_fraction 0.0, 실제 약 0.001736.

warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor를 scalar로 변환하는 경고 1건이다.

## 해석

- 이번에 추가한 source-boundary fidelity 단위 검증과 sine false-hole support 회귀를 포함해 통과 수는 이전 checkpoint 대비 2건 늘었다.
- 실패한 annulus/trimmed fitter 모듈은 이번 변경 범위가 아니며, default dispatcher나 production code도 변경하지 않았다.
- 전체 suite의 기존 실패를 이 작업 중 조용히 수정하지 않는다. 별도 소유자/범위 확인이 필요한 공용 회귀다.

## 다음 작업

- 1-cell raster tolerance의 point count/seed sweep을 추가하고 deterministic review evidence를 보강한다.
- source-boundary fidelity threshold는 sweep을 근거로 제안하되 아직 자동 gate로 승격하지 않는다.
- 현재 국소 구현 진행률은 약 86%다.
## Attribution 보정 — annulus orientation fixture

초기 checkpoint의 “annulus 실패는 이번 작업과 무관” 판단은 철회한다.

- HEAD의 `torch_component_boundary.py`를 격리 로드한 동일 seed=14 fixture는 `total_orientation_flip_samples = 8`을 재현했다.
- 현재 extractor에서 `frame_margin=0.0`도 `8`, canonical `frame_margin=0.05`는 `0`이다.
- 따라서 failure는 이번 Boundary-first observed-support frame 확장에 직접 따른 **intentional contract change로 stale해진 fixture**다.
- fixture는 old bad-path 재현을 요구하지 않고, canonical extractor에서 `orientation flip = 0`, `uv_overlap = false`, `near_degenerate_slice_count = 0`을 검증하도록 갱신했다.

trimmed fitter의 degenerate-fraction 2건은 이 분리 실험과 무관하며 별도 attribution이 계속 필요하다.