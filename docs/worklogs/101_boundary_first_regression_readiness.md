# Worklog 101 — Boundary-first Regression Readiness

## 전체 검증

- 전체 pytest: `473 passed, 3 failed, 1 skipped, 1 warning`.
- Boundary-first ordered loop, central cap, multi-component review, source RMS gate, pole-aware regularity 회귀는 통과했다.

## 남은 실패

- `tests/test_annulus_chart.py`: independent-fit의 알려진 bad-seed orientation-flip 기대값 불일치.
- `tests/test_trimmed_component_fitter.py` 2건: flat fit의 `degenerate_fraction` 기대값 불일치.
- 이 3건은 기본 constructor/공유 geometry 범위이며, isolated Boundary-first runner가 dispatcher나 production training을 변경한 결과가 아니다.

## 결론

- isolated feature-gated benchmark는 review artifact로 사용 가능하다.
- 기본 dispatcher/production integration은 아직 준비되지 않았다.
- 전제: multi-hole planar-domain decomposition, concave interior-support topology, normal/curvature gate, 위 전체 regression 실패 해소.