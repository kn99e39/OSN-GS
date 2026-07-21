# 24. Renderer Multi-Patch 검증

날짜: 2026-07-15

## 범위

active baseline notebook이나 gaussian-splatting project를 수정하지 않고, 이미 commit된 WebRenderer revision 88477a8을 검증했다.

## 확인된 구현

- NurbsGeometry.buildGeometry()가 patches[]의 모든 유효 entry를 사용하며 array가 존재할 때 top-level primary patch를 중복하지 않는다.
- render된 각 patch에는 결정론적인 color variation이 적용된다.
- surface와 iso-line vertex는 patch별로 조립되므로 line segment가 patch boundary를 넘어 연결되지 않는다.
- geometry bound가 모든 유효 patch를 포함하며 기존 camera-reset path에서 사용된다.
- flattened control grid, malformed patch, patch ID, skipped-patch reporting은 tests/nurbs_geometry_smoke_test.js가 검증한다.

## 검증

- static code review로 WebRenderer/util/NurbsGeometry.js와 WebRenderer/main.js의 동작을 확인했다.
- repository에는 single-patch, multi-patch, flattened-grid, malformed-patch를 다루는 Node smoke test가 있다.
- 이 환경에는 node가 없어 runtime 실행은 하지 못했다. browser나 WebGPU runtime test도 시도하지 않았다.

## 남은 Priority 0 작업

- Patch isolate/toggle UI.
- assigned patch ID에 따른 Gaussian coloring.
- sampled surface point, U/V iso-line, control grid, diagnostic curve의 독립 toggle.
- Python/JavaScript 수치 NURBS parity test.
- export provenance field와 artifact-to-run linkage.
- perspective/orthographic 비교 control.

## TODO 갱신

TODO.md에서는 확인된 multi-patch rendering, deterministic patch-color, cross-patch iso-line, all-patch bound 항목만 제거했다.
