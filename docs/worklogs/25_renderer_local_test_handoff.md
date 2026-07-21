# 25. Renderer Local-Test Handoff

WebGPU를 사용할 수 있는 local machine에서 실행한다. active notebook이나 gaussian-splatting은 수정하지 않는다.

## 입력

WebRenderer는 file://가 아니라 HTTP로 제공한다. 두 개의 분리된 patch가 있는 synthetic run에서 일치하는 point_cloud.ply와 nurbs_surface.json을 불러온다(crease 장면이 적합하다). renderer revision, artifact path, 실행 명령/config, seed를 기록한다.

## 테스트와 통과 조건

1. WebRenderer에서 다음을 실행한다: node --check util/NurbsGeometry.js, node --check main.js, node tests/nurbs_geometry_smoke_test.js. 모두 exit 0이면 통과다.
2. Browser NURBS Surface와 NURBS Curves에서 모든 유효 patches[] entry가 표시되고, patch color가 결정론적이며, reset camera가 모든 patch를 포함하고, U/V line이 각 patch에서 끝나는지 확인한다. console error가 없고 두 patch가 모두 보이면 통과다.
3. Diagnostics에서 patch isolate, Gaussian patch-ID color, sampled-surface/U/V/control-grid/base-curve 독립 toggle, perspective/orthographic 비교를 확인한다. 없는 control은 fitting failure가 아니라 NOT IMPLEMENTED로 기록한다.
4. Parity: patch마다 Python과 JS를 네 corner 및 고정 interior UV 다섯 점에서 평가한다. max Euclidean error <= 1e-6이면 통과다. 실패 시 patch ID, UV, position, degree, shape, knot을 기록한다.
5. Provenance: JSON에는 source path, CLI/config, seed, timestamp, file hash가 있어야 하며 screenshot과 validation report도 같은 hash를 식별해야 한다. 누락 field는 NOT IMPLEMENTED다.

console error와 screenshot을 포함한 PASS/FAIL/NOT IMPLEMENTED table을 반환한다. 모든 check를 통과하기 전 renderer image를 NURBS collapse라고 부르지 않는다.

## TODO 갱신

renderer Priority 0 test를 TODO.md에서 이 handoff로 옮겼다.
