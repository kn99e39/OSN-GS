# 33. Renderer SH Streaming

날짜: 2026-07-20

## 수행 내용

- 두 OSN-GS snapshot producer(TorchOSNGSTrainer와 legacy train_with_ws_runtime.py hook)가 model의 active SH degree와 coefficient-major RGB SH data를 내보내도록 확장했다.
- packed WebSocket payload에 shDegree와 평탄화된 shCoefficients를 추가했다. Gaussian 하나당 coefficient 수는 (shDegree + 1)^2이며 tensor layout은 [gaussian, coefficient, RGB]다.
- renderer protocol 문서를 갱신했다. WebGPU renderer는 Gaussian Composition에서 이 field를 사용하며, 해당 field가 없는 기존 snapshot에 대해서는 DC-color 호환성을 유지한다.

## 결과

Live OSN-GS snapshot이 모든 streamed Gaussian을 DC RGB로 축약하지 않고 CUDA rasterizer와 동일한 active SH coefficient를 보존할 수 있다. browser가 갱신된 producer payload를 받으면 WebGPU Composition 경로에서 view-dependent color를 사용할 수 있다.

## 평가

- 두 producer file에 대한 Python AST parsing을 통과했다.
- renderer 측 packed-snapshot SH decoding은 3DGS_Renderer/tests/gaussian_stream_sh_smoke_test.js로 검증한다.
- 이 환경에서는 browser target을 사용할 수 없어 end-to-end WebSocket/browser render는 실행하지 않았다.

## 남은 위험

Degree-3 JSON snapshot은 SH만으로 Gaussian당 float 48개를 전달한다. 이전 stream의 RGB 값 3개와 비교하면 serialization, network, browser upload 비용이 크게 증가한다. cadence와 크기를 제어하려면 stream_every와 stream_max_gaussians를 사용해야 하며, 대규모 장면에서 full-resolution degree-3 streaming을 자주 활성화하기 전에는 binary snapshot format을 검토해야 한다.
