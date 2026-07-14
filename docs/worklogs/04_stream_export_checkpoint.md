# 04. Streaming, Export, and Resume

## 작업 내용

- Stream payload의 `.tolist()`와 JSON materialization을 background worker로 이동했다.
- 모든 NURBS patch와 voxel patch ID를 stream payload에 포함했다.
- PLY 필드 계약을 유지하면서 Python per-row writer를 NumPy vectorized writer로 교체했다.
- Raw Gaussian/optimizer/ADC/multi-patch state를 보존하는 checkpoint v2와 resume 경로를 구현했다.
- Notebook과 독립 CLI에 resume, patch budget, ADC schedule, streaming 옵션을 연결했다.
- Windows CUDA/MSVC 버전 하드코딩을 제거했다.

## 결과

- `STREAM_TO_RENDERER=False`여도 cache snapshot을 남겨 bulk 전송할 수 있다.
- Checkpoint round-trip regression test가 통과했다.
- PLY renderer header regression test가 통과했다.
- Notebook JSON validation이 통과했다.

## 평가

JSON 변환과 파일/network I/O는 worker로 이동했다. GPU-to-CPU snapshot 자체는 일관된 iteration 상태를 확보하기 위해 학습 thread에서 수행하므로, 매우 잦은 full-Gaussian streaming은 여전히 비용이 있다. 기본 1000 iteration 간격을 유지하는 것이 적절하다.
