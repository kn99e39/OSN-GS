# 14. 합성 NURBS 생성자 벤치마크

날짜: 2026-07-13

## 작업

- 프로젝트 루트의 격리된 `nurbs_constructor_benchmark/` 패키지를 추가했다.
- plane, sine sheet, sharp crease의 deterministic synthetic Gaussian-center scene과 analytic residual/normal oracle을 구현했다.
- benchmark가 별도 또는 복사된 constructor가 아니라 실제 `TorchOSNGSPipeline.initialize()`를 호출하도록 구성했다. 따라서 voxel regioning, patch fitting, LSQ/IDW mode, foot-point UV binding, `TorchGaussianModel` 초기화가 production 경로 그대로 검증된다.
- 입력-point foot-point RMS, surface chart residual, normal angular error, patch/control-point 수, finite 상태를 JSON report로 저장했다.
- `--max-fit-rms`, `--max-chart-rms` 선택 인자를 추가해 임계값 초과 또는 non-finite 결과에서 non-zero exit하는 regression gate로 만들었다.
- 사용법과 scene 추가 방법은 `nurbs_constructor_benchmark/README.md`에 기록했다.

## 결과

- 새 파일의 AST syntax parse는 통과했다.
- 현재 작업 환경의 기본 Python에는 `torch`가 설치돼 있지 않아 production pipeline runtime benchmark는 실행할 수 없었다 (`ModuleNotFoundError: torch`). `requirements.txt`는 `torch>=2.1`을 요구한다.

## 평가

Constructor를 복사하지 않았으므로 이후 `osn_gs/surface/torch_nurbs.py` 또는 `osn_gs/core/torch_pipeline.py`의 개선이 같은 benchmark에 즉시 반영된다. 실제 scene에는 없는 ground truth를 synthetic oracle로만 제공해, production 입력/생성 경로와 정량 평가 기준을 분리했다.

## 남은 위험

- 기본 임계값은 의도적으로 강제하지 않았다. 실제 PyTorch/CUDA 환경에서 baseline report를 먼저 기록한 뒤 CI 또는 regression threshold를 정해야 한다.
- analytic chart residual은 symmetric Chamfer distance가 아니다. 부분 관측 coverage나 topology 품질을 더 엄밀히 비교해야 하면 다음 단계에서 reference-surface sampling 및 bidirectional distance를 추가할 수 있다.
