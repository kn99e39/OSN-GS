# 9. CUDA 빌드 사전 점검

날짜: 2026-07-14

## 작업

- CUDA diff Gaussian rasterizer JIT build 이전의 native-toolchain preflight를 추가했다.
- Windows training process에서 x64 MSVC environment를 활성화한 뒤 cl.exe, INCLUDE, LIB를 확인한다.
- PyTorch extension builder가 사용할 CUDA_HOME, nvcc.exe, Python Ninja package도 확인한다.
- train.py와 scripts/train_osn_gs_torch.py가 scene loading과 trainer 초기화 전에 preflight를 호출하도록 연결했다.
- --skip_cuda_build_preflight escape hatch를 추가했지만 기본값은 검증이다.
- 기존 JIT loader도 같은 validation 함수를 사용하므로 preflight와 실제 build의 환경 설정이 분리되지 않는다.
- compiler directory를 process PATH 맨 앞에 강제하고, PyTorch와 동일한 where cl probe가 성공해야만 preflight를 통과하게 했다.

## 결과

현재 Windows 환경에서 preflight는 Visual Studio 18 Community의 HostX64/x64 cl.exe와 CUDA 13.3 nvcc.exe를 확인했고, MSVC PATH를 현재 Python process에 성공적으로 주입했다.

## 평가

이전에는 rasterizer JIT가 내부에서 where cl 실패 후 fallback 차단 오류로 나타났다. 이제 dataset load 및 GPU tensor allocation 이전에 toolchain 원인을 직접 보여주며, MSVC가 설치됐지만 notebook process PATH에 없는 경우에는 자동 활성화한다.

## 검증

- 실제 .venv Python에서 preflight 통과.
- 확인 항목: cl.exe, INCLUDE/LIB, CUDA_HOME, nvcc.exe, Ninja.

## 남은 위험

- preflight는 compiler readiness만 확인하며 CUDA extension 전체 컴파일 성공을 보장하지는 않는다. CUDA/PyTorch/MSVC ABI 오류나 source compile 오류는 이후 JIT build 로그에서 다뤄진다.
