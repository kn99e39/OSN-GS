# 15. Baseline Graphdeco 3DGS: Selector + Buildable On This System

날짜: 2026-07-15

## 목표

동일 데이터셋으로 OSN-GS와 원본 Graphdeco 3DGS(`gaussian-splatting/`)를 비교하기 위해, (1) 노트북에서 두 프레임워크를 선택 가능하게 하고, (2) 구식 의존성(CUDA 11.6/py3.7/torch 1.12 명세)인 baseline을 현재 시스템(torch 2.12+cu130, RTX 5080=sm_120, Windows, VS2022/VS18)에서 실제로 학습 가능하게 만든다.

## 시스템에서 baseline 확장 빌드

`gaussian-splatting`은 `diff_gaussian_rasterization`, `simple_knn`(둘 다 CUDA 확장), `fused_ssim`(선택)이 필요하다. 셋 다 미설치 상태였다.

- **빌드 방법**: VS2022 `vcvars64.bat`로 x64 개발환경 활성화 후 `pip install --no-build-isolation`로 `submodules/simple-knn`, `submodules/diff-gaussian-rasterization`를 설치. `TORCH_CUDA_ARCH_LIST=12.0`(RTX 5080 Blackwell sm_120)로 arch 지정.
- **핵심 수정**: CUDA 13의 CCCL 헤더가 `MSVC 표준 준수 전처리기`를 요구해 `fatal error C1189`로 막힌다. 소스를 건드리지 않고 **MSVC `CL` 환경변수에 `/Zc:preprocessor`를 주입**해 해결(cl.exe가 `CL` 변수를 자동 prepend하므로 직접 cl 호출과 nvcc의 host 컴파일 양쪽에 적용됨). 이는 OSN-GS JIT 빌드가 이미 쓰던 것과 동일한 플래그다.
- `fused_ssim`은 빌드하지 않았다. `gaussian-splatting/train.py`가 부재 시 순수 torch `ssim`로 자동 fallback한다(`FUSED_SSIM_AVAILABLE=False`).

## 검증

- `simple_knn._C.distCUDA2`가 RTX 5080에서 실행, `diff_gaussian_rasterization`·`scene.gaussian_model`·`gaussian_renderer`·`arguments` 전부 import.
- `gaussian-splatting/train.py`를 `DATASET`(185 cameras, 138,766 SfM points)로 30 iteration 실제 학습: 래스터화 forward/backward 정상, loss 0.251→0.215, iter30 train PSNR 16.83, Gaussian 저장, 정상 종료.
- Windows 주의: torch C++ 확장은 `import torch`를 **먼저** 해야 DLL이 로드된다(안 그러면 `DLL load failed`).

## 노트북 선택자 (`colab_train_3dgs.ipynb`)

이미 있던 `FRAMEWORK_MODE`('osn_gs' / 'graphdeco_3dgs')를 로컬 baseline에 맞게 배선:

1. **GS_ROOT 해석**: 로컬(non-Colab)에서 `graphdeco_3dgs`는 `NOTEBOOK_ROOT / 'gaussian-splatting'`(리포에 포함된 로컬 baseline 폴더)을 GS_ROOT로 쓴다. 이전엔 두 모드 모두 `NOTEBOOK_ROOT`가 되어 baseline이 OSN-GS의 train.py를 잘못 실행했다.
2. **빌드 셀**: CUDA 확장 빌드 전에 `TORCH_CUDA_ARCH_LIST`(GPU에서 감지), `CL += /Zc:preprocessor`, `DISTUTILS_USE_SDK`, `VSLANG`를 설정하고 필요 시 MSVC를 in-process 활성화. 이게 없으면 유저 머신에서도 동일한 CCCL C1189로 막힌다. osn_gs 모드는 빌드할 확장이 없어(`if extensions:`) 무영향.

graphdeco 커맨드 빌더/스트리밍 훅/`MODEL_ROOT`(=`output/scene`)는 기존 그대로 사용.

## OSN-GS에 미치는 영향

`diff_gaussian_rasterization`를 venv에 설치하면 OSN-GS의 `diff_gaussian_loader`가 vendored JIT 대신 **설치된 패키지**를 먼저 쓴다(로더의 의도된 우선순위). vendored와 **동일 소스**(forward.cu identical 확인)라 동작 불변이며, 매 실행 JIT 빌드가 없어져 더 빠르다. `tests/` 26개 통과.

## 공정 비교 주의 (중요)

- 현재 OSN-GS 기본은 **VRAM-safe = 반해상도**(`--low_vram`, worklog 14). baseline `gaussian-splatting`은 원본 해상도(단, >1.6K 폭 이미지는 자동으로 1.6K로 축소)로 학습한다. **해상도가 안 맞으면 비교가 불공정**하다.
- 공정 비교를 하려면: OSN-GS를 `--no-low_vram`으로 돌려 전해상도로 맞추거나, baseline에 `-r`/`--resolution`을 줘 OSN-GS의 유효 해상도에 맞춘다.
- baseline은 `lambda_dssim=0.2`로 **L1 + D-SSIM**을 쓰지만 OSN-GS는 **L1 + MSE**(SSIM 없음) — `TODO.md`의 품질 격차 1순위 원인. 비교 시 이 차이를 염두에 둘 것.

## 재현용

빌드 스크립트 `scripts/build_baseline_extensions.bat`로 남겨둔다(VS2022 vcvars + `CL=/Zc:preprocessor` + arch). 다른 GPU면 `TORCH_CUDA_ARCH_LIST`를 해당 compute capability로 바꾼다.
