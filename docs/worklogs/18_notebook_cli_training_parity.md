# 18. 노트북 / CLI 학습 동등성

날짜: 2026-07-15

## 배경

노트북(`colab_train_3dgs.ipynb`, 내부적으로 `train.py` 호출)과 CLI 직접 실행의 학습 결과가 달라지는 문제. 원인은 노트북이 `OSN_*` 값을 train.py에 **명시적으로** 넘기지만, CLI argparse **기본값**이 그와 달라서 인자 없는 CLI 실행이 다른 레시피로 돌던 것. 특히 `densify_until_iter`/`densification_interval`의 CLI 기본이 0이라 **CLI 직접 실행 시 ADC가 통째로 꺼졌다**.

사용자 결정: 공통 기본 레시피를 **VRAM-safe(반해상도, low_vram on, ADC on)**로 통일한다. 즉 인자 없는 CLI 실행이 노트북과 동일한 결과를 내도록 CLI 기본값을 노트북의 유효 설정에 맞춘다.

## 작업

두 CLI 파서(`osn_gs/interop/colab_args.py`의 `build_osn_gs_train_parser` — `train.py`가 사용; `scripts/train_osn_gs_torch.py`의 `build_parser`)의 **결과에 영향을 주는 기본값**을 노트북과 일치시켰다:

- `--densify_until_iter`: 0 → **15000**
- `--densification_interval`: 0 → **100**
- `--visible_surface_resolution_scale`: 1.0 → **4.0**
- `--low_vram`: `store_true`(기본 off) → `BooleanOptionalAction`, **기본 True**. 전해상도 실행은 `--no-low_vram`으로 opt-out. low_vram이 기본 on이므로 `train_resolution_scale`은 자동으로 ≥2(반해상도)로 강제되어 노트북과 동일.

노트북(`colab_train_3dgs.ipynb`)도 low_vram 토글이 새 기본값(True) 하에서 계속 동작하도록 수정: `if OSN_LOW_VRAM: cmd += ['--low_vram']` → `cmd += ['--low_vram'] if OSN_LOW_VRAM else ['--no-low_vram']`. (targeted JSON 편집, UTF-8 보존, `json.loads` 검증 완료.)

## 결과 / 검증

- 두 파서 모두 인자 없이(`-s`만) `densify_until=15000, densification_interval=100, res_scale=4.0, low_vram=True`를 낸다. `--no-low_vram`이 정상 opt-out.
- `train.py --help`, `scripts/train_osn_gs_torch.py --help` 정상. `tests/` 26개 전부 통과.
- 이제 동일 dataset·동일 `--iterations`로 노트북/CLI 어느 쪽을 돌려도 같은 레시피(ADC on, 반해상도, 동일 NURBS/voxel/covariance/ADC 파라미터)로 학습된다.

## 남은 차이 (결과 아님, perf/메모리 전용)

아래는 노트북과 CLI 기본이 다를 수 있으나 **학습 결과(트레이닝된 Gaussian/NURBS)에는 영향이 없는** 워크스페이스/메모리 배치 knob이라 통일하지 않았다:

- `--image_device`: 이미지 스테이징 위치(CPU/auto)만 결정. 학습 수식 불변.
- `--visible_surface_fit_device`: NURBS fitting 워크스페이스(cpu/cuda). 동일 LSQ 수식이라 결과 동치(부동소수 미세차 가능). CLI 기본 `cpu`는 VRAM-safe 취지에 맞음.
- `--*_chunk_size`, 스트리밍/로그/save 주기: 성능·I/O 전용.

또한 `--iterations`/`-s`/`-m`/`--save_iterations` 등은 매 실행 사용자가 지정하는 값이라 기본값 통일 대상이 아니다. 두 경로를 같은 값으로 실행하면 동일 결과.

## 주의 (후속)

- 기본값이 **세 곳**(colab_args.py, scripts/train_osn_gs_torch.py, 노트북 `OSN_*`)에 중복돼 있어 앞으로도 drift 위험이 있다. 한 곳을 바꾸면 세 곳을 함께 맞춰야 한다. 장기적으로는 단일 defaults 모듈로 통합하는 리팩터가 바람직.
- 이 VRAM-safe 기본은 **반해상도 학습**이라, 별도 baseline(Graphdeco `gaussian-splatting/`)이 전해상도로 도는 것과 비교하면 OSN-GS가 불리하다. baseline 대비 품질 분석은 `TODO.md` 참고(그 비교를 공정하게 하려면 baseline도 동일 해상도로 맞추거나 OSN-GS를 `--no-low_vram`으로 돌려야 함).
