# Worklog 60-A: Held-out eval 파이프라인 학습 루프 연결 + smoke test

날짜: 2026-07-22

상태: **연결 및 smoke test 완료. 실제 10k A/B는 사용자 지시 대기.**

## 배경

Worklog 59에서 vendoring한 Graphdeco held-out split/해상도 로직(`osn_gs/data/vendor/graphdeco_scene_split.py`, `load_colmap_scene_with_eval_split`)을 실제 학습 CLI에 연결하고, held-out PSNR/SSIM을 계산하는 평가 헬퍼를 추가한 뒤, CUDA smoke test로 전체 파이프라인이 실제로 동작하는지 확인했다.

## 구현

### `osn_gs/eval/held_out_metrics.py` (신규)

`evaluate_held_out_cameras(rasterizer, model, test_cameras, test_images, device, background=None)`: held-out 카메라마다 렌더링 후 기존 `osn_gs/losses/torch_losses.py::ssim`, `osn_gs/utils/torch_ops.py::psnr_from_mse`를 그대로 재사용해 PSNR/SSIM을 계산(재구현하지 않음). `camera_count`/`psnr_mean`/`ssim_mean`/`per_camera`(카메라별 psnr/ssim/mse) 반환.

- 구현 중 `osn_gs.losses.torch_losses` ↔ `osn_gs.core.torch_pipeline`의 기존 circular import를 발견했다(정상 학습 진입점은 항상 `osn_gs.core`를 먼저 import해서 우연히 문제가 없었을 뿐). `osn_gs.eval.held_out_metrics`가 프로세스에서 가장 먼저 import되는 OSN-GS 모듈일 경우(예: 독립 평가 스크립트, 이 파일 자신의 단위테스트) 이 순서 의존성이 그대로 드러나서, 함수 내부에서 `import osn_gs.core`를 먼저 한 뒤 `torch_losses`를 import하도록 방어적으로 고쳤다(순환 자체를 없애는 근본 수정은 아니고, 안전한 순서를 명시적으로 강제).

### `scripts/train_osn_gs_torch.py`, `train.py`, `osn_gs/interop/colab_args.py`

세 곳 모두(노트북/CLI parity 유지, `[[project_notebook_cli_parity]]`) 동일하게 `--eval`(기본 꺼짐), `--llffhold`(기본 8), `--resolution`(기본 -1), `--resolution_scale`(기본 1.0) 플래그를 추가했다.

- `--eval` 지정 시 `load_colmap_scene_with_eval_split`로 로드해 train-only `TorchScene`으로 학습하고, held-out 카메라는 학습에 전혀 노출되지 않는다.
- `--low_vram`(기본 켜짐) 또는 명시적 `--train_resolution_scale`이 `--eval`과 함께 쓰이면 baseline과 맞춘 해상도 위에 추가로 다시 축소를 걸어버려 공정성이 깨지므로, `--eval` 시 `train_resolution_scale`을 1로 강제 override하고 이유를 로그로 남긴다.
- 학습 완료 후 `evaluate_held_out_cameras`를 호출해 콘솔에 요약을 출력하고, `--disable_output_files`가 아니면 `<output>/held_out_eval.json`에 상세 리포트(카메라별 psnr/ssim/mse 포함)를 저장한다.

## 검증

### 단위테스트

`tests/test_held_out_metrics.py`(3개): 완전 동일 렌더링 시 PSNR=inf/SSIM>0.99, 노이즈 섞인 렌더링 시 유한한 PSNR과 더 낮은 SSIM, 카메라/이미지 리스트 길이 불일치 시 `ValueError`. 페이크 rasterizer/camera로 CUDA/실제 모델 없이 테스트.

전체 스위트 `python -m unittest discover -s tests -p "test_*.py"` → **171 passed, 1 skipped**(기존 158+1 대비 순수 신규 13개, 회귀 없음).

### CUDA smoke test (실제 DATASET, 5 iteration)

```
scripts/train_osn_gs_torch.py -s DATASET --output output/_smoke_eval_test --iterations 5 \
  --eval --no-low_vram --skip_cuda_build_preflight --disable_stream_nurbs --save_interval 5
```

결과:
```
OSN-GS eval-split scene: train=161 test=24 resolution=(1600, 1036) downscale_factor=3.2419
...
OSN-GS torch training complete: iteration=5, loss=1.095579, psnr=13.393, gaussians=138766, uncertain=0
OSN-GS held-out eval (cameras=24, resolution=(1600, 1036), llffhold=8): psnr_mean=13.205 ssim_mean=0.3517
OSN-GS held-out eval report: output\_smoke_eval_test\held_out_eval.json
```

`held_out_eval.json`이 `iteration`/`resolution`/`downscale_factor`/`llffhold`/`camera_count`/`psnr_mean`/`ssim_mean`/`per_camera`(24개 카메라 각각의 image_name/psnr/ssim/mse) 필드로 정상 직렬화됨을 확인했다. 5 iteration만 돈 결과라 PSNR 자체는 낮지만(초기 학습 단계), 파이프라인이 실제 CUDA rasterizer로 end-to-end 동작한다는 것이 목적이었다. smoke test 산출물은 삭제했다(재현 가능, 저장 불필요).

## 남은 사항

- 실제 10k iteration OSN-GS/Graphdeco A/B 실행 자체 — 사용자 지시 대기, 이번 세션에서 하지 않음.
- baseline(`gaussian-splatting/train.py`) 쪽은 이미 자체적으로 `--eval --llffhold`를 지원하므로 별도 작업 불필요.
