# Worklog 62: Graphdeco–OSN-GS Lockstep Training Parity

## 방법

`scripts/devtools/lockstep_parity_harness.py`. baseline `GaussianModel`을 실제로 초기화한 뒤 그 raw tensor(xyz/scaling/rotation/opacity/SH)를 그대로 OSN-GS `TorchGaussianModel.replace_tensors()`에 이식해 **양쪽이 byte-identical한 초기 상태**로 시작한다(ADC 비활성화라 population 크기가 끝까지 불변이라 Gaussian index가 한 run 내내 동일 개체를 가리킴). baseline `Scene.__init__`이 `random.seed(0)`으로 카메라 리스트를 섞은 뒤, baseline 학습 루프의 pop-based 카메라 선택 알고리즘을 그대로 재현해 **두 프레임워크가 매 step 동일 카메라**를 쓰게 했다. 각 side는 자기 자신의 실제 render/loss/optimizer 코드(baseline: `gaussian_renderer.render`+`utils.loss_utils`, OSN-GS: `OSNGaussianRasterizer.render`+`image_reconstruction_loss`)를 그대로 호출한다 — 재구현 없음.

## 최초 발산 지점

Step 1(첫 forward, 아직 어떤 gradient도 적용되지 않은 시점)에서 이미 **렌더된 이미지 자체가 달랐다**(`image_mean_abs_diff=0.00051`, loss b=0.26439 vs o=0.25700). 카메라 파라미터를 직접 대조한 결과 두 개의 실제 코드 결함을 확인:

1. **FoVy가 downscale된 해상도에서 재계산됨.** `osn_gs/data/colmap_scene.py`의 `camera_fovs()`가 focal을 `downscale`로 나누고 이미 반올림된 downscale 후 width/height를 사용해 FoV를 다시 계산하고 있었다. baseline(`utils/camera_utils.py::loadCam`)은 COLMAP 원본 해상도에서 **한 번만** FoV를 계산하고 이후 어떤 renders/resize에도 절대 재계산하지 않는다. FoV는 해상도와 무관한 각도량이라 baseline 방식이 맞고, OSN-GS 쪽은 width/height가 서로 다른 반올림 오차를 가질 때(비정수 downscale factor, 예: 3.2419) FoVx는 우연히 일치했지만 FoVy는 어긋났다(b=0.8226821382 vs o=0.8221548647).
2. **Ground-truth 이미지 resize filter가 다름.** OSN-GS `load_image_tensor()`가 `Image.BILINEAR`를 명시 지정했지만, baseline `PILtoTorch()`는 `resample` 인자를 아예 안 줘서 Pillow 기본값(RGB 이미지는 `BICUBIC`)을 쓴다. 실측 `gt_mean_abs_diff=0.0049`(동일 shape, 동일 소스 픽셀, 동일 목표 해상도인데도) — 순수 resize filter 불일치.

## 수정

`osn_gs/data/colmap_scene.py`:
- `camera_fovs()`가 항상 COLMAP 원본 카메라 해상도(`colmap_camera.width`/`.height`)만 쓰도록 두 호출부(`load_colmap_scene`, `load_colmap_scene_with_eval_split`)를 수정, `downscale` 나눗셈 제거.
- `load_image_tensor()`의 두 resize 경로(`target_size` 지정/`downscale>1`) 모두 `Image.BILINEAR` → `Image.BICUBIC`.

Surface reconstruction/reliability 코드는 건드리지 않았다(이번 fix는 순수 COLMAP 데이터 로더 범위).

## 짧은 replay 재검증

같은 harness를 fix 적용 후 재실행:

- **Step 1**: `image_mean_abs_diff=0`, `gt_mean_abs_diff=0.0`, `b_loss=0.26439112` vs `o_loss=0.26439148`(7자리까지 일치), xyz/scaling/rotation/opacity gradient norm 전부 6자리 이상 일치 — **최초 발산이 float32 noise 수준으로 사라짐.**
- **Step 2~10**: 미세한 차이가 다시 나타나 서서히 커짐(step10 `image_mean_abs_diff=0.0046`) — 카메라 순서는 고정돼 있으므로, `diff_gaussian_rasterization`의 backward가 atomicAdd 기반이라 실행마다 미세한 부동소수점 비결정성이 있고, 이게 비선형 SGD로 누적되는 것으로 보인다(두 프레임워크가 같은 설치된 CUDA 확장을 공유하므로 코드 차이가 아니라 커널 자체의 특성).
- **Step 600 (ADC 없이, 순수 population 통계)**: `anisotropy median` baseline=1.581 vs osn_gs=1.588(거의 일치), `p99` 7.12 vs 7.24 — fix 이전 real training에서 관측됐던 iteration 600의 27~37 대 baseline 전체 population p99 8.6라는 극단적 격차가 **완전히 사라졌다.** 개별 Gaussian 단위 tensor diff(`MEAN_TENSOR_DIFF`)는 step이 갈수록 커지지만(카오스적 궤적 발산, 예상된 현상), **population 수준 통계는 baseline과 계속 근접 추적한다** — 3k+ 이후 screen-size prune 폭주를 유발했던 이상 anisotropy 성장의 근본 원인이 이 두 데이터 로더 결함이었음을 뒷받침한다.

## 테스트

```text
python -m pytest -q
775 passed, 1 skipped, 1 warning, 18 subtests passed in 240.87s
```

(worklog 61의 772 passed에서 `tests/test_colmap_scene_camera_fov_resolution_independence.py` 3개 순증)

신규 focused 회귀 테스트: `camera_fovs()`가 `downscale` 인자와 무관하게 동일 결과를 내는지, PINHOLE/SIMPLE_PINHOLE 모델별 fx/fy 사용이 올바른지 직접 검증.
