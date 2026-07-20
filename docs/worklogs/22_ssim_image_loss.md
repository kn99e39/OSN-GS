# 18. D-SSIM Image Loss (Baseline Quality-Gap Cause 1)

날짜: 2026-07-15

## 배경

`TODO.md`의 baseline 품질 격차 분석에서 **1순위 원인**: OSN-GS의 image loss가 `0.8·L1 + 0.2·MSE`로, 원본 Graphdeco 3DGS의 `(1-λ)·L1 + λ·(1-SSIM)`(λ=0.2)와 달리 **SSIM(구조 유사도)이 코드베이스 전체에 없었다**. 3DGS 품질의 구조적 선명도는 D-SSIM에서 크게 나오고, MSE는 저주파 평균 오차만 줄여 결과가 흐려진다.

## 작업

- `osn_gs/losses/torch_losses.py`에 원본 3DGS `utils/loss_utils.ssim`을 그대로 이식한 순수 torch `ssim`(window 11, σ=1.5, C1=0.01², C2=0.03², separable Gaussian window, `(C,H,W)`/`(N,C,H,W)` 모두 허용, window 캐시). `losses/__init__`에 export.
- `image_reconstruction_loss(image, target, lambda_dssim=0.2)`가 `(1-lambda_dssim)·L1 + lambda_dssim·(1-SSIM)`을 반환하도록 변경. MSE는 loss에서 빠지고 **PSNR 계산용으로만** 함께 반환(기존 소비처 유지).
- `TorchTrainingConfig`: `lambda_l1=0.8`/`lambda_mse=0.2` → **`lambda_dssim=0.2`** 단일 노브(원본과 동일하게 L1 가중 = 1-λ). 트레이너 호출부 갱신. `lambda_l1`/`lambda_mse`는 이 두 파일 밖(CLI/노트북)에서 참조가 없어 그 외 변경 불필요.

## 검증

- **원본과 수치 일치**: 같은 랜덤 이미지쌍에서 `osn_gs.losses.ssim`과 `gaussian-splatting/utils/loss_utils.ssim` 결과 **diff = 0.0**. `ssim(x,x)=1.0`.
- 트레이너 6-iteration CPU smoke: D-SSIM loss가 미분 가능하고 loss 감소(0.5408→0.5340)·PSNR 증가(5.686→5.770). gradient가 SSIM 경로로 흐름 확인.
- `tests/` 26개 통과.
- (주의) `torch_losses`를 core보다 먼저 import하면 기존부터 있던 import-order 순환이 드러난다(정상 학습 경로는 core를 먼저 로드해 무관). 이번 변경이 만든 것 아님.

## 효과 / 후속

- 이제 OSN-GS와 baseline의 image loss가 동일 구성이라, 두 프레임워크 비교가 loss 축에서 공정해졌다.
- **아직 실제 10k A/B 재학습으로 격차 축소를 정량 측정하지 않았다** — `TODO.md` 검증 계획대로 해상도를 맞춰(OSN-GS `--no-low_vram` 또는 baseline `-r`) 재측정 필요.
- 남은 2차 후보: NURBS surface anchor loss 구속(`lambda_surface` ablation), 결정론적 뷰 샘플링. `TODO.md` 참고.
- SSIM은 매 iteration conv2d 3~5회 추가되나 원본 3DGS와 동일 비용이며 품질에 필수. 성능 우려 시 fused-ssim(빌드됨) 연동은 후속 옵션.
