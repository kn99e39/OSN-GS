# TODO: baseline 3DGS 대비 Scene 품질 하락 — 남은 후보

동일 데이터셋 10k에서 OSN-GS가 원본 Graphdeco 3DGS(`gaussian-splatting/`)보다 품질이 낮은 문제. 실행환경 노트북+CUDA(ADC 정상). 정적 코드 대조로 후보를 좁혔고, **최우선 원인이던 image loss의 SSIM 부재는 해결함** — 원본과 동일한 `(1-0.2)·L1 + 0.2·(1-SSIM)` 도입, SSIM은 원본 3DGS와 수치 일치(`docs/worklogs/18_ssim_image_loss.md`). 아래는 남은 2차 후보(미검증)와 검증 계획.

## 남은 후보 1 — NURBS surface anchor loss가 certain Gaussian 위치를 구속 (OSN-GS 고유, 2차)

- `nurbs_surface_loss`가 매 iteration certain Gaussian에 대해 `(gaussian_xyz - patch.evaluate(uv))²`를 최소화한다(`osn_gs/losses/torch_losses.py`, `nurbs_surface_loss`). gradient가 Gaussian `_xyz`로도 흘러 Gaussian을 NURBS 표면 쪽으로 끌어당긴다. `lambda_surface=0.01`(`torch_trainer.py`).
- baseline에는 없는 제약. NURBS fit이 부정확한 영역에서는 이 anchor가 이미지 최적화와 충돌해 fidelity를 떨어뜨릴 수 있다.
- 방향: ablation으로 `lambda_surface=0`과 비교. 품질이 회복되면 이 항을 약화하거나, image residual이 큰 Gaussian에는 anchor 가중치를 낮추는 방식 검토. **단, NURBS는 프레임워크 핵심이라 완전 제거가 아니라 가중/스케줄 조정 방향**(설계 제약은 `docs/architecture.md` 2026-07-10 참고).

## 남은 후보 2 — 학습 뷰 샘플링이 무작위가 아니라 결정론적 순환 (2차)

- 원본: 매 iteration `randint`로 무작위 카메라, 스택 소진 시 재셔플(`gaussian-splatting/train.py:89-94`).
- 우리: `(iteration + offset) % count`로 순차 순환(`osn_gs/data/torch_scene.py:38`). gradient 다양성이 줄고, 100(densify)/3000(opacity reset) 같은 주기 이벤트와 카메라 순서가 고정 위상으로 맞물려 편향이 생길 수 있다.
- 방향: iteration seed 기반 무작위 순열 샘플링(without replacement, epoch 셔플)로 교체.

## 검증 계획

1. **SSIM 적용 상태로 baseline vs OSN-GS 10k 재학습** → PSNR/SSIM 격차가 얼마나 줄었는지 측정. **공정 비교: 해상도 맞춤 필수** — OSN-GS를 `--no-low_vram`(전해상도)로 돌리거나 baseline에 `-r`로 OSN-GS 해상도를 맞춘다(`docs/worklogs/14`, `15`).
2. 남은 후보 1은 `lambda_surface=0` ablation으로 기여도 격리.
3. 부수 확인: run 로그의 `OSN-GS rasterizer backend:`가 CUDA인지(`osn_gs/render/gaussian_rasterizer.py`). fallback이면 screen-space gradient 미제공으로 ADC가 왜곡돼 비교가 apples-to-apples가 아니게 된다.

## 참고 (원인 아님, 원본과 동치 확인됨)

- per-param LR 값·xyz exponential 스케줄, opacity reset(0.01/3000), clone/split 수식, prune 임계(0.005), SH degree 1000마다 증가, background 기본 검정.

---

# NURBS 표면 생성 품질: 세 안건 평가 도구 + 개선 타깃

`nurbs_constructor_benchmark`가 이제 GT 대비 세 안건을 분리 측정한다(`docs/worklogs/16_ground_truth_nurbs_metrics.md`, `nurbs_constructor_benchmark/README.md`). 개선 작업은 이 지표로 before/after를 재는 것을 전제로 한다. 현재 baseline(600pts, seed0, lsq)에서 드러난 우선 타깃:

- **Patch Topology — `crease` 과분할**: patches=4(GT 2), `topology_label_ari=0.223`. voxel boundary가 능선을 필요 이상으로 잘게 쪼갬(`docs/voxel_role.md`, `torch_voxel_regions.py`의 `voxel_boundary_angle_degrees`/connected-component). 목표: ARI↑, patch_count→2.
- ~~**Surface Support — UV trimming 부재**~~ **(해결됨)**: patch별 UV trim 마스크 도입으로 plane 0.239→0.089, sine 0.184→0.092, crease→0.004(구멍 안 늘림). `docs/worklogs/17_uv_trimming.md`. 남은 것: (1) `density_gradient` 0.66은 trimming이 아니라 support tau가 전역 median NN spacing(밀집 클러스터 지배)이라 희박부를 과표기하는 **metric 보정 이슈** → 국소 밀도 적응형 tau로 완화 필요. (2) trim 마스크가 init 시점 UV 기준이라 학습 중 stale → maintenance에서 재계산.
- Accuracy(chamfer_rms)는 네 scene 모두 0.023~0.028로 무난 — 즉 **주 문제는 정확도가 아니라 support와 topology**임을 지표가 말해준다.
