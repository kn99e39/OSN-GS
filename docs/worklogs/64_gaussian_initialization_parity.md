# Worklog 64: Visible Gaussian Initialization Parity

## 목적

worklog 63에서 loader fix가 실사용 3k+ anisotropy/screen-prune 폭주를 회복시키지 못했음을 확인했다. 이번 목표는 production OSN-GS와 Graphdeco의 **iteration-0 Gaussian 초기화** 차이를 확정하고, visible/certain Gaussian 초기화를 baseline-compatible하게 만드는 것이다.

## 원인 추적

### Baseline: `gaussian-splatting/scene/gaussian_model.py::create_from_pcd`

```python
dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)   # 등방(isotropic), 3축 동일
rots = torch.zeros((n, 4)); rots[:, 0] = 1                       # identity quaternion
```

`distCUDA2`는 최근접 3개 이웃까지의 평균 제곱거리다. 세 축 모두 동일한 값이므로 **iteration 0의 anisotropy는 정확히 1.0**이다.

### OSN-GS: `osn_gs/core/torch_pipeline.py::_canonical_initial_covariance`(수정 전 유일 경로)

로컬 PCA 기반 planar-surfel 초기화. eigenvector로 tangent_u/tangent_v/normal 프레임을 만들고:

```python
tangent_scale = spacing * 0.45 * covariance_scale_multiplier   # spacing = sqrt(mean 3-NN dist^2), distCUDA2와 동일 정의
normal_scale = (tangent_scale * 0.04).clamp_min(covariance_min_scale)   # 법선축을 접선축의 1/25로 강제
scales = stack(tangent_scale*1.05, tangent_scale, normal_scale)
```

`normal_scale = tangent_scale * 0.04`가 **모든 새 Gaussian의 iteration-0 anisotropy를 설계상 ~25로 고정**한다. 이 covariance는 원래 canonical visible surface construction(reliability/region-formation)의 입력으로 설계된 것인데, 유일한 경로였기 때문에 Gaussian 모델 자신의 학습 가능한 `_scaling`/`_rotation` 초기값으로도 그대로 재사용되고 있었다 — 이것이 실제 원인이다.

## 구현

`osn_gs/core/torch_pipeline.py`에 `TorchPipelineConfig.gaussian_initialization_mode: str = "baseline_compatible"`을 추가하고, 두 개의 서로 다른 역할을 명시적으로 분리했다.

1. **surface 구성용 covariance** (`_canonical_initial_covariance`, `construction_covariance`) — visible surface construction/reliability가 쓰는 입력. **미변경.** 여전히 항상 로컬 PCA planar-surfel.
2. **모델의 학습 가능한 scale/rotation init** — `_initialize_canonical`(실사용 "initialize" 스케줄, production 기본값)에서 `gaussian_initialization_mode`에 따라 분기:
   - `baseline_compatible`(기본): 신설한 `_baseline_compatible_scale_rotation()` — `_graphdeco_neighbor_mean_dist2`(기존에 이미 `distCUDA2`와 동일하게 구현돼 있던 헬퍼, worklog 62에서 확인)로 `dist2`를 구하고 `scale=sqrt(dist2)`를 3축에 동일 적용, rotation은 identity quaternion. Graphdeco의 `create_from_pcd`와 텐서 단위로 동일.
   - `covariance_knn`(experimental, opt-in): 기존 로컬 PCA planar-surfel 초기화 그대로 유지.
   - 명시적 `covariance_scales`/`covariance_rotations` override(synthetic fixture 등)는 모드와 무관하게 항상 우선한다(변경 없음).

`initialize_deferred`(`adc_post_commit`/`disabled` 스케줄)는 **의도적으로 이 플래그의 영향을 받지 않는다** — 이 경로는 초기화 시점에 surface를 만들지 않고, 이후 `reconstruct_visible_after_adc()`가 그 시점의 `model.get_scaling`/`get_rotation`으로부터 `covariance_from_scale_rotation()`을 계산해 **최초의** canonical surface를 재구성한다(첫 real ADC 이전에는 image loss만으로는 방향 정보가 거의 생기지 않으므로, 이 초기값이 유일한 orientation evidence). 여기에 등방 초기화를 넣으면 그 첫 재구성이 surface를 못 찾고 실패한다 — 이는 surface reconstruction 자체를 바꾸는 것이라 이번 과제의 금지사항에 해당해 제외했다(실제로 처음에는 무조건 이 플래그를 적용했다가 `test_post_adc_transaction_is_detached_rng_neutral_and_observed_only`가 실패해 발견·수정).

### CLI 플래그

`--gaussian_initialization_mode {baseline_compatible,covariance_knn}`을 `osn_gs/interop/colab_args.py`(공유 파서, `train.py`가 사용), `scripts/train_osn_gs_torch.py`(레거시 별도 파서) 양쪽에 동일 기본값으로 추가했다.

## 3-way parity 검증

`scripts/devtools/gaussian_init_parity_harness.py`. baseline(unmodified train.py 로직)과 OSN-GS 두 모드(각각 `pipeline.initialize()`로 실제 초기화, tensor transplant 없음) 모두 baseline의 `random.seed(0)` 카메라 셔플/pop 알고리즘을 재현해 동일 카메라 시퀀스로 step 0/1/100/600을 실행했다(ADC는 baseline과 동일하게 `densify_from_iter=500, interval=100` → step 600에서 첫 ADC 이벤트 포함).

| step | 지표 | Baseline | covariance_knn | baseline_compatible |
|---|---|---:|---:|---:|
| 0 | count / anisotropy median | 138,766 / **1.0** | 138,766 / **26.25** | 138,766 / **1.0** |
| 1 | anisotropy median | 1.0 | 26.25 | 1.0 |
| 100 | anisotropy median | 1.076 | 26.77 | 1.079 |
| 600(ADC 후) | count / anisotropy median / p99 | 147,713 / 1.617 / 8.60 | 155,969 / 28.44 / 70.35 | 146,529 / 1.576 / 7.97 |
| 600 ADC | clone/split/pruned | (before/after만 로깅) | 5,011 / 24,384 / 0 | 4,540 / 6,456 / 5 |

`covariance_knn`은 학습이 진행돼도 anisotropy가 26~28대에 고정되고, ADC의 split 후보가 baseline_compatible의 **3.8배**(24,384 vs 6,456)로 과다 트리거된다 — 3k+ screen-size prune 폭주의 실제 기원이 이것이다. `baseline_compatible`은 step 0/1에서 baseline과 완전히 일치하고, step 600에서도 count(0.8% 차이), anisotropy(2.5% 차이), split 후보 규모 모두 baseline에 근접한다.

절대 스케일 크기는 baseline_compatible이 baseline보다 약 12% 크다(step 0 s_min median 0.0435 vs baseline 0.0390) — `_graphdeco_neighbor_mean_dist2`의 chunked cdist 기반 최근접 이웃 계산과 baseline의 CUDA `distCUDA2` 커널 간 미세한 구현 차이로 보이며, anisotropy/population 동역학 결론에는 영향이 없다. 남은 잔차로 정직하게 기록해 둔다.

## 완료 기준 대조

- baseline-compatible OSN-GS의 iteration-0 scale/rotation이 Graphdeco와 동일한 semantics를 가짐 → **확인**(anisotropy 정확히 1.0 양쪽 동일, identity rotation).
- 600-step population 통계와 첫 ADC 후보 분포가 baseline에 근접 → **확인**(count 0.8% 차이, split 후보 규모가 covariance_knn 대비 훨씬 근접).
- 현재 covariance 초기화가 높은 anisotropy의 원인인지 확정 → **확정**. covariance_knn은 step 0부터 이미 anisotropy 26.25로 시작해 끝까지 baseline과 괴리, normal_scale=tangent_scale*0.04라는 설계상의 강제 비율이 직접 원인.
- focused 및 full pytest 통과 → 신규 `tests/test_gaussian_initialization_parity.py`(8개) 포함 `783 passed, 1 skipped, 1 warning, 18 subtests passed`.

## 테스트

```text
python -m pytest -q
783 passed, 1 skipped, 1 warning, 18 subtests passed
```

(worklog 62의 775 passed에서 `tests/test_gaussian_initialization_parity.py` 8개 순증)

## 남은 작업

과제 지시대로 600-step parity 확인 전에는 3k production replay를 재수행하지 않았다. `gaussian_initialization_mode=baseline_compatible`이 이제 production 기본값이므로, 다음 라운드에서 3k(가능하면 5k/10k) production replay로 실제 anisotropy/screen-prune 회복 여부를 재확인하는 것이 자연스러운 후속이다.
