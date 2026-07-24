# Worklog 76: anisotropy 격차 재평가 — 초기 covariance 1-NN/3-NN 불일치 발견

날짜: 2026-07-23

상태: **기존 산출물 기반 재조사 완료. 강한 root-cause 후보를 발견했지만 학습 ablation 전이므로 확정하지 않음. Production 코드는 변경하지 않음.**

## 배경

Worklog 73-A는 iteration 3000에서 OSN-GS와 baseline의 scale magnitude가 거의 맞았음에도 anisotropy 평균이 `4.80 vs 15.10`으로 남은 현상을 조사했지만 단일 원인을 특정하지 못했다. 이번에는 Worklog 06, 63, 67, 70–73-A와 실제 ADC/covariance 구현, 남아 있는 3k A/B PLY를 다시 대조했다.

## 문제의 정확한 형태

3k PLY의 anisotropy `(max scale / min scale)` 분포:

| 지표 | OSN-GS | baseline |
|---|---:|---:|
| 평균 | 4.7986 | 15.0987 |
| 중앙값 | 3.3194 | 5.3921 |
| p90 | 8.4219 | 19.2291 |

평균만 extreme tail 때문에 벌어진 현상은 아니다. 중앙값과 p90에도 명확한 격차가 있다.

축별 scale 중앙값은 다음과 같다.

| 축 | OSN-GS | baseline |
|---|---:|---:|
| 최소축 | 0.005367 | 0.003339 |
| 중간축 | 0.009574 | 0.007501 |
| 최대축 | 0.020826 | 0.022357 |

최대축은 거의 같고 baseline의 최소축이 더 작다. 따라서 해결할 문제는 global scale magnitude가 아니라 **OSN-GS의 최소축 contraction/flattening이 baseline보다 느린 이유**다.

## 강한 후보: 초기 covariance 계산이 baseline과 동등하지 않음

`osn_gs/core/torch_pipeline.py::_nearest_neighbor_dist2`는 가장 가까운 **1개** 이웃의 squared distance를 사용한다. 반면 Graphdeco `simple-knn::distCUDA2`는 `simple_knn.cu`에서 최근접 **3개** squared distance의 평균 `(best[0] + best[1] + best[2]) / 3`을 반환한다.

동일 DATASET 138,766개 초기 point에 두 구현을 직접 적용한 결과:

| 초기 isotropic scale | OSN-GS 1-NN | baseline 3-NN |
|---|---:|---:|
| 평균 | 0.037273 | 0.057440 |
| 중앙값 | 0.021216 | 0.038980 |
| p90 | 0.085492 | 0.115233 |

동일 ADC `dense_extent=0.04922929`에 대한 초기 분류:

| 분류 | OSN-GS | baseline |
|---|---:|---:|
| clone 쪽 (`scale <= threshold`) | 76.1527% | 59.8670% |
| split 쪽 (`scale > threshold`) | 23.8473% | 40.1330% |

초기 Gaussian의 **16.2857%가 서로 다른 clone/split 영역**에 놓인다. 이 차이는 OSN-GS가 더 작은 isotropic Gaussian과 더 clone-heavy한 계보로 시작하게 하며, 이후 scale magnitude가 비슷해져도 anisotropy 학습 궤적과 개체군 가중치를 바꿀 수 있다.

기존 코드 docstring의 “Original 3DGS initializes ... from nearest-neighbor distance” 설명은 정확하지 않다. Graphdeco-compatible parity 검증에는 3-NN mean mode가 필요하다.

## Worklog 73-A split 카운트 해석 정정

OSN-GS의 `TorchDensityControlReport.split`은 선택된 parent 수가 아니라 생성된 child 수(`parent_count * split_samples`)다. 기본 `split_samples=2`다. Baseline 임시 진단의 `split_candidates`는 parent 수였다.

따라서 Worklog 73-A의 `OSN-GS split≈12,100 vs baseline split_candidates≈6,100`은 2배 격차가 아니라, parent 기준으로 `약 6,050 vs 6,100`이라 사실상 동일하다. 당시 `clone+split` 합계도 서로 다른 단위를 더했으므로 약 20% 성장 차이라는 해석은 정확하지 않다. 관찰된 population 차이는 split보다 clone 쪽을 우선 조사해야 한다.

## 아직 배제되지 않은 차이

1. **ADC gradient source**
   - Worklog 73-A가 비교한 `_xyz.grad.abs().mean()`은 ADC가 직접 사용하는 `xyz_gradient_accum / denom`과 다르다.
   - OSN-GS는 screen-space `viewspace_points.grad`가 없거나 유효한 nonzero 값이 없으면 world-space `_xyz.grad[:, :2]`로 fallback한다.
   - 기존 로그에는 어느 source를 사용했는지 기록되지 않아 fallback 발동 여부가 미확정이다.

2. **xyz spatial LR**
   - Worklog 72 이후 OSN-GS는 point-cloud 기반 `scene_extent≈12.31`을 position LR scale에 유지하고, baseline은 camera extent `≈4.92`를 쓴다.
   - 같은 `_xyz.grad`라도 실제 xyz update는 약 2.5배 다른 schedule을 따른다. 이 차이가 위치 수렴과 후속 scale-axis gradient에 미치는 영향은 별도 ablation이 필요하다.

3. **ADC iteration gradient lifecycle**
   - OSN-GS는 shape transaction 뒤 survivor gradient row를 보존하지만 baseline은 parameter 교체로 해당 ADC iteration gradient가 사라진다.
   - 100 iteration당 1회라 우선순위는 낮지만 exact parity mode에서는 분리 검증할 필요가 있다.

## 평가

- Worklog 73-A의 “더 무거운 개별 Gaussian trajectory 계측 전에는 후보가 없다”는 상태는 갱신됐다.
- 초기 covariance 1-NN/3-NN 불일치는 실제 코드와 실 데이터 수치로 확인된 명확한 parity 차이다.
- 다만 이것이 3k 최소축 contraction 격차의 root cause인지는 동일 조건 재학습 전에는 확정할 수 없다.
- Production 기본값이나 학습 코드는 이번 조사에서 변경하지 않았다.

## 다음 검증 순서

1. Graphdeco-compatible 3-NN mean covariance initialization을 ablation mode로 추가한다.
2. 3k A/B에서 anisotropy 평균뿐 아니라 중앙값/p90, 최소·중간·최대축 분포, held-out PSNR/SSIM, Gaussian 수를 비교한다.
3. ADC 진단을 parent 단위로 통일하고 gradient source(screen-space/fallback), 실제 ADC gradient quantile, clone/split parent의 anisotropy를 기록한다.
4. 격차가 남으면 position LR만 camera extent로 맞춘 ablation을 실행한다.
5. 마지막으로 ADC iteration gradient 보존 차이를 parity mode로 비교한다.

## 남은 위험

- 초기 3-NN scale이 anisotropy를 개선해도 OSN-GS의 point-cloud 기반 spatial LR 설계와 상호작용할 수 있다.
- anisotropy 수치만 baseline에 맞추고 held-out 품질이나 Gaussian 수가 악화될 수 있으므로 단일 metric으로 채택 여부를 정하면 안 된다.
- 현재 3k 실행은 screen-size pruning 활성화 전이므로 장기 안정 상태를 대표하지 않는다. 원인 고립은 3k로 하되 채택 검증은 pruning이 실제 발동하는 더 긴 실행에서도 확인해야 한다.
