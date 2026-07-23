# Proxy-Based Surface Decomposition Stage 3-R Gaussian-Native Support Continuity 조사

날짜: 2026-07-22

상태: **diagnostics-only 조사 완료. Actual pipeline covariance에서 feasibility 실패. Stage 3 재개 및 Stage 4 진행 금지.**

## 목표

Stage 3의 point-gap conflict를 Gaussian scale, rotation, covariance, opacity 기반 신호가 분리할 수 있는지 검증했다. Merge, admissibility 변경, weighted score, production component 변경은 수행하지 않았다.

## 입력 필드 감사

- `TorchGaussianModel`은 `get_xyz`, `get_scaling=exp(_scaling)`, normalized WXYZ `get_rotation`, `get_opacity=sigmoid(_opacity)`를 제공한다. Renderer covariance convention은 `Sigma=(S*R)^T(S*R)`다.
- Covariance 최소 eigenvector는 flattened anisotropic Gaussian에서만 surface normal 후보가 된다. Isotropic covariance에서는 principal axis가 정의되지 않는다.
- `SyntheticGaussianScene`은 `points`, `colors`, analytic evaluation oracle만 가지며 scale, rotation, covariance, opacity가 없다. Color는 이번 조사에서 사용하지 않았다.
- `boundary_first`는 raw points만 사용한다. Renderer export의 scale `exp(-4.6)`, identity rotation, opacity `sigmoid(10)`은 표시용 placeholder다.
- Current production initialization은 KNN nearest spacing을 xyz 세 축에 동일하게 반복한 isotropic scale, identity rotation, opacity 0.12를 사용한다. 따라서 synthetic benchmark 초기 Gaussian에는 surface-aligned anisotropy가 없다. Scale/rotation은 training 중 학습 가능하지만 이번 benchmark에는 학습된 checkpoint가 없다.

## 변경 파일

- `osn_gs/surface/torch_gaussian_support_continuity.py`
- `scripts/devtools/analyze_gaussian_support_continuity.py`
- `tests/test_gaussian_support_continuity.py`
- `artifacts/gaussian_support_continuity_stage3r_pairs.json`
- `artifacts/gaussian_support_continuity_stage3r_summary.json`
- `artifacts/gaussian_support_continuity_stage3r_signals.csv`
- `artifacts/gaussian_support_continuity_stage3r_production_benchmark.json/report.json`

Stage 3 agglomeration, production component builder, Phase 2, NURBS 파일은 수정하지 않았다.

## 구현한 독립 신호

- one-sided, symmetric, pooled Mahalanobis distance와 raw boundary-pair quantile
- covariance condition, eigenvalue floor, principal-axis ambiguity
- k-sigma directional ellipsoid overlap margin과 overlap fraction
- center gap / directional covariance reach
- covariance principal-axis 기반 tangent/normal gap 및 reach ratio
- local Gaussian kernel bridge density의 minimum, mean, integral, endpoint ratio, valley depth
- bridge opacity weighting on/off
- boundary-facing Gaussian count, opacity mass, projected reach, centroid gap
- 기존 Euclidean support gap, gap/spacing, quadratic RMS/error increase, normal angle, layer score
- deterministic cross-distance, selected boundary-pair, bridge-kernel evaluation count

모든 값은 raw diagnostics로 반환하며 merge decision이나 weighted score를 만들지 않는다. Runtime entry는 scene name, GT label/topology/component count를 받지 않는다.

## 기본 config와 sweep 범위

| 항목 | 기본값 | 조사 범위 |
|---|---:|---|
| covariance eigenvalue floor | `1e-10` | 수치 안정성 고정값 |
| relative eigenvalue floor | `1e-6` | 수치 안정성 고정값 |
| ellipsoid sigma factors | 1, 2, 3 | 세 값 모두 독립 기록 |
| support quantiles | 0.02, 0.1, 0.5 | 세 값 모두 독립 기록 |
| bridge samples | 33 | 17, 33, 65 |
| kernel truncation radius | 4 sigma | 3, 4, 6 sigma |
| opacity weighting | off/on | 둘 다 기록 |
| boundary-facing quantile | 0.1 | 고정 |
| max boundary pairs | 32 | 고정 |
| projection mode | covariance principal axis | 고정 |
| dtype | float64 | 고정 |
| production-init scale multiplier | 1.0 | 0.5, 1.0, 2.0 |
| covariance scale noise sigma | 0 | 0.05, 0.15, 0.3 |
| orientation | 원본 | x 37°, y 61°, z 83° |
| density seed | 2 conflict | 0–4 |
| dense fraction | 0.7 | 0, 0.5, 0.7, 0.9 |

이는 diagnostics 설정이며 production threshold/default가 아니다.

## 직접 추출한 pair

Actual production-init covariance pair는 33개다.

- Positive: density-gradient seed 2 final split, mild-curved seed 0 final split, Stage 1 정의의 curved-annulus AABB-touch missing 4 pairs, curved-annulus existing face-smooth 14 pairs
- Negative: disconnected gap 0.02/0.05 seed 0, gap 0.1 seeds 1/2/4의 최초 false merge, close-parallel offset reject 4 pairs, crease normal reject 4 pairs
- Analytic: high-curvature smooth, coplanar disconnected

각 pair는 full raw boundary-pair vector와 aggregate signal을 저장했다. GT/scene label은 signal 계산 후 분포 평가에만 사용했다.

## Synthetic covariance fixture

Production behavior를 바꾸지 않고 diagnostics artifact 안에서만 다음 7개를 생성했다.

- sparse smooth + tangent-aligned elongated covariance
- disconnected coplanar + isotropic covariance
- disconnected coplanar + non-bridging elongated covariance
- parallel layers
- crease
- rotating tangent-frame curved surface
- density-gradient + local-density-varying tangent covariance

Fixture만 보면 pooled Mahalanobis, directional reach, ellipsoid margin AUC는 각각 0.833이지만 모든 신호의 separation margin은 음수였다. Synthetic fixture에서도 공통 단일 threshold는 없었다.

## 핵심 conflict 정량 결과

| pair | point gap/spacing | pooled Mahalanobis q0.1 | directional reach ratio q0.1 | bridge endpoint ratio q0.1 |
|---|---:|---:|---:|---:|
| density seed 2, positive | 5.163 | 1.423 | 1.097 | 0.257 |
| mild curved seed 0, positive | 4.165 | 0.787 | 0.563 | 0.667 |
| disconnected gap .02, negative | 2.387 | 1.064 | 0.760 | 0.740 |
| disconnected gap .05, negative | 3.080 | 1.445 | 1.034 | 0.692 |
| disconnected gap .1 seed 1, negative | 3.496 | 1.556 | 1.204 | 0.299 |
| disconnected gap .1 seed 2, negative | 3.753 | 1.545 | 1.095 | 0.375 |
| disconnected gap .1 seed 4, negative | 3.907 | 1.667 | 1.260 | 0.356 |

Core conflict에서 의미가 유효한 가장 강한 단일 신호는 pooled Mahalanobis q0.1로 ROC-AUC 0.90이었지만 separation margin은 -0.359였다. Density positive 1.423이 disconnected gap .02 negative 1.064보다 나빠 공통 threshold가 없다.

Point gap AUC는 0.0, directional reach는 0.7, bridge endpoint ratio는 0.3이었다. Pooled Mahalanobis는 ranking을 개선했지만 production-grade separation을 만들지 못했다.

## Principal-axis와 opacity 판단

- Actual 33/33 pair에서 covariance가 isotropic이라 `principal_axis_meaningful_for_all=False`였다.
- 원본 orientation에서는 `normal_reach_ratio`가 우연히 core conflict를 분리했지만 이는 eigenvector basis artifact다.
- 회전 후 density normal-reach 범위는 0.090–0.300, disconnected는 0.135–0.576으로 겹쳤다. 따라서 해당 신호는 무효다.
- Actual opacity는 모두 0.12다. Weighted/unweighted bridge endpoint ratio 최대 차이는 `4.44e-16`으로 정보 증가가 없다.

## Bridge density와 민감도

- Samples 17/33/65와 truncation 3/4/6 sweep에서 density conflict의 bridge ratio 범위는 0.253–0.258, disconnected seed 1은 0.288–0.300이었다. 이 제한된 bandwidth/sample 범위에서는 안정적이었다.
- 그러나 covariance scale multiplier 0.5/1/2에서 density ratio는 0.003/0.257/0.817, disconnected는 0.002/0.299/0.822로 함께 움직였다.
- 따라서 bridge density는 sample/truncation보다 Gaussian scale calibration에 강하게 종속되고 두 class가 계속 겹친다.

## Seed, density, noise 안정성

- Density seeds 0–4 hardest connected pair의 pooled Mahalanobis는 0.707–22.695, bridge ratio는 0–0.866이었다.
- Dense fraction 0/0.5/0.7/0.9에서도 pooled Mahalanobis는 1.825–17.856, bridge ratio는 0–0.204로 크게 변했다.
- Covariance scale noise가 principal-axis 방향을 임의로 만들며 normal/tangent decomposition을 불안정하게 했다.
- 회전에는 Mahalanobis와 bridge scalar가 수치적으로 invariant했지만 class separation 자체가 없었다.

## 단일 또는 두 신호 gate 평가

Invalid principal-axis 신호를 제외하고, 모든 positive를 포함하는 monotonic threshold envelope에서 두 independent signal의 AND를 전수 비교했다. 완전 분리 가능한 conjunction은 없었다.

Best 조합은 pooled Mahalanobis와 bridge/directional/ellipsoid 계열이었지만 모두 disconnected gap 0.02를 false positive로 남겼다. 이 threshold는 분석용 envelope일 뿐 runtime config로 채택하지 않았다.

## 계산량과 결정성

Actual 35개 pair(33 production-init + 2 analytic)의 deterministic cost 합계:

- cross distance evaluations: 375,961
- selected boundary pairs: 567
- bridge kernel evaluations: 10,742,886
- 전체 analyzer 실행: 약 6초 CPU

반복 생성 artifact SHA-256:

- pairs: `AC32A9A3023797F227F853B345EC3096BC703240198009A89603EC12E3AE1E75`
- summary: `10D2F01F34D5953F5489C7DB755983BB7187B19413DCA50CDC6784652924DAEC`
- CSV: `668FA5C6D2AD87799C708E936AA402CAE75FA5E09DF76D18BB89F6796C6D8B01`

세 파일 모두 반복 생성과 byte-identical했다.

## 검증

- Focused pytest: 11 passed.
- 전체 unittest: Ran 202 tests, 1 skipped.
- 기존 `torch_nurbs.py:433` warning 외 오류 없음.
- 실제 `osn-gs benchmark --constructor boundary_first` 성공.
- Stage 3와 Stage 3-R benchmark의 공통 10 scenes에서 patch 수, component 수, topology, chart signature가 모두 동일했다.
- Production `curved_annulus`는 2 components (`disk_like`, `complex`) 그대로다.

## 최종 질문 답변

1. **실제 Stage 3 conflict pair를 분리하는 Gaussian-native signal이 존재하는가?** 아니오.
2. **가장 강한 signal은 무엇인가?** Core ranking에서는 pooled Mahalanobis q0.1(AUC 0.90)이지만 separation margin이 음수다. `normal_reach`의 겉보기 분리는 isotropic eigenbasis artifact라 무효다.
3. **Point-based support gap보다 일관되게 우수한가?** Ranking은 일부 개선하지만 seed/density 전체에서 일관된 separation은 아니다.
4. **Covariance가 실제 Gaussian surface support를 의미하는가?** Current synthetic/production initialization에서는 아니다. Isotropic KNN scale이라 surface normal/tangent 정보를 담지 않는다.
5. **Opacity weighting은 도움이 되는가?** 아니다. Constant opacity 때문에 결과가 동일하다.
6. **Bridge density는 bandwidth/sample 설정에 안정적인가?** 조사한 sample/truncation 범위에는 안정적이지만 covariance scale에 과민하고 class overlap을 제거하지 못한다.
7. **공통 threshold 또는 간단한 conjunction이 가능한가?** 의미가 유효한 신호에서는 불가능하다.
8. **Stage 3 agglomeration을 재개할 근거가 생겼는가?** 없다.
9. **다음 방법론 방향은 무엇인가?** Pairwise local evidence를 더 조합하지 말고 neighborhood/manifold-level connectivity, graph path support, local tangent-field transport처럼 다수 이웃의 연속성을 보는 별도 방법론을 검토해야 한다.

## 채택/기각과 중단

**Stage 3-R Gaussian-native pairwise methodology를 actual pipeline 기준으로 기각한다.** Synthetic anisotropic fixture의 개선은 실제 benchmark Gaussian covariance에 존재하지 않으며, actual conflict에서는 분리가 실패했다.

Stage 3 admissibility를 수정하지 않고 Stage 4 integration으로 진행하지 않는다. 사용자의 별도 승인 없이 neighborhood/manifold-level Stage를 시작하지 않는다.
