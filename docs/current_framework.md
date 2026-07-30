# OSN-GS 현재 구현 파이프라인

이 문서는 논문 작성과 연구 보조를 위해, **현재 구현되어 실제 학습 경로에서 작동하는** OSN-GS 파이프라인을 설명한다. 코드 파일 목록이나 미래 설계안은 다루지 않는다. 아직 기본 training path에 연결되지 않은 모듈은 현재 프레임워크의 동작으로 서술하지 않는다.

## 1. 프레임워크의 중심 표현과 방향성

OSN-GS의 최종 학습 표현은 3D Gaussian primitive 집합이다. visible NURBS는 이를 대체하는 출력이 아니라, 관측 Gaussian으로부터 구성되는 canonical parametric intermediate다. 이 intermediate는 관측 geometry를 patch와 parameter domain으로 조직하고, surface-aware loss와 저장된 geometry artifact의 기준을 제공한다.

핵심 방향은 다음과 같이 고정되어 있다.

```text
관측 image / COLMAP geometry
  -> certain Gaussian의 3DGS 최적화
  -> covariance-guided canonical visible-NURBS 구성
  -> patch·UV·ownership binding
  -> surface loss 및 구조적 ADC 이후의 재구성
```

visible Gaussian은 image loss와 표준 3DGS형 ADC로 최적화된다. visible NURBS fitting loss에서는 Gaussian center를 detach하므로, NURBS가 관측 Gaussian 위치를 역으로 끌어당기지 않는다. 즉 관측 geometry에서 surface를 유도하는 one-way 구조다.

## 2. 관측 Gaussian에서 canonical visible surface까지

### 2.1 Gaussian covariance를 구조 증거로 사용

각 Gaussian은 center, anisotropic scale, quaternion rotation을 가지며 이들로 covariance를 만든다. covariance의 principal frame은 local tangent 후보 두 개와 sign-ambiguous normal 후보 하나를 제공한다. OSN-GS는 raster footprint가 아니라 이 covariance frame을 visible-surface 구조 판단의 1차 증거로 사용한다.

고유값의 상대적 크기는 planar surfel, needle-like, isotropic, ambiguous shape를 구분하는 데 사용된다. 따라서 넓고 얇은 surfel은 surface 후보가 될 수 있지만, isotropic blob이나 normal이 정의되지 않는 needle은 동일한 방식으로 신뢰하지 않는다.

### 2.2 대표 표본과 full-neighborhood evidence

대규모 Gaussian 집합 전체에 topology 연산을 수행하지 않는다. 먼저 density-preserving representative selection으로 bounded canonical 표본을 선택한다. 이 표본에서 covariance, reliability, affinity, region 및 boundary topology를 계산한다.

표본화는 단순 랜덤 downsample이 아니다. 공간 점유와 지역 density를 보존하는 representative를 선택해, bounded construction cost와 local geometry evidence 사이의 균형을 유지한다. 대표 표본의 결과는 nearest-representative assignment를 통해 전체 Gaussian 집합으로 전파된다.

### 2.3 Local Surface Decomposition: affinity graph에서 surface region까지

Local Surface Decomposition은 Gaussian cloud를 semantic object로 분할하는 단계가 아니다. local covariance frame과 이웃 관계가 서로 일관된 Gaussian만 하나의 **local surface chart 후보**로 묶고, crease·parallel sheet·ambiguous evidence는 병합 근거가 아니라 분리 또는 보류 근거로 취급한다.

```text
reliable / ambiguous / rejected Gaussian
  -> scale-backed local neighbor candidates
  -> pairwise manifold relation
       {same_surface, crease, parallel_separate, ambiguous, rejected}
  -> multi-edge consensus와 core-edge seeding
  -> reliable-core connected components
  -> ambiguous member의 보수적 consensus attachment
  -> bridge veto / path-consistency / region-merge 검증
  -> local surface-region candidates + unresolved boundary evidence
```

**(1) Local candidate graph.** 후보 쌍은 kNN만으로 확정되지 않는다. mutual kNN 여부와 함께 covariance tangent scale 기반 radius 또는 tangent-footprint overlap 중 하나가 성립해야 한다. 따라서 index proximity만으로는 same-surface evidence가 되지 않는다. 각 후보는 normal alignment, mutual tangent residual, normal-direction separation, footprint ratio를 사용해 `same_surface`, `crease`, `parallel_separate`, `ambiguous`, `rejected` 관계로 분류된다.

**(2) Reliable core.** `same_surface` edge 하나만으로 두 area를 합치지 않는다. shared same-surface neighbor, supporting triangle/path, local density, tangent transport residual, 인접 crease·parallel·rejected contamination을 결합한 consensus가 core-edge를 결정한다. core-eligible edge의 connected component가 초기 local surface region의 backbone이다. 이 때문에 sparse bridge나 한 개의 우연한 affinity edge가 두 patch를 병합하는 것을 막는다.

**(3) Ambiguous attachment.** covariance 자체가 완전히 rejected는 아니지만 core 증거가 부족한 Gaussian은 처음부터 region에 강제 배정하지 않는다. 성장 단계에서 여러 core member로부터 독립적인 support를 얻고 contradiction이 없을 때만 consensus-attached member가 된다. 조건을 만족하지 않으면 `ambiguous_unassigned`로 남는다.

**(4) Over-merge 방지.** component 사이 connection은 bridge veto와 tangent-frame path consistency를 통과해야 한다. local cut으로 큰 component가 갈라지는지, shared-neighbor와 independent cross-edge가 충분한지, direct tangent relation이 backbone path와 모순되지 않는지, near-threshold pair인지가 함께 검사된다. crease와 parallel-separated evidence는 내부 accepted edge가 아니라 boundary conflict evidence로 보존된다.

**(5) Decomposition output.** 각 `SurfaceRegionCandidate`는 core member, consensus-attached member, rejected/excluded member, internal accepted/ambiguous edge, boundary conflict edge, confidence와 unresolved reason을 보관한다. region state는 core/growing/stable/review/rejected로 구분되며, 이는 확정 object label이 아니라 local geometric evidence의 상태다.

### 2.4 Region boundary와 NURBS 입력의 admission

region별 internal accepted topology를 따라 normal sign을 일관되게 정렬한다. support termination과 full-cloud continuation evidence에서 boundary half-edge candidate를 만들고, directed successor compatibility를 통해 ordered component를 복구한다. `ordered_closed_loop`이면서 outer-boundary candidate이고 branch node가 없는 component만 NURBS materialization에 admission된다.

따라서 Local Surface Decomposition의 출력 전체가 곧바로 NURBS가 되지는 않는다. open chain, branching graph, ambiguous ordering, 작은 review region은 진단 결과로 남고 synthetic closure를 받지 않는다. 조건을 만족하는 region이 하나도 없으면 surface를 억지로 합성하지 않는다.

### 2.5 NURBS patch materialization과 binding

admissible region은 rational tensor-product NURBS patch로 materialize된다. patch가 생성되면 전체 Gaussian에 다음 binding이 전파된다.

```text
Gaussian
  -> visible patch membership
  -> (u, v) foot-point parameter
  -> ownership kind / ownership ID
```

`cluster_ids`는 patch-ID 호환용 값이고, 실제 surface 행동의 기준은 `surface_owner_kind`와 `surface_owner_id` pair다. 이 구분으로 visible patch와 occluded chart가 같은 정수 ID 공간에서 혼동되지 않는다.

## 3. 학습과 surface의 결합 방식

한 iteration은 Gaussian rendering과 image reconstruction을 중심으로 진행된다.

```text
camera/image batch
  -> Gaussian rasterization
  -> image reconstruction loss
  -> optional visible-NURBS fitting loss
  -> backward 및 Gaussian/surface optimizer step
  -> visibility·radius·gradient 통계 축적
  -> Adaptive Density Control
  -> 필요 시 canonical visible-NURBS 재구성
```

image reconstruction loss는 Gaussian appearance, center, opacity, scale, rotation을 직접 최적화한다. visible NURBS loss는 patch control grid와 rational weight가 관측 Gaussian에 맞도록 최적화한다. 이때 Gaussian center는 fitting target으로만 사용된다.

따라서 NURBS는 관측 Gaussian의 별도 geometry prior나 position correction 장치가 아니라, Gaussian 분포를 patch parameterization으로 요약하고 평가하는 중간 구조다.

## 4. ADC와 visible-NURBS lifecycle

ADC는 certain Gaussian에 대해서만 3DGS형 clone, split, prune을 수행한다. view-space gradient 누적값과 scale을 기준으로 작은 high-gradient Gaussian은 clone, 큰 high-gradient Gaussian은 split 후보가 된다. prune은 opacity, screen-space size, world-space scale 조건을 사용한다.

ADC는 Gaussian tensor의 row 수와 순서를 바꾸므로, `stable_gaussian_ids`가 clone/split/prune와 checkpoint 복원을 통과하는 identity를 제공한다. surface binding 및 ownership metadata도 row transport과 함께 유지된다.

visible NURBS lifecycle에는 세 가지 현재 구현 schedule이 있다.

- `initialize`: 학습 시작 때 visible surface를 구축하고, 설정된 surface rebuild interval에 따라 재구축한다.
- `adc_post_commit`: initial surface 없이 시작하며 clone/split/prune가 실제 commit된 구조적 ADC 뒤에만 canonical 재구성을 시도한다. no-op ADC와 opacity reset만으로는 재구성하지 않는다.
- `disabled`: visible NURBS를 만들지 않는 Gaussian-only control 경로다.

재구성은 candidate state에서 patch registry, membership, UV, ownership을 계산한 뒤에만 commit한다. reliability/region/boundary 조건 미달, empty surface, optimizer setup 실패는 fail-closed로 처리된다. 즉 stale NURBS, stale binding, stale surface optimizer를 비우고 Gaussian 학습과 ADC는 계속한다.

## 5. 실제 구현 수식

### 5.1 Gaussian parameterization과 covariance

Gaussian의 학습 파라미터를 center \(\boldsymbol{\mu}\), raw log-scale \(\boldsymbol{\ell}\), raw quaternion \(\mathbf q\), opacity logit \(o\)라고 하면 계산 시 사용하는 scale, rotation, opacity는

\[
\mathbf s=\exp(\boldsymbol{\ell}),\qquad
\hat{\mathbf q}=\frac{\mathbf q}{\lVert\mathbf q\rVert_2},\qquad
\alpha=\sigma(o).
\]

정규화 quaternion이 만드는 회전행렬을 \(R\)이라 하면 covariance는

\[
\Sigma=R\,\mathrm{diag}(s_x^2,s_y^2,s_z^2)R^\top.
\]

고유값 \(\lambda_1\ge\lambda_2\ge\lambda_3\)에 대해 구현에서 쓰는 shape 지표는

\[
\mathrm{planarity}=\frac{\lambda_2}{\max(\lambda_3,\epsilon)},\quad
\mathrm{elongation}=\frac{\lambda_1}{\max(\lambda_2,\epsilon)},\quad
\mathrm{isotropy}=\frac{\lambda_3}{\max(\lambda_1,\epsilon)}.
\]

### 5.2 Rational tensor-product NURBS

control point \(P_{ij}\), weight \(w_{ij}\), U/V B-spline basis \(N_{i,p}(u)\), \(M_{j,q}(v)\)에 대한 patch 평가는

\[
S(u,v)=
\frac{\sum_{i,j}N_{i,p}(u)M_{j,q}(v)w_{ij}P_{ij}}
     {\sum_{i,j}N_{i,p}(u)M_{j,q}(v)w_{ij}}.
\]

basis는 clamped-uniform knot vector에서 Cox-de Boor recurrence로 계산한다.

\[
N_{i,0}(u)=\mathbf{1}[t_i\le u<t_{i+1}],\qquad
N_{i,p}(u)=
\frac{u-t_i}{t_{i+p}-t_i}N_{i,p-1}(u)+
\frac{t_{i+p+1}-u}{t_{i+p+1}-t_{i+1}}N_{i+1,p-1}(u).
\]

zero-width knot denominator는 0 contribution으로 처리하고, parameter는 마지막 knot 바로 아래로 clamp한다.

### 5.3 Image reconstruction과 visible-surface fitting

rendered image \(I\), target \(T\), D-SSIM weight \(\lambda\)에 대해

\[
L_{\mathrm{image}}=(1-\lambda)\lVert I-T\rVert_1+
\lambda(1-\mathrm{SSIM}(I,T)).
\]

certain Gaussian \(x_k\)의 patch/UV surface anchor를 \(S_{c_k}(u_k,v_k)\), 평균 scale을 \(\bar{s}_k\)라 하면 fitting 항은

\[
L_{\mathrm{fit}}=
\frac{1}{|A|}\sum_{k\in A}
\min\left(
\frac{\lVert x_k-S_{c_k}(u_k,v_k)\rVert_2^2}
     {\max(\bar{s}_k,10^{-4})^2},100\right).
\]

최종 visible surface loss는

\[
L_{\mathrm{surface}}=w_s\left(L_{\mathrm{fit}}+0.1L_{\mathrm{smooth}}\right).
\]

여기서 \(x_k\)는 detach되어 있으므로 surface loss는 NURBS parameter만 갱신한다.

### 5.4 Uncertain Gaussian 손실

uncertain Gaussian \(x_k\)와 해당 surface anchor \(S(u_k,v_k)\), confidence \(c_k\)에 대해 anchor prior는

\[
L_{\mathrm{anchor}}=w_a\,\mathrm{mean}_k\left[(x_k-S(u_k,v_k))^2(1-c_k)\right].
\]

image residual MSE를 \(r\)이라고 하면 confidence target과 loss는

\[
c^*=\mathrm{clamp}(e^{-r},0,1),\qquad
L_{\mathrm{confidence}}=w_c\,\mathrm{mean}_k(c_k-c^*)^2.
\]

이 손실 함수는 구현되어 있지만, 기본 training path는 occluded surface에서 uncertain Gaussian을 자동 생성·append하지 않는다.

### 5.5 Adaptive Density Control

visible Gaussian \(i\)의 누적 gradient 통계는

\[
g_i=\frac{G_i}{\max(D_i,1)},\qquad
G_i\leftarrow G_i+\lVert\nabla_{\mathrm{screen}}\rVert_2,\qquad
D_i\leftarrow D_i+1.
\]

\(g_i\ge\tau\)이고 최대 scale \(s_i^{\max}\)가 scene extent \(E\)의 dense threshold \(\rho E\) 이하이면 clone, 초과하면 split 후보가 된다.

\[
\mathrm{clone}:s_i^{\max}\le\rho E,\qquad
\mathrm{split}:s_i^{\max}>\rho E.
\]

split child는 parent scale \(s\)에 비례한 Gaussian noise offset을 quaternion 방향으로 회전해 배치하고, child scale은 `split_samples`를 \(m\)으로 하여

\[
s_{\mathrm{child}}=\frac{s}{0.8m}
\]

로 설정된다. uncertain row는 clone/split하지 않으며 낮은 confidence에서만 cleanup prune한다.

## 6. 현재 기본 경로의 범위

현재 `train.py` 기본 경로는 COLMAP scene에서 certain Gaussian을 초기화하고, CUDA Gaussian rasterization, image loss, ADC, canonical visible-NURBS reconstruction, checkpoint/PLY/NURBS artifact 저장을 수행한다.

uncertain Gaussian proposal·append·ownership·loss 관련 구현은 저장소에 존재하지만, 기본 trainer는 그것을 자동 생성 단계로 orchestration하지 않는다. uncertain-to-certain promotion도 수행하지 않는다. 따라서 이 문서는 visible Gaussian과 canonical visible-NURBS lifecycle을 현재 프레임워크의 주 파이프라인으로 기술한다.
