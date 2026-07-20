# OSN-GS NURBS Construction: Voxel-Driven Architecture Migration Plan

> **Historical record — not the governing plan.** This document preserves the original voxel-driven migration rationale, Stage 1 baseline, and associated decision gates. Do not use it to authorize new implementation work or infer the current architecture direction. The governing plan is [OSN_GS_Final_Boundary_First_NURBS_Direction.md](OSN_GS_Final_Boundary_First_NURBS_Direction.md), which supersedes this document wherever they differ.

## 0. 문서 목적

이 문서는 OSN-GS의 현재 NURBS construction 구현을 코드 기준으로 감사한 뒤, 다음 두 아키텍처를 **사용자의 명시적 승인에 따라 순차적으로 구현**하기 위한 코딩 에이전트용 작업 명령서다.

- **Stage 1 — Architecture 1**
  - 품질 개선 여부를 빠르게 검증하는 실험적 baseline
  - `raw Gaussian count` 기반 voxel 활성화/분할
  - **active leaf voxel마다 독립 NURBS patch 생성**
  - voxel 경계를 patch/support boundary의 직접적인 제약으로 사용
  - patch 수 증가나 seam 비용보다 topology/support 복원 가능성 검증을 우선
- **Stage 2 — Architecture 2**
  - Stage 1의 결과를 바탕으로 확장하는 개선 구조
  - confidence-weighted support mass와 geometric/topological criteria 도입
  - compatible voxel의 patch 병합
  - active/uncertain/inactive 상태
  - boundary voxel refinement와 sub-voxel density-gradient boundary
  - 장기적인 품질·확장성·안정성을 목표로 함

핵심 원칙은 다음과 같다.

> 현재 구현을 추측해서 수정하지 말고 먼저 실제 코드 경로, 데이터 흐름, 수학적 동작을 확인한다.  
> Stage 1은 품질 검증을 위해 의도적으로 단순하고 직접적인 구조를 사용한다.  
> Stage 2는 Stage 1의 benchmark 결과가 확보된 뒤에만 진행한다.

---

# 1. 절대 준수 사항

## 1.1 단계별 중단 지점

에이전트는 다음 순서를 반드시 지켜야 한다.

1. **Current Architecture Audit 수행**
2. 감사 결과를 사용자에게 보고
3. **사용자의 명시적 승인 전까지 코드 수정 금지**
4. 승인 후 **Stage 1만 구현**
5. Stage 1 benchmark 결과와 변경사항 보고
6. **사용자의 명시적 승인 전까지 Stage 2 구현 금지**
7. 승인 후 Stage 2 구현

다음과 같은 임의 진행은 금지한다.

- 감사와 동시에 리팩터링
- Stage 1 구현 도중 Stage 2 구조를 미리 혼합
- voxel-per-patch 구조를 “비효율적”이라는 이유로 임의 제거
- raw count를 support mass로 임의 교체
- 기존 NURBS fitter를 근거 없이 전면 재작성
- benchmark 결과 없이 최적화 우선 적용
- 기존 synthetic scene의 성공 결과를 깨뜨리는 광범위한 변경

## 1.2 프로젝트 구분

- **OSN-GS**: visible Gaussian으로 NURBS surface를 구성하고 occluded surface로 확장하는 연구 프레임워크

현재 작업은 오직 **OSN-GS NURBS construction**에 관한 것이다. 다른 프로젝트 및 작업과 혼용하지 않는다.

## 1.3 우선순위

1. 실제 구현 파악
2. correctness
3. benchmark observability
4. geometry/support/topology 실패 원인 분리
5. 품질 개선 확인
6. 이후 patch 수, continuity, 성능, memory 최적화

---

# 2. Current Architecture Audit

## 2.1 목표

현재 NURBS construction이 실제로 어떤 파일, 클래스, 함수, 데이터 구조를 통해 동작하는지 확인한다.

문서나 기존 설명을 사실로 가정하지 않는다. 아래의 “예상 구조”를 코드와 대조하고, 맞는 부분과 다른 부분을 구체적으로 보고한다.

## 2.2 현재 예상 구조 — 검증 대상

```text
Gaussian centers + color
    ↓
adaptive voxel bootstrap
    ↓
voxel centroid 및 normal 추정
    ↓
normal discontinuity 기반 patch connected components
    ↓
patch별 control-point budget 할당
    ↓
PCA 기반 초기 UV parameterization
    ↓
IDW control-grid seed
    ↓
regularized LSQ fitting
    ↓
foot-point projection 기반 UV correction 반복
    ↓
모든 Gaussian을 patch ID와 UV에 binding
    ↓
support/trim mask 생성 또는 갱신
    ↓
export / viewer / training lifecycle
```

다음 세부 설명도 반드시 코드로 검증한다.

- density-adaptive voxel이 진짜 recursive octree인지, coarse/fine 2단계 구조인지
- voxel subdivision 기준이 Gaussian count인지, density quantile인지, 다른 지표인지
- voxel centroid가 raw Gaussian 대신 fitting observation으로 사용되는지
- voxel normal 추정 방식
- patch adjacency가 face adjacency인지, edge/corner adjacency도 포함하는지
- patch 분할이 normal angle threshold connected component인지
- Gaussian covariance orientation이 현재 topology에 사용되는지
- patch별 U/V control resolution 산정 방식
- aspect ratio가 control grid에 실제 반영되는지
- `visible_surface_resolution_u/v` 또는 유사 설정의 상한
- PCA parameterization의 정확한 축 선택과 normalization
- IDW seed 계산식
- LSQ data term, second-difference regularization, Tikhonov anchoring
- construction 시 rational weight가 모두 1인지
- foot-point projection 반복 횟수와 stopping condition
- Gaussian별 patch ID 및 UV binding 저장 위치
- support mask 또는 UV trim mask의 현재 생성 방식
- 내부 hole topology가 보존되는지
- viewer가 모든 `patches[]`를 렌더링하는지
- viewer가 trim/support mask를 실제 geometry 및 iso-line에 적용하는지
- Python과 JavaScript의 Cox–de Boor 평가가 동일한지
- `maintain_surface_from_certain` 또는 동등 lifecycle에서 UV 및 mask 갱신 순서
- training 중 control grid와 rational weights의 trainability
- initialization benchmark가 학습 전 결과인지

## 2.3 에이전트가 제출할 감사 보고서

### A. 실제 데이터 흐름

실제 호출 순서를 파일과 함수 단위로 작성한다.

```text
entry point
→ file:function
→ file:class.method
→ ...
```

### B. 구현 매핑 표

| 예상 기능 | 실제 파일/함수 | 현재 동작 | 예상과의 차이 | 위험 |
|---|---|---|---|---|
| voxel bootstrap | ... | ... | ... | ... |
| patch segmentation | ... | ... | ... | ... |
| UV parameterization | ... | ... | ... | ... |
| NURBS fitting | ... | ... | ... | ... |
| support mask | ... | ... | ... | ... |

### C. 수정 영향 범위

Stage 1 도입 시 변경해야 할 파일과 변경하지 말아야 할 파일을 구분한다.

### D. 현재 benchmark 재현

최소한 다음 scene을 기존 코드로 재실행하거나, 실행 방법을 확인한다.

- rectangular plane
- elongated plane
- sine surface
- planar hole
- crease

각 scene에 대해 다음을 기록한다.

- patch count
- control-grid resolution
- point-to-surface residual
- surface-to-GT residual
- support/unsupported ratio
- connected component count
- hole count
- renderer 출력 경로
- 실행 config, seed, JSON hash 또는 provenance

## 2.4 Audit 완료 조건

다음이 충족되기 전까지 Stage 1 구현을 시작하지 않는다.

- 실제 코드 경로가 확인됨
- 현재 patch 생성 기준이 확인됨
- 현재 voxel 계층 구조가 확인됨
- 현재 support mask 및 hole 처리 여부가 확인됨
- benchmark entry point가 확인됨
- Stage 1 변경 지점과 regression 위험이 정리됨
- 사용자가 Stage 1 진행을 명시적으로 승인함

---

# 3. Stage 1 — Architecture 1

## 3.1 Stage 1의 목적

Stage 1의 목적은 최종적인 최적 구조를 만드는 것이 아니다.

> Adaptive voxel을 적극적인 surface decomposition 단위로 사용하고, active voxel마다 독립 NURBS patch를 생성했을 때 planar hole, non-rectangular support, local curvature 및 scene density variation에서 실제 품질 향상이 발생하는지 검증한다.

Stage 1에서는 patch 수 증가, seam 수 증가, continuity 비용, memory overhead를 알고도 감수한다. 이것은 성능 최적화가 아니라 **표현력과 correctness를 검증하는 experimental baseline**이다.

## 3.2 Stage 1의 핵심 정책

### 3.2.1 Raw count 기반 voxel 분류

각 voxel `V`에 포함된 Gaussian 수를 다음과 같이 정의한다.

\[
N_V = |\mathcal{G}_V|
\]

초기 구현에서는 confidence-weighted support mass를 사용하지 않는다.

설정값은 반드시 config/CLI에서 변경 가능해야 한다.

```text
voxel_min_gaussian_count
voxel_max_gaussian_count
voxel_max_depth
voxel_min_size
```

기본 의미:

- `N_V < voxel_min_gaussian_count`
  - surface evidence 부족
  - 해당 voxel은 NURBS patch 생성 대상에서 제외
- `voxel_min_gaussian_count <= N_V <= voxel_max_gaussian_count`
  - active leaf voxel
  - 해당 voxel 내부 Gaussian으로 local plane 및 NURBS patch 생성
- `N_V > voxel_max_gaussian_count`
  - 해당 voxel만 subdivision
  - max depth 또는 min size까지 반복

초기 `voxel_min_gaussian_count` 후보는 사용자가 제안한 작은 값, 예를 들어 10 전후로 두되 하드코딩하지 않는다.

### 3.2.2 Recursive adaptive voxel hierarchy

현재 coarse/fine 2단계 구조가 존재한다면 실제 recursive subdivision이 가능한 구조로 확장한다.

안전한 방식 중 하나를 선택한다.

- 기존 voxel builder에 recursive mode 추가
- 별도의 experimental voxel builder 추가
- feature flag로 기존/신규 구조 선택

필수 조건:

- deterministic subdivision
- seed에 독립적인 공간 partition
- parent/child 관계 저장
- leaf voxel ID 안정성
- voxel AABB 저장
- leaf별 Gaussian index 보존
- empty 또는 under-threshold leaf 기록 가능

### 3.2.3 Voxel 내부 local Gaussian plane

각 active leaf voxel 내부 Gaussian center로 local plane을 추정한다.

최소 요구 사항:

- centroid
- covariance
- eigenvectors/eigenvalues
- plane normal
- tangent basis
- point-to-plane residual
- thickness 또는 smallest eigenvalue 기반 진단값

Stage 1에서는 plane의 존재를 기본 가정으로 사용하되, 가정이 깨지는 경우를 숨기지 않는다.

로그와 export에 다음 값을 포함한다.

```text
local_plane_rms
local_plane_max_error
eigenvalue_ratio
normal
tangent_u
tangent_v
```

복잡한 3차원 구조가 감지되더라도 임의의 최종 처리법을 추가하지 않는다.

허용되는 Stage 1 처리:

1. depth/size 여유가 있으면 해당 voxel 추가 subdivision
2. 더 이상 subdivision할 수 없으면 `complex_leaf`로 표시
3. 해당 leaf를 skip할지 임시 fitting할지는 config로 분리
4. 결과와 개수를 명확히 보고

사용자가 별도로 설계한 복잡 구조 처리법을 넣을 수 있도록 extension point를 남긴다.

### 3.2.4 Active leaf voxel마다 독립 NURBS patch 생성

Stage 1의 중요한 실험 조건이다.

> 서로 인접하고 같은 평면에 가까워도 active leaf voxel을 자동으로 병합하지 않는다.

각 active leaf voxel에 대해 독립적인 tensor-product NURBS patch를 생성한다.

목표:

- voxel별 local surface reconstruction 검증
- hole 내부 empty voxel이 patch를 생성하지 않도록 하여 support topology를 직접 복원
- local density variation에 따라 patch resolution이 적응하는지 확인
- Mip-NeRF 360 계열 장면에서 locality의 장점 검증

각 patch provenance:

```text
patch_id
source_leaf_voxel_id
parent_voxel_path
voxel_aabb
gaussian_indices
gaussian_count
local_plane_descriptor
control_grid_resolution
fit_metrics
```

### 3.2.5 Patch domain과 voxel boundary의 관계

Stage 1에서 patch/support boundary는 voxel 경계에 의해 직접 규제된다.

권장 초기 방식:

1. local plane과 voxel AABB의 교차 영역 계산
2. 해당 교차 polygon 또는 그 local 2D bounds를 patch의 support domain으로 사용
3. Gaussian UV를 voxel-local tangent frame에 매핑
4. NURBS patch가 voxel AABB 밖으로 과도하게 확장되지 않도록 boundary constraint 또는 clipped evaluation 적용

초기 구현의 목적은 완전한 CAD-grade trimming이 아니라 다음을 보장하는 것이다.

- patch가 자신의 source voxel 영역을 벗어나 전체 scene으로 확장되지 않음
- empty voxel에는 patch가 없음
- active voxel union이 coarse support topology를 형성함
- planar hole 내부의 inactive voxel이 실제 hole로 남음

boundary를 구현하는 방법은 현재 fitter와 renderer 구조를 감사한 뒤 가장 덜 침습적인 방식을 선택한다.

가능한 구현:

- UV support mask
- local plane–AABB intersection mask
- sampled mesh clipping
- boundary-constrained control grid
- trimmed evaluation

선택한 방식과 이유를 보고서에 명시한다.

### 3.2.6 Control-grid resolution

각 voxel patch의 control-grid resolution은 최소한 다음을 고려한다.

- voxel-local Gaussian count
- local tangent-plane extent
- NURBS degree 최소 조건
- 현재 global U/V cap
- subdivision depth

Stage 1에서는 지나친 자동 최적화를 하지 않는다. resolution policy는 단순하고 관측 가능해야 한다.

예시 정책:

```text
base resolution
+ Gaussian count bucket
+ local aspect-ratio allocation
```

모든 규칙은 config에 노출하고 로그에 기록한다.

### 3.2.7 기존 fitter 재사용

가능한 한 기존의 다음 요소를 재사용한다.

- PCA 또는 tangent-plane UV initialization
- IDW control-grid seed
- regularized LSQ
- second-difference smoothness
- seed anchoring
- foot-point UV refinement
- Gaussian-to-patch UV binding

단, 기존 fitter가 multi-patch 또는 작은 local patch를 처리하지 못하면 최소 범위에서 수정한다.

기존 fitting 알고리즘을 전면 교체하지 않는다.

## 3.3 Stage 1 데이터 흐름

```text
Confident 또는 synthetic Gaussian set
        ↓
Recursive raw-count adaptive voxel hierarchy
        ↓
Leaf voxel Gaussian count
        ├─ below minimum → inactive / skip
        ├─ within range → active leaf
        └─ above maximum → subdivide
        ↓
Active leaf local plane estimation
        ├─ planar enough → continue
        └─ complex → subdivide or mark complex_leaf
        ↓
One active leaf = one NURBS patch
        ↓
Voxel-local UV parameterization
        ↓
Existing NURBS seed + LSQ + UV correction
        ↓
Voxel-constrained patch/support boundary
        ↓
All voxel patches exported and rendered
```

## 3.4 Stage 1 viewer 및 export 요구 사항

반드시 모든 voxel patch를 렌더링한다.

필수 기능:

- patch별 deterministic color
- source voxel AABB toggle
- local plane toggle
- control grid toggle
- sampled surface toggle
- U/V iso-line toggle
- Gaussian-to-patch assignment color
- inactive/active/subdivided/complex voxel 색상 구분
- patch isolate
- voxel isolate
- support mask toggle
- boundary toggle

viewer가 primary patch만 표시하는 구조라면 이를 먼저 수정한다.

export에는 최소한 다음이 포함되어야 한다.

```text
run provenance
voxel hierarchy
leaf states
voxel AABBs
Gaussian indices/counts
local plane descriptors
patch control grids
knots/degrees/weights
patch-local UVs
support masks or clipping information
fit metrics
```

## 3.5 Stage 1 benchmark

### 3.5.1 필수 synthetic scenes

- rectangular plane
- elongated plane
- sine surface
- mild curved sheet
- density-gradient surface
- planar hole
- triangle
- U-shape
- crease
- close parallel sheets

### 3.5.2 Planar Hole 핵심 검증

필수 metric:

- predicted patch count
- active leaf count
- inactive enclosed leaf count
- hole count
- connected component count
- Euler characteristic
- false-fill ratio
- hole IoU
- support coverage ratio
- unsupported surface ratio
- boundary Chamfer
- boundary Hausdorff
- point-to-surface residual
- surface-to-GT residual

### 3.5.3 Geometry와 support 분리 측정

각 benchmark에서 반드시 다음을 분리한다.

- geometry fitting error
- support-domain error
- topology error
- voxel decomposition error
- renderer/export error

### 3.5.4 Ablation

최소한 다음 설정을 비교한다.

- 기존 constructor
- Stage 1 voxel-per-patch
- `voxel_min_gaussian_count` 변화
- `voxel_max_gaussian_count` 변화
- max depth 변화
- voxel boundary constraint on/off
- complex leaf skip/fit

## 3.6 Stage 1 완료 조건

Stage 1 구현 후 에이전트는 멈추고 사용자에게 보고한다.

필수 보고 내용:

1. 변경 파일
2. 새 config
3. 실제 데이터 흐름
4. 기존 구조와의 차이
5. benchmark 결과
6. planar hole의 hole 복원 여부
7. sine curvature 품질 유지 여부
8. patch 수와 memory/time 증가량
9. seam 또는 gap 현상
10. failure cases
11. Stage 2로 넘어가기 전에 해결해야 할 blocker

Stage 2는 자동으로 시작하지 않는다.

---

# 4. Stage 2 — Architecture 2

## 4.1 Stage 2의 목적

Stage 2는 Stage 1에서 확인된 표현력과 품질 개선을 유지하면서 다음 문제를 해결한다.

- voxel-per-patch로 인한 patch 수 폭증
- patch 간 seam과 gap
- raw count threshold의 데이터셋 의존성
- sparse but valid boundary voxel 손실
- dense but low-confidence Gaussian 영역 오인
- smooth curvature 과분할
- close parallel sheet 병합
- voxel-grid-aligned boundary
- fixed threshold instability
- real COLMAP/3DGS Gaussian의 품질 편차

목표 architecture:

```text
Confident Gaussian set
        ↓
Adaptive voxel hierarchy
        ↓
Leaf voxel classification
  ├─ active planar surface cell
  ├─ uncertain boundary cell
  ├─ complex cell → subdivide
  └─ inactive cell
        ↓
Local plane / normal estimation
        ↓
Surface-cell adjacency graph
        ↓
Compatible voxel merging
        ↓
NURBS patch candidate
        ↓
Coarse NURBS geometry fitting
        ↓
Active voxel union projected to patch UV
        ↓
Coarse support mask and topology
        ↓
Boundary-cell refinement using
confidence-weighted density gradient
        ↓
Outer contour + hole contours
        ↓
Support-masked or trimmed NURBS patch
```

## 4.2 Stage 2 전환 조건

Stage 2를 시작하기 전에 Stage 1 결과로 다음 질문에 답할 수 있어야 한다.

- voxel-per-patch가 planar hole을 복원했는가?
- 기존 sine/plane geometry accuracy를 유지했는가?
- patch boundary가 실제 support를 과도하게 침식하지 않는가?
- raw count threshold가 scene/density 변화에 얼마나 민감한가?
- patch 수와 seam이 실제로 어느 정도 문제인가?
- 어떤 voxel이 병합 가능하고 어떤 voxel이 분리되어야 하는가?
- boundary voxel의 해상도가 충분한가?
- local plane residual이 어떤 failure를 예측하는가?

## 4.3 Confidence-weighted support mass

Stage 2에서는 raw count 외에 effective support mass를 도입한다.

\[
M_V = \sum_{i \in \mathcal{G}_V} c_i
\]

초기 confidence 후보:

\[
c_i = c_i^{opacity}c_i^{planarity}c_i^{normal}c_i^{eligibility}
\]

모든 항을 한 번에 활성화하지 않는다.

권장 순서:

1. raw count 유지
2. planarity confidence 추가
3. opacity/confidence 추가
4. real-data eligibility는 synthetic 안정화 후 별도 활성화

Gaussian covariance에서 normal candidate를 얻을 때는 최소 scale 축을 사용하되 normal confidence를 함께 계산한다.

예:

\[
c_i^{planarity}=1-\frac{s_{min}}{s_{mid}+\epsilon}
\]

구현에 더 적합한 안정적인 지표가 있다면 감사 후 선택하고 이유를 기록한다.

## 4.4 Leaf voxel 상태 확장

Stage 2에서는 이진 active/inactive 대신 다음 상태를 사용한다.

```text
ACTIVE
UNCERTAIN
INACTIVE
COMPLEX
```

### ACTIVE

- 충분한 raw count 또는 support mass
- local plane residual이 허용 범위
- normal coherence가 높음

### UNCERTAIN

- count/mass가 낮지만 인접 active surface와 일관됨
- density gradient가 큰 boundary 후보
- sparse support 가능성

### INACTIVE

- surface evidence 부족
- 인접 surface와도 일관성 없음

### COMPLEX

- 높은 plane residual
- multiple normal modes
- crossing/multi-layer 가능성
- 추가 subdivision 또는 별도 처리 필요

상태 전이와 threshold에는 hysteresis를 적용할 수 있도록 설계한다.

## 4.5 Adaptive subdivision criteria 확장

Stage 2 subdivision은 count 하나만 사용하지 않는다.

### Computational trigger

\[
N_V > N_{max}
\]

### Geometric trigger

- point-to-plane residual
- eigenvalue ratio
- normal variance
- multiple normal modes
- local NURBS fitting residual

### Topological trigger

- active/inactive evidence 혼재
- density gradient가 큼
- hole 또는 external boundary 후보
- 인접 voxel state 불일치

### Resolution trigger

- local Gaussian scale 대비 voxel이 너무 큼
- local nearest-neighbor spacing 대비 voxel이 너무 큼

각 trigger를 독립적으로 logging하여 어떤 이유로 subdivision되었는지 추적 가능하게 한다.

## 4.6 Gaussian orientation과 voxel normal 융합

각 Gaussian의 covariance normal과 voxel PCA normal을 결합한다.

- `n_i^G`: Gaussian covariance smallest-axis normal
- `n_V^P`: voxel-local PCA normal
- `n^S`: coarse NURBS normal

초기 topology 구축:

\[
n_V = \operatorname{Fuse}(n_V^P, \{n_i^G\})
\]

coarse fitting 이후:

\[
n^S = \frac{S_u \times S_v}{\|S_u \times S_v\|}
\]

다음 mismatch를 진단한다.

\[
E_{normal}=1-|n_V^T n^S|
\]

이 값은 다음 실패를 구분하는 데 사용한다.

- wrong patch assignment
- invalid Gaussian orientation
- NURBS underfitting
- chart split 필요
- complex/multi-layer cell

## 4.7 Compatible voxel merging

Stage 2에서는 leaf voxel을 local surface cell로 취급하고 호환 가능한 voxel들을 graph로 병합한다.

각 voxel descriptor:

```text
centroid
local plane normal
tangent basis
planarity confidence
support mass
raw count
density estimate
local thickness
Gaussian indices
state
AABB
subdivision depth
```

인접 voxel 연결 기준 후보:

### Spatial adjacency

- face adjacency 우선
- edge/corner adjacency는 별도 ablation

### Normal compatibility

\[
\theta_{ij}=\arccos(|n_i^T n_j|)<\tau_n
\]

### Plane offset compatibility

\[
d_{ij}=|(c_j-c_i)^T n_i|<\tau_d
\]

### Tangential continuity

neighbor displacement가 tangent plane에 일관되는지 확인한다.

### Support continuity

중간에 inactive gap이 없는지 확인한다.

### Curvature smoothness

normal 변화가 smooth한지, crease처럼 discontinuous한지 구분한다.

병합 결과 connected component가 하나의 NURBS patch candidate가 된다.

## 4.8 Boundary score와 hysteresis

인접 surface cell `i,j`의 boundary score 후보:

\[
B_{ij}=w_nB_{ij}^{normal}+w_dB_{ij}^{distance}+w_oB_{ij}^{normal\ offset}+w_sB_{ij}^{scale}+w_\rho B_{ij}^{density}+w_cB_{ij}^{confidence}
\]

normal discontinuity:

\[
B_{ij}^{normal}=1-|n_i^Tn_j|
\]

normal offset:

\[
B_{ij}^{normal\ offset}=\frac{|(c_j-c_i)^T n_i|}{h_i+\epsilon}
\]

hysteresis:

```text
B > tau_high  → boundary 확정
B < tau_low   → 연결 유지
중간 영역      → 주변 graph consistency로 결정
```

Stage 2에서는 threshold를 hardcode하지 않고 scene/patch-relative normalization을 지원한다.

## 4.9 Coarse support topology

병합된 voxel component를 patch UV에 투영하여 coarse support mask를 만든다.

- active voxel: valid support
- uncertain voxel: boundary refinement 대상
- inactive voxel: invalid
- enclosed inactive component: hole 후보

반드시 outer boundary와 inner hole contour를 구분한다.

필수 topology metric:

- connected component count
- hole count
- Euler characteristic
- hole IoU
- false-fill ratio
- coverage
- unsupported ratio

## 4.10 Sub-voxel boundary refinement

Voxel face는 boundary의 coarse bracket으로 사용한다. 최종 boundary를 voxel face에 그대로 고정하지 않는다.

active/uncertain/inactive 경계 voxel 내부에서 confidence-weighted Gaussian density를 계산한다.

\[
D(x)=\sum_i c_iK_i(x)
\]

boundary는 density threshold 또는 hysteresis crossing으로 정제한다.

\[
D(x(t^*))=\tau_D
\]

가능한 구현:

- voxel edge interpolation
- local plane 위 2D density sampling
- marching squares
- marching cubes 후 local plane projection
- patch UV density contour

목표:

- voxel staircase artifact 감소
- confidence 높은 Gaussian density gradient에 boundary 정렬
- planar hole의 smooth inner contour 복원
- external support boundary 개선

## 4.11 Boundary 종류 분리

### Patch topology boundary

- crease
- chartability failure
- smoothness discontinuity
- close/crossing sheet 분리

결과: 서로 다른 NURBS patch

### Surface support boundary

- object outer boundary
- hole
- U-shape 내부
- observed support 종료

결과: 동일 patch 내부 support mask 또는 trim contour

### Controlled extrapolation boundary

- observed support 외부
- occluded surface extension 가능 영역

결과: `M_obs`와 `C_ext` 사이의 transition

목표 patch 표현:

\[
\operatorname{Patch}_k=(S_k(u,v),M_{obs,k}(u,v),C_{ext,k}(u,v))
\]

## 4.12 Boundary-constrained refitting

Stage 2에서는 coarse NURBS geometry를 유지하면서 boundary constraint를 LSQ에 추가할 수 있다.

\[
\min_P \sum_i\|S(u_i,v_i)-x_i\|^2+\lambda_sR_{smooth}+\lambda_aR_{anchor}+\lambda_bR_{boundary}
\]

초기에는 hard constraint보다 soft boundary term을 우선한다.

중요:

- hole boundary는 tensor-product domain edge로 강제하지 않는다.
- hole은 support mask/trim contour로 처리한다.
- crease boundary는 patch segmentation에 사용한다.
- outer chart boundary만 control-grid edge 제약 후보가 될 수 있다.

## 4.13 Stage 1 → Stage 2 migration 요구 사항

Stage 1 구현 시 다음을 보존한다.

- stable voxel IDs
- parent/child hierarchy
- leaf descriptor
- local plane descriptor
- patch-to-source-voxel mapping
- per-leaf raw count
- fit diagnostics
- support mask export
- feature flag
- baseline constructor 유지

Stage 2에서는 이를 기반으로 다음을 추가한다.

- support mass
- confidence fields
- multi-state leaf classification
- subdivision reason
- adjacency graph
- merge component ID
- boundary score
- uncertainty
- density field
- refined contour
- merged patch provenance

## 4.14 Stage 2 benchmark

Stage 1 benchmark를 모두 재실행한다.

추가 필수 scene:

- crescent
- trapezoid
- wedge
- L-shape
- cylinder strip
- sphere cap
- saddle
- strongly bent sheet
- T-junction
- disconnected surfaces
- crossing sheets
- thin shell front/back

비교 대상:

1. 기존 constructor
2. Stage 1 voxel-per-patch
3. Stage 2 merged topology
4. Stage 2 + support mass
5. Stage 2 + sub-voxel boundary refinement

필수 비교 지표:

- geometry accuracy
- support topology
- patch count
- seam/gap
- continuity
- memory
- construction time
- threshold sensitivity
- density/seed stability
- failure mode

## 4.15 Stage 2 완료 조건

- Stage 1의 planar hole 개선을 유지
- patch 수가 의미 있게 감소
- seam/gap이 Stage 1보다 감소
- sine 및 curved surface accuracy 유지
- close parallel sheets 분리
- crease 과분할 완화
- hole topology 보존
- density variation에 대한 threshold 민감도 감소
- boundary staircase artifact 감소
- regression benchmark 통과
- architecture 및 worklog 문서화

---

# 5. 구현 전략

## 5.1 Feature flags

기존 구조와 비교 가능하도록 최소한 다음 mode를 제공한다.

```text
legacy
voxel_patch_stage1
voxel_topology_stage2
```

기본값을 즉시 변경하지 않는다. benchmark로 검증한 후 결정한다.

## 5.2 코드 경계 제안

실제 파일 구조를 감사한 후 이름은 조정할 수 있지만 책임은 분리한다.

```text
Input Eligibility
Adaptive Voxel Hierarchy
Voxel Surface Analyzer
Voxel Patch Builder              # Stage 1
Voxel Surface Graph              # Stage 2
Voxel Component Merger           # Stage 2
NURBS Geometry Fitter
Support Domain Estimator
Boundary Refiner
Patch/Surface Export
Benchmark + Viewer
```

## 5.3 테스트 전략

### Unit tests

- voxel subdivision
- Gaussian assignment conservation
- deterministic leaf IDs
- plane fitting
- local UV transform
- AABB-plane intersection
- active/inactive classification
- hole extraction
- adjacency
- merge criterion
- boundary interpolation

### Regression tests

- legacy output unchanged under `legacy`
- sine accuracy does not regress beyond configured tolerance
- planar hole false-fill decreases
- patch export/viewer count matches constructor count
- Python/JavaScript surface parity
- deterministic seed behavior

## 5.4 실패를 숨기지 말 것

다음 상황은 fallback으로 덮지 말고 진단값을 남긴다.

- underconstrained local patch
- degenerate plane
- insufficient Gaussian
- complex voxel at max depth
- LSQ ill-conditioning
- invalid Jacobian
- UV fold-over
- empty support mask
- viewer/export mismatch

---

# 6. 에이전트 응답 형식

각 단계가 끝나면 다음 형식으로 보고한다.

## Summary

무엇을 확인하거나 구현했는지.

## Actual Architecture

파일/함수 기준 실제 구조.

## Changes

변경 파일과 책임.

## Design Decisions

선택한 구현 방식과 대안 대비 이유.

## Benchmark Results

before/after 표.

## Failure Cases

재현 가능한 실패와 로그/이미지 경로.

## Risks

correctness, performance, lifecycle, compatibility 위험.

## Next Gate

다음 단계에 필요한 사용자 승인 또는 결정.

---

# 7. 최초 실행 명령

우선 **Current Architecture Audit만 수행**한다.

현재 코드를 수정하지 말고 다음을 제출한다.

1. 실제 NURBS construction 호출 흐름
2. voxel 생성 및 subdivision의 실제 구현
3. patch segmentation의 실제 구현
4. UV, IDW, LSQ, foot-point correction의 실제 구현
5. support/trim mask와 planar hole 처리의 실제 구현
6. viewer/export의 multi-patch 및 mask 적용 여부
7. Stage 1을 구현할 때 변경해야 할 파일과 예상 위험
8. 기존 benchmark 재현 결과 또는 정확한 실행 명령
9. 예상 구조 설명 중 틀린 부분의 정정
10. Stage 1 implementation plan

보고 후 멈추고 사용자의 승인을 기다린다.
