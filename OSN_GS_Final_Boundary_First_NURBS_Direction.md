SN-GS Final NURBS Construction Direction

## Boundary-First Topology, Component-Level Geometry Fitting, and Topology-Aware Chart Generation

---
## Document Status

This is the **governing implementation plan** for all future OSN-GS NURBS
construction work. Its phase gates, implementation prohibitions, benchmark
matrix, and approval requirements take precedence over earlier NURBS migration
plans and roadmap notes when they disagree.

The historical voxel-per-patch plan remains useful as the record of the Stage 1
baseline and its measurements, but it does not authorize further implementation
work. See
[OSN_GS_Voxel_Driven_NURBS_Migration_Plan.md](OSN_GS_Voxel_Driven_NURBS_Migration_Plan.md).
---

---

# 0. 문서 목적

이 문서는 OSN-GS의 NURBS construction 최종 방향을 확정하고, 코딩 에이전트가 현재 구현을 파괴하지 않은 채 단계적으로 전환하도록 지시하기 위한 실행 계획서다.

최종 방향은 다음 문장으로 요약한다.

> **Boundary가 surface topology와 chart coordinate를 결정하고, Gaussian 전체가 NURBS geometry를 결정한다.**

즉, boundary curve가 surface geometry를 단독으로 생성하지 않는다. Boundary는 다음을 결정한다.

- observed support topology
- outer boundary와 inner hole boundary
- crease 및 chart split 위치
- patch layout
- U/V parameterization 방향
- boundary-conforming iso-line 구조
- occluded extension의 기준 방향

실제 3D geometry는 component에 속한 Gaussian 전체를 이용한 기존 NURBS fitting으로 결정한다.

---

# 1. 프로젝트 범위

## 1.1 OSN-GS

OSN-GS는 observed Gaussian으로부터 NURBS 기반 visible surface를 구성하고, 해당 surface의 구조적 prior를 이용해 occluded surface에 Gaussian을 배치하는 연구 프레임워크다.

## 1.2 GS-Insp

GS-Insp는 별도의 Gaussian Splatting inspector 프로젝트다.

현재 작업은 오직 **OSN-GS의 NURBS construction 및 support topology**에 관한 것이다. 두 프로젝트를 혼용하지 않는다.

---

# 2. 현재 구현 상태

## 2.1 Legacy constructor

현재 legacy 경로의 대략적인 구조는 다음과 같다.

```text
Gaussian centers
→ coarse/fine 2-stage adaptive voxel bootstrap
→ voxel centroid 및 normal
→ normal discontinuity connected components
→ patch별 budget
→ patch point set의 PCA min-max rectangular UV domain
→ IDW seed
→ regularized LSQ
→ foot-point UV correction
→ Gaussian-to-patch binding
→ untrimmed NURBS evaluation
```

주요 한계:

- voxel은 segmentation과 budget 산정에만 사용됨
- voxel AABB가 patch domain에 영향을 주지 않음
- patch domain은 PCA min-max rectangle
- hole, U-shape, crescent 등의 non-rectangular support를 geometry domain이 덮음
- viewer의 geometry 및 iso-line이 support mask를 적용하지 않던 문제가 존재했음
- topology와 geometry가 분리되어 있지 않음

## 2.2 Stage 1 coarse voxel-per-patch baseline

현재 구현된 Stage 1-A~E는 다음 구조다.

```text
recursive raw-count voxel hierarchy
→ active leaf마다 독립 NURBS patch
→ leaf 내부 raw Gaussian fitting
→ local plane–voxel AABB intersection polygon
→ patch-local UV support mask
→ global patch union metric
```

확인된 장점:

- raw Gaussian local fitting으로 geometry accuracy 크게 개선
- sine surface curvature 재현 향상
- planar hole false-fill 감소
- patch overlap 감소
- close parallel sheet 분리 유지

확인된 한계:

- leaf voxel마다 독립 patch가 생성되어 patch 수 증가
- active-active interface에서 seam 및 tiny false hole 발생
- 실제 hole boundary가 voxel 해상도보다 작으면 완전 복원 실패
- hole이 명시적 boundary loop가 아니라 patch union의 빈 공간으로 간접 표현됨
- voxel별 독립 UV frame으로 sign/axis/frame mismatch 가능
- density-refined boundary가 component 전체가 아니라 leaf 단위로 적용될 경우 seam을 근본적으로 제거하기 어려움

이 구현은 폐기하지 않는다.

다음 역할로 유지한다.

- local fitting accuracy baseline
- voxel support topology 실험 baseline
- 최종 architecture의 ablation 비교군
- legacy와 최종 구조 사이의 중간 검증 경로

---

# 3. 최종 설계 원칙

## 3.1 Geometry, topology, parameterization의 분리

최종 patch 표현은 단순한 NURBS surface 하나가 아니다.

\[
\mathcal{P}_k =
\left(
\Omega_k,
\Phi_k,
S_k,
M_{\mathrm{obs},k},
C_{\mathrm{ext},k}
\right)
\]

각 항의 의미:

- \(\Omega_k\): canonical parametric domain
- \(\Phi_k\): physical surface support를 parametric domain에 대응시키는 chart mapping
- \(S_k(u,v)\): NURBS geometry
- \(M_{\mathrm{obs},k}(u,v)\): observed valid support
- \(C_{\mathrm{ext},k}(u,v)\): controlled extrapolation confidence

각 모듈의 책임:

### Topology

- surface component
- outer boundary
- inner hole
- crease
- disconnected surface
- chart split

### Parameterization

- U/V orientation
- boundary correspondence
- iso-line flow
- Jacobian quality
- seam location

### Geometry

- control points
- rational weights
- curvature
- Gaussian residual
- smoothness

### Support

- valid observed surface
- hole
- outer support boundary
- invalid domain

이 책임들을 다시 하나의 PCA rectangle에 묶지 않는다.

---

# 4. 최종 아키텍처

```text
Synthetic / Eligible Gaussian Set
        ↓
Adaptive Voxel Surface Scaffold
        ↓
Voxel Local Surface Analysis
        ↓
Compatible Surface-Cell Graph
        ↓
Physical Surface Components
        ↓
Component-Level Boundary Extraction
  ├─ outer loops
  ├─ inner hole loops
  ├─ crease boundaries
  └─ unresolved/complex boundaries
        ↓
Topology Classification
  ├─ disk-like
  ├─ annulus
  ├─ multi-hole
  ├─ disconnected
  ├─ crease-separated
  └─ non-chartable / complex
        ↓
Chart Layout Selection
  ├─ trimmed single component patch
  ├─ boundary-conforming quad chart
  ├─ O-grid multi-patch
  ├─ cut-to-disk charts
  └─ safe fallback
        ↓
Boundary-Conforming Parameterization
  ├─ harmonic coordinate field
  ├─ Coons/Gordon seed
  ├─ boundary correspondence
  └─ Jacobian validation
        ↓
Component Raw Gaussian NURBS Fitting
  ├─ IDW or boundary-based seed
  ├─ regularized LSQ
  ├─ boundary constraints
  └─ foot-point UV refinement
        ↓
Observed Support Mask / Trim Loops
        ↓
Boundary-Aligned Extension Charts
        ↓
Occluded Surface Extension
```

---

# 5. 핵심 역할 분담

## 5.1 Voxel의 역할

Voxel은 최종 NURBS patch가 아니다.

Voxel은 다음 역할을 수행한다.

- local Gaussian grouping
- local plane hypothesis
- local normal observation
- density/support evidence
- adaptive resolution
- boundary location bracket
- surface-cell adjacency node

최종 관계:

```text
Voxel = local surface evidence
Component = physical surface region
Chart = parameterization unit
NURBS patch = geometry representation unit
```

다음 등식은 최종 구조에서 성립하지 않는다.

```text
leaf voxel == NURBS patch
```

단, Stage 1 voxel-per-patch 경로는 ablation으로 유지한다.

## 5.2 Boundary의 역할

Boundary는 surface geometry의 부수적인 clipping 정보가 아니다.

Boundary는 다음을 결정하는 선행 구조다.

- component topology
- hole 존재 여부
- patch/chart layout
- U/V coordinate 방향
- chart seam
- boundary-conforming iso-line
- extension direction

Boundary는 두 단계로 구성한다.

### Coarse boundary

active/inactive voxel interface

### Refined boundary

Gaussian density transition 또는 confidence-weighted support transition

```text
voxel interface
→ topological bracket
→ density refinement
→ sub-voxel boundary curve
```

## 5.3 Gaussian의 역할

Gaussian 전체는 최종 surface geometry를 결정한다.

Boundary curve만으로 surface를 확정하지 않는다.

각 component \(C\)의 Gaussian 집합:

\[
\mathcal{G}_C =
\bigcup_{V_i \in C} \mathcal{G}_{V_i}
\]

를 이용해 다음을 최적화한다.

\[
\min_{\mathbf P}
\sum_{i \in \mathcal G_C}
\left\|
S(u_i,v_i)-x_i
\right\|^2
+
\lambda_s R_{\mathrm{smooth}}
+
\lambda_a R_{\mathrm{anchor}}
+
\lambda_b R_{\mathrm{boundary}}
\]

Boundary는 parameterization과 constraint를 제공하고, Gaussian residual이 control points를 결정한다.

---

# 6. Boundary-First가 의미하는 것

Boundary-first는 다음 순서를 의미한다.

```text
component 생성
→ boundary 추출
→ topology 판정
→ chart layout 결정
→ parameterization
→ NURBS fitting
```

다음을 의미하지 않는다.

```text
boundary curve만 추출
→ Gaussian 내부 정보를 무시
→ curve interpolation만으로 최종 surface 확정
```

Coons/Gordon surface는 최종 surface라기보다 다음 용도로 우선 사용한다.

- boundary-conforming initial seed
- structured control-grid initialization
- interior iso-line initial layout
- LSQ 안정화

최종 geometry는 Gaussian 전체에 대한 fitting으로 재조정한다.

---

# 7. Topology별 최종 표현 정책

## 7.1 Disk-like component

예:

- plane
- triangle
- trapezoid
- mild curved sheet
- U-shape without enclosed hole

우선순위:

1. boundary-conforming quadrilateral chart 가능성 검사
2. boundary anchor 또는 corner 선택
3. four-sided boundary segmentation
4. Coons 또는 harmonic parameterization seed
5. component-level Gaussian LSQ fitting

Boundary가 지나치게 복잡하거나 quadrilateral chart가 불안정하면 trimmed component patch를 fallback으로 사용한다.

## 7.2 Annulus component

예:

- planar hole
- ring-like support
- 하나의 inner hole을 가진 connected sheet

두 표현을 모두 유지한다.

### Correctness baseline

```text
single component NURBS geometry
+ outer trim loop
+ inner trim loop
```

목적:

- seam 없는 안정적인 geometry
- topology 정확성 검증
- existing fitter 재사용
- 최종 구조의 기준선

### Boundary-conforming target

```text
outer boundary
+ inner boundary
→ segment correspondence
→ O-grid multi-patch
```

각 O-grid patch:

- outer boundary segment
- inner boundary segment
- 두 radial connector
- boundary-conforming U/V field
- component Gaussian subset 또는 weighted global fitting

목적:

- iso-line이 boundary 형상을 따름
- radial/tangential coordinate 확보
- hole topology 명시
- occluded extension 방향과의 정합

초기 구현은 correctness baseline을 먼저 통과시킨 뒤 O-grid를 활성화한다.

## 7.3 Multi-hole component

가능한 정책:

- trimmed component patch baseline
- hole별 cut graph
- multi-chart decomposition
- 복잡도가 높으면 topology fallback

하나의 rectangular chart에 모든 hole을 억지로 넣지 않는다.

## 7.4 Crease-separated component

crease boundary는 support trim이 아니라 patch topology boundary다.

결과:

- 서로 다른 NURBS patch
- shared boundary curve
- 필요 시 \(C^0\), \(G^1\), \(C^1\) continuity 조건

## 7.5 Close or crossing sheets

공간적으로 가깝더라도 다음 기준으로 component를 분리한다.

- local plane offset
- normal orientation
- tangent compatibility
- sheet thickness
- Gaussian assignment
- multi-layer evidence

XY projection union metric만으로 topology를 판단하지 않는다.

---

# 8. Iso-Line 정책

## 8.1 의미

NURBS iso-line은 단순한 viewer grid가 아니라 chart coordinate의 시각화다.

\[
u = \mathrm{constant},
\qquad
v = \mathrm{constant}
\]

Boundary-conforming architecture에서는 iso-line이 다음을 반영해야 한다.

- boundary tangent direction
- boundary-normal 또는 radial direction
- chart topology
- extension coordinate

## 8.2 Disk-like chart

가능한 목표:

- 한 iso-line family는 opposing boundary segment를 연결
- 다른 family는 나머지 boundary segment를 연결
- boundary 형상을 따라 smooth하게 변화

## 8.3 Annulus O-grid

권장 coordinate 의미:

- \(u\): inner/outer boundary를 따라가는 periodic tangential coordinate
- \(v\): inner boundary에서 outer boundary로 진행하는 radial coordinate

또는 extension 목적에 따라 반대로 정의할 수 있다.

필수 검증:

- iso-line crossing 없음
- orientation sign consistency
- periodic seam consistency
- Jacobian lower bound
- boundary correspondence 안정성

## 8.4 Trimmed fallback

Trimmed component patch에서는 underlying iso-line이 boundary를 완전히 따르지 않아도 된다.

단:

- invalid support에서 geometry/index를 생성하지 않음
- hole을 가로지르는 iso-line segment를 생성하지 않음
- valid run 단위로 polyline 분할
- outer/inner trim contour를 별도로 표시

---

# 9. Occluded Extension과 Observed Geometry의 분리

Observed surface representation과 occluded extension chart를 반드시 동일하게 만들 필요는 없다.

권장 구조:

```text
Observed component NURBS
        ↓
Selected boundary segment
        ↓
Boundary tangent
+ surface normal
+ outward support direction
        ↓
Boundary-aligned extension chart
        ↓
Occluded NURBS strip
```

표현:

\[
S_{\mathrm{obs}}(u,v)
\]

\[
S_{\mathrm{ext},j}(s,t)
\]

여기서:

- \(s\): boundary tangential coordinate
- \(t\): boundary 밖으로 진행하는 extension coordinate

장점:

- observed surface를 안정적인 trimmed NURBS로 유지 가능
- 전체 observed parameterization을 extension 목적에 맞게 왜곡하지 않음
- occluded extension은 boundary 정렬 좌표를 직접 사용
- boundary별 confidence와 extension policy 분리 가능

최종 연구 방향에서 이 구조를 주요 후보로 유지한다.

---

# 10. 구현 단계

---

# Phase 0 — Current State Preservation

## 목표

현재 구현을 비교군으로 완전히 보존한다.

필수 constructor mode:

```text
legacy
voxel_patch_stage1
boundary_component_trimmed
boundary_component_charted
```

`legacy`와 `voxel_patch_stage1`의 출력은 변경하지 않는다.

필수:

- golden regression 유지
- 기존 benchmark artifact 유지
- 기존 Stage 1 report 보존
- 모든 새 기능은 feature flag로 추가
- trainer 기본 mode 변경 금지

완료 후 다음 단계로 진행한다.

---

# Phase 1 — Surface-Cell Component Builder

## 목표

active voxel들을 physical surface component로 병합한다.

## 1.1 입력

Stage 1 hierarchy의 leaf descriptor를 재사용한다.

```text
leaf_id
AABB
Gaussian indices
raw count
centroid
local plane normal
tangent basis
plane residual
eigenvalue ratio
state
```

## 1.2 adjacency

초기에는 face adjacency만 사용한다.

edge/corner adjacency는 별도 ablation으로 둔다.

## 1.3 compatibility

최소 조건:

### Normal compatibility

\[
\theta_{ij}
=
\arccos(|n_i^\top n_j|)
<
\tau_n
\]

### Plane offset

\[
d_{ij}
=
\frac{|(c_j-c_i)^\top n_i|}
{h_i+\epsilon}
<
\tau_d
\]

### Support continuity

- active-active shared face
- inactive gap 없음
- complex leaf는 별도 정책

### Smooth curvature

normal angle만으로 smooth curved sheet를 과분할하지 않도록 local normal-field smoothness를 진단한다.

초기 Phase 1에서는 단순 threshold를 사용하되, 모든 merge/split 이유를 export한다.

## 1.4 출력

```text
component_id
member_leaf_ids
Gaussian indices
component adjacency
component AABB
aggregate plane descriptor
boundary leaf IDs
merge diagnostics
```

## 완료 조건

Planar Hole:

- physical component count = 1
- enclosed inactive region = 1

Crease:

- component count가 GT 2에 접근
- legacy의 과분할 악화 없음

Close parallel sheets:

- component가 서로 병합되지 않음

보고 후 다음 단계 진행 여부를 사용자에게 확인한다.

---

# Phase 2 — Component-Level Boundary Extraction

## 목표

patch별이 아니라 component 전체에서 outer/inner boundary를 먼저 추출한다.

## 2.1 Coarse support

component member voxel의 plane–AABB intersection polygon을 component frame 또는 provisional UV에 투영한다.

```text
component voxel polygon union
→ coarse support mask
```

active-active shared face는 boundary가 아니다.

## 2.2 Refined support

component Gaussian으로 density field를 생성한다.

초기 구현:

- center-based kernel density
- raw count weight
- local NN spacing 기반 bandwidth
- configurable threshold
- marching squares contour extraction

후속 구현:

- opacity
- planarity
- normal confidence
- eligibility
- hysteresis
- local density normalization

## 2.3 contour hierarchy

반드시 다음을 구분한다.

- outer loops
- inner hole loops
- tiny artifact loops
- unresolved open contours

border-connected background와 enclosed background를 분리한다.

## 2.4 topology descriptor

```text
connected_component_count
outer_loop_count
hole_count
Euler characteristic
loop area
loop perimeter
loop nesting hierarchy
```

## 2.5 artifact

- voxel union
- raw density
- threshold field
- outer contour
- hole contour
- artifact contour
- world-space boundary points

## 완료 조건

Planar Hole:

- component = 1
- outer loop = 1
- significant hole loop = 1
- tiny false-hole area 감소
- false-fill이 Stage 1 coarse baseline보다 감소

Plane/Sine:

- significant hole = 0
- false tiny hole 감소

---

# Phase 3 — Trimmed Component Correctness Baseline

## 목표

component-level boundary를 사용해 안정적인 geometry와 support topology 기준선을 만든다.

## 3.1 Geometry fitting

component의 모든 raw Gaussian을 모아 기존 fitter를 재사용한다.

```text
component raw Gaussian union
→ component initial frame
→ initial UV
→ existing IDW seed
→ regularized LSQ
→ foot-point correction
```

## 3.2 Support

component outer/inner contour를 UV support mask 또는 trim loop로 적용한다.

## 3.3 중요 계약

- control grid는 hole을 가로질러도 정상
- rendered surface는 hole을 덮으면 안 됨
- iso-line은 hole에서 끊겨야 함
- hole을 control-grid topology로 억지로 표현하지 않음

## 3.4 metric

### Geometry

- point-to-surface
- surface-to-GT
- normal error
- Jacobian degeneracy
- control-grid collapse

### Support

- IoU
- false-fill
- uncovered ratio
- hole IoU
- hole count
- Euler error
- boundary Chamfer/Hausdorff

### Architecture

- geometry patch count
- component count
- seam/gap
- construction time
- memory

## 완료 조건

Planar Hole:

- geometry patch count = 1
- component count = 1
- significant hole count = 1
- active-active seam = 0
- geometry accuracy가 Stage 1보다 의미 있게 악화되지 않음

Sine:

- Stage 1에서 얻은 높은 curvature accuracy 유지

이 단계가 최종 correctness baseline이다.

---

# Phase 4 — Boundary-Conforming Chart Generator

## 목표

Boundary 형상을 따르는 parameterization과 iso-line을 생성한다.

이 단계는 trimmed baseline을 대체하기 전에 별도 experimental mode로 구현한다.

## 4.1 topology classification

```text
disk_like
annulus
multi_hole
crease_split
complex
```

## 4.2 disk-like chart

초기 후보:

- boundary corner/anchor detection
- four-sided boundary segmentation
- Coons patch seed
- harmonic map
- Gaussian LSQ refinement

필수 검증:

- UV overlap 없음
- Jacobian positive
- boundary correspondence
- aspect distortion
- Gaussian fitting residual

## 4.3 annulus chart

초기 목표는 O-grid multi-patch다.

### Input

- outer loop
- inner loop

### Layout

- loop segmentation count: 4 또는 8부터 시작
- inner/outer segment correspondence
- radial connector curve
- per-cell quadrilateral chart

### Seed

- Coons patch 또는 transfinite interpolation

### Refinement

- component Gaussian fitting
- shared boundary constraint
- seam consistency

### Coordinate

권장:

- tangential periodic coordinate
- radial coordinate

## 4.4 periodic seam

annulus에는 적어도 하나의 periodic seam 또는 multi-patch junction이 필요하다.

필수:

- deterministic seam placement
- low-curvature 또는 low-confidence region 우선
- seam provenance
- seam metric
- viewer toggle

## 4.5 continuity

초기:

- shared boundary \(C^0\)

후속:

- tangent-plane \(G^1\)
- 가능하면 control-point 기반 \(C^1\)

초기 correctness를 위해 \(C^0\)부터 시작하고, continuity 최적화는 별도 단계로 둔다.

## 완료 조건

Planar Hole O-grid:

- outer/inner boundary를 정확히 따름
- significant hole = 1
- iso-line이 boundary conforming
- patch seam이 허용 threshold 이하
- Jacobian fold 없음
- trimmed baseline 대비 geometry accuracy 유지

---

# Phase 5 — Boundary-Aligned Extension Charts

## 목표

Observed surface chart와 별개로 occluded extension을 위한 boundary-local chart를 생성한다.

## 5.1 boundary segment selection

다음 종류를 구분한다.

- observed outer support boundary
- inner hole boundary
- crease boundary
- invalid/untrusted boundary
- potential occlusion boundary

모든 boundary가 extension 대상은 아니다.

## 5.2 local frame

각 boundary sample에 대해:

- boundary tangent
- surface normal
- in-surface outward normal
- confidence

를 계산한다.

## 5.3 extension chart

\[
S_{\mathrm{ext}}(s,t)
\]

- \(s\): boundary tangential parameter
- \(t\): outward extension parameter

## 5.4 controlled extrapolation

\[
C_{\mathrm{ext}}(s,t)
\]

를 별도 필드로 유지한다.

초기 기준:

- boundary confidence
- local curvature continuation
- normal consistency
- support density decay
- maximum extension distance

## 완료 조건

- observed geometry와 extension geometry가 명시적으로 분리됨
- extension이 support mask 바깥에서만 생성됨
- boundary tangent 방향이 안정적임
- extension confidence export 및 visualization 가능

---

# Phase 6 — Stage 2 Robustness Improvements

다음은 기본 architecture가 동작한 이후에 추가한다.

## 6.1 Raw count → support mass

\[
M_V =
\sum_{i\in\mathcal G_V}
c_i
\]

confidence 후보:

- opacity
- Gaussian planarity
- covariance normal confidence
- geometric consistency
- training eligibility

## 6.2 Voxel state

```text
ACTIVE
UNCERTAIN
INACTIVE
COMPLEX
```

## 6.3 Adaptive threshold

- patch-relative threshold
- quantile threshold
- local density normalization
- high/low hysteresis
- scale-aware bandwidth

## 6.4 Gaussian orientation fusion

- covariance smallest-axis normal
- voxel PCA normal
- fitted NURBS normal

## 6.5 geometric/topological subdivision

- point-to-plane residual
- normal variance
- density transition
- multi-layer evidence
- local fitting residual

## 6.6 lifecycle

- UV refresh 후 support mask 갱신
- patch merge/split lifecycle
- orphan cleanup
- residual backtracking
- ADC 이후 binding refresh

## 6.7 performance

- Python recursive hot path 제거
- vectorized hierarchy
- payload size 제어
- maintenance cadence 최적화
- snapshot copy 최적화

---

# 11. 구현 금지 사항

다음 방식으로 metric을 맞추지 않는다.

- GT hole을 참조한 scene-specific special case
- morphology closing으로 모든 작은 hole 강제 제거
- fixed hole count 강제
- patch별 threshold 하드코딩
- hole을 control grid collapse로 표현
- active-active interface를 support boundary로 취급
- boundary curve만으로 최종 geometry 확정
- boundary 추출 전에 PCA rectangle을 최종 domain으로 확정
- Stage 1 결과를 삭제하거나 덮어쓰기
- legacy 기본 경로 변경
- benchmark 없이 trainer/ADC에 즉시 통합

---

# 12. Benchmark Matrix

## 12.1 Required modes

```text
legacy
voxel_patch_stage1
boundary_component_trimmed
boundary_component_charted
```

## 12.2 Required scenes

### Rectangular baseline

- plane
- elongated plane
- sine
- density gradient
- mild curved sheet

### Support topology

- triangle
- trapezoid
- wedge
- L-shape
- U-shape
- crescent
- planar hole

### Chartability

- curved ribbon
- cylinder strip
- sphere cap
- saddle
- strongly bent sheet

### Multi-surface topology

- crease
- T-junction
- disconnected surfaces
- close parallel sheets
- crossing sheets
- thin shell front/back

## 12.3 Required metrics

### Geometry

- point-to-surface residual
- surface-to-GT residual
- Chamfer RMS
- normal error
- Jacobian minimum
- Jacobian condition
- fold-over count
- control-grid degeneracy

### Support

- coverage
- unsupported ratio
- false-fill
- support IoU
- hole IoU
- hole count
- connected components
- Euler characteristic
- boundary Chamfer
- boundary Hausdorff

### Chart

- UV overlap
- area distortion
- angle distortion
- iso-line crossing
- periodic seam mismatch
- boundary correspondence error

### Multi-patch

- patch count
- seam gap
- seam normal mismatch
- overlap
- duplicate coverage
- failed merge
- spurious split

### System

- construction time
- peak memory
- payload size
- viewer parity
- deterministic reproducibility

---

# 13. Viewer 및 Export 요구 사항

## 13.1 Viewer layers

- Gaussian centers/ellipses
- voxel hierarchy
- active/inactive/complex cells
- surface component IDs
- coarse voxel boundary
- refined density boundary
- outer loops
- inner hole loops
- chart layout
- NURBS control grid
- U iso-lines
- V iso-lines
- support mask
- periodic seam
- extension chart
- patch isolate
- component isolate

각 layer는 deterministic color와 toggle을 가져야 한다.

## 13.2 Export

```text
run provenance
constructor mode
config
voxel hierarchy
surface-cell descriptors
component membership
boundary loops
loop hierarchy
topology classification
chart layout
chart seams
control grids
knots/degrees/weights
Gaussian UV binding
support mask
extension confidence
metrics
failure diagnostics
```

## 13.3 Python/JavaScript parity

반드시 다음을 비교한다.

- Cox–de Boor evaluation
- support query
- trim application
- iso-line segmentation
- chart seam
- bounds/resetCamera
- patch/component count

---

# 14. 승인 게이트

에이전트는 각 Phase 종료 후 멈추고 보고한다.

## Phase 1 보고

- component builder 구조
- merge/split 기준
- planar hole component 수
- crease component 수
- close sheet 분리
- failure cases

## Phase 2 보고

- boundary extraction 흐름
- outer/inner loop
- density definition
- hole topology metric
- artifact contours
- threshold sensitivity

## Phase 3 보고

- trimmed component 결과
- geometry/support metric
- active-active seam 제거 여부
- sine accuracy
- planar hole correctness

## Phase 4 보고

- chart generator
- topology별 layout
- O-grid 결과
- Jacobian
- seam
- boundary-conforming iso-line
- trimmed baseline 비교

## Phase 5 보고

- extension chart
- boundary frame
- confidence
- observed/extension separation

사용자 승인 없이 다음 Phase로 자동 진행하지 않는다.

---

# 15. 최초 Agent 실행 명령

현재 `legacy` 및 `voxel_patch_stage1` 구현을 유지하고, 우선 **Phase 1 — Surface-Cell Component Builder**만 구현하라.

## 요구 사항

1. Stage 1 hierarchy leaf를 graph node로 재사용
2. face adjacency 생성
3. normal compatibility 계산
4. plane offset compatibility 계산
5. active-active support continuity 확인
6. deterministic connected component 생성
7. component provenance export
8. merge/split reason export
9. legacy와 voxel_patch_stage1 수치 불변 regression
10. benchmark-only 구현
11. trainer/ADC 통합 금지

## 필수 benchmark

- plane
- sine
- planar_hole
- crease
- close_parallel_sheets
- density_gradient

## 반드시 보고할 값

- voxel leaf count
- component count
- component별 Gaussian count
- component별 member leaf
- GT component count
- component assignment ARI
- merge error
- split error
- runtime
- failure cases

## 기대되는 Planar Hole 결과

```text
physical component count = 1
outer support loop candidate = 1
inner inactive region candidate = 1
```

단, Phase 1에서는 아직 boundary curve와 NURBS refit을 구현하지 않는다.

보고 후 멈추고 사용자의 승인을 기다린다.

---

# 16. 최종 결정 요약

최종 OSN-GS NURBS construction 방향은 다음과 같다.

```text
Boundary-first topology
+
Component-level Gaussian geometry fitting
+
Topology-aware chart generation
+
Boundary-aligned extension charts
```

핵심 설계 규칙:

1. Voxel은 local evidence이지 최종 patch가 아니다.
2. Boundary를 geometry fitting 이전에 추출한다.
3. Boundary는 topology와 parameterization을 결정한다.
4. Gaussian 전체가 NURBS geometry를 결정한다.
5. Hole은 support topology로 명시한다.
6. Disk-like surface와 annulus surface에 같은 chart를 강제하지 않는다.
7. Trimmed component patch를 correctness baseline으로 유지한다.
8. Boundary-conforming O-grid를 annulus의 최종 chart 후보로 개발한다.
9. Observed surface와 occluded extension chart를 분리할 수 있다.
10. 각 단계는 benchmark와 승인 게이트를 통과한 뒤 진행한다.
