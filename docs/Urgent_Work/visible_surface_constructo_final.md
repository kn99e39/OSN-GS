Visible Surface Construction을 완성 수준으로 끌어올리기 위한 개발 방향

1. 현재 구현 수준

현재 OSN-GS의 Visible Surface Construction은 학습된 Gaussian으로부터 covariance 기반의 국소 표면 구조를 해석하고, 신뢰할 수 있는 Gaussian들을 surface region으로 묶은 뒤, 해당 region의 관측 경계를 복원하여 실제 평가 가능한 NURBS surface를 생성하는 end-to-end 경로를 갖추고 있다.

현재 구현된 전체 흐름은 다음과 같다.

Observed Gaussians
→ Covariance Frame Extraction
→ Structural Reliability Evaluation
→ Surface Manifold Affinity
→ Consensus-Aware Surface Region Formation
→ Local Support-Termination Boundary Evidence
→ Ordered Boundary Recovery
→ Boundary-first Materialization Adapter
→ Visible NURBS Surface Fitting
→ Geometric Validation

Clean plane과 smooth curved sheet와 같은 제한된 synthetic fixture에서는 stable region 하나로부터 ordered closed boundary loop를 복원하고, 이를 TorchNURBSSurface로 materialize한 뒤 evaluate(uv)를 수행하는 데 성공했다. Rotation, uniform scale, input-order shuffle, covariance sign-equivalent representation에 대한 canonical tangent-frame invariance도 targeted fixture 범위에서는 보완되었다.

그러나 현재 구현은 전체 visible scene을 안정적으로 NURBS surface network로 변환하는 시스템은 아니다. 실제 trained 3DGS snapshot의 4,000-Gaussian crop에서는 consensus-aware region이 80개만 형성되었으며, 가장 큰 region도 29개 Gaussian에 불과했다. 전체 입력 중 3,409개, 약 85%의 Gaussian은 ambiguous_unassigned 상태로 남았다.

따라서 현재 구현의 정확한 의미는 다음과 같다.

신뢰 가능한 일부 Gaussian core로부터 admissible한 closed-boundary surface patch를 복원하는 experimental visible-surface construction path.

이는 다음과는 구분되어야 한다.

실제 학습 장면의 주요 visible surface를 충분한 coverage와 일반적인 topology로 안정적으로 복원하는 완성형 visible-surface construction system.

현재 구조를 완성 수준으로 끌어올리려면 단순히 기존 threshold를 완화하거나 NURBS fitter만 고도화해서는 안 된다. Reliable-core coverage, boundary interpretation, topology decomposition, multi-patch fitting, real-scene validation을 순서대로 확장해야 한다.

2. 완성 수준의 정의

Visible Surface Construction의 완성도를 전체 Gaussian 중 region에 포함된 Gaussian 비율만으로 판단해서는 안 된다. 학습된 3DGS에는 실제 surface geometry를 나타내는 Gaussian 외에도 다음과 같은 Gaussian이 포함될 수 있기 때문이다.

낮은 opacity의 보조 Gaussian

view-dependent appearance를 표현하기 위한 Gaussian

surface normal을 명확하게 나타내지 않는 isotropic Gaussian

과도하게 크거나 작은 Gaussian

floater 또는 optimization noise

서로 다른 깊이와 surface가 혼재된 Gaussian

따라서 완성 수준을 평가할 때에는 최소 다음 coverage를 분리해야 한다.

Gaussian-count coverage

Opacity-weighted coverage

Rendering-contribution-weighted coverage

Image-space coverage

World-space surface-area coverage

Region별 reliable/ambiguous/rejected coverage

완성형 시스템의 목표는 모든 Gaussian을 강제로 surface에 할당하는 것이 아니다. 대신 주요 visible geometry에 기여하는 Gaussian과 surface area의 대부분을 올바른 surface region 또는 명시적인 unresolved 상태로 설명할 수 있어야 한다.

즉 완성 기준은 다음에 가깝다.

주요 visible surface를 구성하는 Gaussian들을 높은 purity의 surface region으로 복원하고, 일반적인 surface topology를 disk-like chart network로 분해하여 유효한 NURBS patch network로 materialize할 수 있는 상태.

3. Reliable Core에서 전체 Surface Region으로의 확장

현재 가장 우선적으로 해결해야 할 병목은 reliable_core_only coverage이다.

현재 구조에서는 intrinsic reliability와 contextual consistency가 높은 Gaussian만 stable core가 되며, 주변의 많은 Gaussian은 ambiguity를 이유로 region에 포함되지 않는다. 이 방식은 false merge를 방지하는 데에는 효과적이지만, 실제 surface가 작은 core island들로 파편화되는 문제가 있다.

이를 해결하기 위해서는 threshold를 단순히 낮추는 대신, reliable core에서 주변 Gaussian으로 confidence를 전파하는 core-seeded region expansion이 필요하다.

권장 구조는 다음과 같다.

Reliable Core Seed
→ High-Confidence Attached Shell
→ Medium-Confidence Surface Shell
→ Unresolved Boundary Shell
→ Explicit Unknown/Conflict State

각 ambiguous Gaussian을 특정 region에 포함할지 판단할 때에는 다음 evidence를 함께 사용해야 한다.

Region surface에 대한 tangent-plane residual

Gaussian covariance normal과 region normal의 정합성

Gaussian footprint와 local spacing의 호환성

Accepted neighbor 중 해당 region에 속한 비율

Reliable core까지 연결되는 local topology path

가까운 경쟁 region의 존재

Close-parallel 또는 crease conflict

Gaussian opacity

Multi-view rendering contribution

Camera visibility와 depth consistency

이 문제는 unary와 pairwise evidence를 이용한 labeling 문제로 표현할 수 있다.

Unary Evidence:
Gaussian g가 surface region R에 얼마나 잘 맞는가?

Pairwise Evidence:
인접 Gaussian g_i와 g_j가 같은 surface label을 가져야 하는가?

구현 후보로는 다음과 같은 방법이 적절하다.

Confidence-aware iterative region growing

Constrained label propagation

Graph-cut 또는 CRF 기반 region assignment

Belief propagation

Conflict-veto를 포함한 constrained union-find

Surface projection과 reassignment를 반복하는 EM-style assignment

중요한 점은 UNKNOWN 상태를 제거하지 않는 것이다. 모든 ambiguous Gaussian을 강제로 할당하면 close-parallel surface, crease, thin geometry 사이에서 false merge가 증가할 가능성이 높다. 목표는 ambiguous 상태를 없애는 것이 아니라, 충분한 evidence가 있는 Gaussian만 안전하게 region에 편입하는 것이다.

4. Boundary Recovery와 Region Coverage의 공동 개선

현재 boundary recovery는 reliable core topology를 기반으로 한다. 따라서 reliable-core coverage가 낮으면 실제 surface가 계속 존재함에도 core가 끝나는 지점이 boundary처럼 보일 수 있다.

실제 Surface Continuation 존재
→ Reliability가 낮아 core가 중단됨
→ False Support-Termination Boundary 생성

이를 해결하기 위해서는 region expansion과 boundary recovery를 독립적으로 처리해서는 안 된다. Boundary는 단순히 reliable core가 끝나는 지점이 아니라, 동일한 surface manifold의 관측 가능한 continuation이 실제로 끝나는 지점이어야 한다.

최소한 다음 상태를 구분해야 한다.

Genuine observed surface termination

Reliability frontier

Sparse sampling gap

Occlusion-related observation boundary

Geometric crease

Parallel-surface conflict

Rejected or noisy adjacency

Low-rendering-contribution region

Covariance와 local topology는 geometry evidence를 제공하지만, 실제 visible boundary를 판단하려면 camera 및 renderer evidence가 추가되어야 한다.

활용 가능한 observation evidence는 다음과 같다.

여러 camera view에서 해당 방향의 surface support가 반복적으로 관측되는지

해당 방향의 camera ray가 실제 free space를 통과하는지

Boundary candidate가 image-space silhouette과 대응하는지

Depth discontinuity와 일치하는지

주변 Gaussian의 screen-space footprint가 continuation을 암시하는지

단순히 covariance reliability만 낮아진 영역인지

해당 Gaussian이 실제 rendering에 얼마나 기여하는지

이를 통해 observed_support_termination과 unresolved_sampling_gap을 더 안정적으로 구분해야 한다.

5. Hard Sector Quantization에서 Continuous Circular Gap으로의 전환

현재 support-termination detection은 transported tangent frame에서 주변 continuation direction을 sector로 양자화하는 방식에 기반한다. Worklog 16와 123에서는 canonical tangent frame, angular-margin sharing, cyclic missing-run normalization을 도입해 rotation, scale, input-order, covariance sign ambiguity를 보완했다.

그러나 sector discretization 자체는 여전히 다음 요소에 민감할 수 있다.

Sector 개수

Sector boundary 위치

Angular margin

Local density

Floating-point perturbation

Curvature에 따른 tangent transport 오차

완성 수준에서는 고정된 sector bin보다 continuous circular support-gap analysis를 사용하는 것이 더 적절하다.

각 Gaussian에서 accepted local neighbor를 tangent plane에 투영한 뒤 다음 과정을 수행할 수 있다.

Neighbor direction의 continuous angle을 계산한다.

Angle을 circular order로 정렬한다.

인접 angle 사이의 angular gap을 계산한다.

각 neighbor의 거리, confidence, footprint, density를 반영한다.

유효한 missing-support interval을 continuous angular interval로 표현한다.

Interval의 중심과 폭으로 boundary direction과 confidence를 산출한다.

즉 다음과 같은 표현에서:

Sector 4가 비어 있음

다음과 같은 표현으로 전환한다.

Tangent angular interval [θ_a, θ_b]에서
충분한 same-surface continuation evidence가 존재하지 않음

이 방식은 sector 시작각이나 bin boundary에 대한 민감도를 줄이고, curved surface에서 geometric support gap을 더 직접적으로 표현할 수 있다.

6. 단일 Closed Loop에서 Surface Chart Atlas로의 확장

현재 NURBS materialization은 사실상 다음 조건을 만족하는 경우에만 성공한다.

Stable Reliable Region
+ Ordered Closed Loop
+ Outer Boundary Candidate
+ No Branch
+ No Self-Intersection
+ Sufficient Interior Support

Open chain, branching boundary, multiple loop, unresolved boundary role과 같은 topology는 unsupported_topology 또는 review_required로 종료된다.

그러나 실제 장면에서는 다음 topology가 일반적으로 나타난다.

Open surface boundary

Multiple boundary loops

Inner holes

Crease

T-junction

Crossing surface

Closed smooth surface

하나의 NURBS patch로 표현하기 어려운 넓거나 복잡한 곡면

일부 boundary가 관측되지 않은 surface

따라서 장기적으로는 다음 가정을 버려야 한다.

Surface Region 하나
→ NURBS Surface 하나

대신 다음 구조가 필요하다.

Surface Region
→ Topology Analysis
→ Disk-like Chart Decomposition
→ Ordered Boundary per Chart
→ NURBS Patch Network

각 topology에 대한 처리 방식은 다음과 같다.

단일 Closed Loop

현재의 Boundary-first single-patch 또는 cap/fan materialization을 유지할 수 있다.

Multiple Loops

Outer loop와 inner loop의 ownership 및 nesting을 복원해야 한다. 이후 다음 중 하나를 선택할 수 있다.

Trimmed NURBS surface

Hole 주위를 여러 disk-like chart로 분할

Inner loop를 포함하는 structured patch network

Open Boundary

Open boundary가 다음 중 무엇인지 구분해야 한다.

실제 open surface

Reliable coverage 부족

Sampling gap

Occlusion boundary

Parameterization seam 필요

실제 open surface라면 open chart fitting을 지원해야 하며, coverage 부족이라면 region expansion으로 되돌려야 한다.

Branching Boundary

Branching component를 단순히 unsupported로 폐기하지 않고 다음으로 분해해야 한다.

Chart junction

Crease junction

T-junction

Competing boundary interpretation

Topology ambiguity

Closed Smooth Surface

Sphere-like closed surface에는 직접 관측된 outer boundary가 존재하지 않는다. 이 경우 parameterization을 위해 derived seam을 생성해야 한다.

Derived seam은 observed boundary와 구분되어야 하며, 다음 provenance를 가져야 한다.

Parameterization-derived

Paired seam

Not an observed surface termination

Reversible or reviewable cut

7. Surface Region을 Disk-like Chart로 분해

NURBS patch fitting을 안정적으로 수행하려면 각 chart가 가능한 한 topological disk에 가까워야 한다.

필요한 흐름은 다음과 같다.

Surface Manifold Region
→ Topology and Distortion Analysis
→ Cut/Seam Graph Generation
→ Disk-like Surface Charts
→ Ordered Chart Boundaries
→ NURBS Patch Fitting

Chart split 기준은 voxel이나 PCA rectangle이어서는 안 된다. 다음과 같은 surface evidence를 기준으로 해야 한다.

Observed surface boundary

High curvature

Crease

Topological handle

Parameterization distortion

Fitting residual concentration

Foldover risk

Jacobian degeneration

Support crossing

Derived seam requirement

Voxel은 spatial neighbor search나 broad-phase acceleration에는 사용할 수 있지만, final surface region이나 chart topology를 결정하는 기준이 되어서는 안 된다.

8. Boundary-first Builder의 Multi-Patch Fitter 확장

현재 materialization adapter는 admissible한 단일 closed loop와 observed interior support를 canonical LSQ fitter에 전달하여 하나의 TorchNURBSSurface를 생성한다.

완성 수준에서는 이를 patch-network fitter로 확장해야 한다.

Robust Chart Parameterization

가능한 방법은 다음과 같다.

Harmonic parameterization

Discrete conformal parameterization

Least-Squares Conformal Mapping

Geodesic-aware local coordinates

Boundary-constrained parameterization

PCA UV는 초기값이나 diagnostic 용도로 사용할 수 있지만, canonical chart parameterization으로 사용해서는 안 된다.

Robust Surface Fitting

Fitting은 다음 constraint를 함께 사용해야 한다.

High-weight observed boundary constraint

Reliable interior point constraint

Covariance-derived normal constraint

Surface smoothness regularization

Curvature regularization

Jacobian or foldover barrier

Robust loss against outliers

Iterative foot-point and UV correction

Adaptive control-grid resolution

고정된 control-grid 크기를 모든 surface에 적용하기보다 surface complexity와 residual에 따라 control resolution을 조절해야 한다.

Adaptive Patch Splitting

다음 문제가 발생하면 하나의 patch를 계속 왜곡시키기보다 chart를 분할해야 한다.

Residual이 특정 영역에 집중됨

Normal alignment error가 큼

Parameterization distortion이 큼

Jacobian이 퇴화함

Foldover가 발생함

Curvature가 하나의 control grid로 표현하기 어려움

Boundary 또는 support crossing이 발생함

Patch Continuity

인접 patch 사이에는 다음 continuity 조건이 필요하다.

일반적인 shared boundary: C0 continuity

Smooth continuation: 필요 시 C1 continuity

Crease: C0 continuity만 유지

Shared boundary ownership

Stable patch adjacency

Patch-boundary provenance

9. Review 상태를 Recovery Policy로 연결

현재 review_required, unsupported_topology, boundary_recovery_failed, fit_failed, validation_failed는 대부분 파이프라인 종료 상태다.

완성형 시스템에서는 failure state가 다음 처리 단계로 연결되어야 한다.

open_boundary
→ open-chart fitting 또는 region coverage expansion

branching_boundary
→ chart decomposition 또는 junction analysis

multiple_loops
→ outer/inner ownership recovery 또는 hole-aware decomposition

derived_seam_required
→ seam generation

insufficient_observed_support
→ neighboring support propagation 또는 region reassignment

fit_failed
→ adaptive control-grid refinement 또는 chart split

validation_failed
→ reparameterization, split, refit

다만 fallback은 무제한으로 반복해서는 안 된다. 각 recovery는 다음을 기록해야 한다.

Recovery reason

Recovery policy

Attempt count

Changed topology

Added or removed evidence

Final unresolved reason

Recovery budget을 초과하거나 evidence가 부족하면 정직하게 unresolved로 남겨야 한다.

10. Real-Scene Evaluation 체계 구축

현재 synthetic fixture는 개별 알고리즘의 positive 및 negative control로 중요하지만, 실제 시스템의 완성도를 증명하기에는 충분하지 않다.

실제 trained Gaussian snapshot에서 단계별 yield를 측정해야 한다.

Input Gaussians
→ Structurally Usable Gaussians
→ Region-Assigned Gaussians
→ Boundary-Recovered Regions
→ Admissible Charts
→ Materialized NURBS Patches
→ Geometrically Validated Patches
→ Render-Relevant Surface Coverage

각 단계에서 최소 다음 지표를 기록해야 한다.

Structural Stage

Intrinsic reliability distribution

Contextual consistency distribution

Reliable, ambiguous, conflict, rejected count

Covariance anisotropy distribution

Region Stage

Region count

Region size distribution

Assigned Gaussian coverage

False merge rate

False split rate

Region purity

Competing-region ambiguity

Boundary Stage

Boundary candidate count

Genuine termination versus sampling-gap distribution

Closed/open/branch/ambiguous component count

Boundary precision and recall

Boundary invariance

Chart and Materialization Stage

Admissible chart count

Materialization attempt count

Materialized patch count

Fit failure count

Validation failure count

Patch count per region

Control-point count

Boundary and interior residual

Foldover, normal-flip, Jacobian failure count

Coverage Stage

Gaussian-count coverage

Opacity-weighted coverage

Rendering-contribution-weighted coverage

Image-space reconstruction coverage

World-space surface-area coverage

특히 실제 scene에서는 Gaussian 수보다 rendering contribution과 visible surface area를 중심으로 coverage를 해석해야 한다.

11. 권장 개발 순서

Visible Surface Construction을 완성 수준으로 끌어올리기 위한 권장 순서는 다음과 같다.

Phase 1 — Real-Scene Core-to-Shell Coverage Expansion

Reliable Core
→ Confidence-Aware Region Propagation
→ Ambiguity Taxonomy
→ Expanded Surface Regions

첫 목표는 85%의 ambiguous Gaussian을 무조건 할당하는 것이 아니다. Rendering과 geometry에 의미 있게 기여하면서 특정 surface region과 일관된 Gaussian을 안전하게 확장하는 것이다.

Phase 2 — Continuous Boundary Support Recovery

Canonical Transported Tangent Frame
→ Continuous Circular Support Gap
→ Observation/Visibility Evidence
→ Genuine Boundary versus Sampling Gap

Hard sector quantization 의존을 줄이고, geometry와 camera evidence를 결합하여 실제 surface termination을 판정한다.

Phase 3 — General Surface Chart Decomposition

Closed/Open/Multi-Loop/Branching Region
→ Topology Analysis
→ Cut and Seam Graph
→ Disk-Like Chart Atlas

Outer/inner loop, open surface, branch, derived seam을 포함하는 일반적인 topology를 chart 단위로 변환한다.

Phase 4 — Multi-Patch NURBS Fitting

Chart Parameterization
→ Adaptive NURBS Fitting
→ Patch Splitting
→ C0/C1 Reconciliation
→ Geometric Validation

Region 하나를 patch 하나로 제한하지 않고, surface complexity에 따라 patch network를 생성한다.

Phase 5 — Real-Scene Acceptance and Production Integration

다양한 실제 scene에서 다음 유형을 검증한다.

Planar architecture

Smooth curved object

Thin structure

Close parallel surface

Crease

Hole

Cluttered geometry

Sparse observation

Occlusion-heavy region

이 단계에서 충분한 real-scene coverage와 geometric validity가 확인된 이후 production dispatcher, trainer, renderer, checkpoint와 연결해야 한다.

12. 완성 판정 기준

다음 조건을 만족해야 Visible Surface Construction을 완성 수준이라고 평가할 수 있다.

주요 rendering-contributing Gaussian과 visible surface area의 대부분이 surface region 또는 명시적인 unresolved 상태로 설명된다.

단일 closed loop뿐 아니라 open boundary, multiple loop, hole, branch, derived seam topology를 chart atlas로 처리할 수 있다.

실제 trained scene에서 의미 있는 수와 면적의 visible NURBS patch가 생성된다.

Close-parallel false merge와 phase-alias shortcut이 실제 scene에서도 안정적으로 억제된다.

Rotation, translation, uniform scale, input-order shuffle, covariance sign-equivalent representation에 대해 결과가 불변이다.

Boundary fidelity, interior fidelity, normal consistency, foldover, Jacobian, self-intersection validation을 통과한다.

하나의 patch로 표현할 수 없는 region은 adaptive chart split과 multi-patch fitting으로 처리된다.

실패 component는 무조건 폐기되지 않고 원인에 맞는 제한된 recovery path 또는 명시적인 unresolved state를 갖는다.

Region, boundary, chart, patch, NURBS surface 사이의 provenance를 end-to-end로 추적할 수 있다.

Synthetic fixture 성공뿐 아니라 real-scene 단계별 yield와 rendering-relevant coverage가 검증된다.

13. 핵심 결론

현재 Visible Surface Construction의 가장 큰 병목은 NURBS fitter가 아니라 reliable_core_only region coverage이다.

현재처럼 실제 4,000-Gaussian crop에서 3,409개 Gaussian이 ambiguous로 남고, region의 최대 크기가 29개에 불과하면 boundary recovery와 NURBS fitting을 아무리 정교하게 만들어도 실제 장면에서 충분한 surface를 생성하기 어렵다.

따라서 다음 구현 목표는 다음과 같이 정의하는 것이 가장 적절하다.

Covariance-guided reliable core를 보존하면서 observation evidence, rendering contribution, manifold consistency를 이용해 ambiguous Gaussian을 안전하게 surface region으로 확장한다.

그 이후에 continuous boundary recovery, general chart decomposition, multi-patch NURBS fitting을 순차적으로 추가해야 한다.

최종적으로 필요한 구조는 다음과 같다.

Trained Visible Gaussians
→ Reliable Core Detection
→ Confidence-Aware Core-to-Shell Region Expansion
→ Continuous Geometry/Observation Boundary Recovery
→ General Surface Topology Analysis
→ Disk-Like Chart Atlas
→ Adaptive Multi-Patch NURBS Fitting
→ Patch Continuity Reconciliation
→ Geometric and Rendering Validation
→ Production Integration

즉 현재 구현은 Visible Surface Construction의 핵심 경로를 증명한 prototype이며, 완성형 시스템으로 발전시키기 위해서는 coverage 확장과 일반적인 topology 처리가 다음 핵심 연구 과제가 된다.