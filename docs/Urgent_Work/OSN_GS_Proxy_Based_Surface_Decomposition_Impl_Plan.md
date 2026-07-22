# OSN-GS Phase 1 Methodology Replacement Plan
## Adaptive Local Surface Cells + Variational Proxy-Based Surface Decomposition

**진행 상태 (2026-07-22)**
Stage 0 기준선, Stage 1 diagnostics-only quadratic proxy, Stage 2 diagnostics-only Spatial Candidate Graph 검증을 완료했다. Production component membership과 기본값은 변경하지 않았다. Stage 2는 `curved_annulus` 누락 smooth pair 및 기존 face-smooth pair recall 1.0을 유지했으나 coarse leaf scene에서 dense/complete candidate graph가 발생함을 확인했다. 근거는 `docs/worklogs/62_proxy_decomposition_stage2_candidate_graph.md`, `artifacts/proxy_decomposition_stage2.json`, `artifacts/proxy_decomposition_stage2_unified_benchmark.json/report.json`에 있다. **Stage 3 Merge-Only Agglomeration은 사용자 승인 대기 상태이며 아직 구현하지 않았다.**

**문서 목적**
이 문서는 OSN-GS의 현재 Phase 1 Surface-Cell Component Builder를, 축 정렬 voxel face adjacency에 의존하는 방식에서 **local geometric proxy 기반 surface decomposition** 방식으로 교체하기 위한 개발 지침이다.

본 문서는 사용자, ChatGPT, Codex가 동일한 설계 기준과 검증 절차를 공유하도록 작성되었다.
Codex는 이 문서를 구현 지시의 우선 기준으로 사용한다.

---

# 1. 배경

현재 OSN-GS의 Phase 1은 adaptive voxel hierarchy로 Gaussian들을 local leaf로 나눈 뒤, leaf AABB 사이의 face adjacency와 local plane compatibility를 이용해 surface component를 구성한다.

기존 구조는 평면 및 near-planar synthetic scene에서는 작동했지만, `curved_annulus`에서 다음 문제가 확인되었다.

- 실제 표면은 하나로 연속되어 있음
- 표면이 axis-aligned voxel의 z-band 경계를 가로지름
- 연속된 두 leaf가 voxel 격자에서는 face가 아니라 corner 방향으로만 접촉
- 두 leaf 사이의 face-adjacency candidate edge 자체가 생성되지 않음
- merge threshold를 완화해도 후보 edge가 없으므로 component split이 유지됨
- 결과적으로 하나의 curved annulus가 `disk_like` 및 `complex` component로 과분할됨
- downstream annulus O-grid 및 coupled fitting 경로에 진입하지 못함
- 두 개의 `trimmed_rect_fallback` surface가 독립 fitting되어 실제 surface에 crack이 발생함

진단 결과 모든 생성된 adjacency edge는 merge 조건을 통과했다.

```text
edge_reason_counts = {'merged': 14}
```

따라서 문제는 merge threshold가 아니라 **axis-aligned face adjacency를 surface connectivity의 대리 기준으로 사용한 구조적 한계**다.

---

# 2. 핵심 문제 정의

현재 Phase 1은 다음 두 역할을 동시에 수행한다.

1. Gaussian을 계산 가능한 local cell로 공간 분할
2. 실제 surface component의 연결성 결정

하지만 이 두 역할은 분리되어야 한다.

```text
Spatial decomposition != Surface decomposition
```

Voxel hierarchy는 다음 목적에는 유효하다.

- local Gaussian support 구성
- local frame 및 plane 추정
- density/support 통계 계산
- 공간 검색 가속
- 계산량 분산

그러나 voxel face adjacency가 곧 실제 surface connectivity라는 가정은 curved surface에서 성립하지 않는다.

---

# 3. 새로운 Phase 1의 목표

새 Phase 1은 다음 원칙을 따른다.

> Adaptive voxel hierarchy는 atomic local surface cell을 생성하는 데만 사용하고, 실제 surface component는 geometric proxy distortion을 기준으로 구성한다.

새 canonical flow:

```text
Observed Gaussian primitives
→ Adaptive local surface cells
→ Spatial candidate region graph
→ Variational proxy-based region agglomeration
→ Surface components
→ Topology and boundary extraction
→ NURBS chart construction
→ Coupled NURBS fitting
```

---

# 4. 방법론적 기반

새 구조는 다음 연구 흐름의 공통 원리를 참고한다.

- Variational Shape Approximation
- Point-set Variational Shape Approximation
- Local surface reconstruction
- Partition-of-Unity reconstruction
- Local chart/atlas decomposition
- Quadric or polynomial proxy segmentation

핵심 원리는 다음과 같다.

> 어떤 voxel과 face로 닿았는지가 아니라, 여러 local surface cells가 하나의 smooth geometric proxy로 함께 설명될 수 있는지를 기준으로 region을 구성한다.

Voxel은 final topology boundary가 아니라 local approximation domain이다.

---

# 5. 설계 원칙

## 5.1 단일 canonical methodology

다음과 같은 예외 분기를 추가하지 않는다.

```text
if face-adjacent:
    rule A
elif diagonal:
    rule B
elif curved:
    rule C
```

모든 candidate region pair에 동일한 merge criterion을 적용한다.

## 5.2 Voxel hierarchy 유지

현재 adaptive voxel bootstrap과 leaf Gaussian membership은 유지한다.

교체 대상은 voxel hierarchy 자체가 아니라 다음 부분이다.

- face-contact 기반 adjacency generation
- face adjacency 기반 component union
- local plane threshold 중심 merge decision

## 5.3 Proxy와 final representation 분리

```text
Quadratic proxy
= decomposition 및 merge 판단용

NURBS
= 최종 surface representation
```

Quadratic proxy가 최종 surface를 제한해서는 안 된다.

## 5.4 Production 교체 전 diagnostics-first

초기 구현은 반드시 opt-in 또는 benchmark-only여야 한다.

기존 production 경로는 승인 전까지 유지한다.

## 5.5 Scene-specific, GT-dependent logic 금지

금지 사항:

- `curved_annulus` 전용 branch
- scene name 기반 조건
- GT component 수를 runtime에 사용
- GT chamfer를 merge objective에 사용
- 특정 seed 전용 correction
- face adjacency 실패 시에만 실행되는 fallback

---

# 6. 목표 아키텍처

## Phase 1-A — Atomic Surface Cell Extraction

현재 adaptive voxel hierarchy를 이용해 local surface cell을 만든다.

각 cell은 최소한 다음 정보를 가진다.

```text
cell_id
gaussian_indices
centroid
aabb
local_frame
plane_normal
plane_residual
sampling_scale
support_mass
state
parent/level provenance
```

추가 diagnostics:

```text
point_count
median_nn_spacing
covariance_eigenvalues
planarity_score
local_curvature_proxy
```

Atomic cell은 최종 component가 아니다.

---

## Phase 1-B — Spatial Candidate Graph

모든 cell pair를 비교하지 않고, spatial search로 가까운 pair만 candidate로 만든다.

허용 가능한 구현:

- centroid KD-tree radius search
- AABB distance search
- spatial hash
- 기존 voxel index 기반 neighborhood search

Candidate generation은 계산량 절감 장치일 뿐, surface connectivity 판정 자체가 아니다.

### Candidate 조건의 기본 원칙

Candidate radius는 local scale에 맞춰 정규화한다.

예:

```text
candidate_distance <= candidate_radius_factor *
                     max(cell_A_scale, cell_B_scale)
```

단, 해당 값은 benchmark에서 sweep하고 문서화해야 한다.

### Candidate graph diagnostics

각 pair에 대해 다음을 기록한다.

```text
pair_id
cell_a
cell_b
centroid_distance
aabb_distance
support_gap
scale_normalized_gap
candidate_source
```

`candidate_source`는 분석용 provenance이며 merge rule 분기에 사용하지 않는다.

---

## Phase 1-C — Geometric Proxy Model

### 최초 proxy

초기 구현은 local quadratic height field를 사용한다.

Region point set을 local frame으로 변환해:

```text
z = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
```

를 regularized least squares로 fitting한다.

### Local frame

Region covariance 또는 weighted PCA로 tangent frame을 계산한다.

주의:

- normal sign ambiguity 처리
- degenerate covariance 검사
- multi-layer region은 quadratic height field로 표현하기 어려우므로 invalid proxy로 표시
- fitting 전에 scale normalization 수행

### Proxy output

```text
frame
coefficients
normalized_rms_residual
normalized_max_residual
condition_number
point_count
support_extent
valid
```

### Plane proxy 비교

Quadratic이 불필요한 경우를 분석하기 위해 plane residual도 함께 유지한다.

초기 prototype에서는 quadratic proxy만 merge decision에 사용해도 되지만, plane/quadratic 비교 수치는 반드시 기록한다.

---

## Phase 1-D — Agglomerative Region Merge

초기 상태:

```text
one atomic cell = one region
```

각 candidate region pair에 대해 merged proxy를 fitting한다.

Merge cost의 기본 정의:

```text
delta_error =
    normalized_proxy_error(region_i ∪ region_j)
    - weighted_error(region_i)
    - weighted_error(region_j)
```

단순 total residual이 아니라 반드시 다음을 고려한다.

- point-count normalization
- component scale normalization
- local spacing normalization
- pre-merge 대비 error 증가량
- model validity
- layer consistency
- support gap

### 권장 merge score 구성

첫 pass에서는 단일 weighted score보다 항목별 diagnostics를 우선한다.

```text
merged_proxy_error
error_increase
scale_normalized_gap
normal_variation
layer_separation_score
proxy_condition
```

최종 merge decision은 문헌 기반 criterion을 검토한 뒤 확정한다.

### Merge process

```text
1. 초기 region 구성
2. candidate pair별 merge cost 계산
3. 가장 낮은 cost pair 선택
4. merge가 admissible하면 병합
5. 새 region proxy 재계산
6. 영향을 받는 candidate edge 갱신
7. 더 이상 admissible merge가 없을 때 종료
```

### 구현 요구

- deterministic ordering
- deterministic tie-breaking
- stale priority-queue entry 무효화
- region provenance 유지
- merge history 저장
- 동일 입력에서 동일 결과 보장

---

# 7. Merge Admissibility

최종 production criterion은 feasibility pass 후 확정한다.

초기 prototype에서는 아래 조건을 각각 분리해 측정한다.

## 7.1 Proxy fitness

병합된 point set이 하나의 quadratic surface로 충분히 설명되는가.

## 7.2 Support proximity

두 region의 실제 Gaussian support 사이 간격이 local sampling scale 대비 작은가.

Centroid distance만 사용하지 않는다.

권장:

- symmetric nearest-neighbor quantile
- closest boundary-support quantile
- local spacing-normalized gap

## 7.3 Smooth curvature compatibility

Normal angle 자체가 아니라 distance 대비 normal variation을 측정한다.

```text
normal_change_rate = normal_angle / support_distance
```

단, 이 값 하나만으로 merge하지 않는다.

## 7.4 Layer consistency

가까운 평행 surface의 잘못된 병합을 막아야 한다.

연결 방향이 tangent 방향인지 normal 방향인지 기록한다.

```text
normal_offset_A = abs((centroid_B - centroid_A) dot normal_A)
normal_offset_B = abs((centroid_B - centroid_A) dot normal_B)
```

두 sheet가 normal 방향으로 분리되어 있으면 merge하지 않아야 한다.

## 7.5 Proxy validity

다음은 merge reject 사유가 될 수 있다.

- ill-conditioned fitting
- multi-valued height field
- severe multi-layer evidence
- insufficient support
- extreme extrapolation
- excessive residual concentration

모든 reject reason을 worklog에 집계한다.

---

# 8. 단계별 구현 계획

## Stage 0 — Baseline Freeze and Interface Audit

상태: **완료 (2026-07-22, Worklog 60).** Production 변경 없음.

### 목표

현재 Phase 1/2 interface와 baseline 결과를 고정한다.

### 작업

- 현재 production commit 기록
- 기존 benchmark scene별 component 수/topology 기록
- `curved_annulus` leaf/AABB/component provenance 저장
- Phase 1 output dataclass/interface 확인
- Phase 2가 기대하는 필드 목록 정리
- 기존 face adjacency 결과를 regression baseline으로 저장

### 산출물

```text
docs/worklogs/<new_id>_proxy_decomposition_stage0.md
artifacts/proxy_decomposition_baseline.json
```

### 종료 조건

- 기존 production 경로가 byte-identical하게 재현됨
- 새 구현의 삽입 지점이 문서화됨

---

## Stage 1 — Diagnostics-Only Quadratic Proxy

상태: **완료 (2026-07-22, Worklog 61).** 10개 신규 단위 테스트와 당시 전체 171개 테스트 통과(1 skip). Stage 2는 후속 승인 후 Worklog 62에서 완료했다.

### 목표

Region proxy fitting이 대상 scene을 구분할 수 있는지 확인한다.

### 구현

신규 파일 권장:

```text
osn_gs/surface/torch_surface_proxy.py
```

함수 예시:

```python
fit_quadratic_surface_proxy(points, weights=None, regularization=...)
evaluate_quadratic_proxy(proxy, points)
merge_proxy_diagnostics(region_a, region_b, points)
```

### 검증 대상

필수 synthetic cases:

- planar sheet
- mild curved sheet
- curved annulus
- sharp crease
- parallel double layer
- disconnected close patches
- density gradient
- sparse boundary
- high curvature sheet

### 보고

각 pair/region별:

- plane error
- quadratic error
- merged error increase
- condition number
- layer score
- normalized support gap

### 종료 조건

최소한 다음 경향이 확인되어야 한다.

- curved-annulus 연결 pair는 낮은 merged quadratic error
- crease pair는 더 높은 merged error 또는 residual concentration
- parallel layers는 layer consistency에서 분리 가능
- disconnected close patches는 support gap에서 분리 가능

Production 변경 금지.

---

## Stage 2 — Spatial Candidate Graph

상태: **완료 (2026-07-22, Worklog 62).** Diagnostics-only candidate generation과 회전·point count·adaptive leaf resolution·density gradient·parallel-layer distance sweep을 완료했다. Production component membership은 변경하지 않았으며 Stage 3는 사용자 승인 대기다.

### 목표

Face adjacency와 무관하게 relevant region pair를 안정적으로 후보로 생성한다.

### 구현

신규 파일 권장:

```text
osn_gs/surface/torch_surface_candidate_graph.py
```

함수 예시:

```python
build_surface_cell_candidate_graph(
    cells,
    points,
    radius_factor,
    max_neighbors,
)
```

### 요구

- adaptive cell size 대응
- deterministic neighbor ordering
- duplicate pair 제거
- face/edge/corner relation은 diagnostics로만 기록
- curved-annulus의 끊어진 두 leaf pair가 candidate에 포함되어야 함
- parallel layers에 과도한 all-to-all candidate가 생성되지 않아야 함

### 종료 조건

- curved-annulus missing edge가 후보에 들어옴
- candidate recall이 충분함
- candidate pair 수가 통제 가능함
- 기존 face edges도 대부분 포함됨

---

## Stage 3 — Merge-Only Agglomeration Prototype

### 목표

Proxy-based merge만으로 curved-annulus 과분할을 해결할 수 있는지 검증한다.

### 구현

신규 파일 권장:

```text
osn_gs/surface/torch_surface_decomposition.py
```

핵심 구조:

```python
build_proxy_surface_components(
    cells,
    points,
    candidate_graph,
    config,
)
```

Config 예시:

```text
candidate_radius_factor
max_candidate_neighbors
proxy_regularization
max_normalized_proxy_error
max_error_increase
max_normalized_support_gap
max_layer_separation
min_region_support
```

모든 설정값은 공개적으로 기록한다.

### 주의

이 Stage에서는 split refinement를 구현하지 않는다.

### 필수 결과

- `curved_annulus`가 하나의 component로 병합되는지
- planar 4개 annulus scene이 기존 topology를 유지하는지
- crease가 과병합되지 않는지
- parallel layer가 합쳐지지 않는지
- disconnected close patch가 합쳐지지 않는지

### 종료 조건

위 다섯 조건 중 하나라도 구조적으로 실패하면 production 통합로 넘어가지 않는다.

---

## Stage 4 — Downstream Interface Integration

### 목표

새 component 결과가 기존 Phase 2~5 pipeline에 정상 전달되는지 확인한다.

### 작업

- component membership 변환
- component frame 계산
- Phase 2 boundary input 연결
- topology classifier 연결
- annulus routing 확인
- coupled boundary fitting 연결
- legacy face-adjacency path와 A/B 가능하도록 opt-in config 유지

### 중요

`curved_annulus`가 한 component가 되더라도 Phase 2의 planar density-mask boundary extraction이 실패할 수 있다.

따라서 아래를 별도 판정한다.

```text
Phase 1 component recovery success
Phase 2 topology recovery success
Annulus routing success
NURBS fit success
```

하나의 aggregate success로 섞지 않는다.

### 종료 조건

- component가 올바르게 구성됨
- Phase 2가 curved component에서 의미 있는 boundary/topology를 추출함
- annulus route 또는 적절한 chart route로 진입함
- crack이 제거되거나 명확히 감소함
- 기존 planar scenes regression 없음

---

## Stage 5 — Performance and Determinism

### 목표

Production 적용 가능한 계산 복잡도인지 확인한다.

### 측정

- cell 수
- candidate edge 수
- proxy fitting 횟수
- merge 횟수
- total Phase 1 runtime
- peak memory
- region size distribution
- deterministic repeatability

### 최적화 후보

승인 전에는 구현하지 말고 병목 확인 후 적용한다.

- sufficient statistics caching
- incremental normal-equation update
- stale candidate lazy invalidation
- bounded neighbor count
- batched proxy fitting
- CPU/GPU execution 비교

---

## Stage 6 — Production Adoption Gate

### 필수 benchmark

최소:

- 기존 4 planar annulus scenes × 5 seeds
- curved_annulus × 5 seeds
- mild_curved_sheet × 5 seeds
- crease scenes
- parallel-layer scenes
- disconnected-close scenes
- density/sparsity variations

### 필수 비교

```text
current face-adjacency Phase 1
vs.
proxy-based Phase 1
```

### 지표

#### Component correctness

- component count
- over-segmentation
- under-segmentation
- pairwise co-membership precision/recall
- topology classification

#### Geometry

- chamfer_rms
- false_fill
- coverage
- boundary conformance

#### Chart validity

- orientation flips
- outer flips
- condition p95/p99/max
- min normalized singular value
- near-degenerate count
- seam gap

#### Performance

- runtime
- memory
- candidate edge count
- merge count

### 채택 조건

- curved-annulus over-segmentation 해소
- crease/parallel/disconnected scene의 under-segmentation 없음
- 기존 planar scene의 material regression 없음
- downstream NURBS fitting 안정성 유지
- runtime 증가가 허용 범위 내
- 동일 입력에서 deterministic

Production default 전환은 사용자 승인 후에만 수행한다.

---

# 9. Split Refinement 정책

초기 구현에서는 merge-only로 제한한다.

Split은 다음 상황에서만 별도 Stage로 승인한다.

- 잘못 병합된 region을 merge criterion만으로 방지할 수 없음
- initial voxel partition에 따라 결과가 지나치게 민감함
- large curved component 내부에서 proxy residual이 국소 집중됨
- broad real-data test에서 under-segmentation이 반복됨

Split 후보:

- residual-driven seed insertion
- VSA-style reassignment
- binary region split
- spectral cut on region graph

Split 구현은 현재 scope 밖이다.

---

# 10. 테스트 계획

## Unit Tests

### Quadratic proxy

- exact plane
- exact paraboloid
- noisy curved sheet
- degenerate line-like points
- two parallel layers
- deterministic fit

### Candidate graph

- face-neighbor pair
- corner-neighbor pair
- curved z-band crossing pair
- distant pair exclusion
- adaptive-size leaves
- duplicate pair removal

### Merge engine

- deterministic tie
- stale queue entry
- cyclic candidate updates
- provenance preservation
- one-region termination
- no-valid-merge termination

## Integration Tests

- `curved_annulus` becomes one component
- planar annulus remains one annulus component
- crease remains split
- parallel sheets remain split
- disconnected close patches remain split
- downstream topology field remains valid
- existing test suite stays green

## Regression Tests

현재 전체 suite를 모든 Stage 종료 시 실행한다.

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

# 11. Worklog 규칙

각 Stage마다 한국어 worklog를 작성한다.

필수 구조:

```text
목표
변경 파일
구현 내용
사용한 설정값
진단 결과
정량 결과
성공/실패
새로 발견된 문제
채택/기각 판단
production 변경 여부
다음 승인 요청
```

모든 threshold와 weight를 기록한다.

숨겨진 default나 benchmark-only override를 만들지 않는다.

---

# 12. 예상 파일 구조

신규 권장:

```text
osn_gs/surface/
├─ torch_surface_proxy.py
├─ torch_surface_candidate_graph.py
├─ torch_surface_decomposition.py
```

수정 가능성이 높은 파일:

```text
osn_gs/surface/torch_voxel_hierarchy.py
osn_gs/surface/torch_component_boundary.py
nurbs_constructor_benchmark/boundary_first.py
nurbs_constructor_benchmark/runner.py
tests/test_surface_proxy.py
tests/test_surface_candidate_graph.py
tests/test_surface_decomposition.py
```

기존 파일의 public behavior는 production 승인 전 변경하지 않는다.

---

# 13. 시간 및 난이도 추정

## Feasibility prototype

범위:

- quadratic proxy
- candidate graph
- merge-only prototype
- synthetic diagnostics

예상:

```text
약 1주
```

## Production-ready first version

추가:

- downstream integration
- deterministic merge
- benchmark validation
- performance 측정
- regression hardening

예상:

```text
약 2~4주
```

## Paper-grade validation

추가:

- split/merge refinement
- broad synthetic scenes
- real 3DGS data
- baseline comparison
- sensitivity analysis
- scalability study

예상:

```text
약 4~8주 이상
```

---

# 14. 중단 기준

아래 중 하나가 발생하면 무리하게 production 통합하지 않고 사용자에게 보고한다.

- quadratic proxy가 curved surface와 crease를 안정적으로 구분하지 못함
- parallel layers가 반복적으로 merge됨
- candidate graph가 지나치게 dense해짐
- component는 복원되지만 Phase 2 boundary/topology가 구조적으로 실패함
- threshold가 특정 scene에만 맞춰져야 함
- merge order에 따라 결과가 크게 달라짐
- runtime 또는 memory가 비현실적으로 증가함

---

# 15. Codex 실행 지침

1. 먼저 repository와 현재 Phase 1/2 interface를 읽는다.
2. Stage 0 baseline을 먼저 기록한다.
3. production default를 변경하지 않는다.
4. 한 Stage가 끝날 때마다 worklog를 작성하고 멈춘다.
5. 새 문제가 발견됐다고 즉시 scope를 확장하지 않는다.
6. scene-specific branch, fallback, retry selector를 만들지 않는다.
7. 모든 threshold, score, normalization을 코드와 worklog에 명시한다.
8. `curved_annulus`만 통과시키는 것이 아니라 crease/parallel/disconnected negative controls를 함께 검증한다.
9. GT는 benchmark 평가에만 사용하고 runtime merge decision에는 사용하지 않는다.
10. production adoption은 사용자의 별도 승인 후에만 수행한다.

---

# 16. 최초 실행 요청

Codex는 우선 **Stage 0과 Stage 1만 수행**한다.

## Stage 0

- 현재 component builder와 Phase 2 interface audit
- baseline scene/component/topology dump
- `curved_annulus` leaf provenance 및 missing candidate relation 기록

## Stage 1

- diagnostics-only quadratic proxy 구현
- curved/smooth/crease/parallel/disconnected pair에 대한 proxy fitting 비교
- production path 변경 없음
- 전체 unit test 유지

Stage 1 종료 후 다음 내용을 보고하고 멈춘다.

```text
1. Quadratic proxy가 smooth curved pair와 crease/parallel pair를 구분하는가
2. 어떤 normalized metric이 가장 분리력이 있는가
3. 어떤 metric은 불안정하거나 scene-specific한가
4. Stage 2 candidate graph 구현으로 진행할 근거가 충분한가
5. 예상되는 downstream Phase 2 위험은 무엇인가
```

Stage 2 완료 결과를 보고한 뒤 멈춘다. 사용자의 승인 없이 Stage 3로 진행하지 않는다.
