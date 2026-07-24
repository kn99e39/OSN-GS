# OSN-GS Phase D — Parametric Continuation Domain: Design

상태: **DESIGN REVISION 3. 큰 방향과 revision 2의 핵심 수학 계약(world-space outward 공식, sampled-grid source of truth, first-order canonical geometry, 상태 이름 등)은 승인됐다. 이 문서 상태로는 구현 착수가 아직 승인되지 않았다.** Gate C는 최종 승인됐다(`docs/worklogs/79_observation_evidence_phase_c_gate_c_round2.md`). 이 문서는 `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md` §6(Phase D)의 상세 설계다. Production pipeline/trainer 연결, Phase C evidence와의 실제 결합, Phase E candidate 생성, Phase F NURBS fitting, diagnostics helper 실제 리팩터는 이 문서의 범위 밖이며 별도 승인 전에는 시작하지 않는다.

## 0. 이 문서가 근거하는 audit

아래 API 목록은 `osn_gs/surface/torch_nurbs.py`, `osn_gs/surface/torch_patch_boundary.py`, `osn_gs/surface/torch_boundary_reconciliation.py`, `osn_gs/surface/torch_annulus_chart.py`를 직접 읽어 확인했다. 존재하지 않는 API를 가정한 곳은 모두 "§9 Prerequisite"로 명시했다.

- `TorchNURBSSurface`(`torch_nurbs.py:207-238`): `control_grid (U,V,3)`, `weights (U,V)`, `degree_u/degree_v`(control point 수보다 크면 evaluation 시점에 자동으로 낮아짐, `_effective_degree`), `uv_support_mask`, knot vector는 저장되지 않고 `(n_u,n_v,degree_u,degree_v,dtype,device)` 키로 lazily cache된다.
- `evaluate_with_derivatives(uv) -> (S, S_u, S_v)`(`:342`), `evaluate_with_second_derivatives(uv) -> (S, S_u, S_v, S_uu, S_uv, S_vv)`(`:365`) — 둘 다 rational quotient rule의 analytic closed form이다. Degree가 부족한 축은 정확히 0인 텐서를 반환한다(생략이 아니라 명시적 0).
- `normals(uv) -> normalize(S_u x S_v)`(`:425`). Mean/Gaussian curvature를 계산하는 메서드는 없다 — 필요하면 `S_uu/S_uv/S_vv`에서 직접 유도해야 한다.
- Knot vector는 **clamped uniform**이며 domain이 `[0,1]`에 하드코딩돼 있다(`_bspline_basis_pair`가 `u`를 `[knots[0], knots[-1]-eps]`로 clamp, `torch_nurbs.py:64`; `_basis_tables`가 입력 `uv`를 먼저 `[0,1]`로 clamp, `:307-308`). **`[0,1]`을 넘어서는 domain 확장(reparameterize) 헬퍼는 어디에도 없다.** Span-detection이 마지막 반복 knot에서의 폐구간을 전제하므로 knot-insertion 없이 단순히 범위를 늘릴 수 없다. (이번 개정에서도 이 결론은 변하지 않는다 — §4.6.)
- `boundary_control_indices(resolution_u, resolution_v, edge, start, stop)`(`torch_nurbs.py:843-874`) — 사각형 patch 한 edge(`u0/u1/v0/v1`)의 row-major control-grid 인덱스.
- `SharedBoundaryConstraint`/`fit_coupled_patch_graph_lsq`(`torch_nurbs.py:182-204, 877-899+`) — Phase B의 generic joint solver. Phase D는 이것을 호출하지 않는다.
- `PatchBoundarySegment`(`torch_patch_boundary.py:30-79`) — `boundary_id, patch_id, source_kind, uv, world, inner_uv, inner_world, tangent_world, inward_tangent_world, normal_world, closed, orientation, interior_side="left", state, control_edge, adjacent_patch_id, adjacent_boundary_id, confidence: dict, provenance: dict`. State는 `unclassified|reconciled_internal|unsupported|extension_candidate`(`BOUNDARY_STATES`). **`patch_id`는 정수이고, 실제 `TorchNURBSSurface` 객체 참조는 들어있지 않다** — Phase D는 caller가 `patch_id -> TorchNURBSSurface` 매핑을 별도로 제공해야 한다(§2, §9). `PatchBoundarySegment`는 **연속 boundary curve 함수가 아니라 ordered `uv`/`world` samples만 갖는다** — 이번 개정의 §3이 sampled grid를 canonical source of truth로 삼는 이유다.
- `surface_jacobian_validity(surface, resolution, relative_minimum)`(`torch_boundary_reconciliation.py:193-218`) — `‖S_u x S_v‖`(면적) 기반의 단순 proxy. 진짜 singular value는 아니다.
- `_jacobian_diagnostics(surface, resolution, eps, characteristic_length, collect_samples)`(`torch_annulus_chart.py:124-224`) — `J^T J`의 closed-form eigenvalue로 진짜 `sigma_min/sigma_max`를 계산하고, per-slice self-consistent `reference_normal` 기반 `orientation_flip_count`를 함께 낸다. **입력이 `TorchNURBSSurface` 객체이고 내부에서 `surface.evaluate_with_derivatives`를 직접 호출하며, singular-value 계산과 orientation-consistency 계산이 하나의 함수에 섞여 있다** — Phase D는 이 함수를 그대로 호출할 수 없고(continuation domain은 `TorchNURBSSurface`가 아님), 리팩터 시에도 두 책임을 하나로 합치지 않는다(§9 Prerequisite, 사용자 교정에 따라 두 개의 별도 helper로 분리).
- Legacy/미사용: `predict_torch_occlusion_curves`, `build_torch_surface`, `sample_torch_occluded_surface`(모두 자체 docstring에 "Stage 2용 legacy helper"로 명시, production/test 호출부 없음, 단일 방향 global occlusion 추정이라는 pre-reset 전제를 그대로 가짐). **Phase D는 이 셋을 재사용하지 않는다.** `pca_parameterize_points`/`fit_torch_base_curves`는 실제로 살아있지만 "point cloud에서 UV를 재추정"하는 도구이므로 Phase D의 "Gaussian point cloud에서 local geometry를 재추정하지 않는다" 원칙상 사용하지 않는다.
- 과거 확장 설계 문서(`OSN_GS_Phase5_Boundary_Aligned_Extension_Plan.md`, `OSN_GS_Final_Boundary_First_NURBS_Direction.md`)는 구체적 수식 없이 구조적 아이디어만 갖고 있었다: `S_ext(s,t)`/`C_ext(s,t)` 분리 표기, `s`=boundary tangential parameter/`t`=outward parameter라는 좌표계 관례, boundary별 local frame(tangent/normal/outward-normal/confidence). 이 구조적 아이디어는 premise-independent이므로 재사용한다.

## 1. 승인된 Phase D 방향 (사용자 확정 사항)

Phase D는 최종 parametric surface나 NURBS chart를 만들지 않는다. 역할은 다음으로 제한한다.

```text
Visible NURBS boundary
→ boundary-local world-space sampled continuation strip
→ geometric validity 및 uncertainty metadata
```

확정 사항:

1. Boundary마다 독립된 sampled continuation strip 하나를 생성한다.
2. Boundary pairing, strip overlap, 공동 bounded region 구성은 Phase E 소유다.
3. Phase C `ObservationEvidence`는 Phase D에서 호출하지 않는다. 호환 가능한 world sample interface만 제공한다.
4. 최종 constrained occluded NURBS chart는 Phase F 소유다.
5. Control-net/knot-domain extrapolation은 Phase D에서 하지 않는다.
6. Gaussian point cloud에서 local geometry를 재추정하거나 local proxy를 다시 fit하지 않는다.
7. Canonical continuation geometry는 first-order만 사용한다.
8. Second derivative는 actual strip position에 적용하지 않고 curvature-growth 및 uncertainty diagnostic에만 사용한다.
9. Production pipeline/trainer에는 연결하지 않는다.
10. Facing/normal similarity는 hard gate로 쓰지 않는다 — 어떤 validity 조건도 이 값을 reject 기준으로 쓰지 않는다.
11. One-sided extrapolation은 초기 범위에서 제외한다 — Phase D 산출물은 "아직 검증되지 않은 continuation hypothesis"일 뿐 확정된 occluded geometry가 아니다(§5의 상태 의미론 참고).

## 2. 입력 계약

```python
class ContinuationDomainBuildError(RuntimeError):
    """Grid/AABB를 구성할 수 없는 필수 계산 실패. 입력 계약 위반(ValueError)과
    사후 품질 문제(state=degenerate/rejected)의 중간 범주다 -- §2.3, §5."""


def build_continuation_domain(
    surface: TorchNURBSSurface,
    boundary: PatchBoundarySegment,
    *,
    expected_patch_id: int | None = None,
    extent_multiplier: float = 1.0,
    local_surface_scale: float | None = None,
    arclength_epsilon: float = 1e-6,
    t_count: int = 5,
) -> ContinuationDomain: ...
```

- `surface`: source patch의 실제 NURBS 객체. `PatchBoundarySegment.patch_id`로는 얻을 수 없으므로 caller가 별도 매핑(`dict[int, TorchNURBSSurface]`)에서 조회해 전달한다 — 이는 새 데이터 구조가 아니라 이미 존재하는 두 산출물(patch surfaces list, boundary list)을 나란히 갖고 있는 caller의 책임이다(§9).
- `expected_patch_id`: 선택적 self-consistency 체크. 제공되면 `expected_patch_id == boundary.patch_id`를 검증한다. **`surface` 자체는 `patch_id` 필드를 갖지 않으므로, 잘못된 surface가 잘못된 boundary와 짝지어 전달되는 caller 실수를 이 함수 하나만으로 완전히 막을 수는 없다** — 이 한계를 §9에 명시한다.
- `extent_multiplier`, `local_surface_scale`: §4.5.
- `arclength_epsilon`: §2.2 — 인접 샘플 간 world distance가 이 값 이하이면 zero-length segment로 간주해 `ValueError`. 단순 `eps` clamp로 잘못된 tangent를 만들지 않기 위한 명시적 reject 임계값이다(scale-aware 값을 caller가 넘길 수도 있고, 기본값은 절대 tolerance).
- `t_count`: outward 샘플 개수.

**사용하는 `boundary` 필드**(전부 이미 존재, 신규 필드 없음): `uv`, `world`(canonical `s`-domain의 물리적 근거), `inner_uv`/`inner_world`(outward 부호 선택에만 사용, §4.2), `closed`(순환 여부), `state`(아래), `confidence`(uncertainty로 전달). `tangent_world`/`inward_tangent_world`/`normal_world`는 재사용하지 않고 교차검증(cross-check)에만 쓴다 — Phase D는 boundary tangent를 world-arclength 기준으로 독립적으로 다시 계산한다(§4.1).

### 2.1 Closed boundary의 중복 종료점 제거(정규화, 검증 아님)

`torch_patch_boundary.py`의 `_canonicalize_closed_loop`는 닫힌 loop의 `uv`/`world`를 **마지막 샘플이 첫 샘플과 같은 값으로 중복**되도록 만든다(`ordered + [ordered[0]]`). Phase D의 canonical representation은 이 중복을 저장하지 않는 ordered unique samples다 — `build_continuation_domain`은 `boundary.closed`이고 `boundary.world[-1]`이 `boundary.world[0]`과 (부동소수 오차 이내로) 같으면 그 마지막 샘플을 제거한 뒤 자신의 `s_count`/`s_world`/`world` 등을 구성한다. 이 stripping은 **정규화 단계**이며 §2.2의 validity 검사 대상이 아니다 — stripping 이후의 unique-sample 집합에 대해서만 §2.2를 적용한다(closing segment는 "마지막 unique sample → 첫 sample"로 별도 계산, §4.1).

### 2.2 입력 계약 위반 (eager `ValueError`, 단일 정책)

다음은 전부 **`ValueError`를 즉시 발생**시키며, `ContinuationDomain`을 생성하지 않는다. §2.3의 build 실패, §5의 사후 품질 문제와는 별개의 범주다.

- `boundary.state == "reconciled_internal"` — 이미 봉합된 내부 seam이므로 continuation 대상이 아니다. 다른 `unclassified`/`unsupported`/`extension_candidate`는 모두 후보로 받는다(현재 `unsupported`/`extension_candidate`로의 실제 분류기는 없으므로 사실상 `unclassified` 전부가 후보가 된다 — "true object-boundary classifier는 범위 밖"이라는 기존 원칙과 일치하는 의도적 선택이며, 분류 대신 validity로 걸러낸다).
- `expected_patch_id`가 제공됐는데 `boundary.patch_id`와 불일치.
- `boundary.uv`/`boundary.world`/`boundary.inner_uv`/`boundary.inner_world`의 shape 불일치(샘플 수가 서로 다름, §2.1 정규화 이후 기준).
- 최소 sample 수 미달(§9 최소 boundary sample 계약: open `<3`, closed `<4` unique samples, §2.1 정규화 이후 기준).
- **인접 boundary 샘플 간 world distance가 `arclength_epsilon` 이하**(open: 모든 adjacent pair; closed: adjacent pair 전체 + closing segment(마지막 unique sample → 첫 sample)). 자동 deduplication/sample repair는 하지 않는다 — 발견 즉시 거부한다.
- `local_surface_scale`을 명시적으로 넘겼는데 non-finite 이거나 `<= 0`.
- `extent_multiplier`가 non-finite 이거나 `<= 0`.

### 2.3 Grid 구성 자체의 실패 (`ContinuationDomainBuildError`)

입력은 계약을 통과했지만(§2.2), 그럼에도 grid/AABB/extent를 구성할 최소 조건을 만족하지 못하는 경우다. **이 경우 `ContinuationDomain` 객체 자체를 반환하지 않는다** — `state=rejected`인 완전한 객체를 반환하는 것과 다르다.

- automatic `local_surface_scale` derivation 실패(§4.5 — 유효 scale 후보가 2개 미만).
- Surface evaluation 자체가 boundary 전체에서 실패(예: `evaluate_with_derivatives` 호출이 전 샘플에서 예외/전부 non-finite).
- Boundary 전체에서 tangent/outward direction을 단 하나도 만들 수 없음(모든 샘플이 degenerate).
- 위 결과로 finite한 continuation grid 또는 AABB를 전혀 구성할 수 없음.

### 2.4 Grid 구성 후의 품질 문제 (`state=degenerate`/`rejected`인 `ContinuationDomain` 반환)

Grid/AABB/provenance가 실제로 구성된 뒤에 발견되는 부분적 품질 문제(일부 방향만 invalid, Jacobian collapse, 일부 샘플만 non-finite, orientation 비일관성, 과도한 second-order growth)는 **예외를 던지지 않고** `state=degenerate` 또는 `state=rejected`인 `ContinuationDomain`을 반환한다(§5).

## 3. 출력 계약

`ContinuationDomain`의 canonical source of truth는 **sampled grid**다. `PatchBoundarySegment`가 연속 함수가 아니라 ordered samples만 갖는다는 사실(§0)과 대칭을 이룬다 — 임의의 `(s,t)`를 아무 때나 analytic하게 재평가하는 continuous closed-form API는 두지 않는다.

```python
STATE_VALID = "valid"
STATE_DEGENERATE = "degenerate"
STATE_REJECTED = "rejected"
CONTINUATION_STATES = {STATE_VALID, STATE_DEGENERATE, STATE_REJECTED}

@dataclass
class ContinuationDomain:
    """Boundary-local world-space sampled continuation strip.

    ``state == STATE_VALID``는 이 strip이 수치적으로 유효한 continuation
    hypothesis라는 뜻일 뿐, occluded surface나 Phase E의 occluded-region
    candidate로 승인됐다는 뜻이 아니다(§5).
    """

    domain_id: str                  # f"{source_boundary_id}:continuation"
    source_patch_id: int
    source_boundary_id: str
    closed: bool                    # boundary.closed 그대로

    s_count: int
    t_count: int
    s_world: Any                    # (s_count,) -- boundary를 따른 cumulative world arclength (§4.1). s_world[0] = 0.
    boundary_length: float          # 전체 boundary 길이 -- open: s_world[-1]; closed: s_world[-1] + closing segment 길이 (§4.1)
    t_world: Any                    # (t_count,) -- [0, continuation_extent] 범위의 world-space distance 샘플

    world: Any                      # (s_count, t_count, 3) -- first-order 샘플 grid. world[:, 0, :] == boundary.world (불변식)
    tangent_s: Any                  # (s_count, t_count, 3) d(world)/ds, world-arclength 기준
    tangent_t: Any                  # (s_count, t_count, 3) d(world)/dt; first-order이므로 t에 무관, outward_tangent_world를 broadcast
    normal: Any                     # (s_count, t_count, 3) normalize(tangent_s x tangent_t); invalid 지점은 zero vector(NaN 금지, normal_valid_mask로 표시)

    outward_tangent_world: Any      # (s_count, 3) -- t=0에서의 unit outward 방향(§4.2)

    normal_valid_mask: Any          # (s_count, t_count) bool
    direction_valid_mask: Any       # (s_count,) bool -- outward 방향 계산 자체의 유효성
    sample_valid_mask: Any          # (s_count, t_count) bool -- finite 여부 포함 전체 유효성

    local_surface_scale: float      # §4.5 -- 여러 raw scale 후보의 robust aggregate
    continuation_extent: float      # = extent_multiplier * local_surface_scale (실제 t_world 최대값)
    extent_multiplier: float

    aabb_min: Any                   # (3,) -- Phase E broad-phase용
    aabb_max: Any                   # (3,)

    state: str                      # CONTINUATION_STATES
    reason: str
    validity: dict[str, Any]        # §5 -- Jacobian singular value/condition, orientation consistency, parameter distortion
    uncertainty: dict[str, float]   # second_order_growth_ratio, second_order_displacement_at_extent(§4.3, intrinsic curvature 아님), inner_probe_distance passthrough, boundary confidence passthrough
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.state not in CONTINUATION_STATES:
            raise ValueError(f"Unknown continuation-domain state: {self.state!r}")

    def payload(self) -> dict[str, Any]: ...
```

**보조 API(선택적, source of truth 아님)**: 임의의 `s` 조회가 꼭 필요하면 아래처럼 별도의 module-level 함수로만 제공한다 — `ContinuationDomain`의 메서드로 두지 않고, boundary의 원본 샘플에 대해서만 정의한다.

```python
def interpolate_boundary_arclength(boundary: PatchBoundarySegment, s_query: Any) -> tuple[Any, Any]:
    """boundary.world를 cumulative world arclength 기준으로 piecewise-linear 보간한다.

    Analytic closed-form이 아니라 명시적으로 piecewise-linear 근사다.
    ContinuationDomain의 s_world/world grid를 대체하지 않는다 -- grid가
    canonical source of truth이고, 이 함수는 grid 샘플 사이의 임의 s 조회가
    필요할 때만 쓰는 보조 API다.
    """
```

**Phase C evidence와 연결 가능한 query sample representation**(interface만, 호출하지 않음): `ContinuationDomain.world.reshape(-1, 3)`는 `classify_world_samples(evidence, world_points)`가 받는 `(N,3)` 형태와 이미 일치한다. `reshape(s_count, t_count)`로 되접으면 grid 형태의 evidence map이 된다. **Phase D는 이 호출을 하지 않는다.**

## 4. Continuation 구성 방법

### 4.1 `s` 좌표 — cumulative world arclength (단위 고정, closed boundary 계약 포함)

`s`는 sample index, normalized parameter, arclength를 혼용하지 않고 **cumulative world arclength**로 고정한다. `s_world`는 §2.1에서 중복 종료점을 제거한 unique-sample 집합(`s_count`개) 위에서 정의한다.

```text
s_world[0] = 0
s_world[i] = s_world[i-1] + ||world[i] - world[i-1]||      for i = 1 .. s_count-1

boundary_length (open)   = s_world[s_count-1]
boundary_length (closed) = s_world[s_count-1] + ||world[0] - world[s_count-1]||   (closing segment 포함 전체 perimeter)
```

`s_world[i]`는 "첫 sample부터 i번째 sample까지의 누적 길이"만을 의미하고, closing segment 길이는 `s_world`가 아니라 `boundary_length`에만 반영된다 — 두 표현을 혼용하지 않는다.

Boundary world tangent `T(s_i)`도 world-arclength 기준으로 계산한다.

- **Open boundary**: interior 샘플은 central difference(`(world[i+1]-world[i-1]) / (s_world[i+1]-s_world[i-1])`), 양 endpoint는 one-sided difference.
- **Closed boundary**: `boundary_length`를 이용한 periodic central difference. 인덱스 `i`의 이전/다음 이웃을 순환시키되, 순환된 이웃의 "periodic s" 값을 `boundary_length`만큼 offset해 사용한다:

```text
이웃 인덱스가 wrap돼서 (i-1) < 0 이면:
  previous sample의 periodic s = s_world[s_count-1] - boundary_length

이웃 인덱스가 wrap돼서 (i+1) >= s_count 이면:
  next sample의 periodic s = s_world[0] + boundary_length = boundary_length

T(s_i) = (world[next] - world[previous]) / (periodic_s[next] - periodic_s[previous])
```

(예: `i=0`일 때 previous는 인덱스 `s_count-1`이며 periodic s는 `s_world[-1] - boundary_length`; `i=s_count-1`일 때 next는 인덱스 `0`이며 periodic s는 `s_world[1]`이 아니라 `boundary_length`다 — "다음 sample의 periodic s = s_world[1]"이라는 표현은 `i=0`에서 `next=1`인 일반 interior 케이스를 가리키며, wrap이 실제로 일어나는 endpoint에서는 위 offset 공식을 쓴다.)

Sampling density가 달라져도 `T`와 이후 `d(outward)/ds`의 크기가 임의로 변하지 않도록 모든 미분은 world-arclength 간격으로 나눈다(sample-index 기준 유한차분 금지).

### 4.2 Outward 방향 — world-space에서 직접 정의(UV perpendicular 폐기)

이전 개정의 UV-space perpendicular `(b,-a)` 공식은 폐기한다. UV parameterization이 skewed/non-uniform하면 UV에서 수직인 방향이 world tangent plane에서 수직이라는 보장이 없기 때문이다.

각 boundary 샘플에서 다음 순서를 사용한다(전부 world-space 연산):

```text
1. Analytic S_u, S_v를 evaluate_with_derivatives(uv)로 평가한다.
2. Boundary world tangent T는 §4.1의 cumulative-world-arclength 기준 값을 그대로 쓴다.
3. Surface normal N = normalize(S_u x S_v).
4. Tangent plane 안의 cross-boundary 방향 후보 C = normalize(N x T).
5. inner_world - boundary_world와 내적해 outward 부호를 선택한다:
   dot(C, inner_world - boundary_world) > 0  이면  outward = -C
   그 외에는                                        outward = C
```

이 공식은 UV metric 왜곡에 영향받지 않는다 — `N`과 `T`가 둘 다 world-space 벡터이고 `C = N x T`는 자동으로 world tangent plane 안에서 `T`에 수직이다. UV axis swap, UV scale/skew, loop reversal에 대해 동일한 world-space 결과가 나와야 한다는 것이 §7의 필수 invariant다.

**Degenerate 처리**: `N` 또는 `T`의 norm이 거의 0이면(`eps` 미만) 조용히 임의 방향을 만들지 않는다 — 해당 `s`의 `direction_valid_mask`를 `False`로, `outward_tangent_world[s]`를 zero vector로 기록한다(NaN 금지).

### 4.3 Second-order 방향 diagnostic — UV 투영, position에 사용하지 않음

**명칭과 의미(중요)**: 아래에서 계산하는 값은 surface의 intrinsic curvature(mean/Gaussian/normal/geodesic curvature) 추정치가 아니다. World-metric으로 정규화한 하나의 outward 방향에 대한 **directional second-order continuation diagnostic**일 뿐이다. 이 문서와 구현 전체에서 `second_order_displacement_at_extent`/`second_order_growth_ratio`라는 명칭을 쓴다(이전 개정의 `curvature_displacement`/`curvature_growth_ratio` 명칭은 논문 맥락의 surface curvature와 혼동될 수 있어 폐기한다). 이 값을 논문에서 쓰는 surface curvature와 동일한 것처럼 표현하지 않는다.

이 diagnostic을 위해 world outward 방향을 UV 방향으로 투영해야 하는 경우에만 다음 최소제곱 문제를 쓴다(이 결과는 §4.4의 canonical position에 들어가지 않는다).

```text
J = [S_u S_v]                       (3x2, at this boundary sample)
q_raw = argmin_q ||J q - outward_tangent_world||²     ((J^T J) q_raw = J^T outward_tangent_world)
q = q_raw / ||J q_raw||             (반드시 정규화: ||Jq|| = 1)
```

`t`가 world-length 단위이므로 directional second derivative의 단위를 맞추려면 `||Jq||=1` 정규화가 필수다.

```text
D²S[q,q] = q_u² S_uu + 2 q_u q_v S_uv + q_v² S_vv

second_order_displacement(t) = 0.5 * t² * D²S[q,q]
second_order_displacement_at_extent = second_order_displacement(continuation_extent)     (벡터)
second_order_growth_ratio = ||second_order_displacement_at_extent|| / max(continuation_extent, eps)   (스칼라)
```

`second_order_displacement_at_extent`(벡터 크기)와 `second_order_growth_ratio`(스칼라)만 `uncertainty`에 기록하며, 둘 다 위 정의를 그대로 docstring/metadata에 명시한다: "Directional second-order continuation diagnostic. Not an intrinsic mean, Gaussian, normal, or geodesic curvature estimate." **`world`/canonical position에는 어떤 형태로도 더하지 않는다** — §4.4가 유일한 canonical position 공식이다.

### 4.4 Canonical geometry — first-order sampled grid only

```text
world(s_i, t_j)    = boundary_world(s_i) + t_j * outward_tangent_world(s_i)
tangent_t(s_i, t_j) = outward_tangent_world(s_i)                              (t에 무관 -- first-order 정의)
tangent_s(s_i, t_j) = T(s_i) + t_j * d(outward_tangent_world)/ds (s_i)
```

`d(outward_tangent_world)/ds`는 §4.1과 동일하게 world-arclength 간격으로 나눈 유한차분이다 — surface geometry 재추정이 아니라 이미 analytic하게 계산된 `outward_tangent_world(s)` 벡터장 자체의 이산 도함수이므로 원칙 위반이 아니다.

**2차 position variant는 canonical output으로 제공하지 않는다.** `ContinuationDomain.world`는 항상 1차 근사이며, 곡률 정보는 §4.3의 diagnostic 값으로만 존재한다.

### 4.5 `local_surface_scale`과 `continuation_extent` — canonical 집계식(구현 시 임의로 정하지 않도록 고정)

세 개념을 구분한다.

```text
inner_probe_distance:
  boundary.confidence["inner_distance_median"] 등 -- inward/outward 부호와 local
  differential probe(§4.2의 inner_world)에만 쓰인다. extent로 재사용하지 않는다.

local_surface_scale:
  아래 canonical 공식(§4.5.1)으로 확정한다.

continuation_extent:
  = extent_multiplier * local_surface_scale.  실제로 strip이 만들어지는 world 최대 거리.
```

#### 4.5.1 `local_surface_scale`의 canonical 집계식

각 scale source를 먼저 하나의 scalar로 축약한다.

```text
L_boundary = median(positive adjacent boundary segment lengths)
             (closed boundary는 closing segment(마지막 unique sample -> 첫 sample, §4.1) 포함)

L_inner    = median(positive ||inner_world - boundary_world|| 값)

L_control  = median(source NURBS control_grid에서 u/v 방향 인접 control-point
             edge 길이(예: control_grid[i,j]-control_grid[i+1,j],
             control_grid[i,j]-control_grid[i,j+1]) 중 positive한 전체 값)
```

그 후:

```text
valid_scales = {L_boundary, L_inner, L_control} 중 finite 이고 positive인 값만 모은 집합

local_surface_scale = median(valid_scales)
```

**Automatic scale derivation은 `len(valid_scales) >= 2`일 때만 허용한다.** `len(valid_scales) < 2`이면 `ContinuationDomainBuildError`를 던진다(§2.3) — `ContinuationDomain`을 만들지 않는다.

Caller가 `local_surface_scale`을 명시적으로 제공하면 이 자동 집계 전체를 건너뛴다. 이 경우 explicit 값은 반드시 finite이고 positive여야 하며, 아니면 `ValueError`다(§2.2 — 입력 계약 위반, 사후 build 실패가 아니다).

이 계약이 보장해야 하는 invariant(§7 fixture로 검증):

- `inner_probe_distance` 하나만 바꿔도(예: `inner_uv`를 다른 offset으로 재계산) `local_surface_scale`/`continuation_extent`가 그 값에 정확히 선형 비례하지 않는다 — `L_boundary`/`L_control`이라는 다른 독립 후보가 있기 때문이다.
- Boundary sampling density가 달라져도 동일 geometry에서 `L_boundary`(median이므로 스케일 자체는 유지)가 과도하게 변하지 않는다.
- Control grid resolution만 달라져도 `L_control`(median이므로)이 임의로 급변하지 않는다.

`extent_multiplier`의 기본값은 이 설계 문서에서 고정하지 않는다 — isolated benchmark에서 검증할 configurable value로 둔다(예: 1.0을 잠정 시작점으로 제안하되 확정 아님).

**Multi-scale strip은 이번 Phase D 최소 구현 범위에 넣지 않는다.** Phase E recall이 부족할 경우 후속 검토한다.

### 4.6 Knot vector 확장 정책

**확장하지 않는다.** §0에서 확인했듯 clamped knot vector를 `[0,1]` 밖으로 늘리는 기존 헬퍼가 없고, span-detection 로직이 폐구간을 전제하므로 안전하게 확장할 수 없다. `ContinuationDomain`은 `TorchNURBSSurface`가 아니라 §4.4의 닫힌형 수식으로 생성되는 sampled grid다(§8의 대안 비교에서 이 선택의 근거를 상세히 다룬다).

### 4.7 Corner/endpoint 처리

Open(비순환) boundary의 양 끝점에서는 `d(outward_tangent_world)/ds`의 유한차분이 한쪽 이웃만 가지므로 one-sided difference를 쓴다(§4.1과 동일 정책). 서로 다른 두 boundary가 만나는 corner는 Phase D에서 합치지 않는다 — 각 edge가 독립적으로 자신의 strip을 끝점까지 만들고, corner 연결 여부는 Phase E의 pairing 몫이다(§6).

## 5. 유효성 조건과 상태 의미론

### 상태 이름

`candidate`는 Phase E의 occluded-region candidate와 혼동되므로 사용하지 않는다.

| 상태 | 의미 |
|---|---|
| `valid` | Sampled strip이 finite하고 local differential validity(아래 표)를 만족한다. **occluded surface 또는 occluded-region candidate로 승인됐다는 뜻이 아니다** — continuation hypothesis가 수치적으로 유효하다는 뜻일 뿐이다. |
| `degenerate` | Grid/AABB/provenance는 실제로 구성됐지만 Jacobian collapse, partial direction degeneracy, excessive second-order growth 등 부분적 품질 문제가 있다. |
| `rejected` | Grid/AABB/provenance는 구성됐지만(§2.3의 `ContinuationDomainBuildError`와 달리 객체 자체는 만들어짐) 사후 검사에서 전체적으로 신뢰할 수 없다고 판정됐다(예: 구성 후에도 전역적으로 non-finite인 비중이 과도함). **`local_surface_scale` 자동 도출 실패처럼 grid 자체를 만들 수 없는 경우는 `rejected`가 아니라 §2.3의 `ContinuationDomainBuildError`다** — 이 둘을 혼동하지 않는다. |

### 유효성 조건

| 조건 | 판정 | 처리 |
|---|---|---|
| Jacobian magnitude / collapse | `(tangent_s, tangent_t)`의 2x2 `J^T J` singular value — `compute_parametric_jacobian_metrics`(§9 Prerequisite 1) 재사용 | `sigma_min < eps` → 해당 `(s,t)` 샘플의 `sample_valid_mask=False`로 기록 |
| Orientation flip/fold-over | 인접 `s` 샘플 간 `normal(s,t)` 부호 일관성 — `compute_orientation_consistency`(§9 Prerequisite 2, closed 여부에 따라 순환/비순환 비교) | flip 카운트 기록, hard reject 아님(soft) |
| Derivative collapse | 위 Jacobian singular value와 동일 지표 | 위와 동일 |
| Excessive second-order growth | §4.3의 `second_order_growth_ratio`(intrinsic curvature 아님, directional diagnostic) | 비율이 임계값(기본 0.5, 미확정) 초과 시 `reason`에 기록. `continuation_extent`를 국소적으로 줄이는 것은 이번 canonical 범위에 포함하지 않음(uniform `continuation_extent` 유지, uncertainty만 표시) |
| Self-intersection | **Phase D 범위 밖**(마스터 플랜 §8 Phase F 소유). Jacobian collapse/second-order-growth 지표만 로컬 proxy로 제공 | 결과에 `self_intersection_checked: false` 명시 |
| Source visible surface로의 역침범 | Canonical(1차, 작은 extent) 범위에서는 second-order-growth 지표로 근사. 독립적 nearest-point 검사는 향후 개선으로 미룸 | 위와 동일하게 미검증임을 명시 |
| Adjacent boundary 간 domain overlap | Phase D는 검사하지 않는다 — Phase E의 pairing 입력으로 제공만 한다(§6) | 해당 없음 |
| Parameter-space distortion | `torch_annulus_chart._parameter_quality`와 동일한 anisotropy/orthogonality 공식(§9) | uncertainty에 기록 |
| Numerical stability(grid 구성 후) | 모든 나눗셈에 기존 관례와 동일한 `eps` clamp, `torch.isfinite` 전수 검사 | Grid가 구성된 후 전역적으로 non-finite 비중이 과도하면 `state=rejected`, `reason="non_finite_domain"`(grid 자체를 만들 수 없는 경우는 §2.3 `ContinuationDomainBuildError`) |
| Invalid domain 폐기 여부 | **폐기하지 않는다** — `PatchReconciliationResult`가 실패한 pair도 보존하는 기존 관례를 따른다 | `degenerate`/`rejected` 도메인도 항상 반환값에 포함, 필터링은 호출자 책임 |

### NaN 대신 mask

Degenerate normal/tangent/direction을 NaN으로 저장하지 않는다. 무효 위치는 zero vector로 저장하고 `normal_valid_mask`/`direction_valid_mask`/`sample_valid_mask`로 유효성을 명시적으로 표현한다.

## 6. Boundary pairing과 domain interaction

**Phase D는 pairing을 수행하지 않는다.** 마스터 플랜 §7(Phase E)이 "두 개 이상의 strip 교차/근접/공동 bounded region을 candidate로 유지한다"를 명시적으로 소유한다. Phase D는 boundary 하나당 독립된 `ContinuationDomain` 하나만 만든다.

Phase E가 필요로 할 최소 interface(Phase D가 미리 제공):

- `ContinuationDomain.world`(world-space sampling) — proximity/intersection 판정의 원재료.
- `ContinuationDomain.aabb_min`/`aabb_max` — broad-phase 필터. `torch_voxel_hierarchy.VoxelNode`의 `aabb_min`/`aabb_max` 관례와 필드명을 통일했다.
- `source_patch_id`/`source_boundary_id` — provenance.

Corner에서 인접한 두 edge의 strip이 자연스럽게 이어지는지(§4.7)는 이번 Phase D 범위에서 검증하지 않는다.

## 7. Fixture 및 테스트 계획

| Fixture | 기대 invariant | 실패 조건 |
|---|---|---|
| Planar boundary(사각형 edge) | §4.2 world-space 공식으로 얻은 outward가 interior 반대쪽을 향함; second_order_growth_ratio ≈ 0 | Outward가 interior로 나감, 또는 ratio가 0이 아님 |
| Smoothly curved boundary(sine 등) | `continuation_extent` 이내에서 second_order_growth_ratio가 임계값 이내 | 비율 폭주 |
| Rotated plane | World-space 결과가 회전 전과 정확히 일치(회전만큼) | 회전에 따라 outward가 달라짐 |
| **UV axis swap** | Outward world 방향이 UV 축을 바꿔도 동일 | 방향이 달라짐 |
| **UV scale/skew** | UV parameterization을 비균일하게 rescale/skew해도 동일 world geometry에서 outward가 동일(§4.2가 world-space 전용 공식이므로 이 invariant가 핵심 검증 대상) | 방향이 달라짐(구 UV-perpendicular 공식의 결함 재발 감지용) |
| Orthogonal surfaces(두 patch가 90도로 만남) | 각 patch의 continuation domain이 독립적으로 생성됨 — **이 fixture는 geometry accuracy가 아니라 "normal/facing hard gate가 없다"는 policy-regression test다** | Facing/normal 차이를 이유로 domain 생성이 거부됨(정책 위반) |
| Oblique surfaces | 위와 동일(policy-regression) | 위와 동일 |
| Annular/radial boundary(내/외 edge만, seam 제외) | `v0`/`v1`에서만 domain 생성, `u0`/`u1`(reconciled seam)은 `ValueError`(§2.2) | Seam에서도 domain이 생김, 또는 예외 대신 조용히 skip됨 |
| Reversed parameter direction(loop reversal) | World-space 결과가 방향과 무관하게 동일(`s` 샘플 순서만 반전) | Outward 부호가 반전 방향에 따라 달라짐 |
| **Boundary resampling density 변경** | 같은 위치에서 strip 방향과 second-order diagnostic이 허용오차 내 일치(§4.1의 world-arclength 정규화가 sample density에 안정적인지 확인) | Density를 바꾸면 방향/second-order 값이 유의미하게 달라짐 |
| Degenerate derivative/normal/tangent(`S_u`≈0, annulus 내부 극점 근방 등) | `direction_valid_mask=False`/`sample_valid_mask=False`로 감지, **NaN 없이** zero vector + mask로 표현, `state=degenerate`, 예외로 죽지 않음 | NaN 출현, 또는 예외 발생, 또는 감지 실패 |
| High-curvature fold-over(합성 극단 곡률 patch) | second_order_growth_ratio가 임계값을 초과함을 실제로 감지 | 감지 실패 |
| **Artificial reconciled boundary(negative control)** | `state="reconciled_internal"`인 boundary에 `build_continuation_domain` 호출 시 **`ValueError`**(입력 계약 위반, §2.2) | 조용히 domain을 만들거나 degenerate 상태로 넘김 |
| Unsupported open boundary(positive control) | `state="unclassified"`에서 정상적으로 `state=valid` domain 생성 | 생성 거부 |
| Corner/endpoint boundary | Open boundary 양 끝에서 one-sided difference로 finite `tangent_s`; `world[:, 0, :] == boundary.world`(정확한 등식, 불변식) | 끝점에서 NaN/Inf, 또는 `t=0` 행 불일치 |
| **`t_world` 정확성** | `t_world`가 실제 `world`에서 측정한 world distance(`‖world[i,j] - world[i,0]‖`)와 일치 | 불일치(단위 변환 버그) |
| **Closed boundary closing segment** | `boundary_length`가 `s_world[-1] + closing segment 길이`와 정확히 일치; §4.1의 periodic central difference가 closing segment를 포함해 계산됨 | `boundary_length`가 closing segment를 누락하거나 이중 계산 |
| **Open boundary adjacent duplicate(negative control)** | 인접 샘플 world distance가 `arclength_epsilon` 이하인 open boundary에 `build_continuation_domain` 호출 시 **`ValueError`** | 조용히 통과하거나 잘못된 tangent(예: 매우 큰 값)를 생성함 |
| **Closed boundary closing-segment zero-length(negative control)** | 마지막 unique sample과 첫 sample이 (중복 제거 후에도) `arclength_epsilon` 이하로 겹치는 closed boundary에 `build_continuation_domain` 호출 시 **`ValueError`** | 위와 동일 |
| **`local_surface_scale`/`continuation_extent` 안정성** | `inner_probe_distance`만 변경(예: `inner_uv`를 다른 offset으로)해도 `local_surface_scale`/`continuation_extent`가 그 하나의 probe에 비정상적으로 비례하지 않음(§4.5.1의 `L_boundary`/`L_control`이라는 다른 독립 후보가 있으므로); boundary sampling density 변경 시 `L_boundary` 안정성; control grid resolution 변경 시 `L_control` 안정성 | 하나의 probe distance에 정확히 비례해서만 변함(다중-후보 aggregate가 실제로는 단일 probe로 축소됐다는 뜻) |
| **`local_surface_scale` 자동 도출 실패(negative control)** | 유효 scale 후보(`L_boundary`/`L_inner`/`L_control` 중 finite·positive)가 1개 이하인 극단적 합성 fixture에서 `build_continuation_domain` 호출 시 **`ContinuationDomainBuildError`**(`ValueError`도 `ContinuationDomain(state=rejected)`도 아님) | 잘못된 예외 타입, 또는 `state=rejected`인 `ContinuationDomain` 객체가 반환됨 |
| **최소 sample 수 fixture** | Open 3개/closed 4개 unique sample에서 `d/ds`, endpoint 차분, orientation diagnostic이 계산 가능함(§9) | 계산 불가 또는 예외 |
| **상태 승격 금지** | 모든 fixture에서 `state`가 `valid`/`degenerate`/`rejected` 중 하나일 뿐 `occluded_candidate`나 `validated chart`류 값으로 나오지 않음(§5) | 그런 상태값이 나타남 |

## 8. 대안 비교

| 방식 | 정확도 | NURBS 일관성 | 구현 복잡도 | 안정성 | 디버깅 난이도 |
|---|---|---|---|---|---|
| **Control-net extrapolation**(실제 knot vector를 `[0,1]` 밖으로 확장) | 이론상 최고 | 최고 | **높음** — clamped knot span-detection을 다시 설계해야 함(§0), 기존에 없는 low-level 수학(knot insertion 등) | 중간(B-spline 수치 버그 위험) | 어려움 |
| **World-space derivative-seeded sampled strip**(이 문서가 권장하는 방식, §4) | 1차: 국소 정확, `t` 커질수록 오차 증가(Taylor truncation) | `TorchNURBSSurface`가 아님 — sampled grid가 candidate 산출물(Phase F가 진짜 NURBS chart를 별도로 만듦) | **낮음** — closed-form 대수 연산, 새 low-level B-spline 수학 불필요 | 높음(단순 대수 연산, eps clamp만 신경쓰면 됨) | 쉬움(수식이 짧고 명시적) |
| **Local fitted proxy**(합성 점을 다시 `fit_torch_visible_surface_lsq`로 NURBS fit) | 오히려 **낮음** — 이미 analytic하게 정확한 함수를 근사 fit으로 다시 근사하는 순환적 정보 손실 | 최종적으로 NURBS이긴 함 | 중간(재사용은 쉽지만 IDW 시딩 등 point-cloud fitting 전용 가정이 안 맞음) | 낮음(fit 자체의 degenerate/fold 위험 추가) | 중간 |

**권장**: World-space derivative-seeded sampled strip(§4)을 Phase D의 유일한 구현으로 채택한다. Control-net extrapolation은 Phase F에서 재검토할 가치가 있지만 Phase D 범위에서는 과잉 설계다. Local fitted proxy는 어느 단계에서도 채택하지 않는다.

## 9. 권장 최소 구현 범위

### Prerequisite(구현 착수 전 필요, 순수 리팩터/무동작-변경)

1. **`compute_parametric_jacobian_metrics(deriv_a, deriv_b, eps, scale) -> dict`**: `torch_annulus_chart.py:124-224`의 `J^T J` closed-form eigenvalue 계산(`sigma_min/sigma_max/condition/area` + scale-normalized variant)을 `TorchNURBSSurface`가 아니라 raw `(deriv_a, deriv_b)` 텐서를 받는 순수 함수로 추출한다. `torch_annulus_chart.py`와 Phase D 모듈이 공유한다.
2. **`compute_orientation_consistency(normals, valid_mask, closed, ...) -> dict`**: per-slice/per-sample orientation flip 판정을 별도 함수로 분리한다. **하나의 helper에 두 책임을 합치지 않는다** — annulus의 순환(ring) topology와 continuation strip의 일반적으로 비순환(open)인 topology는 일관성 검사 형태가 다르므로, 이 함수는 각 grid topology의 wrapper가 자신의 topology에 맞게 소유한다(annulus는 자신의 ring-consistency 버전을, Phase D는 자신의 strip-adjacency 버전을 각각 호출).
3. `torch_annulus_chart.py`를 위 두 헬퍼를 호출하도록 리팩터한다. **회귀 검증은 byte-identical을 요구하지 않는다** — 다음을 확인한다: 기존 상태 분류(예: `orientation_flip_count`, `near_degenerate_count`) 동일, 기존 report field 이름 동일, 수치가 합리적 tolerance 내 동일, 전체 suite 회귀 없음.
4. `patch_id -> TorchNURBSSurface` 매핑을 만드는 caller 쪽 관례가 아직 없다(§2) — Phase D 모듈 자체의 책임은 아니지만, 이를 처음 호출할 benchmark/테스트 코드에서 이 매핑을 어떻게 구성할지(예: `list[TorchNURBSSurface]`의 인덱스를 `patch_id`로 그대로 쓰는 기존 Phase B 관례 재사용) §7의 fixture 작성 시 함께 정한다.

### 이번 Phase D에서 실제로 구현할 것

1. 위 prerequisite 1-3(리팩터).
2. `osn_gs/surface/torch_continuation_domain.py`: `ContinuationDomain` dataclass(`boundary_length` 포함), `STATE_*` 상수, `ContinuationDomainBuildError` 예외 타입(§2.3), `build_continuation_domain(...)` entry point, `interpolate_boundary_arclength(...)` 보조 함수, §4.1-4.5의 내부 헬퍼(closed-loop 중복 종료점 stripping, world-arclength tangent + periodic 차분, world-space outward-direction solver, UV 투영 second-order diagnostic, `local_surface_scale` canonical aggregate).
3. `tests/test_continuation_domain.py`: §7의 fixture 전부(신규 closed-loop/adjacent-duplicate/scale-derivation-failure negative control 포함).
4. `osn_gs/surface/__init__.py`에 `ContinuationDomain`, `ContinuationDomainBuildError`, `build_continuation_domain` export 추가(entry point + result/예외 타입만, 내부 헬퍼는 비공개).

### Phase E로 미룰 것

- Boundary pairing, overlap/intersection 판정, 공동 empty-region 지지 여부, candidate conflict graph.
- Facing/normal/curvature를 soft evidence로 실제로 가중해 candidate 우선순위를 매기는 로직.

### Phase F로 미룰 것

- 전체 self-intersection 검사, source-visible-surface 역침범의 독립적 world-space nearest-point 검사, control-net 기반 진짜 constrained NURBS chart 구축.

### Production integration 전 필요한 gate

- 이 설계 자체에 대한 사용자 승인(**이 문서의 목적**).
- Phase D 구현 완료 후 승인 게이트 D(마스터 플랜 §6).
- Phase E/F/G/H는 마스터 플랜이 이미 각각 별도 게이트를 요구.

### 구현 순서(승인 시)

1. Prerequisite 리팩터(§9 1-3) → 기존 annulus 테스트로 회귀 확인(tolerance 기준, byte-identical 아님).
2. Closed-loop 중복 종료점 stripping(§2.1) + `boundary_length`/periodic 차분(§4.1) + adjacent-duplicate 검증(§2.2).
3. World-space outward-direction solver(§4.2) + UV axis swap/scale/skew invariance 테스트.
4. 1차 strip 구성(§4.4) + `s_world`/`t_world`.
5. §4.3 second-order 방향 diagnostic(diagnostic 전용, position에 미반영, 명칭은 `second_order_*`).
6. §5 validity 전체(리팩터된 두 helper 사용) + mask 기반 degenerate 표현 + `ContinuationDomainBuildError`/`degenerate`/`rejected` 3분류(§2.3, §2.4, §5).
7. `local_surface_scale`/`continuation_extent`의 canonical aggregate(§4.5.1).
8. `aabb_min`/`aabb_max` 계산(Phase E 준비).
9. §7 전체 fixture + worklog로 승인 게이트 D 보고 후 정지.

## 10. 승인 게이트 D 보고 형식(구현 후 채울 초안)

| 검증 항목 | Fixture | 결과(구현 후 기입) |
|---|---|---|
| UV axis swap/scale/skew, loop reversal에 대한 world-space invariance | 해당 fixture 3종 | |
| Plane | Planar boundary | |
| Curved | Smoothly curved boundary | |
| Orthogonal/Oblique — normal/facing hard gate 없음 policy 확인 | 해당 fixture 2종 | |
| Degenerate Jacobian/normal/tangent — NaN 없이 mask로 표현 | Degenerate derivative | |
| Closed boundary closing segment가 `boundary_length`에 정확히 반영됨 | Closed boundary closing segment | |
| 최소 sample 수 계약 충족 | 최소 sample 수 fixture | |
| 인접 duplicate/zero-length segment가 `ValueError`로 거부됨 | Adjacent duplicate/closing-segment negative control 2종 | |
| `reconciled_internal`이 `ValueError`로 거부됨 | Artificial reconciled boundary | |
| `local_surface_scale` 자동 도출 실패가 `ContinuationDomainBuildError`로 구분됨(ValueError/rejected와 혼동 없음) | Scale-derivation-failure negative control | |
| `state`가 `occluded_candidate`/`validated chart`로 승격되지 않음 | 전체 fixture 공통 | |
| `local_surface_scale`/`continuation_extent`가 단일 probe에 비정상 종속되지 않음 | 안정성 fixture | |
| Second-order diagnostic이 `second_order_*` 명칭으로 보고되고 intrinsic curvature로 오인되지 않음 | 전체 fixture 공통 | |

이 표는 §7 fixture 구현이 끝난 뒤 실제 수치로 채워 worklog에 옮긴다.
