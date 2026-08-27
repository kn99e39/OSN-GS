# Worklog 127: Evidence-Bounded Projective TSDF for Direct Visible Surface Construction

상태: **완료**
브랜치: `arch/2dgs-coverage-first-surface`
범위: Visible Surface **구성 전제(construction premise)** 자체를 바꾸는 단일 통제 실험 — 역사적 topology/boundary-first 경로 대신 renderer median 관측을 직접 융합하는 evidence-bounded projective TSDF

> **번호 안내**: directive는 "Worklog 125"를 요청했으나 `docs/worklogs/125_fixed_gaussian_visualization_contract.md`와 `output/126_*`이 동시 진행 작업으로 이미 점유되어 있어 이 배치는 **127**로 등록했다. 그 외 어떤 내용도 바꾸지 않았다.

---

## Agent Interpretation of Intent

### DIRECTION

역사적 Visible Surface Construction이 쌓아 올린 중간 추상(primitive locality / KNN / visible component / region / boundary / boundary recovery / chart / parameterization / NURBS)을 **이번 배치에서 개선하지 않는다.** 대신 정확히 하나의 대안 구성 전제만 구현한다:

```
canonical renderer median-depth observations
    -> evidence-bounded projective TSDF
    -> masked zero level-set
    -> Visible Surface geometry
```

이 후보는 **자기 표면이 어디에 존재하는지를 결정하는 데** 기존 visible topology, KNN 관계, region 배정, boundary loop, component recovery, chart 구성을 일절 쓰지 않는다. 그것들은 메시가 만들어진 **이후에만** 진단 비교용으로 등장한다.

### PURPOSE

역사적 구성에 누적된 복잡도가 **과학적으로 필연인지**를 판정한다. 대립 가설은 "dense renderer-grounded visible-surface 관측 자체가 이미 visible geometry를 직접 구성하기에 충분한 정보를 담고 있다"이다. 참이라면 topology/boundary/component/chart는 visible geometry가 **존재하기 전에** 전부 풀려 있을 필요가 없고, 일부는 재구성된 geometry로부터 **사후에 유도**될 수 있다. 목적은 "SDF를 표현 하나 더 추가하기"가 아니라 "evidence-bounded implicit surface가 현재의 topology/boundary-first 구성 전제를 **대체**할 수 있는지 시험하기"이다.

### CENTRAL INTENT

completion heuristic으로 SDF를 성공시키지 않는다. 가장 단순한 renderer-evidence 기반 implicit geometry가 이미 충분한지를 본다. 핵심 질문:

> "직접적 visible 증거가 지역적 권한을 갖지 않는 곳에서는 표면을 발명하기를 **거부**하면서, renderer-native dense surface 관측만으로 쓸모 있는 Visible Surface를 직접 재구성할 수 있는가?"

실패도 유효한 아키텍처 결과다. 보기 좋아질 때까지 field를 튜닝하지 않는다.

### PRESERVE

- 현재 브랜치 이력, canonical 2DGS renderer, 학습된 체크포인트, 161개 학습 카메라
- WL107/109 visible-topology baseline, WL111–119 visible-NURBS baseline, WL120–123 renderer frontier/query 결과
- WL123이 확정한 계약: world-space `x`가 canonical volumetric query이고, canonical renderer median event가 현재의 visible-surface observation frontier 후보이며, exact renderer event identity는 epsilon 없이 보존 가능하다
- Candidate B의 결정 함수와 frozen global aggregation (읽기 전용 비교에만 사용)

### CHANGE ONLY

Visible Surface **구성 후보** 하나. `historical topology/boundary-first geometry` ↔ `evidence-bounded projective TSDF geometry`.

### DO NOT

neural SDF / MLP 학습 / Eikonal loss / NeuS·VolSDF / Poisson reconstruction / watertight 강제 / mesh hole filling / 스크린샷용 smoothing / UNKNOWN voxel 채우기 / SDF 부호를 물체 내부·외부로 해석 / Gaussian covariance normal 사용 / Mesh-Aligned Gaussian normal 도입 / KNN·WL107·109 topology·region id·boundary loop·boundary recovery·chart eligibility를 SDF 구성에 사용 / component 수동 병합 / h 튜닝 / mu sweep / 아티팩트를 본 뒤 support-count 임계 추가 / 문제 영역 블랙리스트 / 실패 영역의 역사적 geometry fallback / TSDF와 역사적 geometry의 하이브리드 / Candidate B 수정 / Occluded Surface completion / occluded space로의 visible surface 외삽 / Trust 구현 / NURBS 재설계 — **전부 하지 않았다.**

### PROMPT-REQUIRED DECISION

| 항목 | 값 | 근거 |
|---|---|---|
| implicit field 종류 | projective TSDF (정확한 Euclidean SDF 아님) | directive §3 |
| 부호 규약 | `s_v(x) = d_v(p) - z_v(x)`, `>0`이 카메라 쪽 | directive §3 |
| 정규화 | `phi_v = clamp(s_v/mu, -1, +1)`, **fusion 목적에 한해** | directive §3 |
| authority 규칙 | `|s_v(x)| <= mu`일 때만. 밖은 `+1`도 `-1`도 아닌 **UNKNOWN** | directive §3 |
| voxel 크기 | `h = 유효 median event footprint(=depth/sqrt(fx·fy))의 전역 중앙값` | directive §4 |
| truncation | `mu = 3h`, 사전 고정 | directive §4 |
| fusion 가중치 | 모든 authoritative 관측에 정확히 **1** | directive §5 |
| 최소 뷰 수 | **없음.** 관측 1개면 충분 | directive §5 |
| cell 자격 | 8개 corner 전부 authority **그리고** `min(phi) <= 0 <= max(phi)` | directive §7 |
| 구성 중 역사적 topology 사용 | **금지** | directive §8 |

### AGENT-INTRODUCED OPERATIONAL CHOICE

증거를 쓰기 전에 전부 공개한다. 어느 것도 field의 **정의**를 바꾸지 않는다.

1. **Candidate voxel 열거 = closure 성장.** field 정의상 authoritative voxel 집합은 명시적으로 열거해야 추출할 수 있다. 임의의 dilation 반경을 고르는 대신, renderer median event의 voxel에서 출발해 **authoritative 집합의 1-voxel 껍질만 반복 검사**하고, 껍질 전체가 새 authoritative voxel을 하나도 내놓지 않으면 멈춘다(= truncation 규칙 자신의 고정점). 반경을 고르지 않으므로 튜닝이 아니고, 검사 대상 집합만 넓힐 뿐 authority는 오직 truncation 규칙만이 부여한다. 사전 실측으로 반경 3에서 authoritative 집합이 이미 22,318,157로 수렴하고 반경 4/5가 각각 +0.09%/+0.013%만 더한다는 것을 확인했다(§부록).
2. **UNKNOWN sentinel + cell 필터로 masked marching cubes 구현.** `skimage.measure.marching_cubes`의 `mask`는 **단일 corner**만 검사하므로(직접 측정, 테스트로 고정) 8-corner 계약을 구현하지 못한다. 대신 UNKNOWN corner를 phi 범위 밖 sentinel로 채워 표준 Lewiner MC를 돌린 뒤 **자격 없는 cell에서 나온 삼각형을 전부 폐기**한다. MC는 cell 단위 연산이므로 남는 삼각형은 sentinel 값과 무관하게 진짜 masked 추출과 **증명적으로 동일**하며, 반대 부호 sentinel 두 번 돌려 결과가 비트 단위로 같음을 회귀 테스트로 강제했다. sentinel 값은 어떤 삼각형에도, 어떤 export에도 도달하지 않는다.
3. **Block 분할 추출 + seam weld.** 64³ cell 블록이 **서로소인 cell**을 소유하므로 삼각형 중복이 없고, 블록 경계의 공유 격자 edge만 두 번 계산된다. MC가 동일 corner 값에서 비트 동일 위치를 주므로 정확 위치 quantize로 seam만 병합한다. 구멍을 닫지 않으며 서로 다른 geometry를 병합하지 않는다(테스트).
4. **Raycast = pixel-centre ray/triangle 교차의 z-buffer 구현.** Candidate B가 비교에 쓰는 것과 **동일한 반올림 픽셀 광선**이므로 두 depth가 같은 광선에서 측정된다. 원근 정확 보간(1/z 선형)을 썼다.
5. **Nearest-surface 거리 = 확장 링 탐색, 반경 3h에서 절단.** 그보다 먼 점은 숫자를 채우지 않고 `NO LOCAL EXTRACTED SURFACE`로 보고한다.
6. **합성 fixture의 depth map은 analytic이다** (canonical 커널 렌더링이 아니다). 이유: S2/S7의 "진짜로 증거가 없는 gap"은 analytic geometry로만 보장할 수 있다(WL120의 S2/S6에서 학습된 표현이 support를 퍼뜨림을 이미 관측). 시험 대상 코드 경로(`scale`/`field`/`extraction`)는 실제 장면과 바이트 단위로 같고 h도 같은 규칙으로 fixture별 유도한다.
7. **Baseline arm의 patch 샘플링.** 역사적 NURBS patch를 균일 24×24 UV 격자로 평가해 같은 신규 지표를 계산할 수 있게 했다. 적합 자체는 전혀 바꾸지 않는 **측정** 선택이다.
8. **Region label**은 WL108–123의 anchor 방식을 그대로 재사용한 **working interpretation**이며 구성 입력이 아니다.

---

## Candidate Scientific Contract

카메라 `v`와 유효 픽셀 `p`로 투영되는 world 점 `x`에 대해

```
z_v(x)   = frozen WL120-123 median-frontier classifier가 쓰는 것과 동일한 camera-space query depth
d_v(p)   = canonical stored renderer median depth
s_v(x)   = d_v(p) - z_v(x)
phi_v(x) = clamp(s_v(x) / mu, -1, +1)          # fusion 전용 정규화
authority(v, x)  <=>  v가 x를 질의할 수 있고  AND  |s_v(x)| <= mu
```

- `s_v > 0` = renderer median surface의 **카메라 쪽**, `s_v < 0` = **뒤쪽**. 이것은 물체 내부/외부가 **아니며** 이 field는 watertight object SDF가 **아니다.**
- authoritative 관측이 하나 이상인 voxel: `phi(x) = authoritative phi_v의 산술평균`, `support_count(x) = 기여 뷰 수`. 가중치는 전부 1이고 angle/opacity/confidence/visibility/normal/component/region 가중은 어디에도 없다.
- authoritative 관측이 없는 voxel: **UNKNOWN**. sparse store에 **부재**로 표현한다. `+1`, `-1`, `0`, 최근접값 중 어떤 것도 넣지 않고, 확산·구멍 채우기·Laplacian/Gaussian smoothing·morphological closing·Poisson fill·fast-marching·neural completion·signed-distance propagation을 전부 하지 않는다.
- cell 자격: 8 corner 전부 authority **그리고** 부호 변화. 결측 corner를 합성하지 않고 UNKNOWN cell을 통과 보간하지 않으며, 추출 후 mesh repair/hole filling/watertight 강제를 하지 않는다. 출력이 열려 있고 끊겨 있고 파편화되어도 그것이 근거 없는 visible surface를 발명하는 것보다 과학적으로 낫다.

**격리(통제 실험의 본체).** `evidence_bounded_tsdf/{scale,field,extraction,mesh_ops,synthetic}.py`는 역사적 topology/boundary/region/chart/KNN/NURBS/Trust/occluded 모듈을 **하나도 import 하지 않는다.** AST 기반 정적 테스트로 강제했다. `attribution.py`만이 메시 생성 **이후에** 그 값들을 읽기 전용으로 본다.

---

## Parameter Derivation

*(측정값은 아래 「Full Real-Scene Reconstruction」에 있다.)*

`h`는 렌더러의 샘플링률에서 한 번 유도되고 다시 선택되지 않는다. 모든 유효 median event에 대해 `footprint = stored_median_depth / sqrt(fx·fy)`이고, `fx`,`fy`는 그 카메라 자신의 FoV에서 graphdeco `focal2fov`의 정확한 역으로 복원한다. `h`는 그 양의 footprint들의 **전역 중앙값**, `mu = 3h`이다. `h`, `mu`, `mu/h` 중 어느 것도 sweep하지 않았고 결과를 본 뒤 다른 백분위로 교체하지 않았다.

---

<!-- MEASURED SECTIONS APPENDED BELOW -->

## Implementation-to-Intent Map

| Directive 요구 | 구현 위치 | 강제 수단 |
|---|---|---|
| §3 projective signed distance 부호 | `evidence_bounded_tsdf/field.py::projective_signed_distance` | `TestProjectiveSignedDistanceSign` |
| §3 truncation과 authority | `field.py::truncated_phi`, `view_authority` | `test_truncation_clamps_only_outside_the_band` |
| §3 UNKNOWN ≠ ±1 | `field.py::SparseProjectiveTSDF` (부재 = UNKNOWN), `lookup`이 NaN 반환 | `test_unknown_voxels_are_absent_never_filled` |
| §2/§3 frozen camera-depth 의미론 | `field.py::project_world_points`, `MIDDEPTH_OFFSET`, `CANONICAL_NEAR_N` | `TestFrozenCameraDepthSemantics` (frozen `project_queries`와 **비트 단위 동일** 검증) |
| §4 h 유도, sweep 없음 | `evidence_bounded_tsdf/scale.py` | `TestScaleDerivation` |
| §4 mu = 3h 사전 고정 | `scale.TRUNCATION_RATIO = 3.0` | 동일 |
| §5 균일 fusion, 최소 뷰 없음 | `field.py::fuse_views` | `test_fusion_weight_is_exactly_one_per_view`, `test_no_minimum_view_threshold_exists_in_the_source` (AST 구조 검사) |
| §5 support_count 회계 | `fuse_views` | `test_support_count_accounting_matches_manual_authority` |
| §6 sparse authority contract | `field.py` 전체 (확산/채우기 코드 없음) | 모듈 docstring + UNKNOWN 테스트 |
| §7 masked cell 추출 | `evidence_bounded_tsdf/extraction.py` | `TestMaskedCellExtraction` 5종 |
| §7 UNKNOWN cell에서 삼각형 없음 | `extract_zero_level_set`의 cell 필터 | `test_no_triangle_comes_from_a_cell_with_an_unknown_corner`, `test_sentinel_choice_cannot_change_kept_triangles` |
| §8 구성 중 역사적 topology 미사용 | 모듈 분리 | `TestConstructionIsolation` (AST import 검사 4모듈 + 소스 토큰 검사) |
| §9 S1–S7 | `evidence_bounded_tsdf/synthetic.py` | `TestSyntheticContracts` |
| §12/§13 증거 재현·raycast | `mesh_ops.py::nearest_surface_distance`, `rasterize_mesh_depth` | `TestMeshOps` (해석적 정답 대조) |
| §16 100% authoritative cell | 추출 계약 자체 | `hallucination_audit.fraction_of_triangles_from_fully_authoritative_cells` |
| §17 mesh 유도 occlusion | `attribution.py::mesh_occlusion_for_view` (epsilon 없음) | 리포트 `sdf_induced_occlusion_audit` |
| §23 결정론 | `field.py` 전체가 RNG 없음 | `TestDeterminism` (chunk-size 불변, 재실행 동일) |


---

## Synthetic Contracts

전부 실제 장면 결과를 보기 **전에** 작성했고, 시험 대상 코드 경로(`scale`/`field`/`extraction`)는 실제 장면과 동일하다. h는 fixture마다 같은 규칙으로 유도된다.

| Fixture | 핵심 계약 | 결과 |
|---|---|---|
| **S1** 단일 열린 평면 패치 | 평면 근처 zero surface, **열린 채 유지**, cap 없음, 지지 footprint 밖 연장 없음 | **PASS** — 면적 1.4299 / GT 1.44 (99.30%), point-to-surface median **0.0068h** p95 0.023h, coverage(≤h) 99.995%, 표면의 max\|z\| = 0.024μ, x/y 경계 초과 **−0.13h**(오히려 안쪽), **cap 면적 비율 0.000**, component 1 |
| **S2** 미지지 gap을 둔 두 동일평면 패치 (**STOP**) | gap을 가로지르는 삼각형 0, 중간에 zero surface 없음 | **PASS** — gap 반폭 = **13.15h = 4.38μ**, **gap-bridging 삼각형 0**, 면적 0.0, component 2, coverage(≤h) 100% |
| **S3** 곡면 열린 시트 | 곡률 복원, 시트를 닫지 않음 | **PASS** — 반경오차 median **0.0293h** p95 0.130h, **호 밖 삼각형 0**, **반대 반구 삼각형 0**, 각도 초과 0.0h, 최대 component 99.97% |
| **S4** 서로 다른 두 깊이 층 | 둘 다 복원, 층간 연결 시트 없음 | **PASS** — front 8,692 / rear 3,142 삼각형, **연결 시트 0**, 층 간격 = 11.69μ, 직접 관측 GT 기준 coverage(≤h) **100%** |
| **S5** 교차뷰 disocclusion | 직접 증거가 있는 곳의 후면을 전역 복원 | **PASS** — blocker 뒤 rear 삼각형 **4,118개**, coverage(≤h) 99.99% |
| **S6** 얇은 구조 | 고정 h가 보존하는지 **보고**(튜닝 금지) | **보고** — 1h/2h/4h/8h 네 기둥 **전부 보존**. 다만 관측가능 GT의 point-to-surface median 0.576h p95 1.42h로 다른 fixture보다 뚜렷이 나쁘고, 재구성 두께가 공칭보다 두꺼움(예: 공칭 1.01h → 재구성 x-범위 3.1h). component 19개로 파편화. **해상도를 바꾸지 않았다.** |
| **S7** 진짜 occluded gap (**STOP**) | 어떤 뷰에서도 직접 median 증거가 없는 구간을 표면으로 잇지 않음 | **PASS** — fixture가 스스로 검증(**strip probe 121개 중 직접 관측 0개**), **gap 통과 삼각형 0** |

모든 fixture에서 `unsupported_triangle_count = 0`, `unsupported_surface_area = 0.0`.

**S7 fixture 결함 1건 공개**: 최초 S7은 occluder가 실제로는 가리지 못해 rear strip의 61개 표본 중 **34개가 직접 관측**되고 있었고, 그 상태에서 gap-bridging 2,600개가 나왔다. 이는 후보의 실패가 아니라 **fixture 설계 오류**였다. occluder 반폭을 0.3→0.62로, 카메라 arc를 0.22→0.10으로 고쳐 "never observed"를 성립시키고, **fixture가 스스로 그 전제를 검증**(`never_observed_samples_verified`)하도록 만든 뒤 재측정해 bridging 0을 얻었다. 이 검증 단계가 없었다면 잘못된 실패를 보고할 뻔했다.

---

## Full Real-Scene Reconstruction

체크포인트/렌더러/카메라 무변경 확인: valid median event **43,817,760**(WL122와 정확히 일치), median representative 합집합 **785,937**(WL119와 정확히 일치).

### Parameter Derivation (실측)

| 항목 | 값 |
|---|---|
| footprint min / p05 / p25 / **median** / p75 / p95 / max | 0.000600 / 0.006864 / 0.008807 / **0.012105** / 0.019641 / 0.030038 / 2.928914 |
| **h (canonical voxel size)** | **0.012105485424399376** |
| **mu = 3h** | **0.036316456273198128** |
| footprint > 2h / 3h / 4h / 5h / 7h / 9h 비율 | 14.51% / 1.210% / 0.0183% / 0.0090% / 0.0055% / 0.0004% |

`h`는 전역 중앙값 한 번으로 확정했고 sweep·재선택이 없다. footprint 분포의 극단 꼬리(최대 2.93 = 242h, depth 1,412 world unit의 하늘/스침 교차)가 존재하지만 중앙값 통계이므로 h에 영향을 주지 않는다.

### Field

| 항목 | 값 |
|---|---|
| seed voxel (renderer event voxel) | 10,614,923 (표현 범위 밖 2개) |
| **authoritative voxel** | **76,720,314** |
| support_count 평균 / 중앙값 / p95 / 최대 | 6.620 / 3 / 27 / 158 |
| support_count = 1 | 27,328,305 (**35.62%**) |
| support_count = 2 / 3 / 4 / 5 | 13.82% / 8.46% / 6.12% / 4.74% |
| 열거 종료 | **닫히지 않음** (60 라운드 상한 도달) |

**닫히지 않은 것에 대한 정직한 보고.** closure 성장은 라운드 3에서 이미 75,409,323(최종의 98.3%)에 도달하고, 라운드 4 이후는 라운드당 전체의 **약 0.014%**씩만 늘어난다. 이 잔여 성장은 **depth 176~1,412 world unit의 극단 far-field median event**(전체 event의 0.018%)에서 나온다 — 그곳은 pixel footprint가 h의 수천 배라 truncation band가 단일 뷰·support_count 1의 매우 넓은 판이 되어 매 라운드 한 겹씩 번진다. 장면 본체는 이미 수렴했다(voxel의 99.26%가 원점 50 world unit 이내, p99 = 20.4). 중요한 것은 방향성이다: **열거가 부족하면 표면이 누락될 뿐, 표면이 발명되지는 않는다.** 얻어진 field는 참 authoritative 집합의 진부분집합이다.

### Extraction

| 항목 | 값 |
|---|---|
| 8 corner 모두 authoritative인 cell | 44,341,747 |
| **자격 cell**(8 corner + 부호 변화) | **21,235,312** |
| 자격 없는 cell에서 나와 **폐기**된 삼각형 | 89,738,359 |
| 자기 cell 밖에 놓인 채 유지된 삼각형 | 52 (전체의 1.2e-6) |
| 처리 블록 | 22,758 후보 중 7,567개에 자격 cell 존재 |
| **정점 / 삼각형** | **28,694,040 / 45,116,659** |
| seam weld로 병합된 중복 정점 | 695,456 |
| **연결 성분** | **582,646** |
| **총 표면적** | **2117.509** |
| 삼각형 면적 / h² 중앙값 | 0.3160 |
| 정점 support: 평균 / 중앙값 / p95 / 최대 | 8.153 / 4 / 33 / 156 |
| support = 1 정점 | 6,134,496 (21.38%) |

hole filling·repair·smoothing·decimation·watertight 강제를 **하지 않았다.**

### Renderer-Evidence Reproduction (전수 43,817,760 event)

| 항목 | 값 |
|---|---|
| **거리 ≤ h** | **89.835%** |
| **거리 ≤ 2h** | **98.455%** |
| 3h 이내에 추출 표면 **없음** | **0.163%** |
| 거리/h: median / p95 / p99 / max | 0.2477 / 1.3625 / 2.1544 / 6.3768 |

영역별 (≤h / ≤2h / event 수):

| 영역 | ≤h | ≤2h | events |
|---|---|---|---|
| table_top | 92.61% | 99.04% | 4,002,173 |
| table_side_curved | 91.47% | 98.73% | 5,218,536 |
| table_legs | 88.42% | 98.02% | 7,303,476 |
| patio | 90.07% | 98.50% | 21,108,132 |
| hedge/background | 87.51% | 98.22% | 6,185,443 |

h와 2h는 고정 해상도에서 유도된 **보고 구간**이며 적합 임계가 아니다.

### Raycast Self-Consistency (161 카메라 전수, pixel-centre ray)

| 항목 | 값 |
|---|---|
| canonical median depth를 가진 픽셀 | 43,817,760 |
| mesh first-hit이 있는 픽셀 | 43,765,357 |
| **ray-hit coverage** | **99.880%** |
| \|depth 오차\|/h: median / p95 / p99 / max | **0.4983** / 12.32 / 202.1 / 5.01e6 |
| signed 오차/h: median / p95 | −0.0832 / +1.6315 |
| \|오차\| ≤ h / ≤ 2h | 67.27% / 79.36% |
| 라스터화 삼각형 / 절두체 밖 / 최대 tier 초과 | 234,881,293 / 2,480,853,826 / **0** |

영역별 \|오차\|/h (median, p95):

| 영역 | median | p95 | pixels |
|---|---|---|---|
| table_top | 0.4000 | 7.91 | 4,002,087 |
| table_side_curved | 0.4555 | 4.58 | 5,210,414 |
| table_legs | 0.5344 | 10.77 | 7,286,170 |
| patio | 0.4776 | 15.04 | 21,097,310 |
| hedge/background | 0.6982 | 14.78 | 6,169,376 |

**꼬리의 귀속(측정)**: 오차가 큰 픽셀에서 메시는 압도적으로 frontier보다 **앞**에 있다 — \|오차\|>5h인 5,134,886픽셀(11.73%)의 **91.6%**, >10h의 92.1%가 mesh-in-front다. 즉 median depth를 재현하지 못한 픽셀의 대부분은 "표면이 없다"가 아니라 **다른 뷰에서 관측된 표면이 이 뷰의 시선을 먼저 가로막는다**는 뜻이다. 이는 재구성이 실제 3D 차폐 geometry로 작동한다는 직접 증거이자, 동시에 얇은 구조(잎·격자)에서 뷰마다 다른 표면이 앞서는 2DGS 특유의 성질이다. **이 관측만으로 "옳다/그르다"를 결론짓지 않는다** — §SDF-Induced Occlusion Audit이 이를 frozen Candidate B와 대조한다.

---

## Hallucination / Unsupported-Gap Audit

| 항목 | 값 |
|---|---|
| **8 corner 모두 authoritative인 cell에서 나온 삼각형 비율** | **100.000%** (계약에 의해, 자격 없는 cell의 삼각형 89,738,359개 전량 폐기) |
| 자기 cell 밖에 놓인 채 유지된 삼각형 | 52 / 45,116,659 (1.2e-6) |
| min 정점 support = 1인 삼각형 | 9,397,223 (**20.83%**), 면적 비중 **22.36%** |
| support = 1 정점 | 6,134,496 (21.38%) |
| 3h 이내에 표면이 없는 median event | 0.163% |

`support_count = 1` 표면은 **삭제하지 않고 그대로 export**했다(`mesh/tsdf_low_support_candidates.ply`, view `TSDF_LOW_SUPPORT_SURFACE`). 이 21%가 실제 얇은 구조(잎·테이블 다리·격자)인지 근거 없는 다리인지는 **사람이 판단할 문제**이며, directive가 금지한 사후 support 임계를 도입해 지우지 않았다.


---

## Qualitative Review Exports

directive §18이 요구한 10개 view를 전부 같은 6개 대표 학습 시점(`preview_png/`에 `<VIEW>__<카메라>.png` 형식)으로 내보냈다. 각 view 폴더에는 색상 의미를 적은 **자체 한국어 README.md**가 있다(스크립트가 직접 작성하므로 누락될 수 없다).

| | View | 내용 |
|---|---|---|
| A | `ORIGINAL_2DGS` | 학습된 SH 색상 그대로의 기준 장면 |
| B | `RENDERER_MEDIAN_SURFACE_POINTS` | 후보가 소비하는 **유일한 증거**(경쟁 아키텍처가 아니라 증거 기준선) |
| C | `NEW_TSDF_VISIBLE_SURFACE` | 추출된 후보 표면 |
| D | `HISTORICAL_VISIBLE_NURBS_BASELINE` | A/B의 arm A(역사적 경로 재생) |
| E | `TSDF_RAYCAST_DEPTH` | first mesh-hit depth. **no-hit은 자유 공간이 아니라 어두운 자홍**으로 구분 |
| F | `MEDIAN_VS_TSDF_DEPTH_ERROR` | signed 오차(빨강=뒤, 파랑=앞), \|오차\|/(2h)로 정규화 |
| G | `TSDF_SUPPORT_COUNT` | 정점별 support_count(빨강=1뷰 → 청록=다수뷰) |
| H | `TSDF_LOW_SUPPORT_SURFACE` | support ≤ 1 표면. **삭제하지 않고 검토용으로 보존** |
| I | `B_VS_TSDF_OCCLUSION_DISAGREEMENT` | frozen Candidate B와 mesh 유도 판정의 불일치 |
| J | `WL121_FRAGMENTATION_CONTEXT_OVERLAY` | WL121의 300개 true-fragmentation context overlay |

추가로 `mesh/`에 **TSDF mesh PLY**, **support-count 색상 PLY**, **low-support/hallucination 후보 PLY** 3종(자체 README 포함), 그리고 `TSDF_FIELD_SLICES/`에 table 구조 / patio / hedge를 지나는 **3개 world-space field slice**를 냈다.

Slice의 색상 규약이 sparse authority contract를 눈으로 검증한다: **파랑 = authority 있고 phi > 0(카메라 쪽)**, **주황/빨강 = authority 있고 phi < 0(뒤쪽)**, **어두운 보라 = UNKNOWN**, **흰색 = zero crossing**. 실제 slice에서 값이 있는 영역은 표면을 감싼 얇은 띠뿐이고 **나머지 전부가 UNKNOWN**으로, 자유 공간처럼 그려지지 않는다.

---

## Qualitative Review Case Table

`review_case_table.json`에 영역별 10건씩(table_top / table_side_curved / table_legs / patio / hedge) 구체적 provenance를 담은 기계 판독 표를 냈다. 각 항목은 world position, source view/pixel, representative id, SDF 값과 authority 여부, support_count, 최근접 추출 표면 거리(및 /h), source view에서의 mesh first-hit depth와 renderer median depth의 차를 포함한다. 외형만으로 판단하지 않도록, 같은 좌표에 대한 Candidate B 상태와 역사적 component는 리포트의 `sdf_induced_occlusion_audit` / `historical_topology_attribution` 절에 함께 둔다.


---

## Baseline A/B

**Arm A 재생 충실성 (무수정 재생, 성공)** — 역사적 topology와 chart 구성이 정확히 재현됐다:

| 항목 | 재생 | 역사적 기준 |
|---|---|---|
| visible component 수 | **559,989** | WL107/109 **559,989** ✔ |
| singleton surfel | **535,910** | WL107/109 **535,910** ✔ |
| largest component 비율 | **0.3677130609868884** | WL119 **0.3677130609868884** ✔ |
| fitted chart 수 | **14,900** | WL119 **14,900** ✔ |
| per-chart residual 중앙값 | **0.004575** | WL119 metric-G **0.004615** (같은 자릿수) ✔ |

fitter 용량(8×4, degree 2/2, correction round 2)과 chart 구성은 WL119 리포트 기록 그대로이며 **비교를 유리하게 만들기 위한 수정은 없다**. 적합된 patch를 균일 24×24 UV 격자로 평가해 8,582,400 정점 / 15,764,200 삼각형의 비교 가능한 geometry를 얻었다(측정 선택이며 적합 자체는 불변).

**공통 질의 집합**: 모든 뷰의 유효 median event에 결정론적 stride 37 → **1,184,316 event**, 양 arm 동일.

| 지표 | **Arm A** (역사적 topology/boundary-first + NURBS) | **Arm B** (evidence-bounded TSDF) |
|---|---|---|
| renderer-evidence coverage ≤h | 58.472% | **89.842%** |
| renderer-evidence coverage ≤2h | 68.200% | **98.486%** |
| 기하 residual/h (median / p95) | 0.3585 / 2.4199 | **0.2479 / 1.3603** |
| ray-hit coverage | **99.996%** | 99.880% |
| \|depth 오차\|/h (median / p95) | 161.95 / 821.37 | **0.4983 / 12.32** |
| 기하 조각 수 | 14,900 fitted patch | 582,646 mesh component |
| 표면적 | 39,942.4 | **2,117.5** |
| **미지지 bridge**(어떤 event로부터도 >2h) | **26.424%** (p95 17.14h, max 17,680h) | **11.899%** (p95 2.77h, max 448h) |

영역별 coverage ≤h:

| 영역 | Arm A | Arm B | events |
|---|---|---|---|
| table_top | 86.00% | **92.58%** | 108,143 |
| table_side_curved | 84.63% | **91.63%** | 141,086 |
| table_legs | 64.40% | **88.35%** | 197,473 |
| patio | 58.49% | **90.08%** | 570,378 |
| hedge/background | 11.53% | **87.52%** | 167,236 |

**해석 가드를 지킨 독해.** directive는 "coverage가 크다고 성공이 아니며, 미지지 채움에서 온 coverage는 성공이 아니다"라고 못박는다. 그래서 coverage와 bridge를 **함께** 읽어야 한다: arm B는 coverage가 1.5배 높으면서 **동시에** 미지지 bridge 비율이 arm A의 절반 이하(11.90% vs 26.42%)이고 그 꼬리도 훨씬 짧다(p95 2.77h vs 17.14h). 즉 arm B의 우위는 채움으로 얻은 것이 **아니다**. 마찬가지로 "조각 수가 적다고 성공이 아니다" — arm B의 582,646 component는 arm A의 14,900 patch보다 훨씬 파편적이며, 이는 arm B가 잘한 점이 아니라 **열린 채로 두었다는 사실의 표현**이다.

arm A의 ray-hit coverage가 99.996%로 더 높은 것은 우위가 아니다. depth 오차 중앙값이 **161.95h**이고 표면적이 arm B의 **19배**(39,942 vs 2,118)라는 사실이 그 이유를 설명한다 — 적합된 NURBS patch가 지지 UV 도메인 밖으로 크게 뻗어 카메라 앞을 잘못 가린다. 시각 산출물(`HISTORICAL_VISIBLE_NURBS_BASELINE` view)에서도 장면 구조를 알아볼 수 없는 흩어진 patch 조각으로 나타난다. 이는 arm A를 폄하하는 관찰이 아니라 WL111–114가 이미 반복 보고한 성질(component 0% coverage, 거대 blob, 미분할 chart)의 기하학적 귀결이다.

**증거 기준선.** raw renderer median surface point cloud는 `RENDERER_MEDIAN_SURFACE_POINTS`로 함께 냈으며, **경쟁 아키텍처가 아니라 두 arm이 공유하는 증거**로만 쓰였다.

---

## SDF-Induced Occlusion Audit

재구성된 메시를 **실제 차폐 geometry**로 사용해(카메라→질의 광선의 first mesh-hit이 질의보다 엄밀히 앞이면 `MESH_OCCLUDED`, **depth epsilon 없음**) frozen Candidate B와 대조했다. Candidate B는 수정하지 않았다.

| 질의 코퍼스 | B_OCC/mesh_OCC | B_OCC/mesh_free | B_OBS/mesh_OCC | B_OBS/mesh_free |
|---|---|---|---|---|
| WL120 원본 4,712 | **646** | **8** | 719 | 3,335 |
| WL121 보충 908 | **14** | **0** | 362 | 524 |
| WL123 generic 21,652 | **3,816** | **53** | 2,848 | 14,923 |

**가장 강한 신호는 한쪽 방향의 거의 완전한 일치다**: B가 OCCLUDED로 판정한 것의 **98.78%(646/654)**를 메시도 독립적으로 OCCLUDED로 본다(WL121은 14/14 = 100%, WL123은 98.63%). B의 occluded domain은 재구성된 3D 표면으로 **기하학적으로 재현된다**.

불일치는 압도적으로 반대 방향(B=OBSERVED인데 mesh=OCCLUDED)이며, WL120에서 727건이다. 그 귀속:

| 귀속 | 건수 |
|---|---|
| 3h 이내에 재구성 표면이 **존재** | **717 (98.62%)** |
| 질의가 authoritative voxel **안**에 있음 | 696 (95.74%) |
| 질의가 UNKNOWN 공간에 있음 | 31 |
| 3h 이내에 표면 **없음**(재구성 누락) | **10 (1.38%)** |
| 최근접 표면 거리/h: median / p95 / max | 0.762 / 2.13 / 4.93 |

즉 이 불일치의 거의 전부는 "없는 표면을 발명해서 가렸다"가 **아니라** "실제로 재구성된 표면이 그 시선을 가로막는다"이다. 이는 §Raycast의 꼬리 귀속(오차가 큰 픽셀의 91.6%가 mesh-in-front)과 **독립적으로 같은 결론**에 도달한다. 어느 쪽이 물리적으로 옳은지 — 얇은 잎/격자 뒤를 B가 OBSERVED로 보는 것이 맞는지, 메시가 가리는 것이 맞는지 — 는 이 배치가 **판정하지 않는다**. directive가 요구한 대로 이것은 진단이며, **Candidate B를 자동으로 대체하지 않는다.**

---

## Historical Topology Attribution (진단 전용)

WL121의 300개 true-fragmentation context를 재생했다(gating 귀속 288/12로 WL121과 동일).

| 항목 | 건수 |
|---|---|
| endpoint A에 h 이내 표면 | 221 / 300 |
| endpoint B에 h 이내 표면 | 237 / 300 |
| midpoint에 h 이내 표면 | 168 / 300 |
| midpoint에 3h 이내 표면 **없음** | 46 / 300 |
| 두 endpoint가 **같은** 추출 mesh component | 173 / 300 |
| 두 endpoint가 **다른** component | 127 / 300 |

**해석 가드**: 같은 component에 있다고 해서 물리적 연속성이 옳다는 뜻이 아니고, midpoint에 표면이 있다고 해서 역사적 분할이 틀렸다는 뜻도 아니다. 역사적 component identity로 SDF를 바꾸지 않았다. 유효한 독해는 "역사적으로 끊긴 300쌍 중 **173쌍(57.7%)**이 새 구성에서는 하나의 표면 조각 위에 놓이고, **46쌍(15.3%)**은 중점 부근에 표면이 아예 없다"는 사실 보고까지다.

---

## Conditional NURBS Handoff

게이트 충족(합성 미지지-gap 계약 통과 + 실제 메시의 renderer-evidence 89.8% / ray-hit 99.9%)으로 **실행**했다. 기존 NURBS fitter를 그대로 쓰고, 기존 `pca_parameterize_points`(역사적 boundary 아키텍처를 되살리지 않는 가장 단순한 기존 parameterization)만 사용했으며, crop은 **공간 ROI(anchor 주변 ±12h 상자)로만** 골랐다 — boundary/region/chart eligibility를 쓰지 않았다.

| Crop | 점 수 | residual median | (/h) | p95 | max | 유한 normal |
|---|---|---|---|---|---|---|
| table_top | 990 | 0.001562 | **0.129h** | 0.011217 | 0.0426 | 100% |
| table_side_curved | 1,108 | 0.009487 | 0.784h | 0.039820 | 0.0567 | 100% |
| patio | 1,435 | 0.008801 | 0.727h | 0.032715 | 0.0513 | 100% |
| table_legs (thin) | 1,517 | 0.012954 | 1.070h | 0.044129 | 0.0656 | 100% |

**네 crop 전부 적합 성공**하고 normal이 전부 유한하다. residual은 평면(table_top) 0.13h에서 얇은 구조(table_legs) 1.07h까지로, 곡면/얇은 구조일수록 나빠지는 예상된 순서다. 질문은 오직 "재구성된 implicit surface가 downstream NURBS 적합의 기하 증거가 될 수 있는가"였고, 답은 **된다**이다.

---

## Implementation Fidelity Statement

**PROMPT-REQUIRED DECISION**과 **AGENT-INTRODUCED OPERATIONAL CHOICE**는 위 「Agent Interpretation of Intent」에 분리해 기록했고, 조작적 선택 8건은 모두 증거를 쓰기 **전에** 공개했다.

**실행 중 발견·수정한 실제 결함 3건 (전부 측정 보고 전):**

1. **S7 fixture가 실제로 가리지 못했다.** occluder 반폭이 부족해 rear strip 61개 표본 중 **34개가 직접 관측**되고 있었고, 그 상태에서 gap-bridging 2,600개가 나왔다. 이는 후보의 실패가 아니라 fixture 설계 오류였다. 기하를 고치고 **fixture가 "never observed" 전제를 스스로 검증**(`never_observed_samples_verified`, strip probe 121/121 미관측)하게 만든 뒤 bridging 0을 얻었다. 이 자체 검증이 없었다면 후보의 거짓 실패를 보고할 뻔했다.
2. **closure 열거의 비용 결함.** 매 라운드 전체 field의 껍질을 검사하던 것을 **직전 라운드 신규분의 껍질만** 검사하도록 바꿨다. 거부는 영구적이므로(field 정의가 변하지 않음) 같은 고정점에 도달하며, 합성 fixture에서 keys/values/support가 **비트 단위 동일**하고 22.7배 빠름을 확인해 회귀 테스트로 고정했다. 실제 장면에서도 라운드별 증가분이 이전 실행과 정확히 일치(+44,072,783 / +17,969,869 / +2,785,257 / +307,385)하고 최종 authoritative가 **76,720,314로 동일**했다.
3. **baseline 재생의 shape 결함.** `accumulate_image_space_pairs`에 평탄화된 representative map을 넘겨 크래시했다. (H,W)로 고친 뒤 topology가 canonical과 정확히 일치함을 확인했다.

**INABILITY TO REALIZE REQUESTED DIAGNOSTIC — 1건.**
directive §6/§7의 authoritative 집합 열거가 **고정점으로 닫히지 않았다**(60 라운드 상한). 잔여 성장은 라운드당 전체의 ~0.014%이고 depth 176–1,412 world unit의 극단 far-field event(전체 event의 0.018%)에 국한되며, 장면 본체는 라운드 3에서 이미 최종의 98.3%에 수렴했다. **다른 양으로 몰래 대체하지 않았다**: 보고한 field는 참 authoritative 집합의 **진부분집합**이고, 방향성이 명확하다 — 열거 부족은 표면을 **누락**시킬 뿐 **발명하지 않는다**. 따라서 이 한계는 §Hallucination의 결론을 약화시키지 않고, §Evidence coverage(89.84%)를 **과소평가** 쪽으로만 편향시킨다.

**변경하지 않은 것**: canonical renderer, 체크포인트, 161 카메라, Candidate B 결정 함수, `aggregate_global`, WL107/109 topology 모듈, 역사적 NURBS fitter와 그 용량. 추적 대상 production 코드 수정 0건. 신규 focused 테스트 **48개 전부 통과**.

**격리 검증**: `scale`/`field`/`extraction`/`mesh_ops`/`synthetic` 5개 구성 모듈이 역사적 topology/boundary/region/chart/KNN/NURBS/Trust/occluded 모듈을 하나도 import하지 않음을 AST 정적 테스트로 강제했고, `field.project_world_points`가 frozen `shared.project_queries`와 **비트 단위 동일**함을 실측 검증했다.

**런타임/자원**: 최종 실행 48.0분(field/mesh/evidence/raycast 캐시 재사용, baseline arm 전체 재생 포함). 캐시 없는 전체 구성은 fusion ~2.5h + extraction ~1h가 추가된다. peak CPU RSS 48.9 GiB, peak GPU allocated 30.9 GiB(reserved 61.8 GiB, 다단계 합산). 성능을 위해 의미론을 바꾸지 않았다.

---

## Architecture Verdict

### **B. IMPLICIT VISIBLE SURFACE GEOMETRY IS VIABLE, BUT CURRENT FIXED PROJECTIVE TSDF IS INSUFFICIENT AS THE CANONICAL CONSTRUCTION**

**A를 선택하지 않은 이유.** directive는 A에 대해 두 조건을 **동시에** 요구한다: (i) 쓸모 있는 실제 장면 visible geometry, (ii) 미지지/occluded gap 표면 조작이 **material하지 않을 것**. (i)은 강하게 충족된다 — 전수 43.8M event의 89.84%가 h 이내, ray-hit 99.88%, 모든 영역에서 arm A를 크게 앞서고, 시각적으로 장면 구조를 그대로 재현하며, downstream NURBS 적합의 증거로 바로 쓰인다. 그러나 (ii)를 "material하지 않다"고 말할 수 없다:

- 메시 표면점 표본의 **11.90%가 어떤 renderer median event로부터도 2h보다 멀다**(최대 448h). arm A(26.42%)보다 훨씬 낫지만 절대적으로 작은 값이 아니다.
- 삼각형의 **20.83%(면적의 22.36%)가 단일 뷰 support**에 서 있다. 이것이 진짜 얇은 구조인지 근거 없는 다리인지 이 배치는 **판정하지 않았고**, directive에 따라 삭제하지도 않았다.
- 열거가 고정점으로 닫히지 않았다(위 INABILITY 참조).

세 가지 모두 "발명했다"는 증거는 **아니다** — 오히려 계약 수준에서는 삼각형의 100%가 8-corner authoritative cell에서 나왔고, 합성 STOP 계약(S2/S7)은 미지지 gap을 한 번도 건너지 않았으며, occlusion 불일치의 98.62%는 실재하는 표면으로 설명된다. 하지만 **"material하지 않음"을 실제 장면에서 적극적으로 입증하지도 못했다.** A는 그 입증을 요구한다.

**C가 아닌 이유.** 후보는 실패하지 않았다. 역사적 경로가 재현하지 못한 것(hedge coverage 11.5% → 87.5%, patio 58.5% → 90.1%)을 topology·boundary·region·chart를 **하나도 풀지 않고** 재현했고, B의 occluded domain을 98.8% 기하학적으로 재현했으며, S1–S7 의미 계약을 전부 통과했다.

**D가 아닌 이유.** 규모 한계가 결론을 좌우하지 않았다. 전 단계가 161뷰 전수로 실행됐고(43.8M event 전수 평가), 두 번의 독립 실행에서 field·mesh·coverage·ray-hit이 동일하게 재현됐으며, baseline arm은 canonical topology와 chart 수를 정확히 일치시켰다. 유일한 미종결(closure)은 방향이 확정된 편향(누락 쪽)이라 verdict를 뒤집지 않는다.

**B의 실질**: implicit·evidence-bounded 구성 전제 자체는 **작동한다.** 부족한 것은 *고정된* projective TSDF라는 구체적 형태다 — 단일 전역 h가 근거리(footprint 0.05h)와 원거리(242h)를 동시에 감당하지 못해 far-field 열거가 닫히지 않고 support=1 표면이 21%까지 남으며, μ=3h가 얇은 구조에서 양면 밴드를 겹치게 만든다(S6에서 재구성 두께가 공칭의 3배).

### SECONDARY NURBS VERDICT: **PROMISING**

네 crop(평면/곡면/patio/얇은 구조) 전부 기존 fitter로 적합 성공, residual median 0.13h–1.07h, normal 100% 유한. 역사적 boundary 아키텍처 없이 공간 ROI만으로 crop을 골랐는데도 동작했다. 이 판정은 **1차 SDF 판정을 결정하지 않으며**, 그 역도 마찬가지다.

---

## Final Report

```
HISTORICAL PREMISE
    renderer evidence -> locality/topology -> region/component
                      -> boundary/chart -> NURBS surface

CANDIDATE PREMISE
    renderer evidence -> evidence-bounded projective TSDF
                      -> visible geometry -> later patching/parameterization -> NURBS
```

**완료 조건에 대한 답:**

1. *"역사적 region/topology/boundary 아키텍처를 먼저 풀지 않고 renderer-native median 관측을 직접 융합해 장면을 덮는 Visible Surface를 만들 수 있는가?"* → **그렇다.** topology·KNN·region·boundary·chart를 **하나도** 쓰지 않고 89.84% evidence coverage / 99.88% ray-hit / 모든 영역 87% 이상을 얻었다(역사적 경로: 58.47%, hedge 11.53%).
2. *"미지지/occluded gap을 통해 표면을 조작하지 않고 엄격히 증거에 갇혀 있는가?"* → **계약 수준에서는 그렇다**(삼각형 100%가 8-corner authoritative cell 출신, 합성 STOP 계약 전부 통과, UNKNOWN 무충전). **실제 장면 수준에서는 미확정**(표본의 11.90%가 >2h, 삼각형 20.83%가 단일 뷰 support). 이 미확정이 A 대신 B를 선택하게 한 결정적 이유다.
3. *"downstream NURBS의 기하 scaffold가 될 만큼 실제 renderer 표면 기하를 재현하는가?"* → **그렇다.** 4/4 crop 적합 성공.
4. *"실제 차폐 geometry로 썼을 때 frozen Candidate B 대비 그럴듯한 Observed/Occluded 분할을 주는가?"* → **그렇다, 한 방향으로 강하게.** B=OCCLUDED의 98.78%를 메시가 독립 재현한다. 반대 방향 불일치(727건)는 98.62%가 실재 표면으로 설명되지만 **어느 쪽이 물리적으로 옳은지는 판정하지 않았다.**

### Architecture-deletion candidates (목록일 뿐, 이 배치에서 제거하지 않음)

Candidate A가 승인**되지 않았으므로** 아래는 삭제 대상이 아니라 **후속 검증 대상**이다. Visible Surface *구성*에 한해, 이번 실험이 없이도 동작함을 보인 것들:

- 구성 이전 단계의 **visible topology 해결**(WL107/109 component/singleton 분해)
- 구성 이전 단계의 **KNN candidate graph / 3D-locality gate / secondary geometric gate**
- 구성 이전 단계의 **region 배정**과 **boundary loop / boundary recovery**
- 구성 이전 단계의 **chart eligibility와 camera-observed chart domain**

이들은 여전히 **구성 이후** 진단·귀속·의미 분할에 유효하며(§Historical Topology Attribution이 그렇게 사용했다), NURBS는 downstream 표현으로 그대로 남는다. **어느 것도 이 배치에서 제거하지 않았다.**

STOP — verdict 이후 topology 삭제, 저장소 리팩터, canonical SDF 채택, Candidate B 대체, Occluded Surface 구성, occluded NURBS로 자동 진행하지 않는다.
