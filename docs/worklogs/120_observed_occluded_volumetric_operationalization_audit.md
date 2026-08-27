# Worklog 120 -- Observed / Occluded Volumetric Operationalization Audit

브랜치: `arch/2dgs-coverage-first-surface`

---

## 1. Agent Interpretation of Intent

**DIRECTION**: camera-supported 3D 재구성 도메인을 OBSERVED / OCCLUDED로 분할하는 **네 개의 독립적인 경쟁 아키텍처 가설**(A. Surface-Hit, B. Median-Depth, C. Geometric Visibility, D. Renderer Reachability)을 구현하고 실험적으로 비교한다. UNRESOLVED는 fail-closed 구현 상태로만 허용된다. 이 넷은 하나의 구현의 네 가지 튜너블 변형이 아니다 -- 각각 별도 모듈, 별도 의미론, 별도 결정 함수를 가진다.

**PURPOSE**: "OBSERVED = 직접 렌더링/광도 감독이 가능하다 / OCCLUDED = 관련 학습 뷰가 질의 영역에 도달하기 전에 차단되어 직접 감독이 불가능하다"는 **논문 수준의 구분**을 이 넷 중 무엇이 지지할 수 있는지를 판정한다. SPARSELY_OBSERVED 재구성 상태는 도입하지 않는다. 관측 다중도(multiplicity)는 진단 메타데이터로만 기록하고 canonical policy branch로 승격하지 않는다.

**CENTRAL INTENT**: 질문은 "어느 구현이 가장 예쁜 숫자를 내는가"가 **아니다**. "현재 canonical 2DGS 증거만으로, 순환 의존·의미론적 모순·증거 기아 없이 의도한 Observed/Occluded 부피 구분을 표현할 수 있는 operationalization이 있는가(없을 수도 있다)"이다. 하나 또는 전부에 대한 부정적 결과도 유효한 결과이며, 아키텍처 성공을 자동으로 주장하지 않는다.

### PRESERVE
- 동일한 학습 모델(`output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt`, 1,190,469 surfel, uncertain 0개), 동일한 161 학습 카메라 집합.
- WL107/109 visible topology -- **코드를 한 줄도 건드리지 않았다**.
- Canonical production renderer(`osn_gs/render/vendor/diff_surfel_rasterization/`), `OSNSurfelRasterizer`, 학습 경로 -- 무수정.
- WL107 진단 sibling(`diff_surfel_rasterization_diag`) -- **무수정**. WL107/109/110/112-119 재생이 비트 단위로 그대로 유지되도록, 이번 배치의 CUDA 추가는 **세 번째 별도 sibling 패키지**로 만들었다.
- WL119 geometry 진단(G2 = median-surfel local intersection 재구성) -- 그대로 재사용.
- 동일한 query bank / 동일한 global aggregation / 동일한 reporting metric을 네 후보 전부에 적용.

### CHANGE ONLY
- 후보별 per-view Observed/Occluded operationalization **그것 하나뿐**.

### DO NOT (전부 준수)
opacity 변경 / opacity hardening / opaque surrogate Gaussian 생성 / WL107·109 topology 수정 / SPARSELY_OBSERVED canonical 상태 추가 / NURBS continuation / occluded NURBS / uncertain Gaussian 생성 / Trust 구현 / 구멍 메우기 / visible component 병합·분할 / dense volumetric grid를 canonical 표현으로 도입 / C만을 위한 mesh 도입 / 한 후보를 유리하게 하려는 threshold 튜닝 / downstream occluded-surface 구축으로 자동 진행 -- **하나도 하지 않았다.**

---

## 2. Candidate Operational Contract

네 후보 모두 **동일한 frozen global aggregation**(section 2 지시)을 쓴다: 관련 뷰 중 **하나라도** OBSERVED이면 GLOBAL=OBSERVED; 관련 뷰가 1개 이상이고 **전부** OCCLUDED이면 GLOBAL=OCCLUDED; 그 외 GLOBAL=UNRESOLVED. 다수결·백분율 투표·최소 뷰 수 규칙·신뢰도 가중·다중도 threshold는 코드 어디에도 없다(`shared.aggregate_global`, `TestGlobalAggregation` 6개 테스트가 이를 직접 강제).

**relevant-view 계약**(section 3): `w <= 0`(유효하지 않은 투영), `camera-space z < near_n`(renderer 자신의 0.2), 이미지 도메인 밖 -- 세 가지를 **서로 구분해서** 기록하고, 셋 다 NON_RELEVANT(= OCCLUDED 증거가 아님)로 처리한다. "관련 뷰 개수" threshold는 없다. 관련 뷰가 0개면 GLOBAL=UNRESOLVED.

### Candidate A -- DIRECT SURFACE OBSERVATION / SURFACE-HIT
1. **OBSERVED**: x가 투영된 픽셀의 median surface event **그 자체**일 때.
2. **OCCLUDED**: 그 픽셀에 qualified surface event가 존재하고 그 event가 같은 카메라 광선 위에서 x보다 앞에 있을 때(x가 그 event가 아닌 경우).
3. **UNRESOLVED**: 그 외 전부 -- 픽셀에 event가 없거나, event가 x와 같거나 뒤에 있는데 x가 그 event가 아닌 경우(**표면 앞의 자유 공간에 대해 A는 아무 말도 할 수 없다**).
4. **primitive**: WL119의 G2 -- `representative_id` / `median_s_u` / `median_s_v`(kernel 자신의 T=0.5 crossing에서 캡처)로부터 `center + s_u*scale_u*t_u + s_v*scale_v*t_v` 재구성.
5. **RENDERER-NATIVE**(s_u/s_v/representative_id는 kernel이 직접 계산) + **RENDERER-GROUNDED**(world point 재구성에 학습된 surfel frame 사용). 둘 다 해당.
6. **가정**: "qualified" = `representative_id >= 0`(kernel 자신의 "어떤 contributor가 T=0.5를 넘었다"). rho3d/rho2d branch로 좁히지 **않는다**(WL119: low-pass event는 분류하되 기각하지 않는다) -- branch는 메타데이터로만 기록.
7. **새 threshold**: 없음. **AGENT-INTRODUCED 수치 규칙 1개**: float32 왕복 동일성 `||x - E|| <= 1e-6 * max(1, ||x||)`. 이것은 의미론적 결합 반경이 아니라 부동소수 정밀도 규칙이다. **실측에서 source view의 hit_distance는 정확히 0.0(최댓값 0.0)** 이었으므로 이 상수는 실제로 아무 일도 하지 않았다.

> **직시한 한계(수리하지 않음)**: 임의의 3D 질의를 "이것이 그 surface event다"라고 판정하는 것은 measure-zero 일치 판정이다. renderer-native하거나 기존 geometry에서 유도되는 "광선 방향 허용 오차"는 존재하지 않는다(surfel footprint는 면내(lateral) 범위이지 광선 방향 범위가 아니다). 커버리지를 늘리려고 허용 오차를 넓히는 일은 하지 않았고, 대신 한계를 그대로 노출한다.

### Candidate B -- MEDIAN-DEPTH PARTITION
1. **OBSERVED**: `query_depth <= median_depth`.
2. **OCCLUDED**: `query_depth > median_depth`.
3. **UNRESOLVED**: 유효한 median event가 없을 때.
4. **primitive**: canonical kernel의 `out_others[MIDDEPTH_OFFSET]` = `median_depth` 그대로.
5. **RENDERER-NATIVE**. 0.5 crossing 규칙 무수정, `T`를 전혀 건드리지 않음.
6. **가정**: 유효성 판정에 A의 `representative_id`를 빌리지 않고 renderer **자신의** 초기화 sentinel을 쓴다 -- `float median_depth = {0}`이고 할당은 `depth >= near_n`(=0.2)을 통과한 contributor에서만 일어나므로 `median_depth > 0`이 정확한 sentinel 검사다.
7. **새 threshold**: 없음.

> 보고서 어디에서도 median depth를 "물리적 first hit"으로 재해석하지 않는다. 그것은 광선 투과도의 절반이 소진된 깊이이며 그 이상도 이하도 아니다.

### Candidate C -- GEOMETRIC VISIBILITY
1. **OBSERVED**: 열린 선분 (camera_center, x)가 어떤 scene geometry와도 만나지 않을 때.
2. **OCCLUDED**: 그 선분이 하나 이상의 surfel의 geometric support와 만날 때.
3. **UNRESOLVED**: relevant view에 대해서는 **결코 반환되지 않는다**.
4. **primitive**: canonical kernel 자신의 alpha cutoff로부터 유도되는 **정확한 유한 geometric support**.
5. **RENDERER-GROUNDED**(support 경계는 canonical, 광선-원판 교차 계산은 이 배치가 수행).
6. **"existing scene geometry"의 사전 감사 결과(section 7 요구)**: 현재 2DGS 표현은 이미 정확한 유한 support를 노출한다. kernel은 `alpha = min(0.99, opa*exp(-0.5*rho)) >= 1/255`일 때만 accept하고, `rho = min(rho3d, rho2d)`인데 이 중 `rho3d = s_u^2 + s_v^2`만이 3D 기하량이다(`rho2d`는 screen-space low-pass floor이며 3D 의미가 없다 -- WL119가 이미 이를 **다른 종류의 관측**으로 규명했다). 따라서 surfel i의 기하 support는
   `{ c_i + a*t_u + b*t_v : (a/scale_u)^2 + (b/scale_v)^2 <= rho_max_i }`, `rho_max_i = 2*ln(255*opacity_i)`
   인 **정확히 정의된 타원 원판**이다. **3-sigma도, k-sigma도, 튜닝된 반경도, 손으로 고른 blocker threshold도 없다.** 실측: rho_max 중앙값 9.338(최대 11.083 = opacity 0.99일 때의 정확한 상한), 비어있지 않은 support를 가진 surfel 1,186,160 / 1,190,469(나머지는 opacity <= 1/255라 kernel 자신이 어디서도 accept하지 않는다).
   rasterizer의 screen-space `radii`/tile binning은 **의도적으로 적용하지 않았다** -- 그것은 래스터화 최적화이지 기하가 아니며, C는 기하 검사로 정의되어 있다.
7. **새 threshold**: 없음. **AGENT-INTRODUCED 수치 규칙 1개**: 열린 선분 guard `t in (1e-6, 1-1e-6)`. 이것이 없으면 R1 anchor가 float 반올림으로 자기 자신을 가릴 수 있다. 실측 `nearest_blocker_t` 중앙값 0.99935 / p95 0.99997로 결정이 이 guard 근처에 몰려 있지 않음을 확인할 수 있게 보고한다.

> **공개하는 의미론적 유보(수리하지 않음)**: opacity 크기와 무관하게 support 교차를 **경성(hard) blocker**로 취급하는 것은 가설 C **자신의 전제**("cumulative alpha transmittance와 무관하게")이지 agent가 도입한 hardening이 아니다 -- opacity 값은 어디서도 수정·대체되지 않았다. 다만 opacity 0.006짜리 surfel이 0.99짜리와 똑같이 절대적으로 차단한다는 뜻이며, 이를 보이기 위해 blocker 개수와 최대 blocker opacity를 메타데이터로 함께 기록한다(결정에는 절대 쓰지 않는다).
>
> 따라서 C의 판정은 "NOT CLEANLY OPERATIONALIZABLE"이 아니라 **"공개된 의미론적 유보를 안고 operationalizable"** 이며, 아래 verdict가 그 유보를 숫자로 정면 평가한다.

### Candidate D -- RENDERER REACHABILITY
1. **OBSERVED**: query depth 이전에 canonical traversal이 **종료되지 않았을** 때.
2. **OCCLUDED**: query depth에 도달하기 전에 canonical termination 조건이 발동했을 때.
3. **UNRESOLVED**: probe가 그 (query, view) 쌍에 대해 아무 값도 쓰지 않았을 때 -- relevant view에서는 발생하면 안 되는 fail-closed 상태이며 발생 시 그대로 계수한다. **실측 발생 0건.**
4. **primitive**: canonical traversal 자신의 종료 이벤트 `if (test_T < 0.0001f) { done = true; ... }`, `test_T = T * (1 - alpha)`.
5. **RENDERER-NATIVE**. 조건·상수 0.0001·순서·수용 검사·depth 규약이 전부 canonical kernel의 것이며, worklog 120 진단 sibling 빌드가 **canonical 지점에서 관측만** 해서 임의 query depth로 재노출한다.
6. **가정**: probe는 canonical kernel이 **ACCEPT한** contributor에서만 해소된다(아래 8절 참조).
7. **새 threshold**: **없음.** 이 모듈은 `T`를 무엇과도 비교하지 않는다. `T`는 메타데이터로만 밖으로 나온다 -- `T`를 결정에 쓰려면 지시가 금지한 바로 그 threshold를 발명해야 하므로 하지 않았다.

**기존 진단 출력으로 재구성 가능한가(section 8 선행 조사)**: 불가능하다. (a) `final_T`는 전체 목록 소진 후의 최종 투과도일 뿐 임의 깊이의 T가 아니고, (b) WL110의 `contrib_ids`/`contrib_post_median` slot 배열은 `OSN_GS_MAX_CONTRIB_SLOTS`=16으로 상한이 있고 WL110이 실제 pixel-view slot의 **97.4%가 truncate**됨을 측정했으며, (c) 잘리지 않았더라도 per-contributor `alpha`와 `depth`가 전혀 없어 투과도 prefix를 만들 수 없다.

---

## 3. Candidate-to-Code Ownership Map

| path | function/class | ownership | 정확한 책임 | 구현한 의미론적 결정 | 이를 덮는 테스트 |
|---|---|---|---|---|---|
| `scripts/devtools/observed_occluded/shared.py` | `project_queries` | SHARED | 투영 + relevant-view 계약만 | **없음** (가시성 경계 판단 없음) | `TestRelevantViewContract`(4) |
| " | `aggregate_global` | SHARED | frozen global 집계 규칙 | 지시 section 2가 명시한 규칙 그 자체 | `TestGlobalAggregation`(6) |
| " | `reconstruct_direct_surfel_intersection_world_point` | SHARED | WL119 G2 재구성 -- **어디**인지만 답함 | 없음(위치 계산) | `TestRelevantViewContract.test_projection_uses_the_rasterizer_pixel_convention` |
| " | `canonical_geometric_support_rho_max` | SHARED | canonical alpha cutoff -> 정확한 support 범위 | 없음(surfel 속성 + canonical 상수) | `TestCanonicalConstants`(2), `TestCandidateC`(7) |
| " | `assign_query_depth_slots` | SHARED | probe slot 순위 배정(순수 장부) | 없음 | `TestQueryDepthSlotAssignment`(2) |
| " | `state_fractions`/`distribution`/`agreement` | SHARED | metric/serialization | 없음 | -- |
| `scripts/devtools/observed_occluded/candidate_a_surface_hit.py` | `classify_view` | **A** | surface-hit 결정 **전용** | "x가 그 event인가 / event가 앞에 있는가" | `TestCandidateA`(6), `TestSyntheticContracts` |
| `scripts/devtools/observed_occluded/candidate_b_median_depth.py` | `classify_view`, `median_depth_map` | **B** | median-depth 결정 **전용** | `query_depth <= median_depth`, sentinel `median > 0` | `TestCandidateB`(5) |
| `scripts/devtools/observed_occluded/candidate_c_geometric_visibility.py` | `classify_view`, `GeometricSceneSupport` | **C** | 경성 line-of-sight 결정 **전용** | "무엇이 blocker인가" = canonical support와의 광선-원판 교차 | `TestCandidateC`(7) |
| `scripts/devtools/observed_occluded/candidate_d_renderer_reachability.py` | `classify_view` | **D** | reachability 결정 **전용** | "canonical traversal이 이미 종료했는가" | `TestCandidateD`(4) |
| `scripts/devtools/observed_occluded/engine.py` | `evaluate`, `build_geometric_support` | SHARED | 렌더 -> 수집 -> 각 후보 호출 -> 기록 | **없음**(소스에 `STATE_OBSERVED`/`STATE_OCCLUDED` 문자열이 아예 없음, 테스트로 강제) | `TestCandidateIsolation`(7) |
| `scripts/devtools/observed_occluded/query_bank.py` | `build_bank`, `region_of_surfel` | SHARED | 결정론적 bank + region 라벨 | **없음**(동일하게 강제) | `TestCandidateIsolation` |
| `scripts/devtools/observed_occluded/synthetic_contracts.py` | `build_contracts`, `run_contracts` | SHARED | S1-S7 fixture + 사전 기대값 | 없음(기대값은 지시/후보 계약에서 유도) | `TestSyntheticContracts`(9) |
| `scripts/devtools/observed_occluded_volumetric_audit.py` | `main` 및 metric 함수들 | SHARED | 오케스트레이션 + 보고 + export | 없음 | 실 scene 재생 |
| `osn_gs/render/torch_surfel_query_depth_diagnostics.py` | `render_with_query_depth_probe` | SHARED(D 전용 소비) | 진단 CUDA sibling 로더/래퍼 | 없음(관측값 노출만) | `TestQDepthCanonicalEquivalence`(3) |
| `osn_gs/render/vendor/diff_surfel_rasterization_qdepth/cuda_rasterizer/forward.cu` | `renderCUDA` 내 probe 블록 | **D** | canonical termination 이벤트를 임의 깊이로 노출 | canonical `test_T < 0.0001f` 관측(새 조건 없음) | `TestQDepthProbeAgainstCanonicalTraversal`(4) |

**하나의 함수가 여러 후보의 의미론을 담고 있는 경우: 없음.**

---

## 4. Shared-Code Semantic Audit

공유 헬퍼가 네 후보 전부가 쓰는 가시성 경계를 몰래 정의하고 있지 않다는 것을 **테스트로 강제**했다(단순 주장이 아님):

- `test_engine_makes_no_observed_occluded_decision` / `test_query_bank_makes_no_observed_occluded_decision`: `engine.py`와 `query_bank.py` 소스에 `STATE_OBSERVED`/`STATE_OCCLUDED` 토큰이 **하나도 없음**을 검사. 두 파일은 상태를 만들 수단 자체가 없다.
- `test_shared_only_decides_states_inside_the_frozen_aggregation`: `shared.py`를 AST로 파싱해 `STATE_OBSERVED`/`STATE_OCCLUDED`를 참조하는 함수가 `aggregate_global`/`agreement`/`state_fractions`(집계·리포팅) **뿐**임을 검사.
- `test_candidate_modules_never_import_each_other`, `test_shared_never_imports_a_candidate`: 후보 간 상호 참조 및 shared -> candidate 역참조가 없음을 AST import 그래프로 검사.
- `test_candidate_b_never_reads_transmittance_or_geometry_support`: B의 코드 본문에 `query_T`/`query_terminated`/`rho_max`/`representative_id`/`blocker`가 없음을 검사.
- `test_candidate_d_never_reads_median_depth_or_geometry_support`: D의 코드 본문에 `median`/`rho_max`/`blocker`가 없음을 검사.

**공유 코드 중 렌더러 출력을 만지는 두 함수에 대한 명시적 정당화**:
- `reconstruct_direct_surfel_intersection_world_point`(WL119 G2): "이 렌더러 이벤트가 3D에서 **어디**인가"에만 답하고 "관측인가 가림인가"에는 답하지 않는다. A만 자기 primitive로 사용하고, query bank가 R1 anchor 배치에 사용한다(지시 9B/R1이 직접 요구). B/C/D는 호출조차 하지 않는다.
- `canonical_geometric_support_rho_max`: **독립적인 두 소비자**가 동일한 canonical 양을 서로 겹치지 않는 이유로 필요로 하기 때문에만 여기 있다 -- C는 blocker 범위로, query bank는 ray-ladder 오프셋의 (튜닝되지 않은, 증거에서 유도된) 단위로. surfel의 속성이자 canonical renderer의 상수이며 아무것도 결정하지 않는다.

**공개하는 편향**: query bank의 R1 anchor는 renderer median surface event 위에 놓인다. 이는 지시 9B/R1이 **직접 지정한 bank 정의**이지 후보의 결정이 아니지만, 동시에 그 event를 primitive로 삼는 **A와 B에 유리한 알려진 편향**이다. 실제로 A의 R1 anchor 보존율 2,640/2,640(100%)은 상당 부분 동어반복이며, 아래 verdict는 이를 감안해 해석한다. 이를 완화하기 위해 bank는 어떤 후보의 primitive도 생성하지 않은 질의류(ray ladder R3/R4, region gap R5, out-of-frustum R6)를 함께 담는다.

---

## 5. Implementation Fidelity Statement

| 후보 | 의도한 가설 | 실제 모듈/함수 | 실제 dataflow | 테스트 | 측정 출력 |
|---|---|---|---|---|---|
| A | 직접 관측된 표면 이벤트만으로 관측 측을 정의 | `candidate_a_surface_hit.classify_view` | diag render -> `representative_id`/`s_u`/`s_v` -> G2 world point -> camera-z -> 동일성/전방성 판정 | `TestCandidateA`(6) + S1-S7 | `per_candidate.A.*` |
| B | renderer median surface depth를 view-local 분할면으로 | `candidate_b_median_depth.classify_view` | `out_others[5]` -> 픽셀 gather -> `<=` 비교 | `TestCandidateB`(5) + S1-S7 | `per_candidate.B.*` |
| C | 경성 기하 line-of-sight | `candidate_c_geometric_visibility.classify_view` | model tensor -> canonical support -> 광선-원판 교차(chunked) | `TestCandidateC`(7) + S1-S7 | `per_candidate.C.*` |
| D | canonical 광선이 query depth까지 기여 가능한가 | `candidate_d_renderer_reachability.classify_view` | qdepth CUDA probe -> `query_terminated` -> 매핑 | `TestCandidateD`(4), `TestQDepthProbeAgainstCanonicalTraversal`(4), `TestQDepthCanonicalEquivalence`(3) | `per_candidate.D.*` |

### PROMPT-REQUIRED DECISION (지시가 규정한 것)
- 네 후보의 OBSERVED/OCCLUDED/UNRESOLVED 정의(section 5-8), frozen global aggregation(section 2), relevant-view 계약(section 3), shared/candidate 코드 분리(section 4), 공통 query bank(section 9), 보고 metric 축(section 10), 단일 승점 금지(section 11).
- B가 0.5 median crossing 규칙을 수정하지 않을 것, C가 새 support 경계를 만들지 않을 것, D가 새 occlusion threshold를 만들지 않을 것.
- R1 anchor를 renderer 관측 표면점 위에 두고 rho3d/rho2d를 분리할 것.

### AGENT-INTRODUCED OPERATIONAL CHOICE (전부 공개)
1. **A의 동일성 규칙**: float32 상대 오차 `1e-6`. 의미론적 결합 반경이 아니라 정밀도 규칙. **실측에서 source view hit_distance 최댓값이 정확히 0.0**이므로 이 상수는 결과에 관여하지 않았다.
2. **C의 열린 선분 guard**: `t in (1e-6, 1-1e-6)`. 자기 차폐 방지용 정밀도 규칙. `nearest_blocker_t` 분포를 함께 보고해 결정이 guard에 몰리지 않음을 확인 가능하게 했다.
3. **질의 픽셀 반올림**: rasterizer 자신의 연속 픽셀 좌표(`((ndc+1)*S-1)*0.5`)에 대한 최근접 반올림.
4. **depth 규약**: camera-space z(`[x,1] @ world_view_transform`의 z). renderer 루프의 `depth` 변수 및 `depths_to_points`가 역투영하는 양과 동일.
5. **query bank 선택 stride/개수**: anchor view stride 16(161뷰 -> 11뷰), branch당 뷰당 anchor 120개, ladder anchor stride 12, ladder 배수 `{-4,-2,-1,-0.5,+0.5,+1,+2,+4}` x **anchor surfel 자신의 canonical support 반경**, region gap 영역당 60개, out-of-frustum 배수 `{4,8}` x scene extent. **RNG 없음** -- 전부 고정 순서 위의 고정 stride. 배수 집합은 사전 확정했고 결과를 본 뒤 조정하지 않았다.
6. **probe slot 용량** `OSN_GS_MAX_QUERY_SLOTS = 8`, 초과 픽셀은 **추가 렌더 패스**로 처리(질의를 버리지 않음). 실측 최대 픽셀당 질의 10개 -> 뷰당 2패스, 총 322 렌더 패스.
7. **C의 chunk 예산** 384 MiB -- 순수 메모리 경계, 결과에 영향 없음.
8. **fallback**: 없음. 어떤 후보도 다른 후보의 결정을 빌리지 않는다.

### 실행 중 발견하고 교정한 구현 결함 1건 (측정 보고 전에 교정)
qdepth probe의 최초 판(revision 1)은 `depth < near_n` 검사 직후, 즉 **수용 검사 이전**에 probe를 해소했다. 그러나 kernel은 tile 목록의 **모든 후보**에 대해 surfel의 **무한 평면**과의 교차 깊이를 계산하므로, 끝내 accept되지 않는 후보가 임의로 큰 교차 깊이를 보고할 수 있다. 실 scene 스모크에서 이 때문에 **probe의 약 99%가 목록 첫 항목에서 T=1.0으로 해소**되어 D가 per-view OCCLUDED 0%라는 퇴화된 결과를 냈다(`accepted_prefix_count` 중앙값 0). 이는 renderer reachability가 아니라 tile 목록 순서의 인공물이다.
교정: probe는 **canonical kernel이 ACCEPT한 contributor에서만** 해소하도록 옮겼다(수용 검사 통과 후 `w = alpha*T` 직전, 그리고 termination 분기 안). 교정 후 `accepted_prefix_count` 중앙값 7, per-view OCCLUDED 25.48%로 정상화되었다. 회귀 방지 테스트 `test_probe_resolves_only_at_accepted_contributors`를 추가했다.

### 남은 유보(교정하지 않고 공개)
canonical tile 목록은 surfel **중심**의 camera-space z로 정렬되므로, accept된 contributor들의 per-pixel `depth`조차 traversal 순서를 따라 정확히 단조가 아니며, rho2d(low-pass) branch로 accept된 이벤트는 자기 footprint에서 멀리 떨어진 `depth`를 가질 수 있다. 이것은 canonical renderer 자신의 depth 의미론(= `median_depth`가 만들어지는 바로 그 의미론)이며, 교정 대상이 아니라 보고 대상이다.

### INABILITY TO REALIZE REQUESTED CONTRACT
**없음.** 지시가 요구한 모든 계약(section 2-11, 13-18)을 요청된 그대로 구현했다. 유일한 근사 대체는 아래 R5 항목에서 명시한다.

---

## 6. Synthetic Contracts (S1-S7)

`scripts/devtools/observed_occluded/synthetic_contracts.py`. 모든 fixture 기하와 **양쪽 기대값**은 실행 전에 확정했고, 결과를 본 뒤 fixture를 조정하지 않았다. 두 기대 열을 절대 섞지 않는다: `expected_global`(지시의 논문 수준 기대)과 `predicted_<X>`(후보 **자신의** 계약이 예측하는 값 -- 이것을 못 맞추면 구현 결함이며 테스트로 강제).

| 계약 | A | B | C | D |
|---|---|---|---|---|
| **S1** 직접 노출된 표면 | OBSERVED ✓ | OBSERVED ✓ | OBSERVED ✓ | OBSERVED ✓ |
| **S2** 표면 앞의 노출된 자유 공간 | **UNRESOLVED ✗(지시 기대 위반)** | OBSERVED ✓ | OBSERVED ✓ | OBSERVED ✓ |
| **S3a** canonical하게 불투명한 차폐 뒤 | OCCLUDED ✓ | OCCLUDED ✓ | OCCLUDED ✓ | OCCLUDED ✓ |
| **S3b** 단일 반투명 primitive 차폐 뒤 | OCCLUDED ✓ | OCCLUDED ✓ | OCCLUDED ✓ | **OBSERVED ✗(지시 기대 위반)** |
| **S4** 교차 뷰 disocclusion | OBSERVED ✓ | OBSERVED ✓ | OBSERVED ✓ | OBSERVED ✓ |
| **S5** 카메라 지원 밖 | UNRESOLVED ✓ | UNRESOLVED ✓ | UNRESOLVED ✓ | UNRESOLVED ✓ |
| **S6** 계층 soft compositing (3 probe) | UNRES, OCC, OCC | **OBS, OCC, OCC** | OCC, OCC, OCC | **OBS, OBS, OCC** |
| **S7** rho3d true-footprint (event, 뒤) | OBS, OCC | OBS, OCC | OBS, **OCC** | OBS, OBS |
| **S7** rho2d low-pass (event, 뒤) | OBS, OCC | OBS, OCC | OBS, **OBS** | OBS, OBS |

**구현 충실도: 9개 계약 x 4후보 전부 자기 계약과 일치(implementation_fidelity_pass 100%).**

핵심 판독:
- **S2 -- A의 구조적 실패**: 표면만 보는 메커니즘은 **관측된 자유 공간을 표현할 수 없다**. 이것이 지시 10E가 짚으라고 한 바로 그 항목이며, A의 가설 자체의 성질이지 구현 결함이 아니다.
- **S3b -- D의 구조적 실패**: opacity 0.99짜리 단일 surfel 뒤에서 T=0.0105로 남으므로 canonical traversal은 종료하지 않고 D는 OBSERVED라고 답한다. 기하적으로는 명백한 차폐인데도 그렇다. (S3a는 canonical하게 불투명한 4겹 스택으로 만들어 D도 OCCLUDED를 낸다 -- 두 fixture는 사전에 함께 설계했다.)
- **S6 -- B와 D의 의미론 차이를 분리**: 첫 contributor(깊이 4.00), median crossing(4.05), canonical termination(5.25)이 **서로 다른 깊이**에 오도록 30겹(alpha 0.3)으로 구성했다. 깊이 4.60 probe에서 **B=OCCLUDED, D=OBSERVED**로 정확히 갈린다. 이 fixture는 결과를 보기 전에 확정했다.
- **S7 -- C의 저역통과 맹점(사전 예측을 정정)**: rho2d 전용으로 accept된 sub-pixel event는 **광선 위에 진짜 기하 support가 전혀 없다**(rho3d=4163.9 vs rho2d=1.0). 따라서 순수 기하 검사인 C는 그 뒤 지점을 OBSERVED로 본다. 최초 손 유도는 C가 OCCLUDED를 낼 것으로 예측했고 그 예측이 틀렸다 -- **fixture나 구현이 아니라 예측을 정정**했으며, 원래 예측과 정정 사유를 코드(`prediction_note`)와 여기에 함께 남긴다.

> Synthetic PASS는 의미론적 정확성만 확립한다. 아키텍처 viability를 증명하지 않는다.

---

## 7. Real-Scene Query-Bank Definition

`output/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_volumetric_audit_report.json` -> `query_bank`. **RNG 없음.** 총 **4,712 질의**, 161 학습 뷰 전부에 대해 평가 -> **758,632 query-view 쌍**.

| 종류 | 개수 | 구성 |
|---|---|---|
| R1 rho3d true-footprint anchor | 1,320 | 11개 anchor view(stride 16) x branch당 120, raster order 균등 stride |
| R1 rho2d low-pass anchor | 1,320 | 동일, rho2d > rho3d인 median event만 |
| R3 behind-surface probe | 880 | anchor surfel 자신의 canonical support 반경의 `+0.5/+1/+2/+4`배 |
| R4 front-of-surface probe | 880 | 동일 반경의 `-0.5/-1/-2/-4`배 |
| R5 region gap probe | 300 | 같은 영역 anchor의 최근접쌍 중점(영역당 60) |
| R6 out-of-frustum control | 12 | scene AABB 중심 ± `{4,8}` x scene extent(91.135), 6축 |

영역 분포: table_top 675 / table_side_curved 838 / table_legs 602 / patio 2,074 / hedge 511 / 미분류 12. 영역 라벨은 WL108-119와 **동일한** anchor fraction 기법을 그대로 재사용했다(보고용 working interpretation이며 결코 ground truth가 아니다).

**R2(교차 뷰 재투영)** 는 별도 질의가 아니라 R1 anchor의 per-view 회계로 구현했다 -- 모든 anchor는 어차피 161뷰 전부에서 평가되므로, source view와 나머지 뷰의 상태를 분리 집계하는 것이 지시가 요구한 바로 그 검사다.

**relevant-view 회계**: RELEVANT 383,322(50.5%) / 투영 무효 95,750 / near 미만 4,738 / 이미지 밖 274,822. 질의당 relevant 뷰 수 중앙값 68, 평균 81.4. **relevant 뷰가 0개인 질의 4개.**

**공개하는 bank 약점 1건**: R6 out-of-frustum control 12개 중 **8개는 실제로는 어떤 카메라의 지원 밖이 아니었다**(scene extent 91.1이 커서 4x/8x 오프셋이 여전히 일부 카메라 절두체 안에 들어옴). 진짜 "relevant 뷰 0개" control은 4개뿐이며, **그 4개에 대해서는 네 후보 전부가 UNRESOLVED를 반환**한다(계약 준수). 이 계약 자체는 S5가 깨끗하게 검증한다. 나머지 8개는 out-of-frustum control이 아니라 평범한 원거리 질의로 재분류해서 읽어야 한다.

**R5에 대한 명시**: 지시는 "현재 fragmentation 사례를 가로지르는" 질의를 요구했다. WL107/109 topology를 이 배치에서 재생하지 않기로 했으므로(topology는 보존 대상이지 입력이 아니며, 후보 결정에 전혀 필요 없다), component 쌍 중점 대신 **같은 영역 anchor의 최근접쌍 중점**으로 근사했다. 이는 요청된 계약의 근사 대체이며 여기 명시한다. 대신 topology 동결 증거는 아래의 fingerprint로 대체 제공한다.

**동결 상태 fingerprint**: 161뷰 전체에서 한 번이라도 median surface representative였던 surfel의 합집합 = **785,937**. WL119가 같은 checkpoint/카메라 집합에 대해 보고한 값과 **정확히 일치**(`matches_worklog_119: true`). 모델·카메라 집합·렌더러가 이 배치에 의해 전혀 변경되지 않았다는 직접 실측 증거다.

---

## 8. Candidate A Results (SURFACE-HIT)

- **per-view 쌍**(relevant 383,322 기준): OBSERVED **0.70%** / OCCLUDED 63.94% / UNRESOLVED 35.36%.
- **global**(4,712): OBSERVED 2,676(56.79%) / OCCLUDED 648(13.75%) / **UNRESOLVED 1,388(29.46%)**.
- **R1 anchor 보존**: 2,640/2,640 GLOBAL OBSERVED, **의미론적 모순 0건**, source view 모순 0건. source view의 `hit_distance` 최댓값 **정확히 0.0**(동일성 상수가 관여하지 않음).
- **교차 뷰**: anchor당 OBSERVED 뷰 수 **최솟값·중앙값·최댓값이 전부 1** -- 즉 각 anchor는 오직 자신을 생성한 뷰에서만 관측된다. OCCLUDED 뷰 중앙값 47, UNRESOLVED 뷰 중앙값 21.
- **ray-order**(220 ladder, 단조 99.55%): 표면 앞(-4x ~ -0.5x)에서 **215-216/220이 UNRESOLVED**. 뒤로 갈수록 OCCLUDED가 113 -> 159 -> 168 -> 180으로 증가.
- **event branch 메타데이터**: rho3d-dominated 293,573 쌍 / rho2d-dominated 89,749 쌍 / event 없음 375,310 쌍.
- **질의 종류별 global**: R4 front probe 880개 중 **862개(98.0%)가 UNRESOLVED**. R5 region gap 300개 중 **280개(93.3%)가 UNRESOLVED**.

**판독**: A의 R1 보존율 100%는 대체로 동어반복이다(bank가 A의 primitive 위에 놓여 있다). 실질적 신호는 그 바깥에 있고, 거기서 A는 **구조적으로 굶는다** -- surface-hit은 measure-zero 판정이므로 생성 뷰 단 하나에서만 OBSERVED가 되고, 표면 앞 자유 공간(R4)과 영역 간극(R5)에서는 거의 전부 UNRESOLVED다. S2가 합성으로 보여준 실패가 실 scene에서 그대로 재현된다.

---

## 9. Candidate B Results (MEDIAN-DEPTH)

- **per-view 쌍**: OBSERVED 35.92% / OCCLUDED 64.08% / **UNRESOLVED 0.00%**(유효 median event가 사실상 모든 relevant 픽셀에 존재).
- **global**: OBSERVED 4,054(86.04%) / OCCLUDED 654(13.88%) / UNRESOLVED 4(0.08%, = relevant 뷰 0개인 질의 4개).
- **R1 anchor 보존**: 2,634/2,640 OBSERVED, GLOBAL OCCLUDED **6건**. source view에서 OCCLUDED로 판정된 anchor **507건(19.2%)**.
- **그 507건의 정체를 정량화했다**: R1 anchor의 source view에서 `query_depth - median_depth`의 분포는 중앙값 **정확히 0.0**, 평균 -1.4e-9, 최대 |1.9e-6|. 2,640건 중 **1,653건이 정확히 0**, 480건이 음수, 507건이 양수다. 양수 쪽 차이의 중앙값은 **4.77e-7(깊이 대비 상대 8.6e-8)**. 즉 이 507건은 의미론적 모순이 아니라 **float32 반올림**이다 -- B의 분할면은 두께가 0이므로 그 면 위의 점은 반올림으로 양쪽에 갈린다. 엄격 부등호 `>`가 동률 근방을 OCCLUDED로 보낸 결과이며, GLOBAL OCCLUDED 6건도 같은 원인이다.
- **교차 뷰**: anchor당 OBSERVED 뷰 중앙값 22(평균 28.9), OCCLUDED 뷰 중앙값 47. OBSERVED와 OCCLUDED 뷰를 동시에 가진 anchor 2,615개 -> 전부 GLOBAL OBSERVED(집계 규칙 정상 동작).
- **ray-order**(단조 87.27%): 표면 앞 4단계 전부 **220/220 OBSERVED**. 뒤 4단계에서 OCCLUDED 113 -> 159 -> 168 -> 180. +4x에서도 40개는 GLOBAL OBSERVED인데, 이는 다른 뷰가 그 지점을 median 앞쪽으로 보기 때문이다(집계 규칙의 정상 결과).
- **영역별 global OBSERVED/OCCLUDED**: table_top 618/57, table_side_curved 737/101, table_legs 525/77, patio 1,759/315, hedge 415/96 -- **영역 간 편차가 작고 일관적**이다.

**판독**: B는 UNRESOLVED가 사실상 0이고(증거 기아 없음), 의미론적 모순도 float 수준(1e-7 상대)에 그친다. 다만 **median depth는 물리적 first hit이 아니다** -- S6에서 median crossing(4.05)이 첫 contributor(4.00)보다 뒤에 있고 termination(5.25)보다 한참 앞에 있다는 것이 그대로 드러난다. B가 그리는 면은 "투과도 절반이 소진된 곳"이며, 이를 가시성 경계로 쓰는 것은 가설 B 자신의 주장이지 렌더러가 보증하는 성질이 아니다.

---

## 10. Candidate C Results (GEOMETRIC VISIBILITY)

- **per-view 쌍**: OBSERVED 10.36% / **OCCLUDED 89.64%** / UNRESOLVED 0%.
- **global**: OBSERVED 767(16.28%) / **OCCLUDED 3,941(83.64%)** / UNRESOLVED 4.
- **치명적 의미론 모순**: R1 anchor 2,640개 중 **2,605개(98.7%)가 자기 자신을 생성한 뷰에서 OCCLUDED**로 판정된다. 그 뷰가 직접 관측해서 그 점을 만들어냈는데도 그렇다. GLOBAL OCCLUDED는 2,556개(96.8%)다. rho3d anchor 1,305/1,320, rho2d anchor 1,300/1,320으로 branch를 가리지 않는다.
- **원인은 명확하다**: relevant 쌍당 blocker 개수 중앙값 **16개**(p95 101, 최대 249)이고, `nearest_blocker_t` 중앙값 **0.99935**다. 즉 차단자들은 대부분 **질의 바로 앞, 같은 표면을 이루는 다른 surfel들**이다. 학습된 2DGS 표면은 겹쳐진 원판들의 두꺼운 수프이므로, 카메라에서 표면 위 한 점까지 가는 광선은 그 점에 닿기 전에 필연적으로 같은 표면의 다른 원판 여럿을 통과한다. 경성 blocker 전제 아래에서는 **모든 표면점이 자기 카메라로부터 가려진다**.
- OCCLUDED 쌍의 최대 blocker opacity 중앙값 0.99997 -- 즉 대부분은 "희미한 surfel이 억지로 막았다"가 아니라 진짜 불투명한 이웃이다. 이는 문제가 opacity 무시 때문만이 아니라 **표면 두께 자체** 때문임을 보여준다.
- **ray-order**: **표면보다 4배 support 반경 앞(-4x)에서 이미 23/220이 OCCLUDED**, -0.5x에서 130/220이 OCCLUDED다. C의 경계는 실제 표면보다 훨씬 앞으로 번져 있다.
- 영역별로도 동일: table_top 159 OBS/516 OCC, patio 335/1,739, hedge 74/437.

**판독**: C는 커버리지가 완전하고(UNRESOLVED 0) 결정론적이지만, **자기 자신이 관측한 표면점을 96.8% 가려졌다고 말한다**. 이는 지시가 "severe semantic contradiction"이라 부른 바로 그 것이며, 허용된 수단(threshold 튜닝, opacity hardening 해제, 허용 오차 도입, 새 표면 표현) **전부가 금지되어 있으므로 이 배치 안에서는 수리 불가능**하다. 수리하려면 정확히 지시가 금지한 "임의의 기하 허용 오차"를 발명해야 한다.

---

## 11. Candidate D Results (RENDERER REACHABILITY)

- **per-view 쌍**: OBSERVED 74.52% / **OCCLUDED 25.48%** / UNRESOLVED 0.00%(fail-closed 상태 발생 0건).
- **global**: OBSERVED 4,708(99.92%) / **OCCLUDED 0(0.00%)** / UNRESOLVED 4.
- **핵심 결과**: D의 per-view 신호는 충분히 실질적이다(97,676 쌍이 OCCLUDED). 그러나 **frozen global aggregation 아래에서 이 bank의 어떤 질의도 모든 relevant 뷰에서 동시에 OCCLUDED가 되지 않는다.** 질의당 relevant 뷰 중앙값이 68개인데, D 기준으로는 그 중 최소 하나는 항상 "traversal이 아직 종료하지 않았다"고 답한다.
- **T 메타데이터**(결정에는 쓰이지 않음): OBSERVED 쌍의 query depth 직전 T 중앙값 **0.8858**(최솟값 1.0e-4), OCCLUDED 쌍의 T 중앙값 **1.43e-4**(최대 9.99e-3). 두 분포가 canonical termination 상수 1e-4를 경계로 깨끗이 갈린다 -- probe가 canonical 조건을 정확히 관측하고 있다는 실측 확인.
- OBSERVED 쌍의 95.9%는 traversal이 실제로 query depth에 **도달**했다(나머지는 그 픽셀의 contributor 목록이 먼저 소진된 경우 = 아무것도 막지 않음).
- `accepted_prefix_count` 중앙값 7, 평균 19.9, 최대 198.
- **ray-order**: 8단계 전부 **220/220 OBSERVED**, 단조성 100%(공허하게). support 반경 4배 뒤까지 가도 global OCCLUDED가 하나도 생기지 않는다.
- **R1 anchor 보존**: 2,640/2,640 OBSERVED, 모순 0건.

**판독**: D는 유일하게 **새 threshold를 하나도 도입하지 않고**(canonical termination 이벤트 그 자체만 사용) per-view 수준에서 의미 있는 이분을 만든다. 그러나 (a) S3b가 보여주듯 **단일 반투명 primitive 뒤를 OBSERVED로 본다** -- canonical termination은 누적 투과도가 1e-4까지 떨어져야 발동하는 매우 보수적인 조건이고, (b) 161뷰 + "하나라도 OBSERVED면 OBSERVED" 규칙 아래에서 **global OCCLUDED가 정확히 0**이 된다. 즉 D는 view-local 신호로는 살아 있지만 **global 수준에서는 아무것도 분할하지 못한다.**

---

## 12. Coverage Accounting

| | A | B | C | D |
|---|---|---|---|---|
| 총 query-view 쌍 | 758,632 | 758,632 | 758,632 | 758,632 |
| relevant 쌍 | 383,322 (50.53%) | 동일 | 동일 | 동일 |
| per-view OBSERVED | **0.70%** | 35.92% | 10.36% | 74.52% |
| per-view OCCLUDED | 63.94% | 64.08% | **89.64%** | 25.48% |
| per-view UNRESOLVED | **35.36%** | 0.00% | 0.00% | 0.00% |
| global OBSERVED | 56.79% | 86.04% | 16.28% | **99.92%** |
| global OCCLUDED | 13.75% | 13.88% | **83.64%** | **0.00%** |
| global UNRESOLVED | **29.46%** | 0.08% | 0.08% | 0.08% |

UNRESOLVED 표본을 분모에서 숨기지 않았다. NON_RELEVANT를 포함한 분모의 표도 리포트에 함께 있다(`per_view_pair_states_including_non_relevant`).

**증거 기아 판정**: A만 STARVED다(per-view OBSERVED 0.70%, global UNRESOLVED 29.46%, R4 front probe의 98.0%가 UNRESOLVED). B/C/D는 UNRESOLVED가 사실상 0으로 기아가 아니다 -- 다만 C는 기아 대신 **과잉 차폐**로, D는 **global 무분할**로 실패한다. 지시 section 12에 따라 A에 대해 threshold 완화·백분율 투표·median fallback·타 후보 결정 차용·근접 추론·flood fill을 **일절 시도하지 않았고**, EVIDENCE-STARVED로 보고하고 원인을 귀속한다: **가설 A의 관측 primitive 자체가 부피를 덮지 못한다(surface-hit은 measure-zero)**.

---

## 13. Positive-Observation Contradiction Accounting

R1 anchor 2,640개(rho3d 1,320 / rho2d 1,320)에 대한 GLOBAL 판정:

| | GLOBAL OBSERVED | GLOBAL OCCLUDED (심각 모순) | GLOBAL UNRESOLVED | source view OCCLUDED |
|---|---|---|---|---|
| A | 2,640 (100%) | **0** | 0 | 0 |
| B | 2,634 (99.77%) | **6** (float 반올림) | 0 | 507 (전부 float 반올림, 상대 8.6e-8) |
| C | 84 (3.18%) | **2,556 (96.82%)** | 0 | **2,605 (98.67%)** |
| D | 2,640 (100%) | **0** | 0 | 0 |

branch별로도 동일한 패턴이다(rho3d anchor C 모순 1,292/1,320, rho2d anchor 1,264/1,320) -- **C의 모순은 low-pass 여부와 무관한 구조적 문제**다.

A의 0건은 상당 부분 동어반복(bank가 A의 primitive 위에 놓임)이므로 D의 0건과 동등한 무게로 읽어서는 안 된다. D의 0건은 독립적인 primitive(canonical termination)로 얻은 것이라 정보량이 더 크다.

---

## 14. Cross-View Accounting (R2)

| | anchor당 OBSERVED 뷰 (중앙/평균/최대) | anchor당 OCCLUDED 뷰 (중앙값) | OBS·OCC 혼재 anchor | 그 결과 GLOBAL | 어떤 뷰에서도 OBSERVED 아님 |
|---|---|---|---|---|---|
| A | **1 / 1.00 / 1** | 47 | 2,616 | 전부 OBSERVED | 0 |
| B | 22 / 28.9 / 153 | 47 | 2,615 | 전부 OBSERVED | 6 |
| C | 0 / 1.13 / 148 | 70 | 82 | 전부 OBSERVED | **2,556** |
| D | 50 / 63.4 / 160 | 13 | 2,355 | 전부 OBSERVED | 0 |

**"일부 뷰에서 가려져도 한 뷰에서 관측되면 GLOBAL OBSERVED"** 라는 의도한 논문 수준 의미론이 네 후보 전부에서 정확히 동작한다(혼재 anchor 전부 GLOBAL OBSERVED). A의 OBSERVED 뷰 수가 상수 1이라는 것은 A가 교차 뷰 관측을 원리적으로 인식하지 못한다는 뜻이고, C의 중앙값 0은 C가 관측 자체를 거의 인식하지 못한다는 뜻이다.

---

## 15. Candidate Agreement / Disagreement Matrix

**GLOBAL 기준**(4,712 질의):

| 쌍 | 동일 상태 | OBSERVED/OCCLUDED 정면 충돌 | resolved/unresolved 불일치 |
|---|---|---|---|
| A vs B | 70.50% | 6 | 1,384 |
| A vs C | 15.70% | 2,588 | 1,384 |
| A vs D | 56.88% | 648 | 1,384 |
| B vs C | 30.24% | **3,287** | 0 |
| B vs D | **86.12%** | 654 | 0 |
| C vs D | 16.36% | **3,941** | 0 |

**per-view 쌍 기준**(relevant only): A-B 64.51% / A-C 63.88% / A-D 26.18% / B-C 74.29% / B-D 61.41% / C-D 35.84%.

혼동 행렬 판독:
- **A vs B**: A의 UNRESOLVED 1,384건이 전부 B에서는 OBSERVED다 -- 정확히 "A는 자유 공간을 표현할 수 없다"는 차이이며 정면 충돌은 6건(B의 float 반올림)뿐이다.
- **B vs D**: D가 B의 OCCLUDED 654건을 전부 OBSERVED로 본다(`OCCLUDED->OBSERVED: 654`). 반대 방향은 0건. 즉 **D의 관측 영역은 B의 관측 영역을 완전히 포함한다**.
- **C vs D**: 두 극단. C의 OCCLUDED 3,941건을 D는 전부 OBSERVED로 본다.
- **B vs C**: C가 B의 OBSERVED 3,287건을 OCCLUDED로 뒤집는다.

**disagreement provenance 예시**(리포트 `disagreement_cases`에 쌍당 최대 40건, query id / world position / camera id / 각 후보 상태 / 해당 primitive / representative id / rho branch 포함):
- `q=122`, R1 rho2d anchor, hedge, cam `DSC07957.JPG`, query depth 13.3529. A=OBSERVED(event depth 13.3529, hit distance **0.0**, rep 859932, branch rho2d), B=OCCLUDED(median 13.3529 -- **동일 값인데 float 반올림으로 뒤집힘**), C=OCCLUDED(blocker 12개, nearest t=0.99978, max opacity 0.998), D=OBSERVED(T=0.6139, reached, prefix 13). B의 507건 source-view 모순의 전형이다.
- `q=2670`, R3 behind probe, hedge, query depth 15.9683. A=OCCLUDED(event 15.6417), B=OCCLUDED(median 15.6417), C=OCCLUDED(blocker 17), **D=OBSERVED(T=0.0854)** -- 표면 뒤 0.33 world unit인데도 canonical traversal은 아직 종료하지 않았다. S3b의 실 scene 판본이다.
- `q=104`, R1 rho3d anchor, patio, query depth 3.7931. A=B=D=OBSERVED, **C=OCCLUDED**(blocker 10개, nearest t=0.9999, max opacity 1.0) -- C가 자기 표면점을 자기 카메라로부터 가린다.

---

## 16. Region-Level Quantitative Results

GLOBAL 상태(OBSERVED / OCCLUDED / UNRESOLVED):

| 영역 | n | A | B | C | D |
|---|---|---|---|---|---|
| table_top | 675 | 383 / 57 / 235 | 618 / 57 / 0 | 159 / 516 / 0 | 675 / 0 / 0 |
| table_side_curved | 838 | 498 / 99 / 241 | 737 / 101 / 0 | 131 / 707 / 0 | 838 / 0 / 0 |
| table_legs | 602 | 366 / 76 / 160 | 525 / 77 / 0 | 68 / 534 / 0 | 602 / 0 / 0 |
| patio | 2,074 | 1,186 / 313 / 575 | 1,759 / 315 / 0 | 335 / 1,739 / 0 | 2,074 / 0 / 0 |
| hedge/background | 511 | 243 / 95 / 173 | 415 / 96 / 0 | 74 / 437 / 0 | 511 / 0 / 0 |

**네 후보 모두 영역 간 상대 패턴이 거의 동일**하다 -- 각 후보의 실패 양상은 영역 특이적이지 않고 **가설 수준의 성질**이다. 굳이 꼽자면 table_legs가 C에서 가장 나쁘고(88.7% OCCLUDED) table_top이 가장 덜 나쁘다(76.4%) -- 얇고 겹치는 구조일수록 blocker 수프 효과가 크다는 해석과 일치하지만, 이 배치는 이를 더 분해하지 않는다.

질의 종류별 GLOBAL(OBSERVED/OCCLUDED/UNRESOLVED)은 section 8-11에 인용했으며 전체는 리포트 `by_query_kind`에 있다.

---

## 17. Qualitative Review Exports

`output/120_osn_gs_observed_occluded_volumetric_audit/` 아래 10개 view. 각 view는 학습 scene 전체를 near-black으로 깔고 그 위에 질의점을 상태/종류/영역 색으로 얹은 PLY + `render.ppm`이며, PNG는 규약대로 **하나의 공유 `preview_png/` 폴더**에 `<VIEW_NAME>.png`로 모았다. 질의 마커 반경은 0.03622(= bank anchor support 반경 중앙값 / 3, 즉 학습 surfel 한 개의 tangent scale 수준)로, 장면을 가리지 않고 개별 점으로 읽히도록 맞췄다.

- `ORIGINAL_2DGS_SCENE` -- 표준 export.
- `QUERY_BANK_BY_KIND`, `QUERY_BANK_BY_REGION` -- bank 자체의 공간 분포(R1/R3/R4/R5/R6, 5개 영역).
- `CANDIDATE_A_GLOBAL_STATE` ~ `CANDIDATE_D_GLOBAL_STATE` -- 녹색=OBSERVED / 적색=OCCLUDED / 회색=UNRESOLVED. 동일 질의, 동일 카메라(`DSC07957.JPG`, 이름순 첫 학습 뷰).
- `DISAGREEMENT_B_vs_D`, `DISAGREEMENT_A_vs_D`, `DISAGREEMENT_C_vs_D` -- 두 후보의 GLOBAL 상태가 다른 질의만, 왼쪽 후보의 상태 색으로.

table / curved table side / patio / hedge 네 영역이 프리뷰 안에서 모두 식별 가능하다. **시각적 유사성만으로는 어떤 주장도 하지 않았다** -- 정량 분포(section 8-16)와 disagreement provenance(section 15)가 근거이고 export는 검토용이다.

`CANONICAL_SUBSET_MEMBERSHIP`은 이번 배치에 **포함하지 않았다**: 그 export는 WL107/109 topology 재생을 요구하는데, 이 배치는 topology를 보존 대상으로만 두고 입력으로 쓰지 않으며 후보 결정에 전혀 필요하지 않다. 동결 증거는 대신 representative 합집합 fingerprint(785,937, WL119와 정확히 일치)로 제공한다.

---

## 18. Candidate-by-Candidate Architecture Verdict

승자를 자동으로 고르지 않는다. 축을 하나의 가중 점수로 합치지 않는다.

### Candidate A -- SURFACE-HIT
- OPERATIONALIZATION FIDELITY: **PASS** (9개 합성 계약 전부 자기 계약과 일치, 동일성 상수는 실측에서 무관여)
- EVIDENCE COVERAGE: **STARVED** (per-view OBSERVED 0.70%, global UNRESOLVED 29.46%, front probe 98.0% UNRESOLVED, anchor당 OBSERVED 뷰 수 상수 1)
- SEMANTIC CONTRADICTION: **LOW** (모순 0건 -- 단, bank 편향 때문에 이 0건은 정보량이 낮다)
- ARCHITECTURE VIABILITY: **NOT VIABLE** -- 의도한 구분은 부피에 대한 것인데 A의 관측 primitive는 measure-zero 표면 사건뿐이다. S2/R4가 같은 실패를 합성·실측 양쪽에서 보여준다. 커버리지를 늘리려면 허용 오차를 발명해야 하고 그것은 금지되어 있다.

### Candidate B -- MEDIAN-DEPTH
- OPERATIONALIZATION FIDELITY: **PASS**
- EVIDENCE COVERAGE: **ADEQUATE** (UNRESOLVED 0.00%)
- SEMANTIC CONTRADICTION: **LOW** (모순 6건, 전부 상대 8.6e-8 수준의 float32 반올림. 단, 분할면의 두께가 0이라 그 면 위의 점은 원리적으로 양쪽에 갈린다는 성질은 남는다)
- ARCHITECTURE VIABILITY: **INCONCLUSIVE** -- 커버리지와 모순 축 모두 통과하지만, B가 그리는 면은 **first-surface 가시성 경계가 아니다**. S6가 median crossing이 첫 contributor와 termination 사이 어딘가라는 것을 통제된 조건에서 직접 보여준다. "그 면을 가시성 경계로 써도 되는가"는 이 배치의 증거만으로는 확정되지 않으며, 이것이 남은 아키텍처 질문의 핵심이다.

### Candidate C -- GEOMETRIC VISIBILITY
- OPERATIONALIZATION FIDELITY: **PASS with CAVEAT** -- 정확한 canonical support(= kernel 자신의 alpha cutoff에서 유도)로 새 경계를 발명하지 않고 구현했다. CAVEAT는 opacity 크기와 무관한 경성 blocker 취급이며, 이는 가설 C 자신의 전제다.
- EVIDENCE COVERAGE: **ADEQUATE** (UNRESOLVED 0.00%)
- SEMANTIC CONTRADICTION: **FATAL** -- R1 anchor의 **98.67%가 자기 생성 뷰에서 OCCLUDED**, 96.82%가 GLOBAL OCCLUDED. 원인은 blocker 중앙값 16개와 nearest_blocker_t 중앙값 0.99935: 학습된 2DGS 표면은 겹친 원판들의 두꺼운 수프여서, 표면 위 한 점으로 가는 광선은 반드시 같은 표면의 다른 원판들을 먼저 통과한다.
- ARCHITECTURE VIABILITY: **NOT VIABLE** -- 현재 표현 아래에서는 경성 line-of-sight가 자기모순적이다. 이를 고치려면 정확히 금지된 것(기하 허용 오차, opacity hardening 해제, 표면 재구성) 중 하나가 필요하다.

### Candidate D -- RENDERER REACHABILITY
- OPERATIONALIZATION FIDELITY: **PASS** -- 새 threshold 0개. probe는 canonical `test_T < 0.0001f`를 canonical 지점에서 관측만 하며, 진단 빌드가 WL107 빌드와 **비트 단위로 동일한** canonical 출력을 낸다는 것을 3개 테스트로 확인했다. 구현 중 발견한 결함 1건은 측정 보고 전에 교정하고 회귀 테스트를 남겼다.
- EVIDENCE COVERAGE: **ADEQUATE (per-view) / DEGENERATE (global)** -- per-view OCCLUDED 25.48%(97,676 쌍)로 신호는 충분하지만, GLOBAL OCCLUDED가 **정확히 0**이다.
- SEMANTIC CONTRADICTION: **LOW** (모순 0건, fail-closed 상태 0건)
- ARCHITECTURE VIABILITY: **NOT VIABLE AS STATED** -- 두 가지 이유. (1) canonical termination은 누적 투과도가 1e-4까지 떨어져야 발동하는 극도로 보수적인 조건이라 **단일 반투명 primitive 뒤를 OBSERVED로 본다**(S3b, 그리고 q=2670 같은 실 scene 사례). (2) frozen aggregation + 161뷰 아래에서 어떤 질의도 모든 relevant 뷰에서 동시에 종료되지 않아 **global 분할이 전혀 생기지 않는다.**

### 종합
**넷 중 어느 것도 의도한 Observed/Occluded 부피 구분을 그대로 지지하지 못한다.** A는 증거 기아, C는 치명적 자기모순, D는 global 무분할로 각각 실패한다. B만이 두 축(커버리지·모순)을 통과하지만, 그 통과가 "median depth가 가시성 경계다"를 증명하지는 않는다 -- S6는 오히려 그 면이 first-surface가 아님을 통제된 조건에서 보여준다. 따라서 **B = INCONCLUSIVE**이며, 이 배치는 승자를 선언하지 않는다.

주목할 구조적 사실 하나: B와 D는 **포함 관계**다(`OCCLUDED->OBSERVED: 654`, 역방향 0). 즉 이 두 가설은 서로 경쟁하는 별개의 경계가 아니라 **같은 축 위의 보수적/공격적 양 끝**이며, 진짜 경계가 있다면 그 사이 어딘가다. 그러나 그 사이를 고르는 일은 정확히 이 배치가 금지한 새 threshold를 도입하는 일이다.

---

## 19. Remaining Architecture Question

**"canonical 2DGS는 표면을 겹친 반투명 원판의 두꺼운 수프로 표현하는데, 이 표현 위에서 '직접 광도 감독이 도달하는가'라는 이분 판정을 세울 수 있는 기준면이 애초에 존재하는가? 아니면 Observed/Occluded는 본질적으로 연속적인 도달량(reachability)이며, 이분화 자체가 새 threshold 없이는 불가능한가?"**

이 배치가 제시한 근거: (a) B와 D가 같은 축 위의 두 끝점이고 그 사이가 654개 질의만큼 비어 있다는 포함 관계, (b) D의 T 분포가 OBSERVED 쌍에서 1.0까지 연속적으로 퍼져 있다는 사실(중앙값 0.886, 최솟값 1e-4), (c) C가 보여준 것 -- 이분화를 기하에 맡기면 표면 두께 자체가 자기모순을 만든다는 것. 세 근거 모두 "경계는 있는데 못 찾았다"보다 **"이분 경계라는 것이 이 표현 위에 자연적으로 존재하지 않는다"** 쪽을 가리킨다.

**이 배치는 이 질문에 답하지 않았고, 답하려 시도하지도 않았다.** visible-topology 수정, occluded-surface 구축, NURBS continuation, uncertain-Gaussian 생성 중 어느 것으로도 진행하지 않고 비교 증거 생산에서 멈춘다.

---

## 20. Exact Branch / Commit / Commands / Output Paths

**브랜치**: `arch/2dgs-coverage-first-surface`
**작업 시작 시점 커밋 SHA(부모)**: `9cd93e51c1c26c57f625d6fc3905540081d5b48c` ("Worklog 119: Visible-NURBS geometry/UV control correction ...")
**이 배치의 커밋 SHA**: `1f8e5b6` ("Worklog 120: Observed/Occluded volumetric operationalization audit -- all four candidates fail, no winner"). 사용자 지시에 따라 동시 진행 중이던 다른 에이전트의 worklog 119-1/119-2/119-3 변경분도 같은 커밋에 함께 담았다(아래 파일 목록의 `M` 항목 및 `torch_exact_knn_performance.py`/`torch_nurbs_performance_batch.py`/`wl119_*` 계열). 커밋 직전 두 작업 영역을 모두 덮는 테스트 166개(WL120 60개 + 렌더러/NURBS 스위트 100개 + 성능 트랙 6개)가 전부 통과했다.

**이 배치가 추가한 파일 (전부 신규, 추적 중인 파일은 0개 수정)**:
```
osn_gs/render/torch_surfel_query_depth_diagnostics.py
osn_gs/render/vendor/diff_surfel_rasterization_qdepth/          (cuda_rasterizer/*.h,*.cu, rasterize_points.*, ext.cpp, setup.py, 패키지 __init__)
scripts/build_surfel_extension_qdepth.bat
scripts/run_with_msvc_env.bat
scripts/devtools/observed_occluded/__init__.py
scripts/devtools/observed_occluded/shared.py
scripts/devtools/observed_occluded/engine.py
scripts/devtools/observed_occluded/query_bank.py
scripts/devtools/observed_occluded/synthetic_contracts.py
scripts/devtools/observed_occluded/candidate_a_surface_hit.py
scripts/devtools/observed_occluded/candidate_b_median_depth.py
scripts/devtools/observed_occluded/candidate_c_geometric_visibility.py
scripts/devtools/observed_occluded/candidate_d_renderer_reachability.py
scripts/devtools/observed_occluded_volumetric_audit.py
tests/test_observed_occluded_volumetric_audit.py
docs/worklogs/120_observed_occluded_volumetric_operationalization_audit.md
```

**이 배치가 수정한 추적 중 파일: 문서/메모리 인덱스 3건뿐** -- `docs/worklogs/README.md`(워크로그 목록 항목 추가), `docs/agent_memory/MEMORY.md` 및 신규 `docs/agent_memory/project_observed_occluded_volumetric_operationalization.md` / `project_worklog120_code_layout.md`(in-repo 메모리 미러). **코드/렌더러/테스트 중 기존 파일은 0건 수정.** `git status`에 보이는 `M`/`D` 항목(`docs/README.md`, `osn_gs/surface/torch_camera_induced_visible_adjacency.py`, `torch_camera_observed_chart_domains.py`, `torch_coverage_first_subset_partition.py`, `torch_nurbs.py`, `scripts/devtools/visible_nurbs_geometry_uv_control_correction.py`, `tests/test_camera_induced_visible_adjacency.py`, `tests/test_nurbs_surface.py`, `tests/test_renderer_native_pixel_surface_chart.py`, `tests/test_visible_nurbs_geometry_uv_control_correction.py`, 삭제된 `reports/primitive_evidence_comparison_*.json`)와 `docs/worklogs/119-1_*.md`, `119-2_*.md`는 **이 배치 시작 시점에 이미 작업 트리에 있던 다른 작업의 변경분**이며 이 배치가 건드리지 않았다.

**canonical/기존 진단 렌더러 무수정 확인**: `osn_gs/render/vendor/diff_surfel_rasterization/`(canonical)과 `osn_gs/render/vendor/diff_surfel_rasterization_diag/`(WL107) 모두 `git status`에 나타나지 않는다. WL107 빌드와의 비트 단위 동등성은 `TestQDepthCanonicalEquivalence`가 12개 필드(render, out_others, radii, representative_id, forward_accepted, contrib_ids, contrib_post_median, contrib_count, median_rho3d/rho2d/s_u/s_v)에 대해 probe on/off 양쪽에서 검사한다.

**진단 CUDA sibling 빌드 명령**:
```
scripts\build_surfel_extension_qdepth.bat 12.0
```
(JIT 빌드, pip 설치 안 함. `torch.utils.cpp_extension.load`는 캐시 히트에서도 `where cl`을 호출하므로 모든 실행이 vcvars 환경을 요구한다 -- 그래서 `scripts\run_with_msvc_env.bat` 래퍼를 추가했다.)

**테스트 명령 및 결과**:
```
scripts\run_with_msvc_env.bat .venv\Scripts\python.exe -m pytest tests\test_observed_occluded_volumetric_audit.py -q
  -> 60 passed

scripts\run_with_msvc_env.bat .venv\Scripts\python.exe -m pytest tests\test_surfel_representative_diagnostics.py tests\test_visible_nurbs_evidence_contract_closure.py tests\test_renderer_native_pixel_surface_chart.py tests\test_representative_only_visible_nurbs.py tests\test_surfel_rasterization_cuda.py -q
  -> 59 passed
```
변경분이 전부 diagnostic/devtools 코드와 새 진단 sibling 렌더러 안에 머무르므로 전체 회귀는 요구되지 않지만(지시 section 17), 기존 렌더러/진단 경로를 쓰는 스위트를 위와 같이 실행해 무영향을 확인했다.

**실 scene 재생 명령**:
```
scripts\run_with_msvc_env.bat .venv\Scripts\python.exe scripts\devtools\observed_occluded_volumetric_audit.py ^
  --checkpoint output\arch_2dgs_coverage_first_surface\2dgs_run1\30000\checkpoint.pt ^
  --out output\120_osn_gs_observed_occluded_volumetric_audit ^
  --device cuda --source-path DATASET --images images_8
```
전체 실행 시간 **110.5초**(그 중 161뷰 평가 107.0초, 렌더 패스 322회). 두 차례 실행에서 모든 수치가 동일했다(결정론적 재생 확인).

**출력 경로**:
```
output/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_volumetric_audit_report.json   (전체 측정치)
output/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz            ((N,V) per-view 상태 + 전체 provenance 배열)
output/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_query_table.json               (질의별 요약 테이블)
output/120_osn_gs_observed_occluded_volumetric_audit/<VIEW_NAME>/iteration_0000001/point_cloud.ply
output/120_osn_gs_observed_occluded_volumetric_audit/<VIEW_NAME>/render.ppm
output/120_osn_gs_observed_occluded_volumetric_audit/preview_png/<VIEW_NAME>.png
output/confirmed/_run_logs/120_observed_occluded_volumetric_audit_run.log
```
직전 worklog들의 export는 규약대로 `output/confirmed/`로 이동했다(118, 119 및 119 GPU 감사 산출물 11개 폴더).

**2단계 검토자를 위한 확인 경로**: 후보 의미론 -> 해석 -> 구현 -> 측정 증거의 연결은 (1) 각 `candidate_*.py`의 모듈 docstring(의도한 가설과 실제 결정 라인이 같은 파일 안에 있음), (2) section 3의 ownership map, (3) section 4가 인용한 격리 테스트, (4) 리포트 JSON의 `per_candidate.<X>` 블록 순으로 직접 대조할 수 있다. **자체 테스트 통과만으로 구현 충실도가 증명되었다고 주장하지 않는다** -- 위 경로가 직접 검사 가능하도록 만드는 것이 이 배치의 목적이다.
