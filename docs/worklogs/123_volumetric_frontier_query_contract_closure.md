# Worklog 123: Volumetric Frontier Query Contract Closure

- 상태: 완료
- 범위: query representation / provenance contract / diagnostic reference audit
- 기준 checkpoint: output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt
- 실행 장면: 161 train cameras, WL120 original bank, WL121 supplemental bank, WL122 median-frontier corpus
- production Candidate B 및 topology 변경: 없음

## 1. Agent Interpretation of Intent

### DIRECTION

Worklog 122가 조건부로 수용한 canonical median-surface event를 renderer-defined visible-surface observation frontier로 유지한다. 이번 배치는 frontier의 의미나 수치 경계를 다시 설계하지 않고, 그 frontier에 대한 query contract를 닫는다.

### PURPOSE

OSN-GS의 전역 volumetric representation은 임의의 3D world-space 위치 x여야 한다. renderer에서 직접 생성된 frontier event에 한해 camera_id, pixel_id, stored_median_depth, 그리고 이미 존재하는 representative_id를 optional provenance로 보존한다.

### CENTRAL INTENT

world-space 3D를 canonical query abstraction으로 유지하면서, renderer-originated median event의 exact identity를 provenance로 보존할 수 있는지 측정한다. Worklog 122의 18.62% closure failure가 zero-thickness renderer-event round-trip에 국한되는지, 일반 임의 3D query를 실질적으로 불안정하게 만드는지, 그리고 heuristic numerical tolerance 없이 닫을 수 있는지를 판정한다.

### PRESERVE

- canonical renderer
- pre-update T > 0.5 median rule
- Candidate B classify_view
- frozen global ANY-OBSERVED aggregation
- checkpoint와 161 cameras
- WL107/WL109 topology
- WL120 original 4,712-query bank
- WL121 supplemental 908-query bank 및 300 true-fragmentation contexts
- WL122 median-frontier evidence와 S1-S5

### CHANGE ONLY

- frozen historical classifier 위에 별도 query representation/provenance layer
- 같은 stored inputs를 이용한 diagnostic float64 reference arm
- Worklog 122 post-median count-vs-weight accounting clarification
- query-contract focused test와 diagnostic artifact

### DO NOT

- Candidate B, T > 0.5, alpha, topology, NURBS 변경
- epsilon, ULP acceptance band, nextafter production correction, percentage threshold 도입
- median depth를 physical first hit으로 호칭
- renderer contribution을 direct physical surface observation으로 호칭
- Surface Representative, Surface Identity, Volumetric Observation을 동일시
- provenance로 다른 view의 visibility, global observation, ownership, surface continuation, trust를 결정
- topology repair, Occluded Surface, occluded NURBS, Trust 구현

### PROMPT-REQUIRED DECISION

측정 후 A, B, C, D architecture verdict 중 하나를 고른다.

### AGENT-INTRODUCED OPERATIONAL CHOICE

결과를 보기 전에 deterministic sampling을 고정했다. renderer-event anchor는 view stride 10, view당 200개로 총 3,400개를 선택했다. generic probes는 anchor ray의 camera-side t=0.5, behind-frontier t=1.5, 그리고 relative offset ±10^-2, ±10^-3, ±10^-4, ±10^-5, ±10^-6, ±10^-7 ladder를 사용했다. generic stability arm에서는 P1 exact-event anchor를 제외했다. float64는 diagnostic only이며, margin sampling stride 991은 저장량 제어일 뿐 판정 임계값이 아니다.

## 2. Worklog 122 Interpretation Corrections

역사적 Worklog 122의 raw measurements는 수정하지 않고 다음 architecture 해석을 교정한다.

1. renderer-visible somewhere는 same-surface redundant contribution을 뜻하지 않는다. Renderer Contribution, Surface Representative, Surface Identity, Volumetric Observation은 별도 개념이다.
2. same frozen visible component는 stronger provenance이지만 physical redundancy를 단독으로 증명하지 않는다.
3. post-median evidence가 overwhelmingly redundant representation이라는 Worklog 122의 문장은 확립된 결론으로 사용하지 않는다.
4. 2.24%는 stated marginal assumptions 아래에서, 어디에서도 median representative가 되지 않은 surfel의 post-median contribution weight에 대한 upper bound일 뿐이다. independent hidden-surface evidence의 정확한 양이 아니다.
5. 39.06% x 72.35% = 28.26%의 accounting을 count와 weight로 분리해 재검산했다. 27.65%에 해당하는 값은 contributor COUNT fraction이 아니라 contribution WEIGHT fraction이다.

## 3. Historical Frontier Preservation

canonical renderer, pre-update T > 0.5 median rule, Candidate B classify_view, global aggregation, checkpoint, 161 cameras, WL107/WL109 topology, WL120, WL121, WL122를 그대로 사용했다.

frozen state fingerprint도 median-surface representative union 785,937로 WL119-WL122 reference와 일치했다. Candidate B decision function과 shared aggregate_global은 호출만 되었고 수정되지 않았다. WL122 S1-S5는 보존했으며, 이번 배치에서 topology를 변경하거나 midpoint를 continuity evidence로 사용하지 않았다.

## 4. Proposed VolumetricQuery Contract

canonical object는 world_position float32 world-space xyz다. 항상 global volumetric representation이며 provenance가 이를 대체하지 않는다.

optional renderer_event_provenance는 camera_id, pixel_id, stored_median_depth, representative_id where already available를 담는다. world_position은 변경하지 않으며 confidence, trust, ownership semantics를 추가하지 않는다.

valid median-event provenance를 가진 query를 source view v에서 평가할 때, carried stored_median_depth와 현재 renderer의 같은 pixel stored median이 exact float32 equality로 일치하면 그 event를 ON_FRONTIER_BY_EVENT_IDENTITY로 표시하고 frozen per-view OBSERVED로 집계한다. 이는 tolerance가 아니라 provenance consistency guard다. stale median은 rejected되고 ordinary frozen comparison으로 남는다.

P1 anchor는 생성 시 view, pixel, stored median, representative를 모두 기록했다. WL121 endpoint는 saved context의 source view, source pixel, representative를 사용하고 source renderer output에서 그 pixel의 stored median을 materialize했다. 이는 world-to-camera 재투영으로 event identity를 재구성한 것이 아니다. WL121 midpoint와 out-of-frustum control에는 provenance를 만들지 않았다.

다른 view는 provenance와 무관하게 normal frozen geometric frontier comparison을 사용한다. global aggregation은 frozen ANY-OBSERVED rule 그대로다.

## 5. Exact Event-Identity Results

161개 view의 모든 source median event를 exhaustive하게 평가했다.

| 항목 | 결과 |
|---|---:|
| total source median events | 43,817,760 |
| historical float32 source OBSERVED | 35,660,438 |
| historical float32 source contradiction | 8,157,322 (18.6165%) |
| provenance-preserved OBSERVED | 43,817,760 |
| provenance-preserved contradiction | 0 |
| provenance applied | 43,817,760 |
| stored-median mismatch rejection | 0 |

historical float32 arm은 Worklog 122의 corpus와 contradiction을 그대로 재현했다. provenance-preserved arm은 모든 source event를 exact ON_FRONTIER identity로 유지했다.

diagnostic float64 reference arm은 source-event pair의 projected pixel을 43,817,760건 모두 동일하게 유지했지만 side는 21,746,643 OBSERVED / 22,071,117 contradiction으로 갈렸다. 이 값은 float64가 canonical이라는 뜻이 아니며, same stored float32 inputs에서 arithmetic arm을 바꾸면 zero-thickness event의 side 의미가 달라진다는 attribution이다. production decision은 이 arm을 사용하지 않는다.

## 6. General Arbitrary-3D Stability

fixed query bank는 총 21,652개다.

| query kind | queries | 설명 |
|---|---:|---|
| P1 renderer-event anchor | 3,400 | exact-event arm 및 cross-view용; generic stability에서 제외 |
| G1 observed free space | 3,400 | anchor ray의 t=0.5 |
| G2 behind frontier | 3,400 | anchor ray의 t=1.5 |
| G3 near-frontier ladder | 5,832 | 486 anchor x 12 fixed offsets |
| G4 WL120 original bank | 4,712 | 원본 bank 그대로 |
| G5 WL121 supplemental bank | 908 | endpoint/midpoint/control 그대로 |

P1을 제외한 generic arbitrary-3D arm의 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| generic query-view pairs | 2,938,572 |
| relevant pairs | 1,590,240 |
| float32/reference exact state agreement | 1,589,122 |
| OBSERVED/OCCLUDED disagreement | 1,118 |
| resolved/unresolved disagreement | 0 |
| projected-pixel agreement | 1,590,200 |
| relevant projected-pixel changes | 40 |
| state disagreement rate over relevant pairs | 0.0703039% |

kind별 relevant pair와 disagreement는 G1 400,703 / 0, G2 217,810 / 1, G3 517,842 / 55, G4 383,322 / 870, G5 70,563 / 192이다. G1 observed free-space는 전 pair가 일치했다.

near-frontier ladder disagreement 수는 offset -1e-2: 0, -1e-3: 0, -1e-4: 2, -1e-5: 0, -1e-6: 1, -1e-7: 13, +1e-7: 35, +1e-6: 3, +1e-5: 1, +1e-4: 0, +1e-3: 0, +1e-2: 0이었다. 이 offset은 측정 설계이며 production tolerance로 사용하지 않았다.

## 7. Numerical Reference Attribution

float32 Candidate B projection/depth와 같은 stored inputs를 사용해 float64 projection/depth reference를 diagnostic으로 계산했다. query-view pair별로 float32 projected row/column, reference projected row/column, 각 arm의 query depth, 해당 pixel의 stored median depth, side, signed margin, ULP attribution을 계산했다.

모든 generic pair의 per-pair fields는 output/123_osn_gs_volumetric_frontier_query_contract/volumetric_query_contract.npz에 보존했다. NPZ의 generic fields는 float32/reference pixel row/column, float32/reference query depth, float32/reference pixel의 stored median, canonical base state, reference state이며, query order와 view order도 함께 저장했다. exhaustive source-event arm은 43,817,760쌍을 streaming으로 평가하고 JSON aggregate를 남겼다.

generic state disagreement 1,118건의 귀속은 다음과 같다.

- 동일 projected pixel: 1,112건. reference signed margin 범위 -1.7828e-7 .. 1.4388e-6, float32 signed margin 범위 -9.5367e-7 .. 9.5367e-7.
- projected pixel change 동반: 6건. reference signed margin 범위 -0.0412571 .. 0.0262196. 해당 6건은 row 재배정이 동반된 discrete raster-pixel boundary case였다. 전체 relevant pair 중 이 경우는 6건이며, 별도 acceptance band나 margin cutoff를 도입하지 않았다.
- 전체 relevant pair에서 pixel change는 40건이었고, 그중 side까지 달라진 것은 6건이었다.

generic disagreement row의 float32 query depth 대 stored median ULP histogram은 0: 1,103, 1: 9, 64: 6이었다. float32 query depth 대 float64 reference depth의 diagnostic ULP histogram은 0: 959, 1: 158, 2: 1이었다. 이것은 측정 결과이며 ULP tolerance로 승격하지 않았다.

## 8. Cross-View Replay

Worklog 122 disocclusion anchor corpus 3,400개를 재생했다.

| 항목 | provenance 없음 | source provenance 유지 |
|---|---:|---:|
| source-view OBSERVED | 2,735 | 3,400 |
| source-view OCCLUDED | 665 | 0 |
| global OBSERVED | 3,381 | 3,400 |
| global OCCLUDED | 19 | 0 |

3,390개 anchor는 적어도 한 다른 view에서 hidden이었다. 역사적 19개 global-OCCLUDED contradiction은 모두 source-event identity 보존만으로 global OBSERVED가 되었다. provenance가 바꾼 것은 source view의 exact event identity뿐이며, 다른 view의 geometric result는 그대로였다. view-count rule은 추가하지 않았다.

## 9. True-Fragmentation Replay

WL121의 300 contexts, endpoint A 300개, endpoint B 300개, midpoint 300개와 8개 out-of-frustum control을 재생했다. gating attribution은 3D-locality rejection 288, secondary geometric gate 12, positive-edge-yet-split 0이었다.

- endpoint A: historical/base 290 OBSERVED / 10 OCCLUDED; provenance layer 300 OBSERVED / 0 OCCLUDED.
- endpoint B: historical/base 296 OBSERVED / 4 OCCLUDED; provenance layer 300 OBSERVED / 0 OCCLUDED.
- midpoint: provenance 없음 상태가 provenance layer와 동일하게 300 OBSERVED / 0 OCCLUDED.
- out-of-frustum control: 8 UNRESOLVED로 유지.

endpoint contradiction은 exact source identity로 사라졌고 midpoint classification은 unchanged였다. midpoint OBSERVED를 surface continuity나 component merge 근거로 해석하지 않았고 topology도 변경하지 않았다.

## 10. Post-Median Accounting Correction

WL122 post-median accepted contributor 전체는 다음과 같다.

| 항목 | COUNT | COUNT fraction | WEIGHT fraction |
|---|---:|---:|---:|
| all post-median contributors | 1,150,990,609 | 100% | 100% |
| depth in front of median | 248,820,747 | 21.617965% | 27.646166% |
| depth at or behind median | 902,169,862 | 78.382035% | 72.353834% |

post-median contribution의 전체 accepted contribution 대비 weight fraction은 39.054929%였다. 따라서 0.39054929 x 0.72353834 = 0.28257739, 즉 28.257739%이다. 두 factor가 모두 contribution-WEIGHT fraction이므로 historical 28.26% claim은 수학적으로 VALID하다.

교정할 문장은 count 248,820,747 옆에 27.65%를 같은 종류의 비율처럼 둔 부분이다. 248,820,747의 비율은 21.62% COUNT이고, 27.65%는 WEIGHT이다. 이 accounting은 post-median contribution을 redundant 또는 independent physical surface evidence로 분류하지 않는다. WL122의 2.24%는 여전히 marginal-distribution upper bound로만 기록한다.

## 11. Synthetic Query Contracts

WL122 S1-S5는 보존했다. query-specific Q1-Q5 결과는 다음과 같다.

- Q1 exact renderer median event + provenance: exact ON_FRONTIER_BY_EVENT_IDENTITY, source OBSERVED, global OBSERVED. PASS.
- Q2 same world coordinate with provenance removed: world position은 Q1과 bitwise identical하고 ordinary float32 result를 그대로 보고했다. 이 fixture는 pass/fail을 강제하지 않는다. 이번 결과에서는 signed margin 0.0, source OBSERVED였다.
- Q3 clearly camera-side point: provenance irrelevant, base/layered 동일, global OBSERVED, signed margin -2.0. PASS.
- Q4 clearly behind point: provenance irrelevant, base/layered 동일, global OCCLUDED, signed margin +2.0. PASS.
- Q5 source event가 다른 view에서 OCCLUDED인 경우: identity는 source view만 settle하고 다른 view는 normal frozen comparison을 사용했다. per-view layered result는 source identity OBSERVED와 other-view OCCLUDED를 함께 보존했고 frozen ANY-OBSERVED global result는 OBSERVED였다. PASS.

Synthetic fixtures는 contract check이지 architecture proof로 사용하지 않았다.

## 12. Implementation Fidelity Statement

이번 배치의 변경은 scripts/devtools/observed_occluded/volumetric_query.py의 query layer와 진단 runner, Q1-Q5 fixture, focused test에 한정된다.

- Candidate B classify_view decision rule은 수정하지 않았다.
- canonical renderer와 global aggregate_global은 수정하지 않았다.
- topology, NURBS, 3D-locality gate, Occluded Surface, Trust는 수정하지 않았다.
- provenance는 source-view exact event identity만 바꾼다.
- reference float64 arm은 diagnostic only다.
- epsilon, tolerance, nextafter correction, ULP band, view-count rule은 추가하지 않았다.
- query bank에는 dense voxel architecture를 만들지 않았다.
- 모든 43,817,760 source events는 exhaustive하게 처리했다.
- final focused regression: 32 passed.
- final 161-camera replay: exit code 0, report generation 17.3 seconds. qdepth build는 기존 local ninja cache를 사용했다.

## 13. Architecture Verdict

### A. WORLD-SPACE VOLUMETRIC QUERY + EVENT PROVENANCE IS A VIABLE CANONICAL QUERY CONTRACT

이 verdict를 선택한다.

근거는 세 층으로 분리된다.

1. renderer-originated frontier event는 world-space round-trip만으로는 8,157,322건이 source contradiction이지만, camera/pixel/stored-median provenance를 유지하면 43,817,760건 전부 exact identity를 보존한다.
2. 3,400 disocclusion anchors의 historical 19 global-OCCLUDED contradiction은 source identity만으로 모두 global OBSERVED가 된다. 다른 view 및 ANY-OBSERVED aggregation은 unchanged다.
3. P1을 제외한 일반 arbitrary-3D query는 1,590,240 relevant pair 중 1,118건만 diagnostic reference와 달랐다. 1,112건은 동일 pixel의 frontier-side arithmetic case이고, 6건은 row 재배정이 동반된 sparse discrete raster boundary case였다. broad free-space instability나 resolved/unresolved drift는 관측되지 않았다. 이 결과를 위해 float32 production rule을 바꾸거나 heuristic tolerance를 만들 필요가 없었다.

6 pixel-changed cases의 observed signed margin은 숨기지 않고 report했지만, 그 값으로 production threshold를 만들지 않았다. 이들은 general query population을 material하게 불안정하게 만드는 별도 volumetric failure mode로 판정하지 않았다. A는 median depth가 physical first hit이라는 뜻이 아니며, post-median contribution을 redundant physical surface evidence로 승인하는 뜻도 아니다.

## 14. Remaining Architecture Question

query representation contract는 여기서 닫는다. 남은 architecture 질문은 이후 모듈이 renderer-defined frontier의 OBSERVED/OCCLUDED label을 어떤 surface/topology semantics로 사용할지이며, 그것은 이번 query-contract batch의 범위를 벗어난다.

특히 다음 배치는 자동으로 시작하지 않는다.

- topology repair
- Occluded Surface construction
- occluded NURBS construction
- Trust implementation
- numerical tolerance tuning

## 15. Exact Branch / Commit / Commands / Outputs

### Branch

arch/2dgs-coverage-first-surface

### Commit

Worklog 123 implementation commit: 0122d42 (Close volumetric frontier query contract).

### Commands

Final full replay:

    $env:Path='C:\Projects\OSN-GS\.venv\Scripts;'+$env:Path; & .\scripts\run_with_msvc_env.bat .venv\Scripts\python.exe scripts\devtools\observed_occluded_volumetric_query_contract.py --checkpoint output\arch_2dgs_coverage_first_surface\2dgs_run1\30000\checkpoint.pt --out output\123_osn_gs_volumetric_frontier_query_contract --device cuda --source-path DATASET --images images_8

Final focused regression:

    $env:Path='C:\Projects\OSN-GS\.venv\Scripts;'+$env:Path; & .\scripts\run_with_msvc_env.bat .venv\Scripts\python.exe -m pytest tests\test_observed_occluded_volumetric_query_contract.py -q

Outputs:

- output/123_osn_gs_volumetric_frontier_query_contract/volumetric_query_contract_report.json
- output/123_osn_gs_volumetric_frontier_query_contract/volumetric_query_contract.npz
- output/123_osn_gs_volumetric_frontier_query_contract/ORIGINAL_2DGS_SCENE
- output/123_osn_gs_volumetric_frontier_query_contract/EVENT_IDENTITY_EFFECT
- output/123_osn_gs_volumetric_frontier_query_contract/NEAR_FRONTIER_LADDER
- scripts/devtools/observed_occluded/volumetric_query.py
- scripts/devtools/observed_occluded_volumetric_query_contract.py
- scripts/devtools/observed_occluded/query_contract_synthetics.py
- tests/test_observed_occluded_volumetric_query_contract.py

초기 replay는 local ninja.exe가 PATH에 없어 Ninja required 오류로 중단되었고, C:\Projects\OSN-GS\.venv\Scripts를 PATH에 추가한 동일 replay가 exit code 0으로 완료됐다. 이는 code/data failure가 아니라 실행 환경 visibility 문제였다.
