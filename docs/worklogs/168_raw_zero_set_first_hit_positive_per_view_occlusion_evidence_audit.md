# Worklog 168 — Raw Zero-Set First-Hit의 Canonical Positive Per-View Occlusion Evidence 감사

## 1. 의도 정렬

W167에서 viable camera-ray blocker로 확인한 historical raw SDF/TSDF zero-set first hit `z_s`를 canonical positive per-view `DirectObservationUnavailable` evidence로 승격할 수 있는지 검사했다. zero-set geometry, iso-value, `h`, `mu`, component population, Candidate-B, W161 paused spatial-domain status, primitive-observation/point-query 분리는 변경하지 않았다. top-20 또는 trusted subset을 만들지 않았고 모든 component를 blocker로 유지했다.

## 2. 구현 충실도

- W167의 `SparseProjectiveTSDF` construction, all-eight-corner zero-set extraction, two-sided exact ray-triangle first-hit primitive를 그대로 호출했다.
- candidate는 유효한 first hit에 대해 strict `0 < z_s < z_q`, analytic ground truth는 strict `0 < z* < z_q`로 정의했다. exact `z_q == z_s`는 `OCCLUDED`가 아니며 semantic epsilon을 추가하거나 sweep하지 않았다.
- W167 single-surface 세 fixture의 analytic-hit, zero-set-hit, premature/delayed, missed/false-hit count를 모두 exact 재현했다.
- query depth를 임의 간격으로 sampling하지 않고, `z_s < z*`이면 `(z_s,z*]`, `z_s > z*`이면 `(z*,z_s]` 전체 disagreement interval을 직접 측정했다.

## 3. Synthetic first-blocker 결과

| fixture | analytic hit | zero-set hit | premature | delayed | missed | false zero-set hit |
|---|---:|---:|---:|---:|---:|---:|
| fronto-parallel plane | 2,500 | 2,500 | 84 | 0 | 0 | 0 |
| oblique plane | 2,362 | 2,208 | 1,306 | 902 | 154 | 0 |
| curved sphere | 1,356 | 1,356 | 0 | 1,356 | 0 | 0 |
| layered two-sheet | 3,364 | 3,364 | 1,038 | 292 | 0 | 0 |

false zero-set hit은 없었지만 strict premature hit은 총 `2,428`개다. fronto-parallel plane의 normalized premature interval은 모두 `8.881784197001252e-15 h`이고, oblique plane은 min/median/mean/p95/max가 각각 `3.0711e-9 / 9.4914e-8 / 1.1645e-7 / 2.6517e-7 / 3.0167e-7 h`다. 값이 작더라도 strict ordering에서 `(z_s,z*]`는 실제 non-empty false-positive occlusion territory이며, 요청상 이를 epsilon으로 제거할 수 없다.

## 4. Multi-surface 및 boundary 결과

layered fixture는 world `z=-0.55` front sheet와 `z=0.65` rear sheet를 가진 동일 support의 two-sheet slab zero-set이다. 두 analytic surface를 모두 만나는 `1,936`개 ray에서 raw zero-set first hit은 전부 front sheet에 더 가까웠고 rear sheet first hit은 `0`이었다. exact rear-surface depth 및 그보다 `0.5` 뒤의 query는 모두 front zero-set surface에 의해 blocked로 분류됐다. target identity나 surface-selection heuristic은 사용하지 않았다.

premature count의 interior/boundary 분리는 fronto-parallel `84/0`, oblique `1,276/30`, sphere `0/0`, layered `970/68`이다. oblique miss `154`개는 모두 support boundary지만, premature counterexample는 interior에도 존재한다. boundary는 fixed 64×64 analytic-hit mask의 exact 8-neighbor adjacency로 attribution만 했고 verdict에서 제외하지 않았다.

## 5. Real-scene non-oracle check

새 real-scene ray replay나 physical hidden-surface label은 만들지 않았다. W167 frozen report를 읽기 전용으로 재사용했으며 3개 camera의 sampled ray `1,259`개가 모두 first hit을 가졌다. first-hit component는 총 `16`개이고 non-top-20 attribution hit `17`개도 모두 active blocker로 유지했다. 이 결과에는 physical ground truth가 없으므로 synthetic strict counterexample를 뒤집는 oracle로 사용하지 않았다.

## 6. Architecture 결과

최종 verdict는 **`RAW_ZEROSET_FIRST_HIT_NOT_PROMOTED_STRICT_PREMATURE_BLOCKER_COUNTEREXAMPLE`**다. `z_s < z_q`만으로 canonical positive per-view occlusion evidence를 선언하면 premature ray마다 `(z_s,z*]`에서 candidate는 blocked이지만 analytic ground truth는 blocked가 아닌 query가 존재한다. 따라서 요청된 strict/no-epsilon 계약에서는 승격하지 않고 counterexample를 보존한 채 중단한다.

Candidate-B, renderer median, W161 global spatial domain, global OBSERVED/OCCLUDED aggregation, final UNRESOLVED semantics, fragment organization, Eligibility, continuation, NURBS는 그대로 보류한다.

## 7. 산출물과 검증

- 결과: `output/168_raw_zero_set_first_hit_positive_per_view_occlusion_evidence_audit/`
- ray-level `fixture_audit.npz`, fixture JSON, 9-section `worklog_168_report.json`
- signed first-hit error 및 complete disagreement interval PNG `8`개, PPM `0`개, UTF-8 README `15`개
- 각 visualization directory는 의미, input/state semantics, palette/legend, 공통 rendering 조건, limitation을 개별 기록한다.
- W168 focused tests는 `5 passed`, W167/W168 combined focused regression은 `16 passed`다. strict ordering/exact boundary, complete interval accounting, fixed boundary split, 세 single-surface historical extraction, layered first-surface semantics를 확인했다.
- W153 replay cache는 `temp/`에 복사하지 않았다.
