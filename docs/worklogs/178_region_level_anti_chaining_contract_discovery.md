# Worklog 178 — W97 Region-Level Anti-Chaining Contract Discovery

## 1. INTENT ALIGNMENT

이번 배치는 immutable W97 Local Surface Decomposition의 원인을 읽기 전용으로 귀속한다. W97 partition, KNN, local-spacing gate, unsigned intrinsic `t_w` relation, deterministic edge order, derived concentration floor, structural core/ownership propagation/fallback/ambiguity, stable Gaussian ID, W154/W155/W171-W177 및 production은 변경하지 않았다. q2/q3, angular span, pathwise rotation, graph support는 관찰값일 뿐 membership rule이 아니다.

## 2. IMPLEMENTATION FIDELITY

신규 `osn_gs/surface/worklog_178_region_merge_diagnostics.py`는 W97이 이미 만든 `CandidateGraph`를 받아 같은 accepted-edge order와 region scatter state를 독립적으로 replay한다. baseline W97 partition이 유일한 subset ID source이고, diagnostic replay는 그 ID를 재판정하거나 feedback하지 않는다. `scripts/devtools/worklog_178_region_level_anti_chaining_contract_discovery.py`가 synthetic/real artifact, full event NPZ, JSON report, PNG review와 README를 생성한다.

Focused verification은 W178 신규와 W97 기존 test를 합쳐 `25 passed`다. 검증한 항목은 baseline subset-ID 불변성, deterministic chronology, q1/q2/q3 normalization, deterministic accepted-graph shortest path, graph-support accounting, five synthetic controls, stable-ID carry-through이다.

## 3. REAL POSITIVE CONTROL GROUNDING — TABLE RIM / SIDE

판정은 `REAL_TABLE_RIM_PROVENANCE_GAP`이다. W145/W155에는 fixed image-space `table_side` annotation과 `PARTIAL / MIXED` qualitative note가 있지만, historical W97에서 하나의 물리적 Table Rim/side로 검토된 stable Gaussian ID population, W97 subset IDs, accepted local edges, region-coherence rejected edges를 잇는 frozen lineage는 없다.

따라서 real Table Rim의 failure stage A/B/C/D/E를 판정하지 않았다. W171 `curved/vase`는 positive substitute로 사용하지 않았다. W155 PNG는 context-only review로 재사용했으며 physical continuity 또는 membership proof가 아니다.

## 4. HISTORICAL NEGATIVE CONTROL GROUNDING — PATIO→HEDGE/BACKGROUND

W96/W97 historical artifact-level evidence는 보존했다. 문서상 W96 giant component는 `894,378 / 1,197,331`이고 patio ground와 hedge/background가 locally plausible chain으로 연결돼 있었다. W97은 동일 historical graph에서 candidate `6,048,719`, spatial `5,156,342`, accepted local `4,015,325` edge, region-coherence rejected merge `553,357`을 기록했으며 giant의 descendant는 `31,564`, largest descendant는 `253,853`이었다.

현재 checkpoint replay는 active row `1,190,469`, W96-like giant `886,782`, W97 descendant `31,414`, recorded rejection `479,781`을 보였다. 하지만 historical population과 current population이 다르고 historical checkpoint hash/semantic stable-ID mask가 저장돼 있지 않아, current stable IDs와 chronology를 exact historical patio→hedge crosswalk라고 주장하지 않는다. 이 real negative evidence의 status도 `PROVENANCE_LIMITED`다.

## 5. TABLE-RIM FAILURE-STAGE ATTRIBUTION

`REAL_TABLE_RIM_PROVENANCE_GAP`. Local graph break, local pairwise failure, region concentration rejection, ownership split 중 어느 것이 Table Rim을 지배적으로 분할했는지는 현재 저장 evidence로 구별할 수 없다. 이 결과는 region-level replacement를 제안할 근거가 아니다.

## 6. W97 REGION-MERGE CHRONOLOGY

각 materialized inter-component event는 stable endpoint IDs, local unsigned alignment/angle, component A/B size, pre-A/pre-B/post q-spectrum, floor, accepted/rejected result, accepted 시 resulting size를 보존한다. Synthetic은 full JSON과 NPZ, current negative replay는 `real_negative_patio_hedge_merge_chronology.npz`에 저장했다. Current negative lineage에는 `1,309,578` inter-component event가 materialized됐다. 이벤트 저장은 W97의 merge decision을 바꾸지 않는다.

## 7. GLOBAL CONCENTRATION TRAJECTORY

W97이 보던 scalar는 q1이다. W178은 모든 materialized state에서 q1/q2/q3 전체를 기록하고 q1만 W97 floor와 비교한 historical result를 보존한다. Smooth curved 및 strongly bent synthetic positive는 local graph가 하나로 연결된 상태에서 q1 floor rejection을 보였고, pathological chain도 같은 stage에서 정지했다. 이 공통성은 scalar concentration이 two cases를 분리한다는 증거가 아니라, real positive provenance가 없는 상태에서 더 이상의 architecture inference를 금지하는 evidence다.

## 8. FULL SCATTER-SPECTRUM DIAGNOSTIC

모든 fixture event의 normalized `(q1, q2, q3)`와 real negative materialized event array를 report/NPZ에 기록했다. q2/q3 threshold, spectrum classifier, curvature-aware concentration은 추가하지 않았다. Full spectrum이 genuine curvature와 pathological chaining을 real control에서 분리한다는 판정은 OPEN이다.

## 9. LOCAL NORMAL-EVOLUTION DIAGNOSTIC

Fixture마다 frozen accepted local graph의 ascending-neighbor BFS shortest path를 사용했다. Per-hop unsigned angle, cumulative local rotation, endpoint angle, edge count, world path length, chord, path/chord ratio를 same schema로 기록했다. Smooth cylinder/quarter-cylinder는 distributed local rotation을, narrow pathological chain은 structurally narrow path support를 가진다. 그러나 real Table Rim path stable IDs가 없으므로 `PATHWISE_NORMAL_EVOLUTION_DISTINGUISHES_CASES`는 선언하지 않았다.

## 10. GRAPH-SUPPORT DIAGNOSTIC

Critical event 직전 component-pair state를 chronology에서 재구성해 accepted local cross-edge count, side별 distinct endpoint, local degree, shared neighbor, critical edge를 제외한 bounded alternate path, articulation-like flag, component-size ratio를 diagnostic-only로 기록했다. Current negative replay의 first rejection은 cross-edge 24개, side endpoint 13/12개, shared neighbor 2개, bounded alternate path 존재, articulation-like false였지만 historical semantic stable-ID crosswalk가 아니므로 new contract evidence가 아니다. bridge veto, shared-neighbor consensus, W150 import, graph-support threshold는 구현하지 않았다.

## 11. SYNTHETIC CONTRACTS

| Fixture | W96 plain CC | W97 | W97 rejection | 결과 |
|---|---:|---:|---:|---|
| planar positive | 1 | 1 | 0 | coherent planar control retained |
| smooth curved positive (120° cylinder) | 1 | 3 | 46 | region concentration fragmentation |
| strongly bent continuous positive (90° quarter cylinder) | 1 | 2 | 23 | region concentration fragmentation |
| pathological chain negative | 1 | 2 | 5 | region concentration stops chain |
| parallel/shortcut negative | 1 | 1 | 0 | unchanged default W97 does not stop parallel shortcut |

Synthetic PASS/FAIL은 real-scene architecture viability가 아니며, geometry를 chosen floor 근처에 맞춰 tune하지 않았다.

## 12. POSITIVE-vs-NEGATIVE MATCHED COMPARISON

모든 synthetic case에 final outcome, failure stage, first critical merge, pre/post concentration, q-spectrum, local-angle statistics, path rotation, cross-edge/endpoints/alternate support/articulation flag, region sizes를 동일 schema로 `output/178_region_level_anti_chaining_contract_discovery/worklog_178_report.json`에 기록했다. Real comparison은 positive가 `REAL_TABLE_RIM_PROVENANCE_GAP`, negative가 `PROVENANCE_LIMITED`이므로 measured real discriminator comparison으로 승격하지 않는다.

## 13. REAL-SCENE VISUAL REVIEW

`output/178_region_level_anti_chaining_contract_discovery/real_scene_qualitative_review/`에는 W155 same-checkpoint renderer context를 재사용한 `GAUSSIAN_SUBSET_WORLD`, `REGION_REJECTION_EVENT`, `RGB_REFERENCE` PNG와 README가 있다. RGB는 secondary context이고 continuity proof가 아니다. Table Rim/side는 context-only로 labelled 됐으며, exact stable-ID lineage가 없는 view를 positive evidence로 해석하지 않는다.

Synthetic case마다 `GAUSSIAN_SUBSET_WORLD`, `LOCAL_ACCEPTED_GRAPH`, `REGION_REJECTION_EVENT`, `NORMAL_FIELD`, `MERGE_TRAJECTORY`, `RGB_REFERENCE` PNG와 per-directory README를 생성했다.

## 14. ARCHITECTURE ATTRIBUTION

지원되는 final attribution은 `PROVENANCE_LIMITED`와 `REAL_TABLE_RIM_PROVENANCE_GAP`이다. Synthetic controls는 W97 global orientation concentration이 accumulated curvature와 narrow chain 모두에서 reject될 수 있음을 보이지만, real Table Rim failure stage가 확보되지 않았으므로 `GLOBAL_ORIENTATION_CONCENTRATION_IS_THE_WRONG_ABSTRACTION`, `SCATTER_SPECTRUM_CONTAINS_DISCRIMINATIVE_STRUCTURE`, `PATHWISE_NORMAL_EVOLUTION_DISTINGUISHES_CASES`, `GRAPH_SUPPORT_DISTINGUISHES_CASES`, `MULTIPLE_SIGNALS_REQUIRED`, `NO_CLEAN_DISCRIMINATOR_FOUND` 중 어느 것도 확정하지 않았다.

## 15. PROMOTED / RETAINED / REJECTED / OPEN

- Promoted: 없음.
- Retained: immutable W96/W97 baseline, intrinsic `t_w`, existing ownership semantics, W154/W155/W171-W177 history, current production.
- Rejected: parameter tuning, new concentration rule, path/bridge/consensus veto, W150 import, W171 curved/vase substitution, TSDF/triangle/chart based split, NURBS/continuation/Eligibility/latent/inferred Gaussian work.
- Open: Table Rim stable-ID lineage recovery; real failure-stage attribution; real positive-vs-negative matched discriminator; evidence가 충분해진 뒤 하나의 minimal next-contract hypothesis를 시험할지 여부.

No new region-level algorithm is reported or implemented by this Worklog.
