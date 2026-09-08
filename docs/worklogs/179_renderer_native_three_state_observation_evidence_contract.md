# Worklog 179 — Renderer-Native Three-State Observation Evidence Contract

## 1. 의도 정렬 (INTENT ALIGNMENT)

Worklog 178을 이어가지 않고 renderer-native observation-evidence layer만 감사했다. surface, W97, TSDF zero-set, NURBS, continuation, trainer와 canonical renderer production behavior는 변경하지 않았다. F를 자동으로 만들지 않고 subject semantics와 independent GT를 검사했다.

## 2. 주제 타입 감사 (SUBJECT-TYPE AUDIT)

| 출력 | 주체 | 사건 | 의미 타입 |
|---|---|---|---|
| `forward_accepted[g]` | primitive-camera pair | canonical forward acceptance | Primitive Observation Evidence |
| `representative_id`, `contrib_ids` | pixel event | median T=0.5 identity | pixel-level primitive event |
| `query_reached`, `query_terminated` | camera-ray point query | accepted reach 또는 termination | Point-Query Occlusion Evidence 후보 |
| `depth_median` | pixel | composited depth | scalar depth proxy |

Primitive event와 point-query event를 서로 relabel하지 않았다.

## 3. 기존 관측 증거 구현 (EXISTING OBSERVATION-EVIDENCE IMPLEMENTATION)

`torch_observation_evidence.py`는 per-camera pixel depth와 `depth_epsilon` band로 free/on-surface/behind를 만든다. 연속 depth와 선택 epsilon에 의존하므로 threshold-free F가 아니다.

W164 diagnostic fork는 canonical forward pass의 `forward_accepted[g]`를 노출한다. 이는 primitive positive contribution이며 false bit에는 blocker 원인이나 identity가 없다. W120–W122 qdepth fork는 `query_T`, `query_terminated`, `query_reached`, `query_prefix_count`, resolution depth/alpha, late-front와 inversion fields를 이미 노출한다.

## 4. Renderer 증거 의미론 (RENDERER EVIDENCE SEMANTICS)

`forward_accepted=1`은 primitive-camera accepted contributor event이며 visible surface support나 surface existence가 아니다. `representative_id/contrib_ids`는 pixel event identity다. qdepth termination은 query depth 전에 canonical T termination이 발생했다는 positive renderer traversal event지만 physical blocker identity를 반환하지 않는다. qdepth reached는 accepted event가 query depth에 도달했다는 뜻이지 query 위치의 surface hit가 아니다. 두 flag가 모두 0이면 exhausted/unresolved fill이고 unknown을 OCCLUDED로 바꾸지 않는다. CUDA median은 E[1/z]를 뒤집은 approximate quantity다.

Primitive Observation Evidence, Point-Query Occlusion Evidence, Visible Surface Support, Surface Existence, Latent Surface Eligibility는 하나의 타입이 아니다.

## 5. 형식 OBSERVED / OCCLUDED / UNRESOLVED 계약 (FORMAL CONTRACT)

paper-level F는 정의하지 않았다. primitive-camera에서는 `forward_accepted=1`만 OBSERVED witness이며 false는 UNRESOLVED로 남겨야 한다. primitive OCCLUDED witness는 없다.

point-query raw event는 `REACHED_ACCEPTED_EVENT`, `TERMINATED_BEFORE_QUERY`, `UNRESOLVED_RENDERER_EVENT`로 결정론적으로 분할된다. 이를 OBSERVED/OCCLUDED/UNRESOLVED로 단순 rename하면 reached에 direct point observation, termination에 physical blocker라는 의미를 발명하게 되므로 승격하지 않았다.

## 6. Threshold-free 감사 (THRESHOLD-FREE CONTRACT AUDIT)

raw qdepth partition은 bool/int flags만 사용하며 새 magnitude threshold를 추가하지 않았다. kernel의 `0.0001f`는 existing canonical termination event다. historical `depth_epsilon`은 semantic state를 바꾸므로 F로 재사용하지 않았다. W165 strict median counterexample와 W168 strict zero-set premature blocker counterexample를 proxy repair 근거로 사용하지 않았다.

## 7. Diagnostic rasterizer 변경 (DIAGNOSTIC RASTERIZER CHANGES)

변경 없음. 기존 diag/qdepth output이 필요한 최소 진단량을 이미 제공한다. canonical renderer, trainer, optimizer, gradient semantics는 건드리지 않았다.

## 8. 합성 true-GT 장면 (SYNTHETIC TRUE-GT SCENE)

`controlled_scene.json`에 foreground blocker z=2, rectangle `[-0.5,0.5]^2`, background wall z=4, rectangle `[-2,2]^2`, front camera `[0,0,0]`, offset camera `[3,0,1]`, fixed 64x64 parameters를 저장했다. GT는 analytic first intersection before query이면 OCCLUDED, earlier mesh intersection이 없으면 OBSERVED, target surface와 admissible event가 모두 없으면 UNRESOLVED다.

`renderer_event_records.json`은 qdepth schema를 가진 four-record control이다. adversarial termination row는 semantic gate fixture이며 live CUDA measurement로 가장하지 않는다. live independent counterexamples는 W165/W168 frozen artifacts를 retained reference로 사용했다.

## 9. Per-camera evidence

각 query slot은 camera, world query, analytic ray fact, qdepth flags, resolution depth/alpha, T, prefix count, late-front/inversion provenance를 가진다. front camera는 blocker/clear control, offset camera는 unsupported control을 제공한다.

## 10. Global aggregation

raw event에 한해 reached witness가 우선하고, reached가 없으며 모든 relevant view가 terminated일 때만 terminated, unresolved view가 있으면 unresolved로 보존한다. majority vote와 confidence-weighted vote가 아니다. paper-level F aggregation으로 승격하지 않았다.

## 11. Depth baseline

Historical median-depth ordering은 reference baseline이다. `depth_epsilon` rule은 threshold-free가 아니며 physical first-hit depth라고 부르지 않았다. W165 fixed plane/oblique/sphere replay의 strict counterexample는 retained했다.

## 12. Transmittance baseline

qdepth termination은 renderer traversal event로 기록했다. transmittance magnitude를 score나 fitted threshold로 만들지 않았다. canonical termination constant는 physical blocker identity나 surface existence certificate가 아니다. raw unresolved를 임의 상태로 강제하는 baseline은 채택하지 않았다.

## 13. Three-state confusion / accounting

승격된 F가 없으므로 3x3 confusion matrix, per-state precision/recall, macro F1은 산출하지 않았다. raw control count는 `REACHED_ACCEPTED_EVENT=1`, `TERMINATED_BEFORE_QUERY=2`, `UNRESOLVED_RENDERER_EVENT=1`이다. relevant subjects=4, reached witness=1, blocker event witness=2, insufficient evidence=1이다. F metrics와 collapse/error counts는 N/A (F rejected)다.

## 14. Counterexamples

W165 curved-sphere coarse representative는 world query `[0.8168298169543378, 0.2722766056514459, -0.32785195413564816]`, pixel `(38,51)`, `m=3.3358161`, `z_query=3.6721480`, `z*=4.0084799`이며 Candidate-B는 OCCLUDED지만 independent GT는 direct access다. Dense sphere에도 `m=3.6581535 < z_query=3.8550744 < z*=4.0519955`가 재현됐다.

W165 finite analytic surface 밖 valid median은 blocker 없는 query를 OCCLUDED로 강제한다. W168 zero-set positive blocker는 fronto-parallel 84, oblique 1,306, layered 1,038 strict premature events를 보였고 W169는 stable oblique geometric front-bias 1,306과 numerical-only 1,122를 분리했다. W179 adversarial raw row는 termination event와 independent GT OBSERVED의 불일치를 보존하지만 live prevalence로 주장하지 않는다.

## 15. Review exports

`output/179_renderer_native_three_state_observation_evidence_contract/`에 `gt_state_map`, `renderer_event_partition`, `evidence_reason_map`, `renderer_event_gt_disagreement` PNG와 각 visualization README, `subject_type_audit.json`, `controlled_scene.json`, `renderer_event_records.json`, `worklog_179_report.json`을 저장했다. `RENDERER_NATIVE_F_MAP`, Gaussian Original/Observed-Occluded pair, F-vs-baseline map은 F gate 실패로 만들지 않았다. W153 replay cache는 복사하지 않았다.

## 16. Testing

W179 focused property tests는 `7 passed`이다. 기존 `test_observation_evidence.py`는 `14 passed, 1 warning`이며 두 묶음 합계는 `21 passed, 1 warning`이다. production/shared renderer API가 변경되지 않아 canonical regression은 실행 대상이 아니다. audit module AST check로 surface/core dependency가 없음을 확인했다.

## 17. Architecture verdict

최종 verdict는 **`NO_RENDERER_NATIVE_THREE_STATE_CONTRACT`**다. stop reasons는 `SUBJECT_TYPE_MISMATCH`와 `OCCLUDED_NOT_POSITIVELY_IDENTIFIABLE`이다. Primitive contributor evidence는 point-query occlusion이 아니며 qdepth raw termination/reachability는 physical blocker/direct point observation semantics를 독립적으로 보장하지 않는다. threshold-free epistemic F를 구현하지 않고 architecture review에서 중단했다. paper novelty를 주장하지 않으며 downstream surface work를 재개하지 않는다.

## 18. Promoted / Retained / Rejected / Open

- Promoted: 없음.
- Retained: `forward_accepted` primitive separation, qdepth raw fields, W160/W164 separation, W165/W168 independent counterexamples, canonical renderer/production behavior, historical surface contracts.
- Rejected: primitive contribution을 point-query occlusion으로 relabel, reached를 direct surface observation으로 relabel, median을 physical blocker로 relabel, zero contribution을 OCCLUDED로 relabel, tuned F threshold, NURBS/continuation/completion 진입.
- Open: independent blocker identity 또는 sound direct point-observation witness를 제공하는 별도 evidence design; architecture review; downstream surface work paused.
