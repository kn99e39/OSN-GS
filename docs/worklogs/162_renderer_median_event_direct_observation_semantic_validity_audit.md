# Worklog 162 — Renderer Median-Event Direct-Observation Semantic Validity Audit

## 상태

W160의 historical Candidate-B median_depth - query_depth ordering을 변경하지 않고, red GLOBAL_OCCLUDED가 image-space projection overlap인지 W155 Gaussian Surface Region query membership인지, 또는 median-event proxy conflict인지 분리 감사했다. 최종 verdict는 RENDERER_PROVENANCE_CONTRACT_GAP이다. query Gaussian의 W155 Region join은 가능하지만 renderer median event를 만든 Gaussian stable ID/Region ID가 기존 artifact에 없어 same-region conflict를 확정할 수 없다.

## 수행 내용

- W160과 동일한 checkpoint, 161개 training camera, camera calibration, W153 confirmed renderer_median_depth_maps.npz, s=median_depth-query_depth, historical Candidate-B/global aggregation을 exact replay했다.
- checkpoint model_raw.stable_gaussian_ids와 W155 gaussian_id_region_status_mapping.npz를 exact join했다. 새 world-distance threshold는 만들지 않았다.
- W155 fixed tabletop review에서 반복적으로 사용된 기존 region_id=1을 frozen tabletop positive control로 사용했다. 이 population은 65,471개이며 world XYZ, Region ID, membership status를 모두 보존했다.
- 세 matched camera의 fixed tabletop/table-side/vase ROI에 투영된 exact global-red query를 A/B/C로 분류했다.
- positive control에 대해 161개 camera별 s distribution과 local state를 저장하고, global-occluded tabletop candidate의 full cross-view raw record를 NPZ/JSON으로 저장했다.
- renderer cache에서 median event contributor identity artifact를 점검했다. depth-only median map은 pixel scalar depth만 저장하며 contributing Gaussian stable ID를 보존하지 않는다.
- W161은 spatial field를 구성하지 않고 OCCLUSION_DOMAIN_CONTRACT_GAP에서 멈췄으므로, W162에서 spatial nearest join이나 synthetic domain을 만들지 않았다.

## 정량 결과

| 항목 | 결과 |
|---|---:|
| W155 region_id=1 tabletop population | 65,471 |
| global OBSERVED / OCCLUDED / UNRESOLVED | 59,274 / 6,197 / 0 |
| fixed tabletop ROI에 투영된 global-red | 869 |
| 그중 A: 기존 tabletop Region 밖 | 625 |
| 그중 B: 기존 region_id=1 | 175 |
| 그중 C: W155 ambiguous membership | 69 |
| cross-view raw global-red tabletop candidate | 6,197 |

세 ROI 전체(global-red projected)를 합친 control accounting은 tabletop 869, table-side/lower 2,859, vase/curved neighbor 4,309이다. tabletop ROI에서 A와 B가 동시에 존재하므로 image/world signal의 조건부 분류는 MIXED다. 그러나 B의 median event가 같은 Region인지 확인할 contributor identity가 없어 최종 분류는 provenance gap으로 보수적으로 유지했다.

tabletop positive control의 global-occluded 6,197개는 161개 relevant local state에 OBSERVED 또는 UNRESOLVED가 없다는 historical all-relevant invariant를 만족한다. GLOBAL_UNRESOLVED population은 0개여서 해당 group의 s 분포는 empty이며 이를 zero나 observed로 채우지 않았다. s는 threshold 없이 min/median/p05/p25/p75/p95/max/exact-zero를 보존했다.

## Synthetic A–E

- A: exact surface alignment은 s=0, historical state는 OBSERVED다.
- B: query가 foreground median event 뒤에 있으면 s<0, OCCLUDED다.
- C: 같은 Region의 서로 다른 depth samples는 ordering만 노출하며 physical event identity를 증명하지 않는다.
- D: image projection overlap이지만 기존 world Region 밖이면 PROJECTS_ON_TABLETOP_BUT_WORLD_LOCATION_ELSEWHERE다.
- E: 한 camera라도 OBSERVED이면 global OCCLUDED가 되지 않는다.

모두 통과했다. 이는 mechanics 검증이지 hidden-surface truth가 아니다.

## 산출물 및 검증

- W162 output README: output/162_renderer_median_event_direct_observation_semantic_validity_audit/README.md
- W162 report: output/162_renderer_median_event_direct_observation_semantic_validity_audit/worklog_162_report.json
- W162 tabletop population NPZ: output/162_renderer_median_event_direct_observation_semantic_validity_audit/w162_tabletop_population_audit.npz
- W162 cross-view raw NPZ: output/162_renderer_median_event_direct_observation_semantic_validity_audit/w162_tabletop_cross_view_raw.npz
- W162 cross-view records: output/162_renderer_median_event_direct_observation_semantic_validity_audit/tabletop_cross_view_records.json
- W162 audit script: devtools/demo/worklog_162_renderer_median_event_direct_observation_semantic_validity_audit.py
- W162 focused test: tests/test_worklog_162_renderer_median_event_direct_observation_semantic_validity_audit.py

PNG 27개, PPM 0개를 생성했고 output의 11개 directory 모두에 UTF-8 README.md를 두었다. PNG는 original_scene, global_state_pure, global_state_overlay, tabletop/state별 view, common_world의 perspective/top/side로 구성되며 camera PNG는 각 visualization directory 바로 아래에 둔다.

focused W162/W160/W161 tests는 11 passed다. local CUDA replay는 약 25.18s에 완료했다. production renderer, checkpoint, Candidate-B, W155 mapping, W154–W161 결과와 spatial field는 변경하지 않았다.

## 남은 위험

GLOBAL_OCCLUDED query가 기존 tabletop Region에 속한다는 사실만으로 median event가 같은 Region의 physical foreground임을 증명할 수 없다. renderer contributor stable ID/Region provenance가 추가되기 전에는 SAME_REGION_MEDIAN_ORDERING_CONFLICT를 계산하거나 MEDIAN_EVENT_PROXY_CONFLICTS_WITH_VISIBLE_SURFACE를 확정하지 않는다. W161 spatial field가 없으므로 W162 result와 W161 spatial cell을 직접 join하지 않았다. 최종 판단에는 human review가 필요하다.
