# Worklog 163 — Renderer Contributor Provenance 및 Median-Event Observation-Semantics Attribution Audit

## 상태

완료. W162의 provenance gap을 existing isolated diff_surfel_rasterization_diag sibling으로 제한적으로 재현했다. Candidate-B, W160 state/cache, W161, W155–W162, production renderer, t_w, Region/TSDF/topology/Boundary First/NURBS/continuation은 변경하지 않았다. W161의 OCCLUSION_DOMAIN_CONTRACT_GAP과 pause 상태도 유지했다.

## 수행 내용

- W155 stable-ID mapping을 checkpoint row와 exact join하고 W155/W156/W157/W162의 Region lineage를 재조정했다. Region 0은 248,086행(accepted CORE+ATTACHED 247,547), Region 1은 65,471행(accepted 65,365)이며 두 Region의 stable-ID overlap은 0이다. W155 fixed tabletop image review는 여러 Region 후보를 포함했고, W156/W157의 primary는 Region 0, W162의 frozen tabletop positive control은 Region 1이다.
- W162가 freeze한 region_id=1 and GLOBAL_OCCLUDED 6,197행을 161개 training camera, 총 997,717 query-camera pair로 재생했다. exact stable ID, query participation, median representative, Region ID, membership, contributor slot/order, uncapped contrib_count를 w163_query_provenance_raw.npz에 저장했다.
- canonical production binding은 contributor identity를 반환하지 않지만, existing diagnostic sibling은 kernel의 동일한 T > 0.5 crossing에서 representative_id와 accepted contributor prefix(K=16), contrib_post_median, uncapped contrib_count를 반환한다. prefix에 query가 없고 count>16이면 NOT_CONTRIBUTOR가 아니라 PROVENANCE_UNAVAILABLE로 남겼다.
- 모든 3개 frozen review camera에서 diagnostic output과 canonical renderer의 RGB/alpha/median depth가 bitwise equal인지 검증했다. median depth도 W162 cache와 valid target pixel에서 bitwise equal이었다.
- required controls인 W162 tabletop/table-side/lower/vase-curved ROI와 W160 frozen background boxes를 같은 provenance contract로 accounting했다.

## 정량 결과

| 항목 | 결과 |
|---|---:|
| W162 target query-camera pair | 997,717 |
| QUERY_IS_EXACT_CONTRIBUTOR | 1,724 |
| QUERY_NOT_CONTRIBUTOR (uncapped count ≤ 16) | 970 |
| QUERY_CONTRIBUTOR_PROVENANCE_UNAVAILABLE | 995,023 |
| truncated pair (contrib_count > 16) | 996,709 |
| median same Gaussian | 2 |
| median same Region, different Gaussian | 904,400 |
| median different Region | 93,315 |
| query at / before / after median | 2 / 0 / 1,722 |
| exact conflict pair | 1,724 |
| unique conflict query stable ID | 183 |

GLOBAL_OCCLUDED tabletop query가 relevant camera의 같은 pixel에서 exact renderer contributor로 참여한 1,724 pair는 기계적으로 OCCLUDED_QUERY_RENDERER_CONTRIBUTOR_CONFLICT로 분류했다. 이를 physical false positive로 단정하지 않았다. 다만 exact conflict와 대규모 truncated/unavailable population이 동시에 존재하므로 최종 architecture result는 MIXED이다. 이는 median-event proxy의 전면적 성공도, unavailable pair의 전면적 non-contributor 판정도 의미하지 않는다.

Control record는 tabletop 869, table-side/lower 2,859, vase/curved 4,309, background 3,778건이다. tabletop ROI는 전부 truncated라 contributor identity가 unavailable이었고, table-side/lower에서는 exact contributor 16건이 확인됐다. diagnostic sibling은 per-pixel per-primitive alpha*T magnitude를 제공하지 않으므로 contribution magnitude에 임의 threshold를 적용하지 않았다.

## Synthetic 및 시각화

- Synthetic A–F가 모두 통과했다: same Gaussian, same-Region different Gaussian, exact query after median, complete-prefix non-contributor, different Region, truncated-prefix unavailable.
- output/163_renderer_contributor_provenance_median_event_observation_semantics_attribution_audit/ 아래에 Original Scene, Observed/Occluded Global State, tabletop Region, global-O tabletop, query contributor, same/different Region median contributor, conflict, common-world visualization을 PNG로 생성했다. 총 27 PNG, PPM 0개이며 각 visualization directory와 common_world에 개별 UTF-8 README를 두었다. camera 파일은 <camera_name_stem>.png를 directory 바로 아래에 둔다.
- common-world plot은 W155 Region 1 전체 65,471행을 표시하고, 그 안에서 W162 target 6,197행과 GLOBAL OBSERVED/GLOBAL OCCLUDED 및 exact conflict subset의 관계를 분리한다.

## 평가 및 남은 위험

- Intent alignment: PASS. Implementation fidelity: PASS. Stop Condition A는 provenance가 기존 diagnostic sibling으로 결정적으로 복원 가능하므로 발동하지 않았다.
- Architecture result: MIXED; mechanical conditional signal: OCCLUDED_QUERY_RENDERER_CONTRIBUTOR_CONFLICT.
- 남은 위험은 truncated pixel의 complete contributor sequence, per-primitive contribution magnitude, renderer median event의 physical first-hit truth이다. Region membership은 observation evidence로 승격하지 않았고, W161 spatial field와의 join도 수행하지 않았다.

## 산출물

- Script: devtools/demo/worklog_163_renderer_contributor_provenance_median_event_observation_semantics_attribution_audit.py
- Test: tests/test_worklog_163_renderer_contributor_provenance_median_event_observation_semantics_attribution_audit.py
- Report: output/163_renderer_contributor_provenance_median_event_observation_semantics_attribution_audit/worklog_163_report.json
- Raw provenance: output/163_renderer_contributor_provenance_median_event_observation_semantics_attribution_audit/w163_query_provenance_raw.npz
- Controls: output/163_renderer_contributor_provenance_median_event_observation_semantics_attribution_audit/control_provenance_records.json
