# Worklog 155 — Intrinsic-Normal Gaussian Surface Region Real-Scene Viability and Fragmentation Attribution Audit

## 상태

`COMPLETE_GAUSSIAN_REGION_VIABILITY_AUDIT`

이번 배치는 W154 Candidate F의 Gaussian-side Local Surface Decomposition을 감사하는 진단 작업이다. 현재 partition을 튜닝하거나 production 경로를 바꾸지 않았으며, downstream 결과도 read-only로 재조인했다. 최종 architecture verdict는 자동 확정이 아니라 `UNRESOLVED`이다. Gaussian fragmentation 신호는 정량화했지만, 한 physical sheet의 과분할/과병합 여부는 matched real-scene review가 필요하다.

## 구현 계약과 입력 경로

- 정확한 active path는 trained 2DGS surfel → `model.get_tangent_u()`, `model.get_tangent_v()`, `model.get_normal()`의 세 번째 회전축 `t_w` → `derive_surface_orientation_from_surfel` → 기존 `partition_surfels_region_coherent` → region ID이다.
- partition에는 checkpoint의 positions, intrinsic `t_w`, stable Gaussian ID와 기존 partition parameter만 입력했다. `t_u`, `t_v`, tangent scale은 provenance로만 운반하고 membership 판단에는 사용하지 않았다.
- covariance construction, covariance minor-axis normal, Gaussian covariance eigendecomposition, `lambda2/lambda3` membership rule은 추가하지 않았다. 기존 region coherence에서 쓰는 `M_R = sum_i n_i n_i^T`의 largest-eigenvalue/trace concentration은 normal scatter accounting이며 Gaussian covariance가 아니다.
- 기존 graph만 재사용했다. 새 KNN graph, spatial graph, threshold, small-region merge/split, boundary-first, NURBS refit은 만들거나 변경하지 않았다.

## Standalone Gaussian replay

입력 checkpoint는 `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt` (`iteration=30000`)이며 1,190,469개 surfel이 모두 active였다. 동일한 `RegionCoherenceConfig()`로 partition을 두 번 독립 실행했다.

- 최종 region ID: `104,977`
- accepted Surface Region: `64,892`
- accepted Gaussian population: `1,146,852`
- replay deterministic: `True`
- 두 실행 mapping SHA256: `06c9e1cbc730f06581895b32ad683e8822c7626eb3de9017fa8f83aaf0248bce`
- W154와의 region/stable-ID/accepted-mask exact join: `True`

## 전역 region accounting

Membership status는 다음과 같다.

| 상태 | Gaussian 수 |
|---|---:|
| core | 1,115,158 |
| attached | 31,694 |
| ambiguous | 3,532 |
| rejected member | 0 |
| unassigned / isolated fallback | 40,085 |

전체 104,977개 final region의 크기는 min 1, median 2, p90 11, p95 20, p99 88, max 248,086이었다. singleton은 40,085개, size ≤ 4는 78,852개, size ≤ 8은 91,533개였다. accepted region만 보면 min 2, median 4, p90 16, p95 31, p99 135, max 248,086이며, accepted population에서 largest region 1/5/10개의 비율은 각각 0.216319 / 0.330285 / 0.358279이다.

이 수치는 Gaussian-side fragmentation/over-merge review의 정량 신호이지 physical-sheet ground truth가 아니다. 특히 많은 singleton과 작은 region은 isolated fallback의 결과로 별도 accounting했으며 자동 병합하지 않았다.

## 기존 graph와 normal coherence

- candidate edges: `6,016,599`
- spatial pass / cut: `5,132,180` / `884,419`
- all-candidate normal-compatible: `4,591,855`
- spatial-edge normal-compatible 및 accepted edges: `3,986,975`
- spatial normal-cut conflict: `1,145,205`
- region-coherence rejected merge: `562,875`
- accepted edge 중 final region 내부 / region boundary crossing: `3,554,646` / `432,329`
- region-internal disconnected component: `0`
- accepted internal `|dot(t_w_i,t_w_j)|`: median `0.980489`, p95 `0.999203`
- spatial-edge `|dot(t_w_i,t_w_j)|`: median `0.966144`, p95 `0.998932`

Boundary/conflict export는 기존 graph의 `normal_cut`과 partition의 `rejected_merge_mask`만 표시하며 새 graph를 구성하지 않았다.

## Real-scene review export

WL145–154에서 고정한 세 카메라 `DSC08043.JPG`, `DSC07960.JPG`, `DSC08003.JPG`를 사용했다. 각 카메라에서 동일 checkpoint/iteration/해상도 `(648,420)`/black background/`OSNSurfelRasterizer`로 다음을 내보냈다.

1. `A_original_scene`
2. `B_intrinsic_tw_normal`
3. `C_accepted_region_ids`
4. `D_membership_status`
5. `E_boundary_conflict`
6. `F_original_plus_region_overlay` 및 fixed review target annotation overlay

검토 대상은 `tabletop`, `table_side`, `vase_neighbor`, WL140의 `background_lower` control box이다. 이 annotation은 membership predicate나 physical-sheet oracle이 아니다. 세 카메라 모두 tabletop/table-side/vase-neighbor에서 region ID `0`/`1`이 큰 비중을 차지하지만, 주변 region ID와 ambiguous/unassigned 상태도 함께 보존했다. `background_lower`도 독립 control로 내보냈다.

Gaussian visualization contract의 필수 `Original Scene`/`Observed-Occluded` matched pair는 W154의 동일 checkpoint/iteration/camera/resolution/background/renderer/row-count 결과를 그대로 복사해 보존했다. W155의 A–F 결과는 이 canonical pair를 대체하지 않는 별도 진단 view다.

## W154 downstream read-only attribution

W154의 `candidate_f_association.npz`, `candidate_f_region_owned_support.npz`, `support_components.json`을 읽어 standalone Gaussian region ID에 재조인했다. partition이나 Candidate F ownership에 feedback하지 않았고 TSDF association/connectivity도 바꾸지 않았다.

- association/support samples: `21,235,312` / `21,235,312`
- native support component records: `495,970`
- accepted Gaussian region별 native component 수: median `2`, p90 `13`, p95 `22`, p99 `69`, max `41,319`
- associated sample 수는 WL154 support 배열의 모든 valid nearest `region_id`를 세며, owned sample 수만 accepted owned entries를 센다.
- accepted region 중 associated TSDF sample이 0인 region: `13,031`
- native component가 0인 region: `13,056`

연결 샘플 수가 큰 region은 region 0 (`3,140,892` associated / `3,133,747` owned / `41,319` components), region 2 (`977,567` / `976,227` / `11,704`), region 5 (`739,957` / `738,742` / `11,008`), region 4 (`208,388` / `208,134` / `2,770`), region 7 (`151,390` / `151,094` / `1,638`) 순이었다. W154의 materialized representative `1,263`개와 abstain accounting은 그대로 유지했다.

## Failure attribution과 결론

다음 네 가설을 합치지 않고 분리해 보존했다.

- A: Gaussian region over-fragmentation
- B: Gaussian region over-merge
- C: plausible Gaussian region이 TSDF support를 fragmented하게 받는 경우
- D: plausible Gaussian region association leakage
- E: mixed attribution

A–D는 정량 신호와 review export를 제공하지만 모두 `HUMAN_REVIEW_REQUIRED`이며, E는 자동 선택하지 않았다. 따라서 이번 배치의 architecture verdict는 `UNRESOLVED`이다. 이는 replay 미완료가 아니라, 물리적 sheet identity를 Gaussian count/component count만으로 판정하지 않도록 한 보수적 결론이다.

## 평가, 보존, 잔여 위험

- 새 diagnostic runner와 W155 focused tests를 추가했다. 시각화 README 생성 테스트를 포함해 기존 region-coherent partition 회귀와 합쳐 `21 passed, 1 warning`이다. warning은 Windows `.pytest_cache` 권한 경고이며 테스트 실패가 아니다.
- 생산 partition, checkpoint, renderer, WL154 Candidate F, TSDF association, NURBS fit, event 1527 blacklist는 변경하지 않았다.
- event 1527 lineage는 `CLEAR_NOT_ON_INTENDED_SURFACE`로 보존했고 `blacklist_applied=false`, 새 event-to-Gaussian correspondence도 만들지 않았다.
- 남은 작업은 사람이 세 카메라의 A–F matched export를 보고 physical-sheet plausibility와 A/B/C/D 중 원인을 판정하는 것이다. 그 전에는 Gaussian region을 병합·분할하거나 production path에 반영하지 않는다.

## 산출물

- 전체 report: `output/155_intrinsic_normal_gaussian_region_viability_audit/worklog_155_report.json`
- stable Gaussian ID → region/status mapping: `output/155_intrinsic_normal_gaussian_region_viability_audit/gaussian_id_region_status_mapping.npz`
- mapping hash: `output/155_intrinsic_normal_gaussian_region_viability_audit/gaussian_id_region_status_mapping.sha256`
- per-region W154 attribution: `output/155_intrinsic_normal_gaussian_region_viability_audit/w155_tsdf_attribution_per_region.json`
- real-scene views: `output/155_intrinsic_normal_gaussian_region_viability_audit/review_views/`
- mandatory visualization pair: `output/155_intrinsic_normal_gaussian_region_viability_audit/mandatory_gaussian_visualization_pair/`
- review render PPM 23개는 같은 경로·이름의 PNG 23개로 변환했고, report의 검토 경로도 PNG를 가리키도록 갱신했다. 원본 PPM 23개는 provenance 보존을 위해 삭제하지 않았다.
- output root부터 pair/view/camera/iteration 하위까지 시각화 관련 디렉터리 39곳 모두에 UTF-8 `README.md`를 추가했다. 각 README는 view 의미, 입력/state semantics, palette/legend, 공통 렌더 조건, review 제한을 설명한다.
