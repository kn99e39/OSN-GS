# Worklog 154 — Gaussian-Region-Owned TSDF Surface Support와 Boundary-First NURBS Construction

## 작업 내용

- `osn_gs/surface/torch_gaussian_region_owned_tsdf.py`에 Candidate F를 격리 구현했다. 2DGS surfel의 학습된 intrinsic `t_w` normal과 기존 `partition_surfels_region_coherent`의 Gaussian region ID를 Gaussian branch의 identity source로 유지한다.
- WL127 sparse projective TSDF의 `keys/value/support_count/h/mu` 계약을 직접 읽고, 여덟 corner가 authoritative이며 zero를 straddle하는 native cell마다 source cell key를 보존한 zero-surface sample을 생성한다. Mesh/Marching Cubes intermediate는 사용하지 않는다.
- 각 TSDF sample을 Euclidean nearest Gaussian에 무조건 association하고, 해당 Gaussian의 기존 region ID를 전달했다. 대규모 replay는 chunked `scipy.cKDTree`, 작은 경우는 `torch.cdist`를 사용하며 stable Gaussian ID tie-break와 distance accounting을 남긴다.
- accepted `core`/비-ambiguous `attached` sample만 native TSDF face adjacency로 connected component화했다. component별 local chart occupancy에서 boundary를 먼저 만들고, closed single-loop와 design-matrix rank가 확인된 경우에만 frozen WL139 `8x4`, degree 2, LSQ fitter를 실행한다. 나머지는 `ABSTAIN_REPRESENTATIVE`와 사유를 보존한다.
- `devtools/demo/candidate_f_gaussian_region_owned_tsdf.py`에 safe checkpoint loader(`weights_only=True`), real replay output, WL145 diagnostic review panel, event 1527 lineage, 그리고 동일 checkpoint/iteration/camera/resolution/background/renderer/row count의 `Original Scene` 및 `Observed-Occluded` pair export를 추가했다.

## 결과 및 평가

- synthetic Candidate F contract: `5 passed`.
- 기존 Gaussian region decomposition 회귀: `16 passed`.
- 저장소 전체 `pytest -q`는 테스트 본문을 `100%`까지 실행했지만, 기존 dirty worktree에서 삭제된 `temp/149_physical_sheet_evidence_vs_chart_extent_failure_attribution/physical_sheet_evidence_vs_chart_extent_failure_attribution_report.json`을 요구하는 historical test가 실패했다. 종료 cleanup에서도 Windows pytest 임시 디렉터리 권한 오류가 발생했다. 이 실패는 Candidate F focused/regression 경로와 무관하다.
- real replay는 동일한 2DGS checkpoint와 WL153 `field.npz`를 사용해 `COMPLETE_CANDIDATE_REPLAY`로 종료했다. direct TSDF zero-surface sample `21,235,312`개 중 Gaussian-region-owned support는 `20,426,913`개, 명시적 `UNOWNED_TSDF_SUPPORT`는 `808,399`개였고, native TSDF face-adjacency component `495,970`개를 만들었다. boundary-first 결과는 `MATERIALIZED_REPRESENTATIVE 1,263`개와 `ABSTAIN_REPRESENTATIVE 494,707`개로 accounting했으며, 총 runtime은 약 `1,045.8s`였다.
- 대규모 association은 `scipy_cKDTree`를 사용했고 rejection radius는 없었다. 결과·component·representative·lineage와 동일 조건의 시각화 pair는 `output/154_gaussian_region_owned_tsdf_boundary_first_nurbs/`에 저장했다.
- 고정 palette는 `OBSERVED=(0.10,0.85,0.35)`, `OCCLUDED=(0.92,0.18,0.18)`, `UNRESOLVED=(0.60,0.60,0.62)`이며 unresolved를 다른 상태로 자동 승격하지 않는다.

## 결정 및 보존한 제한

- TSDF는 patch identity를 재발견하지 않으며, Gaussian center를 observed surface로 사용하지 않는다.
- rejection radius, normal matching/vote, synthetic bridge, global PCA/extrema, latent continuation/occluded surface, event 1527 blacklist, 새 classifier/trust를 추가하지 않았다.
- WL149/WL148/WL145의 historical review와 event 1527의 `CLEAR_NOT_ON_INTENDED_SURFACE`는 lineage/qualitative review용으로만 보존한다. Candidate F geometry의 input, selection, rejection, coloring에는 사용하지 않는다.
- Candidate F는 현재 production `torch_visible_surface_construction` 경로로 승격하지 않은 isolated candidate다. 따라서 `docs/current_framework.md`의 production framework 서술은 변경하지 않았다.

## 남은 위험

- real replay의 native component 계산은 수천만 sample 규모라 약 17분 26초가 걸렸다. 실제 support가 Gaussian region별로 매우 잘게 분절되어 materialized 비율이 낮으므로, 이는 Candidate F의 boundary eligibility/rank guard에 의한 보수적 abstain 결과로 해석해야 한다.
- frozen WL145 semantic review cloud는 correspondence diagnostic일 뿐 자동 semantic correctness 판정이 아니다. 최종 qualitative review는 output pair와 component/boundary exports를 함께 확인해야 한다.
