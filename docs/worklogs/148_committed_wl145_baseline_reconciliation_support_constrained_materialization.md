# Worklog 148 — Committed WL145 baseline reconciliation 및 support-constrained materialization audit

## 상태: 완료 — 격리된 비정규 baseline reconciliation / A-B materialization 감사

WL145의 frozen report, per-view renderer event NPZ, frozen representative를 읽기 전용으로
재생했다. WL145 representative를 다시 fit하지 않았으며, chart/PCA, h/mu, support
occupancy, Candidate B, canonical renderer/checkpoint를 변경하지 않았다.

## Baseline reconciliation

- per-view event union: DSC08043.JPG=754, DSC07960.JPG=330, DSC08003.JPG=502
- exact event union: 1586 points
- event union / frozen fit-input SHA: 79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78
- frozen representative: 96 x 40 = 3840 XYZ/normals
- exact support replay: 314 / 3840, unsupported 3526 / 3840
- support-mask SHA: 23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9
- all-four support cells: 211

역사적 WL145 prose의 248 / 3840은 현재 보존된 executable
_domain_accounting, report, event NPZ, support-mask hash와 일치하지 않는다.
현재 JSON의 supported_regions_four_connected도 211이고, all-four cell 재생도
211이다. 따라서 248은 복구된 baseline이 아니라 provenance-unverified
prose-only 값으로 남겼다. 역사적 WL145 문서는 수정하지 않았다.

## A/B 비교

두 arm은 동일한 frozen representative XYZ와 normals를 공유한다.

- A: 전체 95 x 39 = 3705 cells
- B: WL145에 이미 존재하는 support vertex mask의 네 꼭짓점이 모두 True인 211
  cells만 face로 materialize
- B는 새 vertex를 보간하거나 생성하지 않았고, 새 fit/refit도 없다.
- 면적 convention은 WL145의 기존 per-cell ||du x dv||이다.

| 항목 | A full domain | B support constrained |
|---|---:|---:|
| materialized vertices | 3840 | 274 |
| materialized area | 3.253452 | 0.183408 |
| oracle -> materialized median / p95 | 1.3276h / 2.1729h | 1.3706h / 2.5213h |
| materialized -> oracle median / p95 | 32.4048h / 77.9380h | 0.7721h / 2.0289h |
| oracle coverage <= h / <= mu | 30.52% / 99.12% | 29.19% / 96.53% |
| normal median / p95 | 12.64° / 58.48° | 12.64° / 58.48° |

B는 full rectangle의 94.36% 면적을 제거하고 representative-to-oracle
분포를 크게 줄였지만, oracle coverage는 소폭 감소했다. B domain은
4-connected component 6개(201, 4, 2, 2, 1, 1), isolated cell 2개, 내부 hole
0개다. 이는 support annotation이 full parametric rectangle과 다른 sparse
materialization domain임을 보여주는 결과이며, canonical Surface Membership
성공이나 NURBS trimming의 근거로 승격하지 않았다.

거리 metric은 두 arm의 exported mesh에 포함된 frozen representative
vertex까지의 nearest-vertex metric이다. triangle surface distance로 바꾸거나
새로운 smoothing/interpolation을 적용하지 않았다.

## 실제 Gaussian Scene output

canonical checkpoint와 renderer를 변경하지 않고 WL145와 동일한 세 camera
(DSC08043.JPG, DSC07960.JPG, DSC08003.JPG, 각 648 x 420)에 대해 다음 raw
PNG를 생성했다.

- real_scene_camera_review/<camera>/A...F.png
  - A Gaussian Scene
  - B Gaussian + clean oracle
  - C Gaussian + full-domain representative
  - D Gaussian + support-constrained materialization
  - E/F oracle와 각 representative 결합
- mandatory_gaussian_visualization_pair/<camera>/G_original_scene.png,
  H_observed_occluded.png
  - 동일 checkpoint/iteration/camera/renderer/Gaussian row
  - G는 원래 appearance, H는 fixed green/red/gray state 색상만 변경
- common_world_matched/ raw 3D fixed-view 비교
- chart_space_96x40_diagnostic.png 및 .npz
- materialized_meshes/의 A/B PLY와 frozen_inputs/ replay NPZ

raw overlay point는 near-opaque로 출력했으며, metric population이나 geometry를
display용으로 smoothing하지 않았다.

## 구현 fidelity와 판정

수동 선택은 WL145의 frozen tabletop case/chart와 고정된 세 camera output이다.
heuristic/diagnostic 요소는 UV occupancy의 의미, all-four materialization, 4-connected
topology accounting, nearest-vertex review metric이다. full-reference event union은
기존 WL145 support annotation을 정확히 replay하고 evaluation target으로 확인하는 데만
사용했으며, 새로운 fit/continuation 선택에는 사용하지 않았다.

이 작업은 continuation holdout도, true-occluded prototype도 아니다. 따라서 최종
attribution은 **D. MIXED / INCONCLUSIVE**로 둔다. B의 support-constrained
materialization은 full rectangle의 unsupported tail을 제거하는 유용한 진단이지만,
남은 coverage 저하와 domain fragmentation 때문에 canonical architecture로
승격하지 않는다. 248 기반 mask replacement, automatic Surface Membership,
canonical NURBS trimming/refit, continuation, Occluded Surface 구현은 모두 하지
않았다.

## 검증

- tests/test_wl148_baseline_reconciliation_support_constrained_materialization.py
  focused tests: **4 passed**
- WL148 module syntax check: 통과
- 전체 repository regression: 실행하지 않음
- canonical production code/checkpoint/Candidate B/historical topology: 변경 없음

## 산출물

- 구현: devtools/demo/wl145_baseline_reconciliation_support_constrained_materialization.py
- output: output/148_wl145_baseline_reconciliation_support_constrained_materialization_audit/
