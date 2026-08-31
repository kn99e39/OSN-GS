# Worklog 140 — 실제 Gaussian Scene 정성적 Surface Construction 검증

## 상태

**격리된 비정규 평가 완료.** canonical renderer, checkpoint, Worklog 127 geometry, WL139 fitter/graphness/physical-chart representative는 변경하지 않았다. continuation, pseudo-occlusion, Candidate B, true-occluded prototype은 실행하지 않았다.

## 목적

Worklog 127의 renderer-grounded Visible Surface와 frozen trained Gaussian Scene이 실제로 같은 장면 구조를 가리키는지, 그리고 WL139 representative를 적용할 수 있는 raw ROI가 실제 scene에서 graph-like한지를 정성적으로 확인했다. 이번 배치는 새로운 representative나 Occluded Surface 알고리즘의 검증이 아니다.

## 고정 입력과 review set

- checkpoint: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt`, iteration `30000`, primitive `surfel_2d`, `1,190,469` surfels.
- raw reference: Worklog 127 `RENDERER_MEDIAN_SURFACE_POINTS/iteration_0000001/point_cloud.ply`, `1,212,365` points.
- frozen camera setup: `DATASET/images_8`, `sparse/0`, `-1`, `llffhold=8`, train camera `161`개. 각 ROI의 camera 3개를 raw ROI 투영 coverage와 camera 방향으로만 결정했다.
- WL139의 `h=0.012105485424399376`, `mu=0.03631645627319813` 및 graphness/fitter 설정을 그대로 재사용했다.
- ROI 7개는 fit 전 고정했다. 대표면 품질·metric·continuation·withheld geometry는 ROI/camera 선택에 사용하지 않았다.

주요 ROI는 다음과 같다.

- `curved_table_rim`: 실제 `DSC08111.JPG` raw Gaussian/Visible Surface를 보고 고정한 pixel seed `(150, 95, 450, 315)`에서 선택한 camera-aligned table-side/rim box `(-0.70,1.10,0.35)–(1.30,1.65,1.70)`.
- `historical_wl139_curved_rim_alignment_control`: 기존 WL139 curved-rim 좌표를 read-only로 보존한 대조군.
- `tabletop_planar_strip`, `adjacent_table_side`, `patio_ground_planar`, `wl136_leg_brace`, `hedge_background_complex`.

## 실행 결과

| ROI | points | graphness | representative | 비고 |
|---|---:|---|---|---|
| `curved_table_rim` | 20,181 | `PASS_GRAPH_LIKE`, multimode 0.0891 | 적용 | raw→rep median/p95 `1.678h/10.163h`, rep→raw `2.303h/31.358h`, topology valid. 이는 withheld error가 아닌 representative proximity 진단이다. |
| `historical_wl139_curved_rim_alignment_control` | 1,676 | `PASS_GRAPH_LIKE`, multimode 0.0244 | 적용 | `DSC08111.JPG` 투영 bbox가 `[11.0,317.0,134.1,366.2]`로 확인됐고, raw overlay 수동 검토상 table rim이 아니라 paver/ground에 놓였다. primary 증거로 승격하지 않았다. |
| `adjacent_table_side` | 4,673 | `PASS_GRAPH_LIKE`, multimode 0.0575 | 적용 | raw→rep `1.642h/5.322h`, rep→raw `1.463h/2.590h`; 정성 판단은 사용자 확인 필요. |
| `patio_ground_planar` | 7,552 | `PASS_GRAPH_LIKE`, multimode 0.0579 | 적용 | raw→rep `1.551h/2.762h`, rep→raw `32.105h/79.980h`; proximity만으로 성공을 주장하지 않는다. |
| `tabletop_planar_strip` | 1,554 | `FAIL_MATERIALLY_MULTIVALUED`, multimode 0.2667 | 미적용 | WL139 graphness gate 밖이므로 강제 fit하지 않았다. |
| `wl136_leg_brace` | 4,028 | `FAIL_MATERIALLY_MULTIVALUED`, multimode 0.5738 | 미적용 | thin/multi-sheet 구조에 대해 대표면을 강제하지 않았다. |
| `hedge_background_complex` | 3,635 | `FAIL_MATERIALLY_MULTIVALUED`, multimode 0.4977 | 미적용 | 복합 background는 graph representative 적용 범위 밖이다. |

자동 최종 표기는 `QUALITATIVE ARCHITECTURE VERDICT: USER REVIEW REQUIRED`이다. 현재 환경에서 생성 PNG를 사람처럼 안정적으로 판독해 macro-shape 성공을 자동 확정할 수 없고, historical ROI의 semantic misalignment도 발견됐으므로 A/B/C 성공 판정으로 과장하지 않았다.

## 출력과 시각화

출력 root는 `output/real_gaussian_scene_surface_validation/`이다.

- pass ROI마다 선택된 동일 camera에서 `A_gaussian_scene_only`, `B_gaussian_plus_raw_visible_surface`, `C_gaussian_plus_surface_representative`, `D_raw_evidence_plus_representative`를 생성했다.
- `3d_review/`에 raw Visible Surface, representative shaded/wireframe, raw+representative, analytic normals, chart boundary PNG를 생성했다.
- geometry는 raw/representative PLY와 representative NPZ로 저장했다.
- raw point 시각화는 alpha `0.97`, marker size `3.2`로 near-opaque하게 유지했다. display thinning은 rendering 성능에만 사용했고 metric population은 변경하지 않았다.
- PNG/PLY/NPZ 산출물 검사는 각각 `120/15/4`개였다. camera full-view PNG는 `648×420`이며, crop/3D 진단 이미지는 ROI에 맞는 별도 크기다.

## 구현 충실도와 누수 공개

- 수동 선택: 7개 semantic/chart seed, primary pixel window, case당 camera 3개라는 운영 선택.
- heuristic: raw projected coverage 기반 camera 선택, fixed box/pixel seed, 사람의 Gaussian-scene alignment review.
- full reference 사용: WL127 full raw geometry는 ROI membership, raw target 시각화, graphness 입력으로 사용했다. 이번 WL140은 withheld holdout 평가가 아니므로 withheld subset은 없다.
- 사용하지 않은 것: representative 품질/metric을 이용한 선택, Candidate B, continuation, pseudo-occlusion, true-occluded geometry, Trust, neural/VLM prior, canonical renderer 수정.
- 최종 논문 방법에서 부적절한 부분: 수동 pixel/world ROI, historical 좌표를 semantic truth로 가정하는 것, human review를 자동 gate처럼 취급하는 것, graphness pass와 proximity만으로 실제 scene 정합성/성공을 주장하는 것.
- 모든 변경은 `devtools/demo/real_gaussian_scene_surface_validation.py`, `tests/test_real_gaussian_scene_surface_validation.py` 및 이 문서에 격리했다.

## 검증

- focused tests: `17 passed` (`test_real_gaussian_scene_surface_validation.py`, `test_physical_chart_surface_representative.py`).
- graphness gate가 fit보다 먼저 실행되고, failed ROI에 representative를 강제하지 않는 source contract를 확인했다.
- camera 선택 deterministic, projection identity, display thinning 불변성, fixed WL139 settings, continuation/occluded path 비실행을 확인했다.
- canonical WL139 모듈에 대한 diff와 canonical renderer/checkpoint/Worklog 127 geometry 변경이 없음을 확인했다.

## 결론

이번 배치는 실제 trained Gaussian Scene 위에서 현재 frozen Surface Representative 경로를 raw evidence와 함께 열어 보는 데 성공했지만, semantic camera alignment와 macro-shape에 대한 사람 검토가 남아 있다. 따라서 회의에서 주장할 수 있는 결론은 **“실제 장면 적용 가능성을 점검하는 격리된 qualitative validation이 생성되었고, graphness applicability와 ROI 정합성 문제가 직접 드러났다”**까지다. **Occluded Surface가 해결됐다고 말하지 않는다.**