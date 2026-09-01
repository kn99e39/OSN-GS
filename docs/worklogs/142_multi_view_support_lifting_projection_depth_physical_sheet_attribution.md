# Worklog 142 — Multi-view Support Lifting / Projection / Depth / Physical-Sheet Attribution

## 작업 목적

Worklog 141의 semantic mismatch를 automatic Surface Membership 문제로 바로 확장하지 않고, 다음 세 원인을 분리하는 격리 진단을 수행했다.

- A: projection / coordinate contract failure
- B: visibility / depth-layer failure
- C: true physical-sheet support / membership failure

WL141의 기존 camera, polygon, candidate crop, 2-of-3 mask rule은 MASK_ONLY_BASELINE으로 그대로 재생했다. 새로운 support 후보는 이 baseline 위에 canonical renderer의 기존 depth_median과 candidate point의 camera-space depth를 비교한 MASK_PLUS_DEPTH_SUPPORT뿐이다.

## 구현 및 출력

- 코드: devtools/demo/multi_view_support_lifting_projection_depth_attribution.py
- 테스트: tests/test_multi_view_support_lifting_projection_depth_attribution.py
- 출력: output/142_multi_view_support_lifting_projection_depth_attribution/
- renderer는 OSNSurfelRasterizer.render(...)[depth_median]를 read-only로 사용했다.
- projection은 기존 WL140과 같은 homogeneous row-vector convention을 사용했다.
- depth tolerance는 WL139의 frozen mu를 그대로 사용했으며 WL142 결과로 조정하지 않았다.
- graphness, connected component, appearance/SH, trimming, continuation, Candidate B, Occluded Surface는 support 선택에 사용하지 않았다.

## Projection control 결과

세 control camera에서 projection contract가 통과했다.

- DSC07960.JPG: visible checkpoint center의 nearest-pixel renderer alpha support 0.998481
- DSC08003.JPG: 0.998726
- DSC08111.JPG: 0.998684

따라서 이번 실행에서는 기계적인 row/column, image y 방향, in-frame, camera-space positive-depth contract failure를 관찰하지 않았다. 각 camera의 projected Gaussian-center overlay는 projection_control/에 있다. alpha control은 geometry-independent checkpoint-center control이며 semantic ROI 선택에는 사용하지 않았다.

## WL141 baseline 재현

세 ROI 모두 WL141 report에 저장된 oracle row-ID count/hash와 정확히 일치했다.

| ROI | candidate | MASK_ONLY_BASELINE | 재현 |
|---|---:|---:|---|
| tabletop_top_oracle | 1,554 | 1,367 | exact row-ID hash |
| curved_table_rim_oracle | 20,181 | 17,842 | exact row-ID hash |
| paver_ground_oracle | 7,552 | 6,220 | exact row-ID hash |

WL141 report에는 row-ID 목록 자체가 없으므로 count와 oracle_row_ids_sha256로 재현성을 검증했고, 이 한계는 manifest/report에 명시했다.

## Depth attribution 결과

고정 rule은 mask match가 최소 2개이고, mask-matching renderer depth 비교가 최소 2개이며, 그 중 최소 2개가 abs(point_camera_depth - depth_median) <= mu이고, mask-matching behind/front contradiction이 하나도 없는 경우에만 MASK_PLUS_DEPTH_SUPPORT로 남기는 방식이다.

| ROI | MASK_ONLY | MASK_PLUS_DEPTH | depth로 제거 | mask-matching residual median / h | p95 / h |
|---|---:|---:|---:|---:|---:|
| tabletop | 1,367 | 0 | 1,367 | 596.23 | 818.87 |
| curved rim | 17,842 | 0 | 17,842 | 57.59 | 141.39 |
| paver ground | 6,220 | 0 | 6,220 | -376.96 | 96.72 |

세 mandatory control 모두 MASK_ONLY가 depth-consistent same-sheet support로 승격되지 않았다. 특히 curved rim의 세 camera에서 mask-matching depth relation은 다음과 같이 섞였다.

- DSC08043.JPG: consistent 563, behind 11,840, in-front 7,778
- DSC07960.JPG: consistent 310, behind 15,791, in-front 4,080
- DSC08003.JPG: consistent 233, behind 11,625, in-front 8,323

이는 image-mask agreement가 여러 depth layer를 포함한다는 B 방향의 contamination evidence다. 그러나 depth consistency 자체도 physical sheet identity의 증명은 아니므로 C는 unresolved로 남겼다.

## 산출물

각 ROI에 대해 다음을 생성했다.

- geometry/candidate_spatial_population.ply
- geometry/mask_only_support.ply
- geometry/mask_plus_depth_support.ply
- geometry/removed_by_depth_inconsistency.ply
- geometry/depth_layer_accounting.npz
- camera_overlays/<camera>/A...E.png
- 3d_review/*.png
- case_report.json

raw fixed-view PNG는 기존 Gaussian scene과 support/relation overlay를 직접 비교할 수 있도록 출력했고, 별도의 photometric/SH completion은 만들지 않았다. MASK_PLUS_DEPTH_SUPPORT가 비어 있는 결과도 숨기지 않고 empty PLY와 raw accounting으로 보존했다.

## 결론 및 fidelity

최종 attribution verdict는 F. MIXED / INCONCLUSIVE다.

- projection contract: mechanically PASS
- historical mask-only replay: exact
- depth-layer contamination: 세 ROI 모두 evidence present
- same physical sheet membership: unresolved
- representative replay: explicit human qualitative pass가 없으므로 실행하지 않음
- true-occluded prototype: 실행하지 않음
- canonical renderer, checkpoint, 161 cameras, WL127 geometry, WL139, WL141 masks, Candidate B: 변경 없음

이번 결과는 Occluded Surface 해결이나 final architecture 검증이 아니다. advisor에게는 “projection은 기계적으로 성립하지만, 기존 image-mask support는 depth layer 혼입을 제거하지 못했으며, physical-sheet membership은 아직 열려 있다”는 attribution 결과로 보고한다.

## 검증

- python -B -m py_compile devtools/demo/multi_view_support_lifting_projection_depth_attribution.py
- pytest -q tests/test_multi_view_support_lifting_projection_depth_attribution.py tests/test_oracle_single_surface_support_appearance_evidence.py
- 결과: 10 passed
- 실제 CUDA 실행: failures []
