# Worklog 153 — WL127 Raw Visible Surface 재생성과 construction provenance 회복 감사

## 1. 현재 질문

WL152에서 확인한 vertex-only artifact만으로는 topology를 해석할 수 없었다. 이번 배치는 WL127의 실제 typed `ExtractedSurface`가 어떤 경로로 생성됐는지 식별하고, 동일한 입력·source core로 다시 생성한 뒤 native topology와 construction provenance의 보존 수준을 분리한다.

## 2. WL152 baseline reconciliation

WL152 baseline을 먼저 읽고 고정값을 검사했다. event union은 `1,586`, SHA-256은 `79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78`이며 event `1527`의 `v_min` ownership과 `CLEAR_NOT_ON_INTENDED_SURFACE`를 보존했다. point artifact는 [WL127 point PLY](../../output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/RENDERER_MEDIAN_SURFACE_POINTS/iteration_0000001/point_cloud.ply)의 SHA-256 `fcdd26129737b6610e86837e5138e084ed3cfb95a80d6db2692b9cf70427107a`, `1,212,365` vertices, `0` faces로 재확인했다.

## 3. 역사적 Raw Surface construction contract

기준 commit `943a764`의 `scale.py`, `field.py`, `extraction.py`, package `__init__.py`, `evidence_bounded_tsdf_stages.py`를 Git blob 기준으로 비교했다. 다섯 core blob이 모두 일치하여 `core_exact=true`이다. 현재 `evidence_bounded_projective_tsdf.py`의 후대 diagnostic/export 차이는 replay에 사용하지 않았고, 변경되지 않은 scale/field/extraction core를 직접 호출했다.

고정 contract는 renderer median-depth event seed, pixel-centre unprojection, projective TSDF(`mu=3h`, fusion weight 1, unknown은 sparse-field 부재), 최대 60회 closure growth, block `64`, batch `6`, all-eight-corners authoritative 및 sign-change cell eligibility, Lewiner Marching Cubes, `h*1e-6` quantized seam-only weld이다.

## 4. replay input availability

checkpoint `30000`과 COLMAP `cameras.bin`/`images.bin`, `poses_bounds.npy`, `images_8`의 image-dimension metadata를 확인했다. loader contract는 `images_8`, `sparse/0`, `resolution=-1`, `llffhold=8`이며 전체 185 images에서 train camera 161개를 사용한다. checkpoint SHA-256은 `4e49b916d668acf5eb0a1cc31b979caa86c5fd8743ac9632f36d7e9c1c72c9e2`이다.

역사적 typed vertices/faces/support/value/h 배열과 WL127 `_cache/field.npz`, `_cache/mesh.npz`는 보존돼 있지 않았다. 따라서 typed 결과의 historical byte-exact hash를 직접 비교할 수 없고, source/input-identifiable replay와 WL127 기록 수량 일치로만 의미를 제한했다. qdepth extension은 prebuilt loader input으로만 사용했으며 수정하지 않았다.

## 5. Raw Visible Surface replay

161개 camera의 canonical renderer median-depth channel을 재생하고, WL127의 고정 closure를 60회 그대로 실행했다. authoritative voxel은 `76,720,314`로 WL127 기록과 일치했다. closure는 60회 종료 시에도 `13,972` voxel이 마지막 round에 추가되어 `closed=false`이며, 따라서 결과는 true authoritative set의 strict subset이다. 이 미종결은 surface 발명이 아니라 누락 방향으로만 해석했다.

## 6. typed `ExtractedSurface` contract

재생성 결과는 다음 typed arrays와 `h`를 포함해 `replay_cache/typed_extracted_surface.npz`에 저장했다.

- vertices `28,694,040`
- faces `45,116,659`
- vertex support count
- vertex field value
- `h = 0.012105485424399376`, `mu = 0.03631645627319813`
- companion field cache와 161-view median-depth cache

eligible cells는 `21,235,312`로 WL127 기록과 일치했다. 결과 bundle SHA-256은 `7e5df59a09877fcc1eebd1ab12d4c43a12d3dd352869fdb846c82d64678b495a`이며, replay는 `SEMANTICALLY_EXACT_REPLAY`로 분류했다. 이는 historical typed bundle의 부재 때문에 byte-exact라는 뜻은 아니다.

## 7. construction-provenance lineage

실제 source lineage는 `renderer event → seed voxel → TSDF cell → active cell → Marching Cubes face → seam-welded vertex`까지 의미상 확인됐다. 그러나 WL127의 `fuse_views`는 contributor event/camera ID를 mean field와 support count로 축약하고, extraction/weld는 source-cell/face lineage sidecar를 저장하지 않았다.

따라서 다음은 복구되지 않았다.

- surface element별 event/camera/source-cell ID
- field cell별 camera set
- face/component별 event aggregation

nearest, radius/KNN, normal matching, reprojection voting, distance-threshold 등 post-hoc correspondence는 사용하지 않았다. full reference point PLY는 WL152 hash/shape reconciliation target으로만 읽었고 replay fitting input으로 넣지 않았다.

## 8. native topology accounting

faces와 vertex indices만 사용해 계산했으며 component 선택, merge/split, repair, hole closure, filtering은 하지 않았다.

| 항목 | 값 |
|---|---:|
| vertices / faces | 28,694,040 / 45,116,659 |
| unique edges | 73,751,737 |
| connected components | 582,646 |
| component vertex size (min / median / p95 / max) | 3 / 6 / 35 / 18,897,107 |
| component face size (min / median / p95 / max) | 1 / 4 / 40 / 33,111,314 |
| boundary edges / boundary components | 12,153,565 / 880,791 |
| boundary vertex degree (min / max) | 1 / 6 |
| well-defined boundary loops | 산출 불가; 모든 degree가 2가 아님 |
| non-manifold edges | 35 |
| isolated vertices | 0 |
| degenerate-index / zero-area faces | 28 / 28 |

native topology는 회복됐지만, 이 숫자만으로 physical sheet identity를 만들 수는 없다.

## 9. physical-sheet viability review

기존 검토 대상만 다시 분류했으며 새 membership을 만들지 않았다. `clean_tabletop`, `tabletop_side_relationship`, `vase_or_curved_neighbor`, `background_lower_geometry` 네 case 모두 `NOT_REVIEWABLE`이다. native connected component, support count, boundary, geometry만으로 same sheet/distinct sheet을 추론하지 않았다.

## 10. evidence provenance viability

renderer observation→TSDF cell과 active cell→face의 source semantics는 deterministic하지만 per-element lineage로 저장되지 않았다. 따라서 현재 carrier는 native topology를 제공할 수 있으나 observation provenance carrier로는 불완전하다. 별도의 behavior-neutral lineage sidecar가 향후 필요하며, 그 작업은 이번 배치에서 구현하지 않았다.

## 11. event 1527 trace

event `1527`은 source camera `DSC08003.JPG`, pixel `[259,169]`, historical `v_min` owner, human review `CLEAR_NOT_ON_INTENDED_SURFACE`로 보존했다. blacklist하지 않았다. 다만 WL127 typed carrier에 event/camera/source-cell ID가 없으므로 surface trace는 `EVENT_LEVEL_NOT_AVAILABLE`이다. 기존 review를 replay 결과에 억지로 매핑하지 않았다.

## 12. boundary semantics

계산된 boundary는 face edge incidence에 의한 **topological boundary**일 뿐이다. evidence mask, reconstruction volume, extraction support limit, genuine observed termination 중 어느 원인인지 구분할 ownership contract가 없다. 그러므로 native mesh boundary를 observed physical-surface boundary로 승격하지 않았다.

## 13. architecture verdict

최종 판정은 **`TOPOLOGY_RECOVERED_PROVENANCE_GAP`**이다.

- WL127 typed surface construction은 frozen input/source contract와 기록 수량까지 재생성됐다.
- native topology는 원본 faces에서 회복·회계됐다.
- per-event/per-camera/per-cell provenance와 physical-sheet membership은 여전히 별도 abstraction이 필요하다.
- event 1527은 보존됐지만 element-level trace는 불가능하다.
- NURBS, membership, continuation, Candidate B, canonical Occluded Surface는 실행하지 않았다.
- canonical production code/renderer/checkpoint/camera set은 변경하지 않았다.

## 14. retained / rejected / open

**Retained:** WL152 exact baseline, WL127 point artifact, `943a764` core, frozen checkpoint/cameras, event 1527 review, typed replay cache, native topology accounting.

**Rejected:** point-only PLY를 mesh로 승격, post-hoc event-to-mesh mapping, topology repair, physical-sheet inference, NURBS fitting, canonical production 변경.

**Open:** behavior-neutral per-element lineage sidecar, physical-sheet membership contract, topological boundary와 observed boundary의 deterministic ownership, closure contract의 별도 재검토.

## 구현 충실도 및 산출물

수동 ROI나 continuation extent는 사용하지 않았다. 수동 선택은 없으며, 모든 replay extent/parameter는 WL127 source와 frozen input에서 왔다. heuristic geometry inference는 없었다. full-reference point PLY는 reconciliation/evaluation target으로만 사용했다. 결과는 [output/153_raw_visible_surface_replay_construction_provenance_audit](../../output/153_raw_visible_surface_replay_construction_provenance_audit/)와 번호 보존 복사본 [temp/153_raw_visible_surface_replay_construction_provenance_audit](../../temp/153_raw_visible_surface_replay_construction_provenance_audit/)에 있다. PNG는 geometry를 바꾸지 않는 opaque vertex preview일 뿐이며, 이번 배치는 slide polish를 목표로 하지 않았다.

이 batch에서의 결론은 “WL127 Visible Surface의 typed geometry와 native topology는 재생성 가능하지만, observation provenance와 physical-sheet identity는 현재 carrier에 없다”이다. 이는 canonical Occluded Surface 해결이나 최종 논문 architecture의 검증이 아니다.
