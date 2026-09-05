# Worklog 167-1 — Close Image-Space Review Exports for W167

## 1. 의도 정렬

`167-1`은 W167에 종속된 review-export-only batch이다. 목적은 기존 real-scene first-hit evidence를 실제 RGB camera image에서 사람이 확인할 수 있도록 가까운 overlay/crop으로 재표현하는 것이며, W167 geometry, ray-mesh intersection, query ladder semantics, component ranking, architecture verdict, production behavior는 변경하지 않았다.

## 2. 입력 재사용 대 재생성

- 직접 재사용: W167의 세 camera `*_ray_results.npz` (`pixel`, `status`, `depth`, `world_xyz`, `triangle_id`, `component_id`, `barycentric`), W167 `real_scene_replay_report.json`, W167 frozen W160–W165 ROI definitions, W167 camera metadata, `DATASET/images_8` original RGB frames.
- read-only 재생성: W167의 저장된 hit depth와 동일한 `_real_camera_rays`로 `Q_before`, `Q_surface`, `Q_behind` world point를 복원하고, `full_proj_transform`으로 image-space에 투영했다. W167 mesh, ray intersection, screen-tile broad phase, component labeling, architecture evaluation은 재실행하지 않았다.
- surface query는 저장된 W167 `world_xyz` first-hit point이고, before/behind offset은 W167과 동일한 `0.09684388339519501` world unit이다. Candidate-B median depth나 새 threshold는 사용하지 않았다.
- W167 architecture verdict `REAL_SCENE_REVIEW_REQUIRED`는 report에서 확인하고 그대로 보존했다.

## 3. 시각화 디렉터리 구조

출력은 [`output/167_raw_zero_set_ray_blocker_audit/real_scene/review_views`](../../output/167_raw_zero_set_ray_blocker_audit/real_scene/review_views/)에 저장했다. parent README와 각 visualization directory의 공용 README를 유지하고, 각 디렉터리에는 `DSC07960.png`, `DSC08003.png`, `DSC08043.png`를 직접 배치했다. camera subdirectory나 중복 README는 만들지 않았다.

crop box는 manual tuning 없이 frozen ROI polygon support에서 결정했다. 15% max-span padding, 12-pixel minimum padding, 96×64 minimum readable canvas를 적용했다. W167-associated hit가 tight polygon crop 밖에 남는 경우는 reclassification하거나 crop을 넓히지 않고 report에 공개했다.

## 4. 생성한 시각화 유형

- `first_hit_overlay_full`: 원본 RGB full frame 위에 green projected first-hit pixels.
- 각 target의 `*_first_hit_overlay_cropped`: tabletop, tabletop-vase contact, table-side/lower, vase/curved-neighbor close crop.
- 각 target의 `*_component_provenance_cropped`: green top-20 component와 orange non-top-20 component.
- 각 target의 `*_query_ladder_cropped`: 동일 ray의 cyan `Q_before`, yellow `Q_surface`, red `Q_behind`. 같은 image pixel에 ring/square/cross display marker를 사용해 표시만 구분했다.
- `*_suspicious_hit_spotlight`: target 전체에서 non-top-20 hit가 존재하는 경우에만 생성했다. 현재 네 target 모두 적어도 한 camera에서 non-top-20 hit가 있어 spotlight directory가 있다.

총 PNG는 51개이며, PNG가 primary review artifact다. 기존 W167 산출물은 덮어쓰지 않았다.

## 5. Target / ROI coverage per camera

W167 region ray count와 167-1 target mask count가 모든 camera/target에서 일치했다.

| camera | tabletop | contact | table-side/lower | vase/curved-neighbor |
|---|---:|---:|---:|---:|
| `DSC07960.JPG` | 31 | 222 | 214 | 222 |
| `DSC08003.JPG` | 30 | 181 | 218 | 151 |
| `DSC08043.JPG` | 50 | 220 | 204 | 170 |

저장된 W167 hit records는 세 camera 모두 모든 sampled ray가 `HIT`였고, first-hit projection reprojection error는 약 `1e-5` pixel 수준이다. expected first-hit/query points는 모두 valid image coordinates 안에 있었다.

## 6. Non-top-20 component hit coverage

| camera | tabletop | contact | table-side/lower | vase/curved-neighbor | total |
|---|---:|---:|---:|---:|---:|
| `DSC07960.JPG` | 1 | 5 | 8 | 5 | 19 |
| `DSC08003.JPG` | 0 | 1 | 2 | 1 | 4 |
| `DSC08043.JPG` | 0 | 0 | 1 | 0 | 1 |

target/camera ROI association 기준 non-top-20 hit coverage 합계는 24건이다(겹치는 contact/vase target 포함). unique W167 first-hit fragment는 17건이며, 모두 W167의 attribution-only 결과로 유지했다. fragment 제거·size threshold·false blocker 판정으로 사용하지 않았다. spotlight export는 해당 target에서 적어도 한 camera에 non-top-20 hit가 있는 경우에만 만들었다.

## 7. Human-review relevant observations

- full-frame overlay는 first-hit 위치가 실제 RGB frame의 어느 physical structure에 닿는지 직접 볼 수 있게 한다.
- close crops는 common-world distant view보다 tabletop, vase contact, lower table, vase/curved-neighbor의 실제 image-space alignment를 읽기 쉽다.
- component crop은 top-20과 non-top-20 hit를 분리해, 작은 disconnected component가 의심 위치를 설명하는지 사람이 확인할 수 있게 한다.
- ladder crop은 동일 pixel에서 before/surface/behind 관계가 유지되는지 확인할 수 있다. 표시용 ring/square/cross는 좌표나 query meaning을 변경하지 않는다.
- tight polygon crop 밖에 남는 W167-associated hit가 있다. 특히 contact/vase 및 일부 table-side pair에서 관찰되며, 이는 기존 W167 region association과 intended target support의 관계를 사람이 검토해야 하는 open item이다. 해당 ray를 숨기거나 재분류하지 않았다.
- 따라서 이 batch만으로 physical-surface alignment 성공을 주장하지 않는다.

## 8. Retained / Rejected / Open

- Retained: W167 raw zero-set, saved first-hit records, camera set, ROI intent, component provenance, query offset/relations, `REAL_SCENE_REVIEW_REQUIRED`.
- Rejected: geometry improvement, mesh repair/filtering, new blocker semantics, new component threshold, ray intersection change, NURBS, architecture upgrade, automatic success claim.
- Open: 사람이 각 crop에서 first-hit가 intended visible surface에 놓이는지, non-top-20 spotlight가 meaningful small structure인지 nuisance fragment인지 판단해야 한다.

Focused verification: `tests/test_worklog_167_1_image_space_review_exports.py` — `6 passed` (pytest cache directory 권한 경고 1건은 테스트 실패가 아니다).
