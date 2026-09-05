# Worklog 167-2 — W167 First-hit Geometry의 Depth-Preserving Local 3D Review

## 1. INTENT ALIGNMENT

W167의 raw SDF/TSDF zero-set camera-ray blocker 결과를 바꾸지 않고, 저장된 first-hit geometry를 실제 3D 깊이 순서로 사람이 검토할 수 있는 export layer를 추가했다. W167-1의 image-space reprojection은 보조 확인으로만 유지했다.

## 2. W167 DATA REUSE

`DSC07960.JPG`, `DSC08003.JPG`, `DSC08043.JPG`의 W167 `ray_results.npz`를 그대로 읽었다. `pixel`, `depth`, `world_xyz`, `triangle_id`, `component_id`, `barycentric`, `status` identity를 재계산하거나 재분류하지 않았다. W166의 `historical_sdf_zero_surface_raw.npz`는 local inspection AABB에 들어오는 raw triangle을 읽기 전용으로 조회하는 데만 사용했다. ray intersection, query ladder, component ranking, architecture verdict는 변경하지 않았다.

## 3. WHY W167-1 WAS INSUFFICIENT

같은 ray 위에서 first-hit point를 복원해 원본 pixel로 reproject하면 projection consistency는 확인할 수 있지만, 같은 ray의 3D surface identity를 구분할 수는 없다. 따라서 W167-1은 primary physical-surface evidence가 아니라 secondary projection aid로 명시했다.

## 4. LOCAL 3D VISUALIZATION CONTRACT

카메라 center, 결정론적 ray corridor, 모든 target-associated saved first-hit point, Q_before/Q_surface/Q_behind, hit triangle highlight, raw mesh inspection AABB를 함께 그렸다. AABB는 camera-to-hit population에 고정 비율 padding `0.25`를 적용했다. ray line은 고정 row-major 규칙으로 최대 `32`개만 그리지만 first-hit point는 줄이지 않았다. non-top-20 spotlight에서는 해당 fragment first-hit ray를 전부 유지했다.

PNG mesh stroke는 넓은 AABB에서도 사람이 볼 수 있도록 raw-face 청크별 균등 분산 fixed cap `12,000`을 적용했다. 이는 display-only raster 제한이며, W167 first-hit의 exact triangle/component/world/depth/pixel 행은 sidecar에 모두 남겼다. AABB 밖 geometry는 해당 local frame에서 표시하지 않으며, 그 부재를 물리적 부재로 해석하지 않는다.

## 5. CAMERA / TARGET COVERAGE

4개 target(`tabletop`, `tabletop_vase_contact`, `table_side_lower_geometry`, `vase_foreground_structure`)과 3개 camera의 조합 12개를 모두 처리했다. 각 조합에 대해 `local_3d_cutaway`, `depth_ordered_side`, `local_3d_perspective`, `component_first_hit_spotlight`를 생성했다.

## 6. COMPONENT SPOTLIGHT COVERAGE

W167 exact top-20-by-vertex-count attribution 밖의 unique first-hit fragment `17`개를 모두 spotlight coverage에 포함했다. non-top-20은 물리적 false blocker 판정이 아니라 W167 disconnected-component attribution label로만 표시했다.

## 7. GENERATED REVIEW ARTIFACTS

- [W167-2 review root](../../output/167_raw_zero_set_ray_blocker_audit/real_scene/review_views_167_2/README.md)
- `16` visualization directories, `48` PNG (`1400×920`), 각 디렉터리 공용 UTF-8 `README.md`
- `12` camera/target JSON+NPZ sidecar: exact first-hit rows, camera ray data, query ladder, inspection AABB, crop count
- [machine-readable report](../../output/167_raw_zero_set_ray_blocker_audit/real_scene/review_views_167_2/worklog_167_2_report.json)

## 8. HUMAN-REVIEW QUESTIONS NOW ANSWERABLE

Depth-ordered side view와 local 3D cutaway/perspective에서 각 ray가 처음 닿은 raw zero-set surface/triangle의 위치를 확인할 수 있다. 그 surface가 intended tabletop, vase-contact, table-side/lower, vase/curved-neighbor 물리 surface처럼 보이는지, non-top-20 fragment가 고립 조각인지 coherent local geometry인지 사람이 판단할 수 있다.

## 9. LIMITATIONS

실제 hidden-surface physical ground truth는 없다. 넓은 기존 W167 ROI association은 수정하지 않았으므로 일부 local volume이 넓게 보인다. PNG mesh stroke는 fixed raster cap을 가지며, AABB 밖 remote geometry는 local frame에 표시되지 않는다. 따라서 이 batch만으로 intended physical surface를 자동 확정할 수 없다.

## 10. RETAINED / OPEN

W167의 `REAL_SCENE_REVIEW_REQUIRED` verdict를 그대로 유지했다. W167/W167-1 geometry와 산출물은 덮어쓰지 않았다. 본 batch의 conclusion은 `COMPLETE_W167_2_DEPTH_PRESERVING_LOCAL_3D_REVIEW`; physical surface intent에 대한 최종 판정은 human review로 남겨 둔다.

