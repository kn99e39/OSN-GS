# Worklog 171 — Zero-Set-Derived Visible Structural NURBS Audit with Review Visualizations

## 1. INTENT ALIGNMENT

- 이 batch는 frozen raw zero-set observed support가 bounded visible structural NURBS representative를 직접 만들 수 있는지만 감사했다.
- Real geometry는 W154 `candidate_f_tsdf_surface_samples.npz`의 `world_xyz`와 native cell identity만 사용했다. Existing Gaussian `region_id`는 local support organization에만 사용했으며 Gaussian center는 fit point가 아니다.
- W167 geometry semantics, historical TSDF/extraction, W154 boundary-first WL139 fit family, 모든 selected component, strict `BEHIND_ZEROSET` representation fact와 W161 pause를 유지했다.
- Occluded continuation, production behavior, epsilon/margin, component deletion, largest-component 선택, top-k, trusted subset, smoothing, hole filling은 추가하지 않았다.

## 2. CASE SELECTION

Synthetic control은 fixed plane, origin-symmetric finite parabolic graph, W168 two-sheet slab으로 고정했다. Real case는 W154가 이미 기록한 W145 frozen case를 사용하고, 각 case에서 camera 이름을 정렬해 첫 camera인 `DSC07960.JPG`를 기계적으로 선택했다. 해당 event cloud의 exact XYZ AABB에는 padding을 넣지 않았다.

- `real_tabletop_coherent`: W145 `tabletop_broad_planar_clean`, W154 prior dominant `region_id=1`.
- `real_curved_vase_coherent`: W145 `table_rim_curved_interior_candidate`, W154 prior dominant `region_id=0`.
- `real_mixed_contact_stress`: W145 `tabletop_near_vase_boundary_candidate`, AABB 안의 accepted region 6,781개를 모두 유지.

이 규칙은 W171 topology, fit residual, visualization을 계산하기 전에 확정되며 visual success를 보고 case 또는 component를 고르지 않는다.

## 3. ZERO-SET SUPPORT OWNERSHIP

| Case | Complete selected support | Existing-region context excluded before topology | Selected region 수 | Support 보존 |
| --- | ---: | ---: | ---: | --- |
| synthetic planar | 2,025 | 0 | synthetic 1 | exact hash 일치 |
| synthetic curved | 2,167 | 0 | synthetic 1 | exact hash 일치 |
| synthetic layered | 4,050 | 0 | synthetic 1 | exact hash 일치 |
| real tabletop | 17,965 | 2,970 | 1 | exact hash 일치 |
| real curved/vase | 118,030 | 132,838 | 1 | exact hash 일치 |
| real mixed/contact | 2,092,474 | 0 | 6,781 | exact hash 일치 |

Coherent real case에서 제외된 row는 W154의 prior existing-region organization 밖의 context이며, 선택된 region 안에서는 단 하나의 fragment도 제거하지 않았다. Mixed case는 모든 accepted region과 component를 그대로 입력했다.

## 4. BOUNDARY / DOMAIN VALIDITY

| Case | Native component 수 | Component size median / max | Boundary loop | Closed | Rank |
| --- | ---: | ---: | ---: | --- | ---: |
| synthetic planar | 1 | 2,025 / 2,025 | 1 | yes | 32/32 |
| synthetic curved | 1 | 2,167 / 2,167 | 1 | yes | 32/32 |
| synthetic layered | 2 | 2,025 / 2,025 | 미도출 | no | 미평가 |
| real tabletop | 313 | 2 / 15,189 | 미도출 | no | 미평가 |
| real curved/vase | 1,925 | 2 / 111,601 | 미도출 | no | 미평가 |
| real mixed/contact | 49,872 | 2 / 793,424 | 미도출 | no | 미평가 |

Single native component인 경우에만 W154 `derive_native_support_boundary`를 호출했다. 여러 component가 있으면 largest component를 골라 boundary를 만드는 rescue를 하지 않고 case 전체를 `ABSTAIN_REPRESENTATIVE`로 판정했다.

## 5. SYNTHETIC RESULTS

- Planar single-sheet: 1 component, ordered closed loop 1개, design rank 32/32로 `MATERIALIZED_REPRESENTATIVE`.
- Curved single-sheet: 1 component, ordered closed loop 1개, design rank 32/32로 `MATERIALIZED_REPRESENTATIVE`.
- Layered/non-single-chart: 동일 크기 component 2개가 보존되어 `multiple_native_tsdf_components`로 abstain.

따라서 frozen boundary-first fit 자체는 complete coherent single-chart zero-set support를 materialize할 수 있고, 명시적인 layered support에는 임의 chart를 만들지 않는다.

## 6. REAL-SCENE RESULTS

- Tabletop은 17,965개 support 중 큰 component가 15,189개이지만 나머지를 삭제할 근거가 없고 총 313개 component이므로 abstain했다.
- Curved/vase는 118,030개 중 큰 component가 111,601개이지만 총 1,925개 component이므로 abstain했다.
- Mixed/contact는 2,092,474개, existing region 6,781개, native component 49,872개이며 예상대로 single chart를 만들지 않았다.

큰 component가 대부분을 차지한다는 사실은 complete support가 single-component라는 계약을 대신하지 않는다. 이 batch는 coverage를 높이기 위해 fragment를 삭제하거나 신뢰 집합을 만들지 않았다.

## 7. FIT AND RESIDUAL EVALUATION

| Materialized case | Fit residual median / mean / p95 / max (world) | Fit residual median / p95 / max (`/h`) | Dense support-to-surface median / p95 (`/h`) |
| --- | --- | --- | --- |
| synthetic planar | 1.2014e-7 / 3.2050e-7 / 1.0860e-6 / 1.4457e-5 | 2.4027e-6 / 2.1721e-5 / 2.8915e-4 | 0.43564 / 0.74912 |
| synthetic curved | 6.7925e-4 / 8.2323e-4 / 2.3939e-3 / 1.4777e-2 | 0.013585 / 0.047878 / 0.29554 | 0.43516 / 0.67487 |

`fit_residual`은 fitted UV에서의 exact Euclidean error다. `support_to_surface_dense_reference`는 64x32 dense NURBS sample에 대한 nearest distance이므로 sampling approximation이며 fit residual과 동일한 양으로 해석하지 않는다. Abstain case에는 NURBS와 residual을 생성하지 않았다.

## 8. REVIEW VISUALIZATIONS

`output/171_zero_set_derived_visible_structural_nurbs_audit/review_views/`에 다음 family를 PNG로 저장했다.

- `support_common_world`: case별 complete support의 stable display subset과 existing-region context.
- `boundary_and_domain`: valid chart support/ordered boundary 또는 multi-component abstain projection.
- `nurbs_fit_common_world`: raw support, bounded NURBS, 8x4 control net. 3D common-world가 주 evidence다.
- `residual_review`: materialized support의 residual `/h`; abstain이면 residual 부재를 명시.
- `image_reference_overlay`: fixed real RGB camera 3개와 synthetic reference projection. RGB는 secondary review다.

총 PNG 30개, PPM 0개, README 14개다. 각 visualization directory와 각 case artifact directory에 한글 기반 설명, input/state semantics, palette/legend, shared rendering condition, limitation을 중복해서 기록했다. Camera PNG는 하위 directory 없이 `DSC07960.png`, `DSC08003.png`, `DSC08043.png`로 직접 저장했다.

## 9. ARCHITECTURE RESULT / RETAINED / REJECTED / OPEN

최종 verdict는 **`NO_CURRENT_ZERO_SET_SUPPORT_NOT_RELIABLY_STRUCTURALLY_FIT_READY`**다.

Synthetic coherent support는 materialize할 수 있었으나, 사전 고정된 두 real coherent target 모두 complete selected support가 수백~수천 native component로 분절되어 있었다. 따라서 현재 historical zero-set support를 그대로 사용해 real coherent single-chart structural representative를 일관되게 만드는 계약은 성립하지 않는다.

- Retained: raw zero-set authority, native-cell connectivity, all selected components, W154 fixed fit, W161 pause.
- Rejected: largest-component rescue, fragment filtering, Gaussian-center geometry, threshold sweep, manual cleanup, smoothing/hole fill, occluded continuation, automatic production promotion.
- Open: real coherent target의 fragmentation을 어떤 별도 architecture contract로 다룰지는 후속 결정 사항이다. W171은 이를 수정하거나 최적화하지 않는다.

Focused test 결과는 `5 passed`다. 확인 항목은 synthetic single-sheet materialization, layered/mixed abstention, deterministic pre-fit selection, support hash 보존, boundary/fitter rank, PNG/README artifact 존재다.
