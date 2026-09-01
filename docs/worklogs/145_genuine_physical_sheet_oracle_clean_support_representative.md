# Worklog 145 — Genuine Physical-Sheet Oracle 및 Clean-Support Representative 검증

## 의도

WL141의 historical oracle mask를 자동으로 고치거나 Surface Membership를
구현하지 않고, frozen Gaussian renderer의 `depth_median` event가 실제 하나의
물리 sheet를 지지하는지 먼저 검증했다. 그 뒤 `CLEAR_PHYSICAL_SHEET_ORACLE`
만 unchanged WL139 representative 입력으로 승격했다. continuation,
Occluded Surface, SH/appearance completion은 실행하지 않았다.

## 구현 및 운영 선택

- 구현은 [`genuine_physical_sheet_oracle_clean_support_representative_audit.py`](../../devtools/demo/genuine_physical_sheet_oracle_clean_support_representative_audit.py)와 별도 output 폴더에 격리했다.
- 세 source camera(`DSC08043.JPG`, `DSC07960.JPG`, `DSC08003.JPG`)에서 수동으로 고정한 648x420 image-space interior polygon을 사용했다.
- 각 polygon의 유효 renderer median event를 per-view cloud로 별도 복원하고, WL143의 pixel/depth 역변환을 그대로 사용했다. cloud union에는 KNN, connected component, normal/appearance filtering을 적용하지 않았다.
- WL127 mesh는 immutable provenance/hash와 point-count 확인만을 위해 읽었다. oracle membership, graphness, fitter에는 넣지 않았다. target polygon 위치를 수동 검토할 때 한 source event cloud의 cross-view 재투영을 참고했지만, fitted representative와 withheld geometry는 사용하지 않았다.
- WL139의 `h`, `mu`, 8x4 degree-2, regularization, fit budget을 변경하지 않았다. graphness PASS 이후에만 PCA chart를 graphness/representative용으로 만들었다.

## 결과

### CLEAR control — broad tabletop

`tabletop_broad_planar_clean`은 사람이 source render, common-world cloud, all-target
reprojection을 함께 확인해 `CLEAR_PHYSICAL_SHEET_ORACLE`로 분류했다. 세 cloud의
행 수는 `754 / 330 / 502`이며, pairwise reciprocal median은 각각
`1.83h / 1.31h / 1.67h`였다. p95는 `21.13h / 8.57h / 15.64h`로 renderer
outlier가 남아 있으므로 필터링하지 않고 함께 기록했다.

frozen WL139 graphness는 `PASS_GRAPH_LIKE`였다(`7/417` multimode bins,
`1.68%`, chart coverage `4.45%`, within-bin n median `1.26h`, p95 `4.95h`).
대표 surface의 raw→representative 거리는 median `1.33h`, p95 `2.17h`였고,
fixed-UV fitting residual은 median `0.70h`, p95 `2.90h`였다. topology contract는
valid하고 area inflation은 `1.004`였다.

그러나 representative→raw 거리는 median `32.40h`, p95 `77.94h`였으며,
full chart sample vertex 중 지원된 것은 `248/3840 (6.46%)`, unsupported는
`3592/3840`이었다. 따라서 human review는
`B_VALID_ONLY_ON_SUPPORTED_DOMAIN`으로 기록했다. clean observed patch 위의
대표성은 보이지만, 빈 full rectangle을 실측된 tabletop geometry로 해석할 수 없다.

### 비승격 후보

- `table_rim_curved_interior_candidate`: 얇은 rim/상판/leg 경계가 renderer event에
  함께 나타나 `PARTIAL / MIXED`로 유지했다. WL139 representative를 force-fit하지 않았다.
- `tabletop_near_vase_boundary_candidate`: vase와 dark insert 근처의 depth-layer
  혼입 위험 때문에 `PARTIAL / MIXED`로 유지했다. 별도 자동 제거를 하지 않았다.

## 구현 충실도 및 보존

수동 선택은 camera와 polygon, physical description, same-sheet reason,
distractor/boundary disclosure, human classification이다. heuristic은 post-clear
deterministic PCA chart와 WL139의 기존 representative 해석뿐이다. manual oracle,
manual review, PCA frame은 최종 paper method에서 허용할 수 없다. canonical renderer,
checkpoint, 161 cameras, WL127 geometry, WL139 fitter, Candidate B, historical WL141/WL144
artifact는 변경하지 않았다. WL144의 historical classification은 수정하지 않고,
curved rim을 `PARTIAL / MIXED`로 보정하는 해석만 이 보고서에 기록했다.

## 산출물 및 판정

주요 산출물은 [`output/145_genuine_physical_sheet_oracle_clean_support_representative_audit/`](../../output/145_genuine_physical_sheet_oracle_clean_support_representative_audit/)에 있다. 각 case에 per-view provenance NPZ/PLY, common-world PNG, cross-view reprojection PNG가 있고, clear case에는 raw-only, representative-only, raw+representative, normals, supported/unsupported domain PNG가 있다.

`tests/test_genuine_physical_sheet_oracle_clean_support_representative_audit.py`의
focused tests는 `4 passed`였다. 실제 frozen CUDA 실행도 완료했다.

### 결론

**PARTIAL FEASIBILITY DEMO.** genuinely renderer-grounded tabletop sheet와
unchanged WL139 representative가 관측된 clean support에서는 정합되는 사례를
확보했다. 하지만 supported domain 밖의 representative rectangle은 검증되지
않았고, curved thin sheet와 near-object control은 아직 mixed다. 이는 자동
Surface Membership나 Occluded Surface 해결을 의미하지 않는다.
