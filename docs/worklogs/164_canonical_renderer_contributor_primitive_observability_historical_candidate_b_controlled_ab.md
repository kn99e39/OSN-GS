# Worklog 164 — Canonical Renderer-Contributor Primitive Observability와 Historical Candidate-B Controlled A/B

## 작업

- W163의 `K=16` pixel prefix provenance를 primitive participation 판정에 재사용하지 않고, 기존 isolated diagnostic sibling `render_with_pixel_representative`가 canonical acceptance path에서 반환하는 per-primitive `forward_accepted[g,v]` bit를 1,190,469개 Gaussian × 161개 camera에 대해 수집했다.
- `POINT_QUERY_STATE`를 Gaussian-center의 immutable Historical Candidate-B median-ordering state로 고정하고, 별도 `PRIMITIVE_OBSERVATION_STATE`에 대해서만 `CONTRIBUTED_IN_CAMERA(g,v)=true`인 primitive을 `OBSERVED`로 positive override했다. contribution magnitude, alpha, T, percentage, area, confidence, 새 threshold와 aggregation rule은 추가하지 않았다.
- W160 frozen global aggregation과 W162 Region 1 positive control, W163 provenance artifact를 exact stable ID로 대조했다. W161의 `OCCLUSION_DOMAIN_CONTRACT_GAP` 및 paused status, arbitrary XYZ 의미, TSDF/topology/Boundary First/NURBS/continuation은 유지했다.
- canonical renderer와 diagnostic sibling의 RGB, alpha, median depth를 frozen review camera 6개에서 bitwise 비교했다. diagnostic은 canonical renderer를 대체하지 않는다.
- `Original Scene`, Historical/Candidate global state, global occluded-only, changed transition, contributor primitive, Region 0/1, tabletop-vase contact, common-world visualization을 matched PNG로 생성했다. camera 파일은 각 visualization directory 바로 아래 `<camera_name_stem>.png`으로 저장했고, output root·review root·각 visualization directory에 개별 UTF-8 README를 작성했다.

## 결과

- W160 Historical Candidate-B global state는 exact 재현됐다. 전체 1,190,469개에서 baseline은 `OBSERVED 798,304`, `OCCLUDED 391,457`, `UNRESOLVED 708`이다.
- exact positive contributor는 `46,751,214` primitive-camera pair, 하나 이상의 camera에 기여한 primitive `1,181,613`개, zero-contributor primitive `8,856`개였다. zero-contributor는 `OCCLUDED`의 증거로 해석하지 않는다.
- candidate global은 `OBSERVED 1,186,747`, `OCCLUDED 3,512`, `UNRESOLVED 210`이며, historical `OCCLUDED → candidate OBSERVED` transition은 `387,945`개다.
- frozen W162 Region 1 population `65,471`와 baseline `OBSERVED 59,274 / OCCLUDED 6,197 / UNRESOLVED 0`을 보존했다. candidate는 `OBSERVED 65,453 / OCCLUDED 18 / UNRESOLVED 0`이고, `6,179`개가 `OCCLUDED → OBSERVED`, `18`개가 unchanged `OCCLUDED`였다. historical Region 1 occluded rows의 contributor-camera bins는 `0:18`, `1:1`, `2–5:3`, `6–10:4`, `11–20:12`, `>20:6,159`이다.
- canonical equivalence validation은 6개 camera 모두 RGB/alpha/median depth bitwise equal이었다. synthetic A–F contract도 모두 통과했다.
- architecture verdict는 정량 결과만으로 확정하지 않고 `MIXED`로 보류했다. primitive observation evidence rule은 구현·계측되었지만, physical first-hit truth나 arbitrary XYZ `OCCLUDED` contract를 의미하지 않으며 human qualitative review가 남아 있다.

## 평가

- 이 batch는 “renderer contributor가 있으면 canonical primitive observation evidence가 존재한다”는 제한된 명제를 검증한다. “renderer contributor가 physical visible surface ground truth다” 또는 arbitrary XYZ occlusion semantics를 증명하지 않는다.
- Historical baseline과 Candidate-B를 변경하지 않았고, `POINT_QUERY_STATE`와 `PRIMITIVE_OBSERVATION_STATE`를 per-camera NPZ에서 분리해 재현 가능하게 보존했다.
- 결과물: `output/confirmed/164_canonical_renderer_contributor_primitive_observability_historical_candidate_b_controlled_ab/`

## 검증과 잔여 위험

- focused W164 및 W160–W163 회귀 테스트는 `21 passed`로 완료했다. full-suite 결과는 기존 historical dirty-worktree/Windows 환경 제약과 별도로 해석한다.
- review export는 PNG 33개, PPM 0개이며 각 visualization directory README를 확인했다.
- 최종 architecture 판단, hidden-looking population의 qualitative plausibility, contributor override의 permissiveness는 사람 검토 전까지 열어 둔다. W161 spatial-domain construction과 Gate O2는 계속 paused/open이다.
