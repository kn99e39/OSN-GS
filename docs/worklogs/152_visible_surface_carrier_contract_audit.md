# Worklog 152 — Visible Surface Carrier 계약 감사

## 상태

완료. WL151의 `CONTRACT_GAP` 이후 renderer evidence와 smooth representative 사이에서
Visible Surface Topology를 소유할 수 있는 기존 Raw Visible Surface 표현을
검사했다. canonical 코드와 WL139–WL151 historical 결과는 변경하지 않았다.

## 1. 현재 아키텍처 질문

기존 Raw Visible Surface Geometry가 renderer-grounded evidence와 Boundary First
representative 사이의 topology/boundary ownership carrier가 될 수 있는가?

## 2. WL151 baseline reconciliation

WL149–WL151 frozen 상태는 exact replay했다.

- renderer event union: `1586`
- union SHA-256: `79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78`
- event 1527: `DSC08003.JPG`, pixel `(259,169)`, historical `v_min` owner
- human review: `HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE`
- representative: `3840 x 3`
- support: `314`, mask hash `23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9`
- all-four supported cells: `211`

현재 확인 가능한 WL127 artifact는
`RENDERER_MEDIAN_SURFACE_POINTS/iteration_0000001/point_cloud.ply`이며 SHA-256은
`fcdd26129737b6610e86837e5138e084ed3cfb95a80d6db2692b9cf70427107a`이다.
그러나 matching `_cache/mesh.npz`/`field.npz`가 없어 historical TSDF
`ExtractedSurface` replay 자체는 실행할 수 없었다. 이 사실을 숨기지 않고 carrier
eligibility에 반영했다.

## 3. Renderer-event contract

Renderer event는 camera pixel에서 renderer median depth로 복원된 observation
evidence다. XYZ, median-event depth, source camera/pixel, event normal,
camera provenance는 직접 존재한다.

이것은 physical-sheet identity, visible topology, local region ownership,
boundary ownership을 증명하지 않는다. 특히 1527은 위 human review 값으로만
별도 기록하고 blacklist하지 않았다.

## 4. Canonical Gaussian-region contract

canonical Gaussian node는 position에 더해 covariance 또는 log-scale/rotation,
covariance frame, structural reliability, manifold affinity graph, unique stable
ID를 갖는다. `form_surface_regions`는 same-surface/crease/parallel-separate/
rejected 관계와 consensus/bridge/path semantics로 node-region 및
member/core/attached/rejected identity를 만든다.

renderer event의 normal이나 event ID를 이 Gaussian-region 의미로 재해석하지 않았다.

## 5. Raw Visible Surface contract

현재 이용 가능한 파일은 `point_cloud.ply`이고 header는 binary little-endian,
`element vertex 1212365`뿐이다. `element face`가 없으며 vertex properties는
XYZ와 f_dc/opacity/scale/rot 계열이다.

따라서 실제 entity는 **vertex-only renderer-median point artifact**다. WL127
source의 `ExtractedSurface` dataclass는 `vertices`, `faces`,
`vertex_support_count`, `vertex_field_value`, `h`를 정의하지만, 그 matching
typed mesh replay artifact는 현재 없다.

계약 판정:

- XYZ: `DIRECTLY_PRESENT`
- mesh face, edge/face adjacency, component ID, boundary edge: `ABSENT`
- zero-set sample / mesh vertex: 현재 artifact에 대해서는 `SEMANTICALLY_DIFFERENT`
- source TSDF cell, camera/event provenance, ownership: `ABSENT`
- opacity/scale을 support/confidence로 해석: `SEMANTICALLY_DIFFERENT`

## 6. Evidence → Raw Surface provenance

WL127 source에는 renderer observation → projective TSDF → all-eight-corner masked
marching-cubes → seam-welded vertices/faces라는 deterministic construction이
있다. 그러나 `ExtractedSurface`와 mesh PLY export에는 renderer event ID, source
camera, source pixel, TSDF cell key를 mesh element에 보존하는 필드가 없다.

결과:

- observation → TSDF: source상 `DETERMINISTICALLY_DERIVABLE`
- TSDF → zero-set: source상 `DETERMINISTICALLY_DERIVABLE`
- zero-set → mesh vertex/face: source상 `DETERMINISTICALLY_DERIVABLE`
- mesh element → renderer event/camera set: `ABSENT`
- event 1527 → Raw Surface element: `NOT MAPPABLE UNDER EXISTING CONTRACT`

1527을 삭제하거나 nearest-distance attribution으로 억지 매핑하지 않았다.

## 7. Topology carrier eligibility

available PLY는 face/edge graph가 없으므로 local connectivity, region identity,
physical boundary, pre-fit support domain을 소유할 수 없다. matching TSDF mesh가
source에 정의되어 있다는 사실만으로 현재 point artifact를 topology carrier로
승격하지 않았다.

분류: **INELIGIBLE**

secondary gaps: `RAW_SURFACE_PROVENANCE_GAP`, `PHYSICAL_SHEET_MEMBERSHIP_GAP`.

## 8. Physical-sheet connectivity result

요구된 native connectivity 진단 결과는 vertex `1,212,365`, face `0`이다. 따라서
connected component 수, component size distribution, open/closed boundary 수는
`NOT_DEFINED_WITHOUT_FACES` 또는 `NOT_MEANINGFUL_WITHOUT_FACES`다.

physical-sheet 결과는 **E. INSUFFICIENT_EVIDENCE**다. tabletop이 side/vase/
background/lower geometry와 연결되는지 또는 한 sheet가 fragmentation되는지를
이 artifact의 native connectivity만으로는 판정할 수 없다. 각 point를 component로
보는 것은 새 adjacency를 발명하는 것이므로 하지 않았다.

## 9. Boundary-First eligibility

carrier에 face topology, physical-sheet ownership, source provenance, deterministic
boundary가 없으므로 pre-fit Boundary First gate를 열 수 없다.

- STOP: `RAW_SURFACE_BASELINE_REPLAY_UNAVAILABLE`
- STOP: `RAW_SURFACE_PROVENANCE_GAP`
- STOP: `PHYSICAL_SHEET_MEMBERSHIP_GAP`
- Candidate D eligible: `false`

## 10. Conditional Candidate D

실행하지 않았다. Candidate D는 모든 eligibility gate가 통과한 뒤에만 허용된다.
새 mesh connectivity, physical-sheet membership, camera correspondence,
normal/distance/KNN rule, component filter를 구현하지 않았다.

## 11. ID 1527 trace

1527은 WL149 union과 `v_min` owner로 보존된다. human review는
`CLEAR_NOT_ON_INTENDED_SURFACE`이며 blacklist는 `false`다. 현재 Raw Visible
Surface artifact에는 event/camera/TSDF-cell provenance가 없어, 1527이 intended
sheet·another sheet·no surface 중 어디에 기여하는지 **기존 계약으로는 매핑할 수
없다**.

## 12. Synthetic results

`NOT_RUN`. Candidate D가 ineligible이므로 synthetic fixture를 실행하면 missing
membership/provenance semantics를 임의로 채우게 된다.

## 13. Real-scene quantitative result

Candidate D metric은 `NOT_RUN`이다. available artifact에 대해서만 vertex/face
accounting을 수행했다.

- vertices: `1,212,365`
- faces: `0`
- edge/component/boundary accounting: face graph 부재로 정의 불가
- event-to-surface provenance: `ABSENT`
- event 1527: `NOT MAPPABLE UNDER EXISTING CONTRACT`

## 14. Real-scene qualitative result

Candidate D geometry가 없으므로 새 qualitative review/render는 `NOT_RUN`이다.
WL145/WL149 existing review는 historical output으로만 유지했다.

## 15. Architecture verdict

**INELIGIBLE_CARRIER**

현재 available Raw Visible Surface 표현은 topology carrier가 아니라 vertex-only
point artifact다. source에 ExtractedSurface mesh 구현은 있지만 matching replay가
없고, mesh element와 renderer evidence를 잇는 provenance 및 physical-sheet
membership 계약도 없다.

## 16. Retained / Rejected / Open

- Retained: WL139–WL151, 1586 union, event 1527, renderer semantics, canonical
  TSDF/region/boundary source, available WL127 PLY.
- Rejected: point cloud를 mesh topology로 승격, event attribution heuristic,
  connectivity repair, component filtering, 1527 삭제, Candidate D 구현.
- Open: matching WL127 ExtractedSurface mesh artifact 재생성/freeze, TSDF-cell/event
  provenance 보존, physical-sheet membership, local boundary ownership.

## Intent alignment

WL151 baseline을 exact replay했고, missing contract를 새 heuristic로 채우지 않았다.
geometry, canonical production, WL139–WL151 historical artifact, renderer semantics는
변경하지 않았다.

## Implementation fidelity

WL127 commit `943a764`의 extraction/mesh source, WL149–WL151 baseline, available
PLY header/hash를 읽기 전용으로 감사했다. `Candidate D`, synthetic, real-scene
Candidate D replay, new visualization은 실행하지 않았다. 새 threshold, KNN,
normal-angle/curvature/distance membership, largest-component rule, appearance,
continuation, Occluded Surface는 추가하지 않았다.

## Architecture result

`INELIGIBLE_CARRIER`이며 secondary result는 `RAW_SURFACE_PROVENANCE_GAP`과
`PHYSICAL_SHEET_MEMBERSHIP_GAP`이다. 이번 batch의 missing contract는 정확히
face/edge topology, event/TSDF-cell→surface-element provenance, physical-sheet/local
region identity, boundary ownership이다.

산출물: `output/152_visible_surface_carrier_contract_audit/` 및
`temp/152_visible_surface_carrier_contract_audit/`.

Focused tests: `4 passed`.
