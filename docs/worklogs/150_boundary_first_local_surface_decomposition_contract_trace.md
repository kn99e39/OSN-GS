# Worklog 150 — Boundary First / Local Surface Decomposition 계약 추적

## 상태

완료. 이번 배치는 WL139/WL145/WL148/WL149와 canonical 코드를 읽기 전용으로
감사한 격리 진단이며, geometry·PCA·NURBS·support·renderer 동작을 수정하지
않았다.

## 1. 현재 아키텍처 질문

WL149의 전역 chart에서 off-surface renderer event 1527이 왜 `v_min` owner가
되었는지를 Boundary First와 Local Surface Decomposition의 실제 실행 경로로
추적했다.

## 2. Human review 갱신

사람이 제공한 단일 외부 검토 결과를 다음과 같이 별도 기록했다.

`HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE`

Event 1527은 `DSC08003.JPG`, pixel `(259,169)`, source-local row `443`,
union row `1527`이다. 다른 sparse event에는 이 판정을 일반화하지 않았다.

## 3. 의도된 Boundary First 계약

canonical `construct_visible_nurbs_from_gaussians`는 region topology 뒤에
world-space boundary candidate를 만들고, directed compatibility/order/status를
거쳐 eligible observed outer component만 materialize한다. adapter는 ordered
closed loop와 region-core interior를 **fit 전에** 입력으로 받으며 open/branch/
ambiguous boundary에 synthetic rectangle을 만들지 않는다.

따라서 boundary owner는 전역 PCA extrema가 아니라 region-owned observed
boundary component이다. 후속 occupancy mask는 이 pre-fit 계약과 별개다.

## 4. 의도된 Local Surface Decomposition 계약

bounded construction population이 `form_surface_regions`를 통과하며
`node_region_id`와 `SurfaceRegionCandidate.region_id/member IDs`를 만든다.
이 identity는 `source_region_id`와 supporting source IDs를 거쳐 local fit 및
materialization까지 보존된다. 서로 다른 local surface를 한 전역 PCA union으로
합치는 것은 canonical 계약의 local ownership 수단이 아니다.

## 5. WL139 representative dataflow

WL145는 WL127 mesh를 provenance/hash 확인용으로만 읽고, 수동 image polygon에서
세 camera의 renderer `depth_median` event cloud를 독립적으로 만든다. clear
candidate의 754 + 330 + 502개를
`clean_points = np.concatenate([cloud.points for cloud in clouds], axis=0)`로
1586개 XYZ 배열에 합친다. 이 지점 전후에 canonical region ID, local-surface ID,
boundary owner가 생성되지 않는다.

그 뒤 `_pca_chart_config`가 pooled XYZ에 global PCA를 적용하고
`world_xyz @ axes`의 min/max로 rectangular domain을 정한다. WL139
`fit_physical_chart_surface`는 이 physical chart 좌표와 normal field를 사용해
대표면을 fit한다. WL148 B는 이미 생성된 대표면의 all-four support cell만
나중에 materialize하고, WL149는 이를 변경 없이 replay하여 extrema 영향만
계산했다.

## 6. Event 1527 end-to-end trace

- renderer provenance: camera, pixel, median depth, XYZ, normal은 PRESENT;
  primitive/contributor ID는 ABSENT.
- physical-sheet membership: manual control label은 PRESENT하지만 executable
  canonical membership는 없고, human review 값은
  `CLEAR_NOT_ON_INTENDED_SURFACE`이다.
- Visible Surface topology identity, region ownership, boundary ownership:
  ABSENT.
- PCA 입력 포함: PRESENT. `u=0.8083508388`, `v=-0.5984138976`,
  `n=1.4449360893`; `v_min` owner는 PRESENT/true이다.
- frozen WL149 fixed-axis v-span reduction은 `1.3156265861`, rectangular area
  reduction은 `2.2730185370`이다. 이는 attribution 진단값이며 fit/extent를
  재선택하는 기준으로 사용하지 않았다.

## 7. Pre-fit vs post-fit domain control

이 경로의 실제 구조는 `global PCA rectangle → WL139 representative fit →
WL148 B occupancy/materialization`이다. 즉 WL148 B는 post-fit control이다.

**WL148 B does NOT by itself restore Boundary First semantics.**

## 8. 계약 손실 / bypass 위치

가장 이른 bypass는 WL145 clean-oracle path가 canonical constructor,
`form_surface_regions`, boundary ordering, materialization adapter를 WL139 fit
전에 호출하지 않는 지점이다. 가장 결정적인 identity flattening은 세 cloud를
pooled `clean_points` XYZ 배열로 concatenate하는 지점이며, 이어서 global PCA가
event 1527을 domain owner로 허용한다.

따라서 Boundary First가 이 WL139/WL145 경로에서 active하게 실패한 것이 아니라,
clean-oracle representative control이 그 계약을 우회했다. canonical source에
있는 pre-fit gate 자체를 이 배치에서 재활성화하거나 수정하지 않았다.

## 9. 의도 vs 실제 아키텍처

```text
의도: evidence → local regions → region ownership → observed boundary
      → local chart/support → pre-fit local NURBS → supported materialization

실제: manual polygon → per-view renderer events → pooled 1586 XYZ union
      → global PCA/extrema rectangle → WL139 fit → WL148 post-fit support
      → WL149 attribution replay
```

차이는 local decomposition/boundary owner가 fit 전에 없고, 전역 PCA rectangle이
그 자리를 대신한다는 점이다.

## 10. Historical motivation match

off-surface event가 representative chart extent를 소유하는 현상은 Boundary
First와 local region ownership이 예방하려던 cross-surface/domain ownership
failure와 직접 맞는다: **DIRECTLY TARGETED FAILURE**. 다만 WL139 clean-oracle
global pooling 자체는 그 canonical 계약을 검증하는 실험이 아니므로
**RELATED BUT NOT DIRECTLY TARGETED**로도 별도 분류했다. 이 결과를 canonical
full-scene pipeline failure로 확대하는 것은 **OUTSIDE ORIGINAL CONTRACT**이다.

## 11. Architecture verdict

**B. ARCHITECTURE_BYPASS**

WL139/WL145 path는 Boundary First/Local Surface Decomposition의 executable
ownership contract를 fit 전에 호출하지 않았고, pooled XYZ → PCA extrema →
rectangle abstraction을 사용했다. 이것은 현재 rectangle을 고치거나 event
1527을 삭제했다는 의미가 아니다.

## 12. Retained / Rejected / Open

- Retained: WL139/WL145/WL148/WL149 artifacts, 1586 event union, event 1527,
  PCA axes, representative XYZ/normals, support mask/all-four relation,
  canonical renderer/checkpoint/Candidate B/production continuation.
- Rejected: event filtering, general sparse-event rejection, robust PCA, chart
  clipping, refit, support 변경, continuation, Occluded Surface 구현.
- Open: 현재 renderer evidence에 대한 physical-sheet membership과 region identity의
  publishable 정의, continuation extent/termination/confidence, historical
  Boundary First 의미를 재사용할지에 대한 별도 계약 검토.

## Intent alignment

수동 human update는 event 1527 하나에만 적용했다. 출력은 source/history trace와
frozen replay이며, geometry·실험 parameter·canonical behavior는 변경하지 않았다.

## Implementation fidelity

소스 함수/라인, historical commit, WL149 report/JSON/NPZ를 산출물에 기록했다.
WL149 baseline은 event 1586개, union hash
`79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78`, representative
`3840x3`, support `314`, all-four cells `211`로 exact replay했다. 변경된 것은
새 `devtools/demo` trace 모듈, focused tests, 번호가 붙은 output/temp 복사본,
이 worklog와 README 링크뿐이다.

## Architecture result / 남은 위험

이번 결과는 source/history provenance 질문을 닫았지만 repair batch가 아니다.
canonical region/boundary semantics가 현재 renderer evidence에 물리적으로
적합한지는 아직 검토 대상이며, 이 진단만으로 Occluded Surface 해결이나
architecture success를 주장하지 않는다.

산출물: `output/150_boundary_first_local_surface_decomposition_contract_trace/`
및 보존 복사본 `temp/150_boundary_first_local_surface_decomposition_contract_trace/`.
Focused test: `4 passed`.
