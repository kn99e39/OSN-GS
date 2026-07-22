# Worklog 59: Proxy-Based Surface Decomposition Stage 0 기준선 동결

날짜: 2026-07-22

상태: **Stage 0 완료. Production 경로 변경 없음.**

## 목표

`docs/Urgent_Work/OSN_GS_Proxy_Based_Surface_Decomposition_Impl_Plan.md`의 최초 실행 범위에 따라 현재 Phase 1 component builder와 Phase 2 boundary/topology 인터페이스를 감사하고, proxy 진단 전 기준선을 고정했다.

## 수행 작업

- 현재 production 흐름을 `build_voxel_gaussian_hierarchy -> build_surface_components -> extract_component_boundary -> classify_boundary_result`로 확인했다.
- Phase 1 출력 계약은 `components`, `leaf_component_id`, `edge_decisions`, `component_boundary_faces`, `config`이며, Phase 2가 직접 사용하는 component 필드는 `component_id`, `member_leaf_ids`, `gaussian_indices`, `centroid`, `tangent_u`, `tangent_v`임을 확인했다.
- 기본값(`count=600`, `seed=0`, voxel count 10/150, depth 6, normal 40도, offset ratio 0.5, boundary resolution 64, density threshold 3.0)으로 주요 synthetic scene의 component 수와 topology를 기록했다.
- `curved_annulus`의 mergeable leaf/AABB/component provenance와 서로 다른 component 사이의 최근접 leaf pair를 기록했다.

산출물: `artifacts/proxy_decomposition_baseline.json`

## 기준선 결과

| scene | component 수 | edge reason | Phase 2 topology |
|---|---:|---|---|
| plane | 1 | merged 15 | disk_like |
| planar_hole | 1 | merged 15 | annulus |
| planar_hole_offcenter | 1 | merged 10 | annulus |
| planar_hole_elliptical | 1 | merged 15 | annulus |
| planar_hole_density_gradient | 1 | merged 15 | annulus |
| curved_annulus | 2 | merged 14 | disk_like, complex |
| mild_curved_sheet | 1 | merged 12 | annulus |
| crease | 2 | merged 8, normal 4 | disk_like, disk_like |
| close_parallel_sheets | 2 | merged 8, offset 4 | disk_like, disk_like |
| density_gradient | 1 | merged 4 | disk_like |

## 진단 결과

- `curved_annulus`에서 생성된 14개 Phase 1 candidate edge는 전부 `merged`였다. threshold reject가 component split의 원인이 아니다.
- 서로 다른 두 component 사이에 AABB distance가 0이면서 face contact가 없는 pair가 존재한다. 최근접 예시는 `r07-r52`(centroid distance 0.359), `r05-r50`(0.455), `r05-r52`(0.497)다.
- 따라서 현재 과분할은 local plane compatibility가 아니라 axis-aligned face-contact candidate 누락으로 재현된다.
- `mild_curved_sheet`는 Phase 1에서 이미 component 1개지만 Phase 2에서 `annulus`로 오분류된다. 이 실패는 proxy-based Phase 1만으로 해결됐다고 판정할 수 없으며 Phase 2 loop/topology 문제로 별도 추적해야 한다.

## 평가

Stage 0은 계획 문서의 문제 정의를 실제 production 기본 경로에서 재현했다. Stage 1 quadratic proxy는 production component membership을 바꾸지 않고, smooth curved pair와 crease/parallel/disconnected negative control의 분리력만 측정해야 한다.

## 남은 위험

- AABB 접촉은 candidate 생성 근거일 뿐 실제 surface connectivity 증거가 아니다. Stage 2에서 support gap과 layer consistency가 필요하다.
- Phase 1 component 복원, Phase 2 topology 복원, annulus routing, NURBS fitting을 하나의 성공 지표로 합치면 안 된다.
- 현재 worktree는 다른 작업의 미커밋 변경을 포함하므로 production 파일은 Stage 1에서 수정하지 않는다.
