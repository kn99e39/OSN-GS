---
name: project_zero_set_triangles_never_persisted
description: "W154/W171/W172 계보에는 zero-set 삼각형이 저장된 적 없음 — 셀당 1점(surface-nets 축약), 재추출은 frozen corner_values로 가능"
metadata: 
  node_type: memory
  type: project
  originSessionId: 281e7428-5cda-4d5d-90d9-107197bb75ec
  modified: 2026-09-06T11:38:00.354Z
---

W154 계보(`output/confirmed/154_gaussian_region_owned_tsdf_boundary_first_nurbs/candidate_f_tsdf_surface_samples.npz`)에는 **zero-set 삼각형이 존재한 적이 없다.** 저장된 키는 `source_cell_keys / cell_indices / world_xyz / normals / corner_values / corner_support_count`뿐이고, 셀당 정확히 1점이다.

`extract_tsdf_zero_surface_samples()` (osn_gs/surface/torch_gaussian_region_owned_tsdf.py)는 marching cubes가 아니라 12개 cube-edge 교점의 **평균 1점**을 내는 surface-nets 축약형이다. 저장소에 MC triangle table이 없고, 산출물·코드 양쪽이 `"mesh_intermediate": false`로 명시한다.

**Why:** W173이 §9에서 "native ordered 1D loop population이 없다"고 보고한 것, W174 지시서가 "frozen raw zero-set triangle geometry"를 전제한 것 모두 이 사실에서 비롯됐다. 앞으로도 "native 표면 위상"을 요구하는 지시가 오면 같은 벽에 부딪힌다.

**How to apply:** 삼각형이 필요하면 저장된 `corner_values`(셀당 8코너)에서 **재추출**한다. 원본 TSDF field를 다시 로드할 필요 없다. 각 셀을 자기만의 2×2×2 block으로 marching cubes에 넣으면 ownership이 기하가 아니라 구조적으로 exact해지고, distance/radius/threshold 매칭이 불필요하다 — W174에서 15,189셀 전부 성공, 단위셀 이탈 0건, 저장 world_xyz와 median 0.012h 일치로 검증됨. 구현은 [[project_worklog174_surface_complex_layout]] 참조. 단 이는 "얼어 있는 것을 읽는" 것이 아니라 "같은 field에서 새로 계산"하는 것이므로 보고서에 그 차이를 명시할 것.

관련: [[project_visible_nurbs_evidence_contract_closure]], [[project_worklog174_surface_complex_layout]]
