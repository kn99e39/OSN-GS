# Worklog 148 materialization audit

이 output은 WL145 exact baseline을 먼저 검증한 뒤 동일 frozen WL139 representative에 대해 full-domain(A)와 frozen support-constrained(B)를 비교한 격리 진단이다.

- baseline: 1586 event union, 314/3840 support vertices, exact support-mask hash
- B cell rule: existing support vertices 네 개가 모두 True인 cell만 materialize
- A/B geometry: 동일한 frozen representative XYZ/normals
- `chart_space_96x40_diagnostic.png`는 smoothing/fill 없는 occupancy semantics를 직접 표시한다.
- 실제 장면의 A-F overlay와 mandatory Original Scene/Observed-Occluded pair는 `real_scene_camera_review/` 및 `mandatory_gaussian_visualization_pair/`에 있다.
