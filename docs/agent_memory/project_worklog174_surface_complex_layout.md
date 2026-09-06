---
name: project_worklog174_surface_complex_layout
description: "W174 결과 — native 연결성이 표면 연결성을 일방향 과다추정, W173 witness는 chart 접힘으로 귀속(MIXED_ATTRIBUTION)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 281e7428-5cda-4d5d-90d9-107197bb75ec
  modified: 2026-09-06T11:38:20.489Z
---

W174 (2026-09-06, `arch/2dgs-coverage-first-surface`): W172 tabletop support의 reference zero-set surface complex를 복원해 topology 실패를 귀속했다. 판정 **`MIXED_ATTRIBUTION`**.

**핵심 발견 (재측정 없이 재사용 가능):**

- Reference surface: triangle 31,149 / welded vertex 17,177 / **triangle-connected component 132개**, 최대 성분 91.02%(support row 90.55%). Non-manifold **edge 0개**, vertex 23개. 427 cell이 다중 patch.
- **Native 6-face 연결성은 표면 연결성을 일방향으로 과다추정한다.** 30,821 인접 쌍 중 1,595쌍(5.18%)이 표면 비인접, 그중 262쌍은 다른 component. **역방향은 0.** 이것이 "native component 1개"인 support가 실제로는 132조각인 이유다.
- **W173 height witness(row 4043/4051) 규명**: native cell L1 거리 46으로 멀리 떨어져 있음에도 **같은 triangle component에서 102 edge step으로 연결**. path/chord 비 1.74, 상판→옆면 연속 표면. 판정 `CHART_COLLAPSE_SAME_SURFACE` — support가 이질 구조를 섞은 게 아니라 planar chart가 접은 것.
- Chart distortion: 6,609 occupied bin 중 519개가 서로 다른 component 병합, height span median 0.156h / max 32.89h(W173 수치 재현).

**Why:** W173이 "정상 단일 sheet 확정을 막는 evidence"로 남긴 witness의 성격이 확정됐고, W154 exactly-one-loop gate 외에 native-connectivity 추상화 자체가 별도 실패 지점임이 드러났다.

**How to apply:** 코드는 `devtools/demo/worklog_174_surface_complex.py`(유틸), `..._reference_surface_attribution.py`(분석), `..._review_exports.py`(시각화), `..._reextraction_feasibility_probe.py`(타당성 검증). Vertex welding은 global lattice-edge exact integer key(공간 tolerance 아님) — 인접 셀 병합/원거리 셀 비병합을 합성 fixture로 먼저 검증할 것. 삼각형 부재 배경은 [[project_zero_set_triangles_never_persisted]].

**미해결:** 소수 component가 물리적 별개 물체인지 관측 결손인지 미구분. Per-hole causal provenance 여전히 없음(W173과 동일).

관련: [[project_zero_set_triangles_never_persisted]], [[project_visible_nurbs_evidence_contract_closure]]
