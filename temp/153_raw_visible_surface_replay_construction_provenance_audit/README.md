# Worklog 153 — WL127 Raw Visible Surface replay / provenance audit

이 폴더는 WL127의 vertex-only point PLY와 별도로, 기준 커밋 `943a764`의
typed `ExtractedSurface` replay를 보존하는 진단 전용 산출물이다.

- canonical renderer/checkpoint/161 cameras: 변경하지 않음
- WL152 baseline: event union 1586, event 1527 보존, point PLY 1,212,365 vertices / faces 0
- replay: `replay_cache/` 아래 `field.npz`, `renderer_median_depth_maps.npz`, `typed_extracted_surface.npz`
- provenance: 기존 typed contract에는 per-event/camera/source-cell sidecar가 없음을 명시
- physical-sheet membership / NURBS / connectivity repair: 수행하지 않음

`architecture_verdict.json`과 `raw_visible_surface_replay_construction_provenance_audit_report.json`이
최종 판정이다. PNG는 opaque display-only vertex preview이며 metrics/geometry를 바꾸지 않는다.
