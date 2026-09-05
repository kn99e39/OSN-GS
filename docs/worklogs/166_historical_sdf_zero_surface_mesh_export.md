# Worklog 166 — Historical SDF/TSDF Zero-Level Surface Mesh Export

## 상태

완료 — `COMPLETE_HISTORICAL_ZERO_SURFACE_EXPORT`

## 작업 내용

- W153의 `replay_cache/typed_extracted_surface.npz`를 역사적 source로 확정했다. 이 배열은 기준 commit `943a764`의 renderer-median seed → projective TSDF → all-eight-corner authoritative/sign-changing cell → Lewiner Marching Cubes → `h*1e-6` seam-only weld 경로에서 생성된 typed `ExtractedSurface`이다.
- `devtools/demo/worklog_166_historical_sdf_zero_surface_mesh_export.py`를 추가했다. raw NPZ는 재직렬화하지 않고 byte-for-byte copy하며, OBJ는 source vertex/face row order와 connectivity를 그대로 1-based triangular OBJ로 쓴다.
- source component accounting은 W153의 faces-adjacency-only report를 count와 array shape로 대조해 재사용했다. component 선택, merge/split, hole repair, boundary closure, filtering은 실행하지 않았다.
- `tests/test_worklog_166_historical_sdf_zero_surface_mesh_export.py`에 OBJ round-trip, degenerate/repeated face 보존, occlusion semantics 비검증 계약을 고정했다.

## 결과 및 평가

- 출력: [`output/166_historical_sdf_zero_surface_mesh_export`](../../output/166_historical_sdf_zero_surface_mesh_export/)
- OBJ: `historical_sdf_zero_surface.obj`, 약 2.98 GB
- raw arrays: `historical_sdf_zero_surface_raw.npz`, source NPZ와 SHA-256 동일
- source/export geometry: `28,694,040` vertices, `45,116,659` triangular faces
- OBJ 전수 round-trip: vertex/face count 일치, coordinate mismatch `0`, connectivity mismatch `0`, max absolute coordinate error `0.0`, world bounds 일치
- native topology accounting: `582,646` connected components, `73,751,737` unique edges, `12,153,565` boundary edges, `35` non-manifold edges, `28` degenerate-index/zero-area faces. 이 상태는 historical geometry로 보존했고 수정하지 않았다.
- FBX는 기존 exporter가 없어 `FBX_EXPORT_UNAVAILABLE`로 명시했다. 무거운 외부 의존성은 설치하지 않았다.
- focused verification: `3 passed` (`--basetemp=C:\tmp\osn_gs_w166_pytest`). 첫 실행의 본문은 통과했으나 pytest의 기존 Windows temp cleanup에서 `WinError 5`가 발생해 별도 basetemp로 최종 재실행했다.

## 의미 경계와 남은 위험

- 이 산출물은 외부 viewer용 historical observed-visible reconstruction inspection artifact이며, watertight object mesh가 아니다.
- OBJ/NPZ export는 Occlusion semantics, physical hidden-surface identity, 또는 Observed/Occluded truth를 검증하지 않는다.
- W153 결과는 원본 historical typed array의 byte hash가 아니라 source/input-identifiable semantic replay이다. 이 provenance 한계를 report와 README에 남겼다.
- `docs/current_framework.md`는 current production pipeline을 바꾸지 않는 diagnostic/export-only 작업이므로 수정하지 않았다.
