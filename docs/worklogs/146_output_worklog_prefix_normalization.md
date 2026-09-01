# Worklog 146 — 산출물 폴더 Worklog 번호 정규화

## 작업

실제 `output/` 아래의 Worklog 산출물 루트와 `output/confirmed/` 아래의
검토 완료 산출물 루트에 대응 Worklog 번호를 폴더명 맨 앞에 붙였다.
번호는 기존 저장소 규약에 맞춰 세 자리 숫자를 사용했다.

## 적용 결과

- `output/confirmed/138_scale_separated_visible_surface_representative`
- `output/confirmed/139_physical_chart_surface_representative`
- `output/confirmed/140_real_gaussian_scene_surface_validation`
- `output/141_oracle_single_surface_support_appearance_evidence`
- `output/142_multi_view_support_lifting_projection_depth_attribution`
- `output/143_multi_view_support_lifting_depth_semantics_evidence_aggregation`
- `output/144_per_view_renderer_surface_correspondence_physical_sheet_oracle_audit`
- `output/145_genuine_physical_sheet_oracle_clean_support_representative_audit`

WL127의 `127_...` 및 WL120의 `120_...` 폴더는 이미 번호가 있어 유지했다.
`output/confirmed/`는 검토 상태를 나타내는 컨테이너일 뿐 Worklog 산출물이
아니므로 이름을 바꾸지 않았다.

## 보존 범위와 경로 갱신

`output/osn_gs_scene/`와 `output/arch_2dgs_coverage_first_surface/`는
공유 학습 checkpoint 루트이므로 기존 규약대로 이동·개명하지 않았다.
`devtools/demo`의 WL128–145 기본 출력 경로와 최근 Worklog/README의 참조만
새 경로로 갱신했으며, geometry·metrics·canonical production code는
변경하지 않았다. WL127 입력 경로는 현재 확인된 위치인
`output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/`를 사용하도록
맞췄다.

## 검증

- 실제 폴더 이동 전 source 존재 및 target 미존재를 확인했다.
- 폴더 내부의 PNG/PLY/NPZ/JSON 파일은 이동만 했고 내용은 수정하지 않았다.
- 모든 demo 기본 output root와 최근 문서 경로에 번호 prefix가 반영되었는지
  정적 검색으로 확인한다.

## 남은 위험

`docs/agent_memory/`는 과거 세션 기록 보존 영역이라 기존 경로 문자열을
수정하지 않았다. 과거 기록의 경로는 현재 폴더 구조를 설명하는 최신 참조가
아니며, 실행 가능한 최신 경로는 `docs/output_folder_conventions.md`를 따른다.
