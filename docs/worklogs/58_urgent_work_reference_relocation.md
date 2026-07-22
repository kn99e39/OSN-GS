# Urgent_Work 계획 문서 참조 경로 정리

## 수행 작업

- 루트에서 `docs/Urgent_Work/`로 이동된 계획 문서 네 개를 기준으로 저장소 전체 참조를 검색했다.
- `TODO.md`, `docs/README.md`, benchmark README, Python module/test docstring, 관련 worklog의 `Final Boundary First`, `Phase 5`, `Voxel Driven` 계획 문서 참조를 새 경로로 바꿨다.
- `OSN_GS_Phase4_Hardening_Plan.md`는 이미 완료 처리되어 삭제된 역사 문서이므로, 해당 이름을 언급하는 기존 worklog는 삭제 상태를 설명하는 역사 기록으로 유지했다.

## 결과

- 루트 문서에서의 링크는 `docs/Urgent_Work/...`를 사용한다.
- `docs/` 하위 문서는 `Urgent_Work/...` 또는 `../Urgent_Work/...` 상대 경로를 사용한다.
- benchmark README는 `../docs/Urgent_Work/...`를 사용한다.
- 코드 및 테스트 docstring은 repository-relative `docs/Urgent_Work/...`를 사용한다.

## 평가

- 모든 Markdown의 `Urgent_Work` 링크를 상대 경로로 해석해 검사했고, 누락된 대상은 0개였다.
- 변경한 Python 파일은 `py_compile`을 통과했다.

## 남은 위험

- 향후 계획 문서를 다시 이동하거나 이름을 바꾸면 root/docs/worklogs/benchmark/source docstring의 경로를 같은 방식으로 함께 갱신해야 한다.